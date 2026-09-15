"""Opt-in crowd contribution writer for IDDSI-Open.

This module is the storage half of the crowd pipeline: given media files that
the uploader explicitly opted in to contribute, it

1. enforces configurable size caps *before* reading media content,
2. strips EXIF (and all other image metadata) from JPEG/PNG/WebP images by
   re-encoding the pixel data with Pillow,
3. runs an injectable face/text screening hook on each sanitized image
   (default: an honest no-op that reports nothing and clears nothing),
4. writes one immutable quarantine bundle plus an append-only JSONL index
   with a manifest row whose ``source`` is always ``"real-crowd"`` (never
   ``"real"`` — the frozen release schema's physically-tested enum value), and
5. optionally mirrors the bundle to the *private* Hugging Face dataset repo
   ``lingualeap/iddsi-open-v0`` under a ``crowd/`` prefix.  The push is off
   by default, never creates a repository, and never makes anything public.

Like ``crowd/app.py`` this is a pre-ingestion **crowd staging schema**
(``crowd-staging-1.0.0``): rows here are weak-label quarantined events, not
conforming release rows.  No app.py code is imported or modified.
"""

from __future__ import annotations

import hashlib
import io
import json
import mimetypes
import os
import shutil
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".webm"}

DEFAULT_HF_REPO_ID = "lingualeap/iddsi-open-v0"
DEFAULT_HF_PATH_PREFIX = "crowd"


class CrowdStorageError(ValueError):
    """Base class for safe, user-displayable storage errors."""


class MediaTooLarge(CrowdStorageError):
    """Raised when a file or submission exceeds its configured byte cap."""


class UnsupportedMedia(CrowdStorageError):
    """Raised for unknown formats or media Pillow cannot decode."""


class ScreeningRejection(CrowdStorageError):
    """Raised when the screening hook reports issues for sanitized media."""


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 of ``data`` (content hash)."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file's bytes, streaming in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class UnsupportedImage(CrowdStorageError):
    """Raised when an image file cannot be decoded for sanitization."""


def strip_exif(source: Path) -> bytes:
    """Return sanitized image bytes with EXIF/metadata removed.

    The image is decoded with Pillow and re-encoded from its pixel data, so
    EXIF (JPEG APP2/Exif, PNG eXIf), XMP, ICC and comment segments are all
    dropped.  Non-image files and undecodable images raise ``UnsupportedMedia``
    rather than being passed through unsanitized.
    """
    try:
        from PIL import Image  # noqa: PLC0415 - optional at import time
    except ImportError as exc:  # pragma: no cover - depends on environment.
        raise CrowdStorageError("Pillow is required to sanitize images.") from exc

    source = Path(source)
    suffix = source.suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        raise UnsupportedMedia(
            f"Only {', '.join(sorted(IMAGE_SUFFIXES))} images can be sanitized; got {suffix or '(none)'}."
        )
    try:
        with Image.open(source) as image:
            image.load()
            fmt = (image.format or "").upper()
            if fmt not in {"JPEG", "PNG", "WEBP"}:
                raise UnsupportedMedia(f"Unsupported image format: {fmt}")
            if fmt == "JPEG" and image.mode not in {"RGB", "L", "CMYK"}:
                image = image.convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, format=fmt)
    except UnsupportedMedia:
        raise
    except Exception as exc:
        raise UnsupportedImage(f"Could not decode image {Path(source).name}: {exc}") from exc
    return buffer.getvalue()


# --- Screening hook -------------------------------------------------------
#
# A screening hook receives the path of one *sanitized* image file and returns
# a sequence of issue strings (empty = no issues found).  The default is an
# honest no-op: it implements neither face detection nor OCR, so it reports no
# issues but also never marks a row release-eligible.
ScreeningHook = Callable[[Path], Sequence[str]]


def noop_screening_hook(image_path: Path) -> list[str]:
    """Default no-op screening hook: no detector, no issues, no clearance."""
    del image_path
    return []


@dataclass(frozen=True)
class StorageSettings:
    """Configuration for the contribution writer (HF push defaults OFF)."""

    data_dir: Path
    max_media_bytes: int = 10 * 1024 * 1024
    max_total_bytes: int = 80 * 1024 * 1024
    hf_push_enabled: bool = False
    hf_repo_id: str = DEFAULT_HF_REPO_ID
    hf_path_prefix: str = DEFAULT_HF_PATH_PREFIX

    @classmethod
    def from_env(cls) -> "StorageSettings":
        default_data_dir = Path(__file__).resolve().parent / "data"
        enabled = os.getenv("CROWD_STORAGE_HF_PUSH_ENABLED", "false").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        return cls(
            data_dir=Path(os.getenv("CROWD_STORAGE_DATA_DIR", str(default_data_dir))).expanduser(),
            max_media_bytes=int(os.getenv("CROWD_STORAGE_MAX_MEDIA_BYTES", 10 * 1024 * 1024)),
            max_total_bytes=int(os.getenv("CROWD_STORAGE_MAX_TOTAL_BYTES", 80 * 1024 * 1024)),
            hf_push_enabled=enabled,
            hf_repo_id=os.getenv("CROWD_STORAGE_HF_REPO_ID", DEFAULT_HF_REPO_ID),
            hf_path_prefix=os.getenv("CROWD_STORAGE_HF_PREFIX", DEFAULT_HF_PATH_PREFIX),
        )


@dataclass(frozen=True)
class MediaSpec:
    """One media file offered for contribution, with its capture role."""

    path: Path
    capture_role: str


def _hf_api_factory() -> Any:
    """Lazily construct a HfApi client; never called unless push is enabled."""
    try:
        from huggingface_hub import HfApi  # noqa: PLC0415 - optional dependency
    except ImportError as exc:  # pragma: no cover - depends on environment.
        raise CrowdStorageError(
            "huggingface_hub is required when the crowd HF push is enabled."
        ) from exc
    return HfApi(token=os.getenv("HF_TOKEN") or None)


def push_event_to_private_hf(
    event_dir: Path,
    event_id: str,
    settings: StorageSettings,
    *,
    hf_api: Any = None,
) -> str:
    """Mirror one crowd bundle into the *private* HF dataset repo, ``crowd/`` prefix.

    Safety rules: the repository must already exist and be private (verified
    before any upload); this function never creates a repository and never
    passes any make-public flag.  Returns the ``path_in_repo`` used.
    """
    api = hf_api if hf_api is not None else _hf_api_factory()
    info = api.repo_info(repo_id=settings.hf_repo_id, repo_type="dataset")
    if not bool(getattr(info, "private", False)):
        raise CrowdStorageError(
            "Refusing to upload crowd contributions: the configured HF dataset repo "
            "is not private."
        )
    prefix = settings.hf_path_prefix.strip("/")
    path_in_repo = f"{prefix}/{event_id}" if prefix else event_id
    api.upload_folder(
        repo_id=settings.hf_repo_id,
        repo_type="dataset",
        folder_path=str(event_dir),
        path_in_repo=path_in_repo,
        commit_message=f"Add crowd contribution {event_id}",
    )
    return path_in_repo


class ContributionWriter:
    """Write sanitized, screened crowd contributions into local quarantine.

    Layout (mirrors the app.py staging convention):

    ``<data_dir>/crowd_contributions.jsonl`` — append-only row index.
    ``<data_dir>/incoming/CRD_<stamp>_<id>/manifest.json`` + ``media/...``.
    """

    def __init__(
        self,
        settings: StorageSettings,
        *,
        screening_hook: ScreeningHook = noop_screening_hook,
        hf_push: Callable[[Path, str], Any] | None = None,
    ) -> None:
        if settings.max_media_bytes < 1 or settings.max_total_bytes < 1:
            raise ValueError("Byte caps must be positive")
        self.settings = settings
        self.screening_hook = screening_hook
        # Injectable for tests; the real default uploads to the private HF
        # dataset repo and is only constructed when push is enabled.
        self._hf_push: Callable[[Path, str], Any] = (
            hf_push
            if hf_push is not None
            else lambda event_dir, event_id: push_event_to_private_hf(
                event_dir, event_id, self.settings
            )
        )
        self._append_lock = threading.Lock()

    # -- public API ---------------------------------------------------------

    def save(
        self,
        *,
        media: Sequence[MediaSpec],
        level_user: int,
        sample_kind: str = "food",
        captured_at: datetime | None = None,
    ) -> tuple[dict[str, Any], Path]:
        """Sanitize, screen, and store one opt-in contribution event.

        Returns ``(row, event_dir)`` where ``row`` is the manifest row with
        ``source="real-crowd"`` and ``release_eligible=False``.
        """
        if sample_kind not in {"food", "liquid"}:
            raise CrowdStorageError("sample_kind must be 'food' or 'liquid'.")
        if not 0 <= int(level_user) <= 7:
            raise CrowdStorageError("level_user must be between 0 and 7.")
        self._check_size_caps(media)

        now = captured_at or datetime.now(timezone.utc)
        if now.tzinfo is None:
            raise ValueError("captured_at must include a timezone")
        now_utc = now.astimezone(timezone.utc)
        event_id = f"CRD_{now_utc.strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:12]}"

        self.incoming_dir.mkdir(parents=True, exist_ok=True)
        temporary_dir = self.incoming_dir / f".{event_id}.tmp"
        event_dir = self.incoming_dir / event_id
        media_dir = temporary_dir / "media"
        media_dir.mkdir(parents=True, exist_ok=False)

        try:
            media_rows, screening_issues = self._store_media(media, media_dir, event_id)
            status = (
                "custom_hook_no_issues"
                if self.screening_hook is not noop_screening_hook
                else "noop_default_hook_not_screened"
            )
            row: dict[str, Any] = {
                "schema_version": "crowd-staging-1.0.0",
                "event_id": event_id,
                "captured_at": now_utc.isoformat().replace("+00:00", "Z"),
                "source": "real-crowd",
                "sample_kind": sample_kind,
                "level_user": int(level_user),
                "level_tested_RD": None,
                "level_tested_SLP": None,
                "level_adjudicated": None,
                "media_files": media_rows,
                "test_type": [item.capture_role for item in media],
                "contribution_opt_in": True,
                "license": "CC BY-NC-SA 4.0",
                "privacy_screening": {
                    "status": status,
                    "hook": getattr(self.screening_hook, "__name__", "screening_hook"),
                    "issues": screening_issues,
                },
                "release_eligible": False,
            }
            row["sample_sha256"] = _canonical_hash(row)

            with (temporary_dir / "manifest.json").open("w", encoding="utf-8") as handle:
                json.dump(row, handle, ensure_ascii=False, sort_keys=True, indent=2)
                handle.write("\n")
            temporary_dir.rename(event_dir)

            line = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            self.data_dir.mkdir(parents=True, exist_ok=True)
            with self._append_lock, self.index_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())

            if self.settings.hf_push_enabled:
                self._hf_push(event_dir, event_id)
            return row, event_dir
        except Exception:
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir)
            raise

    # -- internals ----------------------------------------------------------

    @property
    def data_dir(self) -> Path:
        return self.settings.data_dir

    @property
    def index_path(self) -> Path:
        return self.settings.data_dir / "crowd_contributions.jsonl"

    @property
    def incoming_dir(self) -> Path:
        return self.settings.data_dir / "incoming"

    def _check_size_caps(self, media: Sequence[MediaSpec]) -> None:
        if not media:
            raise CrowdStorageError("At least one media file is required.")
        total = 0
        for item in media:
            path = Path(item.path)
            if path.is_symlink() or not path.is_file():
                raise CrowdStorageError(f"Media file is unavailable: {path.name}")
            size = path.stat().st_size
            if size <= 0:
                raise CrowdStorageError(f"Empty media files are not accepted: {path.name}")
            if size > self.settings.max_media_bytes:
                raise MediaTooLarge(
                    f"{item.capture_role} exceeds the per-file cap of "
                    f"{self.settings.max_media_bytes} bytes (got {size})."
                )
            total += size
            if total > self.settings.max_total_bytes:
                raise MediaTooLarge(
                    "Combined media exceeds the submission cap of "
                    f"{self.settings.max_total_bytes} bytes."
                )

    def _store_media(
        self,
        media: Sequence[MediaSpec],
        media_dir: Path,
        event_id: str,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Sanitize images (EXIF strip) and screen them; returns (rows, issues)."""
        media_rows: list[dict[str, Any]] = []
        screening_issues: list[str] = []
        role_counts: dict[str, int] = {}
        for item in media:
            role_counts[item.capture_role] = role_counts.get(item.capture_role, 0) + 1
            suffix = Path(item.path).suffix.lower()
            if suffix in IMAGE_SUFFIXES:
                media_type = "image"
            elif suffix in VIDEO_SUFFIXES:
                media_type = "video"
            else:
                raise UnsupportedMedia(f"Unsupported media format: {suffix or '(none)'}")
            filename = (
                f"{event_id}__{item.capture_role}__t{role_counts[item.capture_role]}{suffix}"
            )
            destination = media_dir / filename
            if media_type == "image":
                sanitized = strip_exif(item.path)
                destination.write_bytes(sanitized)
                # Screen the sanitized copy on disk, not the raw upload.
                issues = list(self.screening_hook(destination))
                if issues:
                    raise ScreeningRejection(
                        "Face/text screening flagged this contribution: "
                        + "; ".join(issues)
                    )
                screening_issues.extend(issues)
                exif_stripped = True
            else:
                shutil.copyfile(item.path, destination)
                exif_stripped = False

            relative_path = (Path("incoming") / event_id / "media" / filename).as_posix()
            media_rows.append(
                {
                    "path": relative_path,
                    "media_type": media_type,
                    "capture_role": item.capture_role,
                    "sha256": sha256_file(destination),
                    "bytes": destination.stat().st_size,
                    "content_type": mimetypes.guess_type(filename)[0]
                    or "application/octet-stream",
                    "exif_stripped": exif_stripped,
                }
            )
        return media_rows, screening_issues
