"""CROWD-CAPTURE research-preview application.

This module intentionally stores uploads in an *incoming quarantine*.  The
privacy screening function is a server-side integration stub, not a face or
OCR model, so no row written here is eligible for public release.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import shutil
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

try:  # Keep storage/unit tests runnable without the optional UI dependency.
    import gradio as gr
except ImportError:  # pragma: no cover - exercised only in minimal installs.
    gr = None  # type: ignore[assignment]

try:  # Package import (tests, repo-root usage).
    from crowd.inference import predict_iddsi_level
    from crowd.flowtest_ui import grade_flow_test
except ImportError:  # Direct execution on Hugging Face Spaces (app.py as __main__).
    from inference import predict_iddsi_level  # type: ignore[no-redef]
    from flowtest_ui import grade_flow_test  # type: ignore[no-redef]


# Wording per r5_reg.md "Safer public-demo boundary" (English source of truth).
DISCLAIMER = (
    "Research demonstration for culinary education. Estimates visual similarity "
    "to IDDSI descriptors; does not perform official IDDSI tests, assess "
    "swallowing, determine suitability for any person, or decide whether food is "
    "safe to consume."
)
# 繁體中文 rendering of the same r5_reg.md disclaimer (identical meaning).
DISCLAIMER_ZH_HANT = (
    "烹飪教學研究示範。本工具僅估算食物外觀與 IDDSI 描述之間的視覺相似度；"
    "並不執行 IDDSI 官方測試、不評估吞嚥能力、不判斷食物是否適合任何人士，"
    "亦不決定食物是否安全食用。"
)
OPT_IN_LABEL = "Contribute this sample to the open IDDSI dataset (CC BY-NC-SA 4.0)"
OPT_IN_LABEL_ZH_HANT = "同意把此樣本貢獻到開放 IDDSI 數據集（CC BY-NC-SA 4.0）"

# Bilingual UI strings for the two research-preview handlers.  The W11 pattern
# shows English and 繁體中文 side-by-side (see the dual DISCLAIMER block); the
# flow-test and photo sections below follow the same convention — every label
# is rendered in both languages rather than behind a runtime toggle, so the
# research-preview wording is never hidden from either audience.
FLOWTEST_SECTION_TITLE = "Flow-Test video measurement (research preview)"
FLOWTEST_SECTION_TITLE_ZH_HANT = "流量測試影片量度（研究預覽）"
FLOWTEST_SECTION_BODY = (
    "Upload one uninterrupted side-view 10 mL / 10 s syringe Flow-Test video. "
    "The deterministic OpenCV grader measures the residual volume and maps it "
    "to an IDDSI flow level, or abstains with machine-readable reasons when the "
    "capture is unclear. An abstain result always means “unclear — repeat the "
    "official physical test”; it is never a pass."
)
FLOWTEST_SECTION_BODY_ZH_HANT = (
    "請上載一段連續、側視的 10 mL／10 秒針筒流量測試影片。此以 OpenCV 實作的"
    "確定性量度工具會量度剩餘容量並對應到 IDDSI 流量級別；如拍攝不清晰，"
    "工具會以機器可讀的原因代碼棄權。棄權結果一律表示「不清晰——請重做官方"
    "實體測試」，絕不代表通過。"
)
FLOWTEST_BUTTON = "Measure Flow-Test video"
FLOWTEST_BUTTON_ZH_HANT = "量度流量測試影片"
FLOWTEST_INPUT_LABEL = "Syringe Flow-Test video (10 mL / 10 s; max 80 MiB)"
FLOWTEST_INPUT_LABEL_ZH_HANT = "針筒流量測試影片（10 mL／10 秒；上限 80 MiB）"
PHOTO_SECTION_TITLE = "Photo visual-similarity estimate (research preview)"
PHOTO_SECTION_TITLE_ZH_HANT = "照片視覺相似度估算（研究預覽）"
PHOTO_SECTION_BODY = (
    "Runs the ordinal trainer stub on the 45° photo. Until a trained checkpoint "
    "ships, it returns a uniform placeholder distribution — not an IDDSI level "
    "and not an accuracy claim."
)
PHOTO_SECTION_BODY_ZH_HANT = (
    "以 45° 食物照片執行序位訓練器預留介面。在已訓練的模型權重推出前，工具只會"
    "回傳均勻的佔位分佈；這並非 IDDSI 級別，亦不代表任何準確度聲稱。"
)
PHOTO_BUTTON = "Run photo estimate"
PHOTO_BUTTON_ZH_HANT = "執行照片估算"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".webm"}
CAPTURE_ROLES = {"plate_photo", "spoon_tilt", "fork_drip", "fork_press", "syringe_flow"}


class CrowdCaptureError(ValueError):
    """Base class for safe, user-displayable submission errors."""


class ConsentRequired(CrowdCaptureError):
    """Raised before media is read when dataset contribution is not opted in."""


class PrivacyAttestationRequired(CrowdCaptureError):
    """Raised when the uploader did not attest to the no-face/no-name rule."""


class MediaValidationError(CrowdCaptureError):
    """Raised for missing, unsupported, or oversized media."""


class RateLimitExceeded(CrowdCaptureError):
    """Raised when a client exceeds the in-memory submission budget."""


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    hf_upload_enabled: bool = False
    hf_repo_id: str | None = None
    max_image_bytes: int = 10 * 1024 * 1024
    max_video_bytes: int = 80 * 1024 * 1024
    max_total_bytes: int = 170 * 1024 * 1024
    rate_limit_count: int = 3
    rate_limit_window_seconds: int = 60 * 60

    @classmethod
    def from_env(cls) -> "Settings":
        default_data_dir = Path(__file__).resolve().parent / "data"
        enabled = os.getenv("CROWD_HF_UPLOAD_ENABLED", "false").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        return cls(
            data_dir=Path(os.getenv("CROWD_DATA_DIR", str(default_data_dir))).expanduser(),
            hf_upload_enabled=enabled,
            hf_repo_id=os.getenv("CROWD_HF_REPO_ID") or None,
            max_image_bytes=int(os.getenv("CROWD_MAX_IMAGE_BYTES", 10 * 1024 * 1024)),
            max_video_bytes=int(os.getenv("CROWD_MAX_VIDEO_BYTES", 80 * 1024 * 1024)),
            max_total_bytes=int(os.getenv("CROWD_MAX_TOTAL_BYTES", 170 * 1024 * 1024)),
            rate_limit_count=int(os.getenv("CROWD_RATE_LIMIT_COUNT", 3)),
            rate_limit_window_seconds=int(os.getenv("CROWD_RATE_LIMIT_WINDOW_SECONDS", 3600)),
        )


@dataclass(frozen=True)
class MediaInput:
    path: Path
    media_type: str
    capture_role: str


@dataclass(frozen=True)
class PrivacyScreeningResult:
    status: str
    release_eligible: bool
    face_detection: str
    visible_text_detection: str
    detail: str


def server_side_privacy_screening_stub(
    media: Sequence[MediaInput],
) -> PrivacyScreeningResult:
    """Integration point for future frame-sampled face detection and OCR.

    It is important that this stub does not return a misleading "passed"
    result.  Uploader-attested samples can be retained in quarantine, but the
    stub always leaves them ineligible for release.
    """

    del media
    return PrivacyScreeningResult(
        status="stub_only_not_cleared",
        release_eligible=False,
        face_detection="not_implemented",
        visible_text_detection="not_implemented",
        detail="Automated face/OCR screening is not implemented; keep this event quarantined.",
    )


class SlidingWindowRateLimiter:
    """Small in-memory limiter suitable for a single Space process."""

    def __init__(self, limit: int, window_seconds: int) -> None:
        if limit < 1 or window_seconds < 1:
            raise ValueError("Rate-limit values must be positive")
        self.limit = limit
        self.window_seconds = window_seconds
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check_and_record(self, key: str, now: float | None = None) -> None:
        current = time.monotonic() if now is None else now
        cutoff = current - self.window_seconds
        with self._lock:
            history = self._events[key]
            while history and history[0] <= cutoff:
                history.popleft()
            if len(history) >= self.limit:
                raise RateLimitExceeded(
                    f"Submission limit reached ({self.limit} per "
                    f"{self.window_seconds // 60 or 1} minutes). Please try later."
                )
            history.append(current)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _as_path(value: Any) -> Path | None:
    """Normalize Gradio filepath values without trusting client filenames."""

    if value is None or value == "":
        return None
    if isinstance(value, (str, os.PathLike)):
        return Path(value)
    if isinstance(value, dict):
        candidate = value.get("path") or value.get("name")
        return Path(candidate) if candidate else None
    candidate = getattr(value, "path", None) or getattr(value, "name", None)
    return Path(candidate) if candidate else None


def _validate_media(media: Sequence[MediaInput], settings: Settings) -> None:
    if not media:
        raise MediaValidationError("At least one required media file is missing.")

    total = 0
    for item in media:
        if item.capture_role not in CAPTURE_ROLES:
            raise MediaValidationError(f"Unsupported capture role: {item.capture_role}")
        if item.media_type not in {"image", "video"}:
            raise MediaValidationError(f"Unsupported media type: {item.media_type}")
        if item.path.is_symlink() or not item.path.is_file():
            raise MediaValidationError("An uploaded media file is unavailable.")

        suffix = item.path.suffix.lower()
        allowed = IMAGE_SUFFIXES if item.media_type == "image" else VIDEO_SUFFIXES
        if suffix not in allowed:
            raise MediaValidationError(
                f"Unsupported {item.media_type} format {suffix or '(none)'}; "
                f"allowed: {', '.join(sorted(allowed))}."
            )

        size = item.path.stat().st_size
        cap = settings.max_image_bytes if item.media_type == "image" else settings.max_video_bytes
        if size <= 0:
            raise MediaValidationError("Empty media files are not accepted.")
        if size > cap:
            raise MediaValidationError(
                f"{item.capture_role} exceeds the {cap // (1024 * 1024)} MiB per-file cap."
            )
        total += size

    if total > settings.max_total_bytes:
        raise MediaValidationError(
            f"Combined upload exceeds the {settings.max_total_bytes // (1024 * 1024)} MiB cap."
        )


def prepare_media_inputs(
    *,
    sample_kind: str,
    level_user: int,
    photo: Any,
    guided_clip: Any,
    guided_clip_role: str | None,
    syringe_video: Any,
) -> list[MediaInput]:
    """Apply capture-protocol requirements and normalize UI values."""

    if sample_kind not in {"food", "liquid"}:
        raise MediaValidationError("Choose food or liquid.")
    if not 0 <= level_user <= 7:
        raise MediaValidationError("The user-declared level must be between 0 and 7.")

    photo_path = _as_path(photo)
    clip_path = _as_path(guided_clip)
    syringe_path = _as_path(syringe_video)
    media: list[MediaInput] = []

    if sample_kind == "food":
        if level_user not in {3, 4, 5, 6, 7}:
            raise MediaValidationError("Food captures use user-declared levels L3-L7.")
        if photo_path is None:
            raise MediaValidationError("A 45° plate photo is required for food captures.")
        if syringe_path is not None:
            raise MediaValidationError("Use a liquid capture for syringe-flow video.")
        media.append(MediaInput(photo_path, "image", "plate_photo"))
        if clip_path is not None:
            if guided_clip_role not in {"spoon_tilt", "fork_drip", "fork_press"}:
                raise MediaValidationError("Choose the physical-test action shown in the clip.")
            media.append(MediaInput(clip_path, "video", str(guided_clip_role)))
    else:
        if level_user not in {0, 1, 2, 3, 4}:
            raise MediaValidationError("Liquid captures use user-declared levels L0-L4.")
        if syringe_path is None:
            raise MediaValidationError("A 10 mL / 10 s syringe-flow video is required for liquids.")
        if clip_path is not None:
            raise MediaValidationError("The optional food-test clip is not used for liquid captures.")
        if photo_path is not None:
            media.append(MediaInput(photo_path, "image", "plate_photo"))
        media.append(MediaInput(syringe_path, "video", "syringe_flow"))

    return media


class ManifestWriter:
    """Write one immutable quarantine bundle plus an append-only JSONL index."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.incoming_dir = self.data_dir / "incoming"
        self.index_path = self.data_dir / "crowd_events.jsonl"
        self._append_lock = threading.Lock()

    def save(
        self,
        *,
        sample_kind: str,
        level_user: int,
        media: Sequence[MediaInput],
        screening: PrivacyScreeningResult,
        captured_at: datetime | None = None,
    ) -> tuple[dict[str, Any], Path]:
        now = captured_at or datetime.now(timezone.utc)
        if now.tzinfo is None:
            raise ValueError("captured_at must include a timezone")
        now_utc = now.astimezone(timezone.utc)
        stamp = now_utc.strftime("%Y%m%dT%H%M%SZ")
        event_id = f"CRD_{stamp}_{uuid.uuid4().hex[:12]}"

        self.incoming_dir.mkdir(parents=True, exist_ok=True)
        temporary_dir = self.incoming_dir / f".{event_id}.tmp"
        event_dir = self.incoming_dir / event_id
        media_dir = temporary_dir / "media"
        media_dir.mkdir(parents=True, exist_ok=False)

        media_rows: list[dict[str, Any]] = []
        try:
            role_counts: dict[str, int] = defaultdict(int)
            for item in media:
                role_counts[item.capture_role] += 1
                suffix = item.path.suffix.lower()
                filename = (
                    f"{event_id}__{item.capture_role}__t{role_counts[item.capture_role]}{suffix}"
                )
                destination = media_dir / filename
                shutil.copyfile(item.path, destination)
                relative_path = Path("incoming") / event_id / "media" / filename
                media_rows.append(
                    {
                        "path": relative_path.as_posix(),
                        "media_type": item.media_type,
                        "capture_role": item.capture_role,
                        "sha256": _sha256_file(destination),
                        "bytes": destination.stat().st_size,
                        "content_type": mimetypes.guess_type(filename)[0]
                        or "application/octet-stream",
                    }
                )

            row: dict[str, Any] = {
                "schema_version": "crowd-staging-1.0.0",
                "event_id": event_id,
                "captured_at": now_utc.isoformat().replace("+00:00", "Z"),
                "source": "real-crowd",
                "sample_kind": sample_kind,
                "level_user": level_user,
                "level_tested_RD": None,
                "level_tested_SLP": None,
                "level_adjudicated": None,
                "media_files": media_rows,
                "test_type": [item.capture_role for item in media],
                "contribution_opt_in": True,
                "license": "CC BY-NC-SA 4.0",
                "no_face_or_name_attested": True,
                "privacy_screening": asdict(screening),
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
            return row, event_dir
        except Exception:
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir)
            raise


def upload_quarantine_bundle(event_dir: Path, event_id: str, settings: Settings) -> None:
    """Optionally mirror one quarantine bundle to a *private* HF dataset repo."""

    if not settings.hf_upload_enabled:
        return
    if not settings.hf_repo_id:
        raise CrowdCaptureError("HF upload is enabled but CROWD_HF_REPO_ID is not set.")
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - depends on optional deployment package.
        raise CrowdCaptureError("huggingface_hub is required when HF upload is enabled.") from exc

    api = HfApi(token=os.getenv("HF_TOKEN") or None)
    info = api.repo_info(repo_id=settings.hf_repo_id, repo_type="dataset")
    if not bool(getattr(info, "private", False)):
        raise CrowdCaptureError(
            "Refusing to upload uncleared crowd media: the configured HF dataset repo is not private."
        )
    api.upload_folder(
        repo_id=settings.hf_repo_id,
        repo_type="dataset",
        folder_path=str(event_dir),
        path_in_repo=f"incoming/{event_id}",
        commit_message=f"Add quarantined crowd event {event_id}",
    )


def submit_contribution(
    *,
    sample_kind: str,
    level_user: int | str,
    photo: Any = None,
    guided_clip: Any = None,
    guided_clip_role: str | None = None,
    syringe_video: Any = None,
    privacy_attested: bool,
    opt_in: bool,
    client_key: str,
    settings: Settings,
    writer: ManifestWriter,
    rate_limiter: SlidingWindowRateLimiter,
    screening_fn: Callable[[Sequence[MediaInput]], PrivacyScreeningResult] = (
        server_side_privacy_screening_stub
    ),
) -> dict[str, Any]:
    """Validate consent before reading media, then save one quarantine event."""

    if opt_in is not True:
        raise ConsentRequired(f'Check "{OPT_IN_LABEL}" before submitting.')
    if privacy_attested is not True:
        raise PrivacyAttestationRequired(
            "Confirm that the media contains no faces, names, badges, documents, screens, "
            "patient information, or identifying speech."
        )

    try:
        level = int(level_user)
    except (TypeError, ValueError) as exc:
        raise MediaValidationError("Choose the level the recipe is intended to be.") from exc

    media = prepare_media_inputs(
        sample_kind=sample_kind,
        level_user=level,
        photo=photo,
        guided_clip=guided_clip,
        guided_clip_role=guided_clip_role,
        syringe_video=syringe_video,
    )
    _validate_media(media, settings)
    rate_limiter.check_and_record(client_key)
    screening = screening_fn(media)
    row, event_dir = writer.save(
        sample_kind=sample_kind,
        level_user=level,
        media=media,
        screening=screening,
    )
    upload_quarantine_bundle(event_dir, row["event_id"], settings)
    return row


def _client_key(request: Any) -> str:
    session_hash = getattr(request, "session_hash", None)
    client = getattr(request, "client", None)
    host = getattr(client, "host", None)
    # Prefer the network client so a page reload does not reset the budget.
    # Direct/API calls without request metadata share a conservative fallback.
    return f"ip:{host}" if host else f"session:{session_hash or 'anonymous'}"


def render_flow_test_markdown(result: dict[str, Any]) -> str:
    """Render one flow-test grade result dict as bilingual Markdown.

    Pure function of the grader's JSON payload — no Gradio dependency — so the
    handler tests exercise it directly.  Always shows the abstention reasons
    (with their human descriptions) when the grader abstains, and never
    presents an abstain as a level or a pass.
    """

    status = result.get("status")
    if status == "error":
        return (
            "❌ **Could not read the video / 無法讀取影片.** "
            f"{result.get('error', 'Unknown error.')}\n\n"
            "No measurement was produced. 未能產生任何量度結果。"
        )

    lines: list[str] = []
    level = result.get("iddsi_level")
    residual = result.get("estimated_residual_ml")
    if status == "graded" and level:
        residual_txt = f"{residual:.2f} mL" if isinstance(residual, (int, float)) else "—"
        lines.append(
            f"🔵 **Flow-Test measurement (research only): {level}** · "
            f"residual ≈ {residual_txt}"
        )
        lines.append(
            f"🔵 **流量測試量度（僅供研究）：{level}** · 剩餘容量約 {residual_txt}"
        )
    else:
        # Abstain — never a pass, never a level.
        lines.append("🟡 **Abstain — unclear, repeat the official physical test.**")
        lines.append("🟡 **棄權——影像不清晰，請重做官方實體測試。**")
        if isinstance(residual, (int, float)):
            lines.append(
                f"(retained for audit only · 僅作審計保留: residual ≈ {residual:.2f} mL — "
                "not a level · 並非級別)"
            )

    reasons = result.get("abstention_reasons") or []
    if reasons:
        descriptions = (result.get("diagnostics") or {}).get("abstention_code_descriptions") or {}
        lines.append("\n**Abstention reasons · 棄權原因:**")
        for code in reasons:
            description = descriptions.get(code, "")
            lines.append(f"- `{code}` — {description}" if description else f"- `{code}`")

    notice = result.get("notice")
    if notice:
        lines.append(f"\n> {notice}")
    return "\n".join(lines)


def handle_photo_estimate(
    photo: Any,
    *,
    predictor: Callable[[Any], Any] = predict_iddsi_level,
) -> str:
    """Run and render the trainer-backed photo stub for the Gradio callback."""

    if photo is None:
        return (
            "❌ **No photo / 未上載照片：** upload a 45° plate photo first. "
            "請先上載一張 45° 食物照片。"
        )
    try:
        result = predictor(photo)
    except Exception:
        return (
            "❌ **Estimation failed / 估算失敗。** No estimate was produced; "
            "please try again later. 未能產生估算結果，請稍後再試。"
        )

    if result.is_placeholder:
        return (
            "🟡 **Stub estimate — no trained model is loaded / 預留估算——未載入已訓練模型。** "
            f"{result.detail} The distribution below is a uniform placeholder, "
            "**not** a model output or IDDSI level. 以下分佈只是均勻佔位值，"
            "**並非**模型輸出或 IDDSI 級別。\n\n"
            + " · ".join(f"L{level}: {prob:.2f}" for level, prob in result.probabilities.items())
        )
    return (
        f"🔵 **Photo visual-similarity estimate (research only) / "
        f"照片視覺相似度估算（僅供研究）：L{result.predicted_level}** "
        f"· model / 模型 `{result.model_kind}` · "
        + " · ".join(f"L{level}: {prob:.2f}" for level, prob in result.probabilities.items())
    )


def handle_flowtest_video(
    video: Any,
    *,
    grader: Callable[[Any], dict[str, Any]] = grade_flow_test,
) -> tuple[dict[str, Any], str]:
    """Return both grader JSON and bilingual Markdown for the Gradio callback."""

    try:
        result = grader(video)
    except Exception as exc:
        result = {
            "video": str(video) if video is not None else None,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "abstention_reasons": [],
            "diagnostics": {"abstention_code_descriptions": {}},
        }
    return result, render_flow_test_markdown(result)


def build_app(settings: Settings | None = None):
    if gr is None or not hasattr(gr, "Blocks"):
        raise RuntimeError("A current Gradio release is required to build the UI.")

    active_settings = settings or Settings.from_env()
    writer = ManifestWriter(active_settings.data_dir)
    limiter = SlidingWindowRateLimiter(
        active_settings.rate_limit_count, active_settings.rate_limit_window_seconds
    )

    def on_submit(
        sample_kind: str,
        level_user: str,
        photo: Any,
        guided_clip: Any,
        guided_clip_role: str,
        syringe_video: Any,
        privacy_attested: bool,
        opt_in: bool,
        request: gr.Request,
    ) -> str:
        try:
            row = submit_contribution(
                sample_kind=sample_kind,
                level_user=level_user,
                photo=photo,
                guided_clip=guided_clip,
                guided_clip_role=guided_clip_role,
                syringe_video=syringe_video,
                privacy_attested=privacy_attested,
                opt_in=opt_in,
                client_key=_client_key(request),
                settings=active_settings,
                writer=writer,
                rate_limiter=limiter,
            )
        except CrowdCaptureError as exc:
            return f"❌ **Not submitted:** {exc}"
        except Exception:
            return "❌ **Submission failed.** No public release was made; please try again later."

        return (
            "✅ **Saved to the incoming research quarantine.** "
            f"Event `{row['event_id']}` is tagged `source=real-crowd`, and your stated level "
            "is stored only as a weak user label. It has not passed face/text screening, "
            "has no physical-test ground truth, and is not release-eligible."
        )

    with gr.Blocks(title="IDDSI-Open CROWD-CAPTURE") as app:
        gr.Markdown("# IDDSI-Open · CROWD-CAPTURE research preview")
        gr.Markdown(f"> **{DISCLAIMER}**")
        gr.Markdown(f"> **{DISCLAIMER_ZH_HANT}**")
        gr.HTML(
            """
            <aside role="alert" style="border:2px solid #9b2c2c;padding:0.8rem;border-radius:0.5rem;">
              <strong>Before recording: no faces or names.</strong><br>
              Frame only the food/liquid, utensils, syringe, and hands if needed. Remove badges,
              labels, order slips, screens, reflections, patient details, location metadata, and
              identifying speech. Do not upload patient/resident media.
            </aside>
            """
        )
        gr.Markdown(
            "Use the level the recipe is intended to be; this is a **weak user label**, not a "
            "tested IDDSI result. For food, upload a 45° photo and optionally one continuous "
            "8–12 s guided test clip. For liquid, upload one uninterrupted 10 mL / 10 s syringe "
            "video; a reference photo is optional."
        )

        with gr.Row():
            sample_kind = gr.Radio(
                choices=[("Food (L3–L7)", "food"), ("Liquid (L0–L4)", "liquid")],
                value="food",
                label="1. What are you capturing?",
            )
            level_user = gr.Dropdown(
                choices=[str(value) for value in range(8)],
                value="4",
                label="2. What level do you say the recipe is?",
            )

        with gr.Row():
            photo = gr.Image(
                type="filepath",
                sources=["upload", "webcam"],
                label="3. 45° photo (required for food; optional for liquid; max 10 MiB)",
            )
            guided_clip = gr.Video(
                sources=["upload", "webcam"],
                label="4. Optional guided food-test clip (8–12 s; max 80 MiB)",
            )
            syringe_video = gr.Video(
                sources=["upload", "webcam"],
                label="5. Syringe-flow video (required for liquid; max 80 MiB)",
            )

        guided_clip_role = gr.Radio(
            choices=[
                ("Spoon tilt", "spoon_tilt"),
                ("Fork drip", "fork_drip"),
                ("Fork press", "fork_press"),
            ],
            label="Which action does the optional food clip show?",
        )
        privacy_attested = gr.Checkbox(
            value=False,
            label=(
                "I confirm these files contain no faces, names, badges, documents, screens, "
                "patient information, location metadata, or identifying speech."
            ),
        )
        opt_in = gr.Checkbox(value=False, label=OPT_IN_LABEL)
        gr.Markdown(
            "Opt-in is voluntary. Submitted media is quarantined; the current server-side "
            "face/text checker is only an integration stub and does **not** clear media for release."
        )
        submit = gr.Button("Submit to quarantine", variant="primary")
        status = gr.Markdown()
        submit.click(
            fn=on_submit,
            inputs=[
                sample_kind,
                level_user,
                photo,
                guided_clip,
                guided_clip_role,
                syringe_video,
                privacy_attested,
                opt_in,
            ],
            outputs=status,
        )

        gr.Markdown(
            f"### {PHOTO_SECTION_TITLE}\n"
            f"### {PHOTO_SECTION_TITLE_ZH_HANT}\n"
            f"{PHOTO_SECTION_BODY}\n\n{PHOTO_SECTION_BODY_ZH_HANT}"
        )
        estimate = gr.Button(f"{PHOTO_BUTTON} · {PHOTO_BUTTON_ZH_HANT}")
        estimate_status = gr.Markdown()
        estimate.click(fn=handle_photo_estimate, inputs=[photo], outputs=estimate_status)

        gr.Markdown(
            f"### {FLOWTEST_SECTION_TITLE}\n"
            f"### {FLOWTEST_SECTION_TITLE_ZH_HANT}\n"
            f"{FLOWTEST_SECTION_BODY}\n\n{FLOWTEST_SECTION_BODY_ZH_HANT}"
        )
        flowtest_video = gr.Video(
            sources=["upload", "webcam"],
            label=f"{FLOWTEST_INPUT_LABEL} · {FLOWTEST_INPUT_LABEL_ZH_HANT}",
        )
        flowtest_run = gr.Button(f"{FLOWTEST_BUTTON} · {FLOWTEST_BUTTON_ZH_HANT}")
        flowtest_json = gr.JSON(label="Flow-Test result JSON · 流量測試結果 JSON")
        flowtest_status = gr.Markdown()
        flowtest_run.click(
            fn=handle_flowtest_video,
            inputs=[flowtest_video],
            outputs=[flowtest_json, flowtest_status],
        )

    return app


# Hugging Face Spaces discovers `demo`; minimal test environments can still
# import the manifest logic without installing a current Gradio build.
demo = build_app() if gr is not None and hasattr(gr, "Blocks") else None


if __name__ == "__main__":
    if demo is None:
        raise SystemExit("Install dependencies from requirements.txt first.")
    demo.launch()
