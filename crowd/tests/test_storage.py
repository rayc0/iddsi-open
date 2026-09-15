"""Tests for the opt-in crowd contribution storage writer (crowd/storage.py).

CPU-only, no network, no real Hugging Face calls: the HF push path is either
disabled (default) or driven by an injected stub uploader / fake HfApi.
"""

from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from PIL import Image

from crowd.storage import (
    DEFAULT_HF_PATH_PREFIX,
    DEFAULT_HF_REPO_ID,
    ContributionWriter,
    CrowdStorageError,
    MediaSpec,
    MediaTooLarge,
    ScreeningRejection,
    StorageSettings,
    UnsupportedImage,
    UnsupportedMedia,
    noop_screening_hook,
    push_event_to_private_hf,
    sha256_bytes,
    sha256_file,
    strip_exif,
)


# -- fixtures ---------------------------------------------------------------


def _settings(tmp_path: Path, **overrides: object) -> StorageSettings:
    values: dict[str, object] = {
        "data_dir": tmp_path / "data",
        "max_media_bytes": 1024 * 1024,
        "max_total_bytes": 2 * 1024 * 1024,
    }
    values.update(overrides)
    return StorageSettings(**values)  # type: ignore[arg-type]


def _jpeg_with_exif(path: Path) -> Path:
    """Write a small valid JPEG carrying a real EXIF payload via Pillow."""
    exif = Image.Exif()
    exif[0x010F] = "Test Camera"  # Make
    exif[0x0110] = "Test Model"  # Model
    image = Image.new("RGB", (16, 12), color=(200, 120, 60))
    image.save(path, format="JPEG", exif=exif)
    return path


def _png_with_exif(path: Path) -> Path:
    exif = Image.Exif()
    exif[0x010F] = "PNG Camera"
    image = Image.new("RGB", (10, 10), color=(10, 200, 30))
    image.save(path, format="PNG", exif=exif)
    return path


def _video_fixture(path: Path) -> Path:
    path.write_bytes(b"\x00\x00\x00\x18ftypmp42fake-video-bytes")
    return path


# -- content hash -----------------------------------------------------------


def test_sha256_file_matches_hashlib_and_is_stable(tmp_path: Path) -> None:
    first = tmp_path / "a.bin"
    second = tmp_path / "b.bin"
    first.write_bytes(b"iddsi-open-crowd-fixture")
    second.write_bytes(b"iddsi-open-crowd-fixture")

    expected = hashlib.sha256(b"iddsi-open-crowd-fixture").hexdigest()
    assert sha256_file(first) == expected
    assert sha256_file(first) == sha256_file(second)
    assert sha256_bytes(b"iddsi-open-crowd-fixture") == expected


# -- EXIF stripping ---------------------------------------------------------


def test_strip_exif_removes_exif_from_jpeg(tmp_path: Path) -> None:
    source = _jpeg_with_exif(tmp_path / "plate.jpg")
    assert b"Exif" in source.read_bytes()  # precondition: EXIF really present
    assert Image.open(source).getexif()[0x010F] == "Test Camera"

    sanitized = strip_exif(source)

    assert b"Exif" not in sanitized
    clean = Image.open(io.BytesIO(sanitized))
    assert not dict(clean.getexif())
    # Geometry survives the re-encode (JPEG is lossy, so pixels are only near).
    original = Image.open(source)
    assert clean.size == original.size
    assert clean.mode == original.mode


def test_strip_exif_removes_exif_from_png(tmp_path: Path) -> None:
    source = _png_with_exif(tmp_path / "plate.png")
    assert Image.open(source).getexif()[0x010F] == "PNG Camera"

    sanitized = strip_exif(source)

    clean = Image.open(io.BytesIO(sanitized))
    assert not dict(clean.getexif())
    # PNG re-encode is lossless: pixel content is preserved exactly.
    assert clean.tobytes() == Image.open(source).tobytes()


def test_strip_exif_is_deterministic(tmp_path: Path) -> None:
    source = _jpeg_with_exif(tmp_path / "plate.jpg")
    first = strip_exif(source)
    second = strip_exif(source)
    assert first == second
    assert sha256_bytes(first) == sha256_bytes(second)


def test_strip_exif_rejects_non_images_and_undecodable_files(tmp_path: Path) -> None:
    video = _video_fixture(tmp_path / "flow.mp4")
    with pytest.raises(UnsupportedMedia):
        strip_exif(video)

    junk = tmp_path / "junk.jpg"
    junk.write_bytes(b"this is not a jpeg")
    with pytest.raises(UnsupportedImage):
        strip_exif(junk)


# -- size caps --------------------------------------------------------------


def test_size_cap_rejects_oversize_before_anything_is_written(tmp_path: Path) -> None:
    huge = tmp_path / "huge.jpg"
    huge.write_bytes(b"x" * 21)
    settings = _settings(tmp_path, max_media_bytes=20)
    writer = ContributionWriter(settings)

    with pytest.raises(MediaTooLarge, match="per-file cap"):
        writer.save(media=[MediaSpec(huge, "plate_photo")], level_user=4)

    assert not settings.data_dir.exists()


def test_total_cap_rejects_combined_oversize(tmp_path: Path) -> None:
    one = tmp_path / "one.jpg"
    two = tmp_path / "two.jpg"
    one.write_bytes(b"x" * 15)
    two.write_bytes(b"x" * 15)
    settings = _settings(tmp_path, max_media_bytes=20, max_total_bytes=20)
    writer = ContributionWriter(settings)

    with pytest.raises(MediaTooLarge, match="submission cap"):
        writer.save(
            media=[MediaSpec(one, "plate_photo"), MediaSpec(two, "fork_drip")],
            level_user=4,
        )


# -- screening hook ---------------------------------------------------------


def test_screening_hook_receives_sanitized_image_and_blocks_on_issues(
    tmp_path: Path,
) -> None:
    source = _jpeg_with_exif(tmp_path / "plate.jpg")
    seen_paths: list[Path] = []
    seen_bytes: list[bytes] = []

    def hook(image_path: Path) -> list[str]:
        seen_paths.append(Path(image_path))
        seen_bytes.append(Path(image_path).read_bytes())
        return ["possible_face_detected"]

    writer = ContributionWriter(_settings(tmp_path), screening_hook=hook)

    with pytest.raises(ScreeningRejection, match="possible_face_detected"):
        writer.save(media=[MediaSpec(source, "plate_photo")], level_user=4)

    assert len(seen_paths) == 1
    # The hook saw the sanitized in-quarantine copy, not the raw upload.
    assert seen_paths[0].suffix == ".jpg"
    assert b"Exif" not in seen_bytes[0]
    # No event bundle or leftover temporary directory survives the rejection.
    incoming = tmp_path / "data" / "incoming"
    assert not any(incoming.iterdir()) if incoming.exists() else not incoming.exists()


def test_screening_hook_noop_default_is_invoked_and_clears_nothing(
    tmp_path: Path,
) -> None:
    source = _jpeg_with_exif(tmp_path / "plate.jpg")
    calls: list[Path] = []

    def recording_hook(image_path: Path) -> list[str]:
        calls.append(Path(image_path))
        return []

    writer = ContributionWriter(_settings(tmp_path), screening_hook=recording_hook)
    row, _ = writer.save(media=[MediaSpec(source, "plate_photo")], level_user=4)

    assert len(calls) == 1
    assert row["privacy_screening"]["issues"] == []
    assert row["privacy_screening"]["status"] == "custom_hook_no_issues"
    # A custom hook with no issues still does not make the row release-eligible.
    assert row["release_eligible"] is False


def test_default_noop_hook_reports_not_screened(tmp_path: Path) -> None:
    source = _jpeg_with_exif(tmp_path / "plate.jpg")
    writer = ContributionWriter(_settings(tmp_path))
    row, _ = writer.save(media=[MediaSpec(source, "plate_photo")], level_user=4)

    assert row["privacy_screening"]["status"] == "noop_default_hook_not_screened"
    assert row["privacy_screening"]["hook"] == "noop_screening_hook"
    assert noop_screening_hook(source) == []


# -- manifest row -----------------------------------------------------------


def test_manifest_row_is_written_with_source_real_crowd(tmp_path: Path) -> None:
    source = _jpeg_with_exif(tmp_path / "plate.jpg")
    video = _video_fixture(tmp_path / "flow.mp4")
    settings = _settings(tmp_path)
    writer = ContributionWriter(settings)
    captured = datetime(2026, 9, 1, 12, 30, 0, tzinfo=timezone.utc)

    row, event_dir = writer.save(
        media=[MediaSpec(source, "plate_photo"), MediaSpec(video, "syringe_flow")],
        level_user=2,
        sample_kind="liquid",
        captured_at=captured,
    )

    assert row["schema_version"] == "crowd-staging-1.0.0"
    assert row["source"] == "real-crowd"
    assert row["sample_kind"] == "liquid"
    assert row["level_user"] == 2
    assert row["level_tested_RD"] is None
    assert row["level_tested_SLP"] is None
    assert row["level_adjudicated"] is None
    assert row["release_eligible"] is False
    assert row["contribution_opt_in"] is True
    assert row["license"] == "CC BY-NC-SA 4.0"
    assert row["captured_at"] == "2026-09-01T12:30:00Z"
    assert len(row["sample_sha256"]) == 64
    assert row["test_type"] == ["plate_photo", "syringe_flow"]

    image_row, video_row = row["media_files"]
    stored_image = event_dir / "media" / Path(image_row["path"]).name
    stored_video = event_dir / "media" / Path(video_row["path"]).name

    # The stored image is the sanitized copy: EXIF gone, hash matches bytes on disk.
    assert image_row["exif_stripped"] is True
    assert b"Exif" not in stored_image.read_bytes()
    assert image_row["sha256"] == sha256_file(stored_image)
    assert image_row["sha256"] != sha256_file(source)  # differs from raw upload
    # Videos pass through verbatim.
    assert video_row["exif_stripped"] is False
    assert video_row["sha256"] == sha256_file(video)
    assert stored_video.read_bytes() == video.read_bytes()

    # manifest.json equals the returned row, and the JSONL index has one line.
    manifest = json.loads((event_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == row
    index_path = settings.data_dir / "crowd_contributions.jsonl"
    lines = index_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == row


def test_rejects_invalid_level_and_sample_kind(tmp_path: Path) -> None:
    source = _jpeg_with_exif(tmp_path / "plate.jpg")
    writer = ContributionWriter(_settings(tmp_path))
    with pytest.raises(CrowdStorageError, match="level_user"):
        writer.save(media=[MediaSpec(source, "plate_photo")], level_user=9)
    with pytest.raises(CrowdStorageError, match="sample_kind"):
        writer.save(
            media=[MediaSpec(source, "plate_photo")], level_user=4, sample_kind="soup"
        )


# -- HF push ----------------------------------------------------------------


def test_hf_push_is_off_by_default(tmp_path: Path) -> None:
    source = _jpeg_with_exif(tmp_path / "plate.jpg")
    pushed: list[tuple[Path, str]] = []

    settings = _settings(tmp_path)
    assert settings.hf_push_enabled is False
    writer = ContributionWriter(settings, hf_push=lambda d, e: pushed.append((d, e)))

    row, _ = writer.save(media=[MediaSpec(source, "plate_photo")], level_user=4)

    assert pushed == []  # disabled flag means the uploader is never invoked
    assert settings.hf_repo_id == DEFAULT_HF_REPO_ID
    assert settings.hf_path_prefix == DEFAULT_HF_PATH_PREFIX


def test_hf_push_enabled_uses_stub_uploader_with_crowd_prefix(tmp_path: Path) -> None:
    source = _jpeg_with_exif(tmp_path / "plate.jpg")
    settings = _settings(tmp_path, hf_push_enabled=True)
    pushed: list[tuple[Path, str]] = []

    def stub_push(event_dir: Path, event_id: str) -> str:
        pushed.append((Path(event_dir), event_id))
        return f"{settings.hf_path_prefix}/{event_id}"

    writer = ContributionWriter(settings, hf_push=stub_push)
    row, event_dir = writer.save(
        media=[MediaSpec(source, "plate_photo")], level_user=4
    )

    assert pushed == [(event_dir, row["event_id"])]


class _FakeHfApi:
    def __init__(self, private: bool) -> None:
        self.private = private
        self.uploaded: dict[str, object] = {}

    def repo_info(self, *, repo_id: str, repo_type: str) -> object:
        assert repo_id == "lingualeap/iddsi-open-v0"
        assert repo_type == "dataset"
        return type("Info", (), {"private": self.private})()

    def upload_folder(self, **kwargs: object) -> None:
        self.uploaded.update(kwargs)


def test_push_refuses_public_repo_before_uploading(tmp_path: Path) -> None:
    fake = _FakeHfApi(private=False)
    settings = _settings(tmp_path, hf_push_enabled=True)

    with pytest.raises(CrowdStorageError, match="not private"):
        push_event_to_private_hf(tmp_path, "CRD_test", settings, hf_api=fake)

    assert fake.uploaded == {}  # nothing was uploaded


def test_push_to_private_repo_uses_crowd_prefix_and_dataset_type(tmp_path: Path) -> None:
    fake = _FakeHfApi(private=True)
    settings = _settings(tmp_path, hf_push_enabled=True)

    path_in_repo = push_event_to_private_hf(
        tmp_path, "CRD_20260901T000000Z_ab12cd34ef56", settings, hf_api=fake
    )

    assert path_in_repo == "crowd/CRD_20260901T000000Z_ab12cd34ef56"
    assert fake.uploaded["repo_id"] == "lingualeap/iddsi-open-v0"
    assert fake.uploaded["repo_type"] == "dataset"
    assert fake.uploaded["path_in_repo"] == path_in_repo
    assert fake.uploaded["folder_path"] == str(tmp_path)
    # No create_repo / make-public flag is ever part of the upload call.
    assert set(fake.uploaded) == {
        "repo_id",
        "repo_type",
        "folder_path",
        "path_in_repo",
        "commit_message",
    }
