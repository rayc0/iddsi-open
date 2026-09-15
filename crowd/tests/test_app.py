from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pytest

from crowd.app import (
    ConsentRequired,
    ManifestWriter,
    MediaInput,
    MediaValidationError,
    PrivacyScreeningResult,
    RateLimitExceeded,
    Settings,
    SlidingWindowRateLimiter,
    submit_contribution,
)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        max_image_bytes=1024,
        max_video_bytes=2048,
        max_total_bytes=3072,
    )


def _stub_result() -> PrivacyScreeningResult:
    return PrivacyScreeningResult(
        status="stub_only_not_cleared",
        release_eligible=False,
        face_detection="not_implemented",
        visible_text_detection="not_implemented",
        detail="test stub",
    )


def test_manifest_writer_records_honest_weak_label_hashes_and_timestamp(tmp_path: Path) -> None:
    source = tmp_path / "plate.jpg"
    source.write_bytes(b"\xff\xd8\xfftiny-cpu-fixture")
    writer = ManifestWriter(tmp_path / "data")

    row, event_dir = writer.save(
        sample_kind="food",
        level_user=5,
        media=[MediaInput(source, "image", "plate_photo")],
        screening=_stub_result(),
    )

    assert row["source"] == "real-crowd"
    assert row["level_user"] == 5
    assert row["level_tested_RD"] is None
    assert row["level_tested_SLP"] is None
    assert row["level_adjudicated"] is None
    assert row["release_eligible"] is False
    assert row["privacy_screening"]["status"] == "stub_only_not_cleared"
    assert len(row["sample_sha256"]) == 64
    datetime.fromisoformat(row["captured_at"].replace("Z", "+00:00"))

    copied = event_dir / "media" / Path(row["media_files"][0]["path"]).name
    expected_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    assert copied.read_bytes() == source.read_bytes()
    assert row["media_files"][0]["sha256"] == expected_hash

    manifest = json.loads((event_dir / "manifest.json").read_text(encoding="utf-8"))
    indexed = json.loads((tmp_path / "data" / "crowd_events.jsonl").read_text(encoding="utf-8"))
    assert manifest == row
    assert indexed == row


def test_opt_in_gate_runs_before_reading_or_saving_media(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    writer = ManifestWriter(settings.data_dir)
    limiter = SlidingWindowRateLimiter(limit=3, window_seconds=3600)

    with pytest.raises(ConsentRequired, match="Contribute this sample"):
        submit_contribution(
            sample_kind="food",
            level_user=4,
            photo=tmp_path / "does-not-exist.jpg",
            privacy_attested=True,
            opt_in=False,
            client_key="test-client",
            settings=settings,
            writer=writer,
            rate_limiter=limiter,
        )

    assert not settings.data_dir.exists()


def test_opted_in_submission_is_saved_but_remains_quarantined(tmp_path: Path) -> None:
    photo = tmp_path / "plate.jpg"
    photo.write_bytes(b"\xff\xd8\xfffixture")
    settings = _settings(tmp_path)

    row = submit_contribution(
        sample_kind="food",
        level_user="4",
        photo=photo,
        privacy_attested=True,
        opt_in=True,
        client_key="test-client",
        settings=settings,
        writer=ManifestWriter(settings.data_dir),
        rate_limiter=SlidingWindowRateLimiter(limit=3, window_seconds=3600),
        screening_fn=lambda _: _stub_result(),
    )

    assert row["contribution_opt_in"] is True
    assert row["source"] == "real-crowd"
    assert row["release_eligible"] is False
    assert (settings.data_dir / "incoming" / row["event_id"] / "manifest.json").is_file()


def test_server_side_size_cap_rejects_before_saving(tmp_path: Path) -> None:
    photo = tmp_path / "oversized.jpg"
    photo.write_bytes(b"x" * 1025)
    settings = _settings(tmp_path)

    with pytest.raises(MediaValidationError, match="per-file cap"):
        submit_contribution(
            sample_kind="food",
            level_user=4,
            photo=photo,
            privacy_attested=True,
            opt_in=True,
            client_key="test-client",
            settings=settings,
            writer=ManifestWriter(settings.data_dir),
            rate_limiter=SlidingWindowRateLimiter(limit=3, window_seconds=3600),
        )

    assert not settings.data_dir.exists()


def test_sliding_window_rate_limit() -> None:
    limiter = SlidingWindowRateLimiter(limit=2, window_seconds=60)
    limiter.check_and_record("client", now=0)
    limiter.check_and_record("client", now=1)
    with pytest.raises(RateLimitExceeded):
        limiter.check_and_record("client", now=2)
    limiter.check_and_record("client", now=61)
