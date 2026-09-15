from pathlib import Path

import pytest

from fixtures.generate import generate_fixture
from train.data import load_manifest, validate_manifest


def test_synthetic_manifest_is_group_disjoint_and_flagged(tmp_path: Path):
    manifest = generate_fixture(tmp_path, per_level_per_split=1, size=16)
    rows = load_manifest(manifest)
    validate_manifest(rows)
    assert len(rows) == 15
    assert all(row["physically_tested"] is False for row in rows)
    assert all(row["release_eligible"] is False for row in rows)
    assert (tmp_path / "SYNTHETIC_ONLY.txt").exists()


def test_group_leakage_is_rejected(tmp_path: Path):
    manifest = generate_fixture(tmp_path, per_level_per_split=1, size=16)
    rows = load_manifest(manifest)
    rows[0]["group_id"] = rows[-1]["group_id"]
    with pytest.raises(ValueError, match="group leakage"):
        validate_manifest(rows)

