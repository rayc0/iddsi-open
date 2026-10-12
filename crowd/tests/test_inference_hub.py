"""Offline tests for the capture pipeline's Hub model-source wiring.

The network path itself is exercised by ``tools/consume_release.py --step
capture`` (pasted output in ``hf/HOW_TO_CONSUME.md`` §5). These tests pin the
decisions around it — a local source stays local, a ``hub:`` source records
provenance, and an unreachable Hub raises instead of quietly degrading to a
local file — by substituting a fake ``eval.hub_source`` module.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import crowd.inference as inference
from crowd.inference import ModelSource, predict_iddsi_level, resolve_model_source

Image = pytest.importorskip("PIL.Image", reason="photo path needs Pillow")

REPO = "lingualeap/iddsi-open-siglip2-v0"
COMMIT = "af40a912ed8c57e1f11518c1063164e173ce5dca"
SPEC = f"hub:{REPO}@v0.7:results/fix1_siglip2_coral_seed17/resolved_config.yaml"


class FakeHubSource:
    """Just enough of eval.hub_source for the capture path's decisions."""

    RELEASE_REPO = REPO
    DEFAULT_REVISION = "v0.7"
    REAL_ENDPOINT = "https://huggingface.co"
    MIRROR_ENDPOINT = "https://<your-mirror>"
    WEIGHT_SUFFIXES = (".pt", ".pth", ".safetensors", ".bin", ".ckpt")

    class HubUnavailable(RuntimeError):
        pass

    def __init__(self, *, weights=(), raise_on_fetch=None, raise_on_list=None):
        self._weights = tuple(weights)
        self._raise_on_fetch = raise_on_fetch
        self._raise_on_list = raise_on_list
        self.fetched = []

    def is_ref(self, value):
        return isinstance(value, str) and value.startswith("hub:")

    def parse_ref(self, spec, *, repo=None, revision=None):
        body = spec[len("hub:") :]
        repo_part, rest = body.split("@", 1)
        revision_part, filename = rest.split(":", 1)
        return SimpleRef(repo_part, revision_part, filename)

    def resolve_local(self, spec, *, kind, repo=None, revision=None, endpoint=None,
                      cache_dir=None, require_network=False):
        self.fetched.append((spec, require_network))
        if self._raise_on_fetch is not None:
            raise self._raise_on_fetch
        return Path("/tmp/released_resolved_config.yaml"), SimpleReceipt()

    def published_weight_files(self, repo=None, *, revision=None, endpoint=None):
        if self._raise_on_list is not None:
            raise self._raise_on_list
        return list(self._weights)


class SimpleRef:
    def __init__(self, repo, revision, filename):
        self.repo, self.revision, self.filename = repo, revision, filename


class SimpleReceipt:
    endpoint = "https://<your-mirror>"
    endpoint_provenance = "env:HF_ENDPOINT"
    resolved_commit = COMMIT


@pytest.fixture()
def fake_hub(monkeypatch):
    def install(**kwargs):
        module = FakeHubSource(**kwargs)
        monkeypatch.setattr(inference, "_hub_source_module", lambda: module)
        return module

    return install


@pytest.fixture()
def config_file(tmp_path, monkeypatch):
    """A released resolved_config.yaml, as the Hub ships it."""
    path = tmp_path / "resolved_config.yaml"
    path.write_text(
        "model:\n"
        "  kind: siglip2\n"
        "  name: /work/models_local/siglip2-base-patch16-224\n"
        "  dropout: 0.1\n"
        "calibration:\n"
        "  threshold: 0.36494785003441793\n",
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------------------
# Local sources stay local
# --------------------------------------------------------------------------

def test_local_source_is_not_a_hub_fetch(fake_hub, tmp_path):
    module = fake_hub()
    source = resolve_model_source(str(tmp_path / "best.pt"))
    assert module.fetched == []
    assert source.kind == "local"
    assert source.local_path == str(tmp_path / "best.pt")
    assert source.weights_published is None


def test_default_source_honours_the_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv(inference.MODEL_SOURCE_ENV, str(tmp_path / "from-env.pt"))
    assert inference.default_model_source() == str(tmp_path / "from-env.pt")


def test_default_source_falls_back_to_the_local_checkpoint(monkeypatch):
    monkeypatch.delenv(inference.MODEL_SOURCE_ENV, raising=False)
    monkeypatch.delenv(inference.CHECKPOINT_ENV, raising=False)
    assert inference.default_model_source() == str(inference._DEFAULT_CHECKPOINT)


def test_predict_with_a_local_source_matches_the_checkpoint_path(tmp_path):
    """A local source must behave exactly like checkpoint= did before."""
    missing = tmp_path / "missing.pt"
    by_source = predict_iddsi_level(None, source=str(missing))
    by_checkpoint = predict_iddsi_level(None, checkpoint=missing)
    assert by_source.status == by_checkpoint.status == "placeholder_no_weights"
    assert by_source.checkpoint == by_checkpoint.checkpoint == str(missing)
    assert by_source.predicted_level is None


# --------------------------------------------------------------------------
# Hub sources record provenance and never invent a level
# --------------------------------------------------------------------------

def test_hub_source_without_published_weights_is_an_honest_placeholder(
    fake_hub, config_file, tmp_path
):
    module = fake_hub()
    monkeypatch_module = module
    # Point the fake fetch at the real fixture file on disk.
    monkeypatch_module.resolve_local = lambda spec, **kw: (config_file, SimpleReceipt())

    source = resolve_model_source(SPEC)

    assert source.kind == "hub"
    assert source.weights_published is False
    assert source.published_weight_files == ()
    assert source.architecture["kind"] == "siglip2"
    assert source.calibrated_threshold == pytest.approx(0.36494785003441793)
    assert COMMIT in source.provenance_line()

    photo = tmp_path / "plate.png"
    Image.new("RGB", (8, 8), (120, 80, 40)).save(photo)
    result = predict_iddsi_level(photo, source=SPEC)

    assert result.status == "placeholder_no_weights"
    assert result.is_placeholder is True
    assert result.predicted_level is None  # never a level with no weights
    assert set(result.probabilities) == {3, 4, 5, 6, 7}
    assert result.extras["hub_repo"] == REPO
    assert result.extras["hub_resolved_commit"] == COMMIT
    assert result.extras["hub_endpoint"] == "https://<your-mirror>"
    assert result.extras["weights_published"] is False
    assert "wk1" in result.detail


def test_hub_source_with_published_weights_says_so(fake_hub, config_file, tmp_path):
    module = fake_hub(weights=["artifacts/best.pt"])
    module.resolve_local = lambda spec, **kw: (config_file, SimpleReceipt())

    source = resolve_model_source(SPEC)
    assert source.weights_published is True
    assert source.published_weight_files == ("artifacts/best.pt",)

    photo = tmp_path / "plate.png"
    Image.new("RGB", (8, 8), (120, 80, 40)).save(photo)
    result = predict_iddsi_level(photo, source=SPEC)
    # The estimate must still not be a level, and the detail must name the file.
    assert result.predicted_level is None
    assert "artifacts/best.pt" in result.detail


def test_unreachable_hub_raises_rather_than_degrading(fake_hub):
    """No silent local substitution: the caller must hear that the Hub failed."""
    fake_hub(raise_on_fetch=FakeHubSource.HubUnavailable("no endpoint reachable"))
    with pytest.raises(RuntimeError):
        resolve_model_source(SPEC)


def test_unknown_weight_status_fails_closed(fake_hub, config_file):
    """An unreachable file list is not evidence that no weights are published."""
    module = fake_hub(raise_on_list=OSError("connection refused"))
    module.resolve_local = lambda spec, **kw: (config_file, SimpleReceipt())
    with pytest.raises(OSError):
        resolve_model_source(SPEC)


def test_require_network_is_threaded_through(fake_hub, config_file):
    module = fake_hub()

    def recording_resolve_local(spec, **kwargs):
        module.fetched.append((spec, kwargs.get("require_network")))
        return config_file, SimpleReceipt()

    module.resolve_local = recording_resolve_local
    resolve_model_source(SPEC, require_network=True)
    assert module.fetched == [(SPEC, True)]


def test_model_source_dataclass_has_weights_only_for_a_real_file(tmp_path):
    present = tmp_path / "best.pt"
    present.write_bytes(b"x")
    assert ModelSource(spec="x", kind="local", local_path=str(present)).has_weights is True
    assert (
        ModelSource(spec="x", kind="local", local_path=str(tmp_path / "nope.pt")).has_weights
        is False
    )
