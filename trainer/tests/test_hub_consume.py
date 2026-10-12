"""Tests for the Hub-consumption path: config loading, endpoint precedence,
receipt honesty, and the ``expect:`` assertions.

Every test here is offline. The network-touching behaviour is exercised by
``tools/consume_release.py --step smoke`` (see ``hf/HOW_TO_CONSUME.md`` §5 for
the pasted output); these tests pin the *decisions* that path makes — which
endpoint is chosen, what a receipt must record, and what a failed assertion
looks like — without depending on a reachable Hub.
"""

from __future__ import annotations

import json

import pytest

from eval import hub_source


CONFIG = """
hub:
  repo: lingualeap/iddsi-open-siglip2-v0
  revision: v0.7
  endpoint: ""
  cache_dir: ""
  require_network: true
artifacts:
  predictions: hub:results/fix1_siglip2_coral_seed17/test_predictions.npz
  threshold: hub:results/fix1_siglip2_coral_seed17/metrics.json
  model_source: hub:results/fix1_siglip2_coral_seed17/resolved_config.yaml
eval:
  bootstrap_samples: 1000
  bootstrap_seed: 2026
expect:
  n: 109
  weighted_kappa_covered: 0.4227
  coverage: 0.8807
"""


@pytest.fixture()
def config_file(tmp_path):
    path = tmp_path / "hub_release.yaml"
    path.write_text(CONFIG, encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Endpoint precedence: an operator-supplied mirror on a CN network, the real
# endpoint elsewhere. The mirror hostname is never named in this file — the
# release leak scan BLOCKs on mirror hostnames, so the test uses a placeholder.
# --------------------------------------------------------------------------

def test_explicit_endpoint_wins(monkeypatch):
    monkeypatch.setenv(hub_source.ENV_ENDPOINT, "https://env.invalid/a")
    monkeypatch.setenv("HF_ENDPOINT", "https://other.invalid")
    chosen = hub_source.resolve_endpoint(
        explicit="https://explicit.invalid", probe=False
    )
    assert chosen.url == "https://explicit.invalid"
    assert chosen.provenance == "explicit-argument"


def test_project_env_wins_over_generic_env(monkeypatch):
    monkeypatch.setenv(hub_source.ENV_ENDPOINT, "https://example.invalid/a")
    monkeypatch.setenv("HF_ENDPOINT", "https://other.invalid")
    chosen = hub_source.resolve_endpoint(probe=False)
    assert chosen.url == "https://example.invalid/a"


def test_generic_hf_endpoint_is_honoured(monkeypatch):
    monkeypatch.delenv(hub_source.ENV_ENDPOINT, raising=False)
    monkeypatch.setenv("HF_ENDPOINT", "https://mirror.example.invalid/")
    chosen = hub_source.resolve_endpoint(probe=False)
    assert chosen.url == "https://mirror.example.invalid"  # trailing slash normalised
    assert chosen.provenance == "env:HF_ENDPOINT"


def test_real_endpoint_is_probed_before_the_mirror(monkeypatch):
    """Outside CN the real endpoint answers first; the mirror is the fallback."""
    monkeypatch.delenv(hub_source.ENV_ENDPOINT, raising=False)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.setenv(hub_source.ENV_MIRROR, "https://mirror.example.invalid")
    answers = {hub_source.REAL_ENDPOINT: True, "https://mirror.example.invalid": True}
    monkeypatch.setattr(hub_source, "_probe", lambda url, repo, timeout: answers[url])
    chosen = hub_source.resolve_endpoint(probe=True)
    assert chosen.url == hub_source.REAL_ENDPOINT
    assert chosen.provenance == "probe:real-endpoint-reachable"


def test_mirror_is_used_when_the_real_endpoint_is_blocked(monkeypatch):
    """This is the CN-network case: huggingface.co is SNI-blocked, the mirror is not."""
    monkeypatch.delenv(hub_source.ENV_ENDPOINT, raising=False)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.setenv(hub_source.ENV_MIRROR, "https://mirror.example.invalid")
    monkeypatch.setattr(
        hub_source,
        "_probe",
        lambda url, repo, timeout: url == "https://mirror.example.invalid",
    )
    chosen = hub_source.resolve_endpoint(probe=True)
    assert chosen.url == "https://mirror.example.invalid"
    assert chosen.provenance == "probe:mirror-reachable"


def test_no_reachable_endpoint_raises(monkeypatch):
    monkeypatch.delenv(hub_source.ENV_ENDPOINT, raising=False)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.setenv(hub_source.ENV_MIRROR, "https://mirror.example.invalid")
    monkeypatch.setattr(hub_source, "_probe", lambda url, repo, timeout: False)
    with pytest.raises(hub_source.HubUnavailable) as excinfo:
        hub_source.resolve_endpoint(probe=True)
    # The message must name both endpoints: no silent local fallback exists.
    assert hub_source.REAL_ENDPOINT in str(excinfo.value)
    assert "https://mirror.example.invalid" in str(excinfo.value)


# --------------------------------------------------------------------------
# hub: references
# --------------------------------------------------------------------------

def test_parse_short_ref_uses_config_defaults(config_file):
    cfg = hub_source.load_consume_config(config_file)
    ref = hub_source.parse_ref(cfg.artifact("predictions"), repo=cfg.repo, revision=cfg.revision)
    assert (ref.repo, ref.revision) == (cfg.repo, cfg.revision)
    assert ref.filename == "results/fix1_siglip2_coral_seed17/test_predictions.npz"
    assert ref.spec.startswith("hub:")


def test_parse_explicit_ref(config_file):
    ref = hub_source.parse_ref("hub:some/repo@v9.9:results/x.json")
    assert (ref.repo, ref.revision, ref.filename) == ("some/repo", "v9.9", "results/x.json")


@pytest.mark.parametrize("spec", ["", "local/path.npz", "hub:", "hub:repo@rev", "hub:repo@rev:"])
def test_bad_refs_are_rejected(spec):
    if spec == "":
        assert hub_source.is_ref(spec) is False
        return
    if spec.startswith("hub:"):
        with pytest.raises(ValueError):
            hub_source.parse_ref(spec)
    else:
        assert hub_source.is_ref(spec) is False


def test_local_path_is_never_treated_as_a_ref():
    assert hub_source.is_ref("results/v07_release/bias_matrix.json") is False
    path, receipt = hub_source.resolve_local("results/x.json", kind="predictions")
    assert receipt is None
    assert str(path).endswith("results/x.json")


# --------------------------------------------------------------------------
# The consume config
# --------------------------------------------------------------------------

def test_config_defaults_to_the_release_repo_and_pinned_tag(tmp_path):
    path = tmp_path / "minimal.yaml"
    path.write_text("artifacts:\n  predictions: hub:results/x.npz\n", encoding="utf-8")
    cfg = hub_source.load_consume_config(path)
    assert cfg.repo == hub_source.RELEASE_REPO
    assert cfg.revision == hub_source.DEFAULT_REVISION
    assert cfg.endpoint == ""  # resolved at run time, never a local fallback
    assert cfg.require_network is False


def test_config_rejects_non_string_artifact(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("artifacts:\n  predictions: 42\n", encoding="utf-8")
    with pytest.raises(ValueError):
        hub_source.load_consume_config(path)


def test_config_rejects_non_numeric_expectation(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("expect:\n  coverage: 'high'\n", encoding="utf-8")
    with pytest.raises(ValueError):
        hub_source.load_consume_config(path)


def test_hub_kwargs_thread_require_network(config_file):
    cfg = hub_source.load_consume_config(config_file)
    kwargs = cfg.hub_kwargs()
    assert kwargs["require_network"] is True
    assert kwargs["repo"] == cfg.repo
    assert kwargs["endpoint"] is None  # "" must not pin an empty endpoint


def test_unknown_artifact_key_names_the_known_keys(config_file):
    cfg = hub_source.load_consume_config(config_file)
    with pytest.raises(KeyError) as excinfo:
        cfg.artifact("nope")
    assert "predictions" in str(excinfo.value)


# --------------------------------------------------------------------------
# The expectations the smoke test asserts
# --------------------------------------------------------------------------

GOOD = {
    "n": 109,
    "macro_f1_with_abstentions": 0.35780803895557994,
    "weighted_kappa_covered": 0.4226804123711341,
    "coverage": 0.8807339449541285,
    "abstention_rate": 0.11926605504587151,
    "ece_all": 0.0906259136378478,
    "ece_covered": 0.08561169349640227,
    "dangerous_underclassification_fnr_covered": 0.3333333333333333,
    "dangerous_underclassification_rate_all": 0.29357798165137616,
}

EXPECT = {
    "n": 109,
    "macro_f1_with_abstentions": 0.3578,
    "weighted_kappa_covered": 0.4227,
    "coverage": 0.8807,
    "abstention_rate": 0.1193,
    "ece_all": 0.0906,
    "ece_covered": 0.0856,
    "dangerous_underclassification_fnr_covered": 0.3333,
    "dangerous_underclassification_rate_all": 0.2936,
}


def test_the_published_metrics_satisfy_the_expectations():
    """The exact numbers the release card publishes, pulled from the Hub."""
    assert hub_source.check_expectations(GOOD, EXPECT) == []


def test_wrong_denominator_fails():
    """108 events is the cross-model comparison set, not the 109-event split."""
    shifted = dict(GOOD, n=108)
    failures = hub_source.check_expectations(shifted, EXPECT)
    assert any(line.startswith("n:") for line in failures)


def test_swapped_metric_fails():
    """kappa over all rows is a different quantity from covered kappa."""
    swapped = dict(GOOD, weighted_kappa_covered=0.4913)
    failures = hub_source.check_expectations(swapped, EXPECT)
    assert any("weighted_kappa_covered" in line for line in failures)


def test_missing_metric_fails():
    failures = hub_source.check_expectations({"n": 109}, EXPECT)
    assert len(failures) == len(EXPECT) - 1


def test_explicit_tolerance_is_honoured():
    exact = {"coverage": {"value": 0.8807339449541285, "tol": 1e-9}}
    assert hub_source.check_expectations({"coverage": 0.8807339449541285}, exact) == []
    # A 1e-4 drift is inside the default tolerance but outside a 1e-9 one.
    tight = {"coverage": {"value": 0.8807, "tol": 1e-9}}
    assert hub_source.check_expectations({"coverage": 0.8807339449541285}, tight) != []
    assert hub_source.check_expectations({"coverage": 0.8807339449541285}, {"coverage": 0.8807}) == []


# --------------------------------------------------------------------------
# Receipt honesty (hazard H-HUBCACHE)
# --------------------------------------------------------------------------

def _receipt(**kwargs):
    base = dict(
        repo=hub_source.RELEASE_REPO,
        revision="v0.7",
        endpoint="https://mirror.example.invalid",
        endpoint_provenance="env:HF_ENDPOINT",
        resolved_commit="af40a912ed8c57e1f11518c1063164e173ce5dca",
        requested_at_utc="2026-10-12T00:00:00Z",
    )
    base.update(kwargs)
    return hub_source.Receipt(**base)


def _file(**kwargs):
    base = dict(
        filename="results/x.json",
        path="/cache/x.json",
        bytes=10,
        sha256="0" * 64,
        already_cached=False,
        snapshot_commit="af40a912ed8c57e1f11518c1063164e173ce5dca",
    )
    base.update(kwargs)
    return hub_source.FetchedFile(**base)


def test_a_fetched_receipt_is_not_all_cache():
    receipt = _receipt(files=[_file()])
    assert receipt.all_from_cache() is False
    assert "WARNING" not in receipt.summary_line()
    assert receipt.to_dict()["all_from_cache"] is False


def test_a_cache_only_receipt_says_so():
    """A receipt that names an endpoint which never answered must say so."""
    receipt = _receipt(
        endpoint="http://127.0.0.1:1",
        endpoint_provenance="env:IDDSI_HF_ENDPOINT",
        resolved_commit=None,
        files=[_file(already_cached=True, snapshot_commit=None)],
    )
    assert receipt.all_from_cache() is True
    assert "served-entirely-from-local-cache" in receipt.summary_line()
    assert receipt.to_dict()["all_from_cache"] is True


def test_mirror_cache_mismatch_is_flagged():
    receipt = _receipt(files=[_file(snapshot_commit="deadbeef")])
    assert receipt.commit_mismatch() is True
    assert "served-snapshot-differs-from-resolved-commit" in receipt.summary_line()


def test_empty_receipt_is_not_all_cache():
    assert _receipt().all_from_cache() is False


def test_receipt_survives_a_json_round_trip(tmp_path):
    receipt = _receipt(files=[_file()])
    out = hub_source.write_receipt(tmp_path / "nested" / "receipt.json", receipt)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["repo"] == hub_source.RELEASE_REPO
    assert payload["files"][0]["sha256"] == "0" * 64
    assert payload["all_from_cache"] is False


# --------------------------------------------------------------------------
# The weight-file question the capture path must answer from the Hub
# --------------------------------------------------------------------------

def test_weight_suffixes_exclude_eval_evidence():
    """test_predictions.npz is eval evidence, not a weight file."""
    assert not any(name.endswith(hub_source.WEIGHT_SUFFIXES) for name in
                   ["results/fix1_siglip2_coral_seed17/test_predictions.npz"])
    assert "best.pt".endswith(hub_source.WEIGHT_SUFFIXES)
    assert "model.safetensors".endswith(hub_source.WEIGHT_SUFFIXES)


def test_unsafe_rfilenames_are_refused():
    for bad in ("/etc/passwd", "../escape.json", ""):
        with pytest.raises(ValueError):
            hub_source.fetch([bad], endpoint="https://example.invalid")
