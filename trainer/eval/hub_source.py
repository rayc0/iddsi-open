"""Resolve and download released IDDSI-Open artifacts from the Hugging Face Hub.

This module is the single implementation of "pull the release from the Hub". It
exists because the release is only genuinely consumed when our own pipelines
read the published bytes instead of a local checkout.

Endpoint selection is explicit and recorded, never silent:

1. an explicit ``endpoint=`` argument;
2. ``IDDSI_HF_ENDPOINT`` (project-specific, wins over the generic variable);
3. ``HF_ENDPOINT`` (the generic huggingface_hub variable);
4. auto-probe the canonical endpoint;
5. probe the mirror named by ``IDDSI_HF_MIRROR``, if the operator set one — a
   CN-side deployment detail that is deliberately NOT hardcoded here, because
   ``ops/scan_release_leaks.sh`` BLOCKs on mirror hostnames;
6. otherwise raise :class:`HubUnavailable`. There is no silent local fallback.

Every fetch writes a receipt that names the endpoint that actually served the
bytes, the resolved commit, and the SHA-256 of each file, so a downstream
metric can always be traced to the exact artifact it was computed from.

No model weights are published with this release (see ``hf/model_README.md``:
the weights are withheld under the wk1 kill rule). What the Hub ships is code,
eval results, and the flow-test grader, so those are the artifacts this module
fetches. Nothing here turns a download into a verdict: the release never issues
one.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

try:  # PyYAML is a declared trainer dependency; keep the import local to the
    # consume-config loader so a minimal install without YAML still resolves
    # plain ``hub:`` references.
    import yaml
except ImportError:  # pragma: no cover - depends on the environment
    yaml = None  # type: ignore[assignment]

# The public face of this release (see README.md and hf/model_README.md).
RELEASE_REPO = "lingualeap/iddsi-open-siglip2-v0"

# Default pinned release tag. ``main`` carries unreleased launch-prep commits, so
# a reproducible consumption pins a tag. Override with IDDSI_HUB_REVISION or
# --hub-revision.
DEFAULT_REVISION = "v0.7"

REAL_ENDPOINT = "https://huggingface.co"
ENV_MIRROR = "IDDSI_HF_MIRROR"


def __getattr__(name: str):
    """Resolve the legacy ``MIRROR_ENDPOINT`` name from the environment.

    Kept for callers (and tests) that read the module attribute directly. It is
    deliberately a function rather than a constant: a constant would put a mirror
    hostname in the shipped source, which ``ops/scan_release_leaks.sh`` BLOCKs.
    """
    if name == "MIRROR_ENDPOINT":
        return os.environ.get(ENV_MIRROR, "")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def mirror_endpoint() -> str | None:
    """The operator-supplied mirror endpoint, or None when unset.

    Kept as a function rather than a module constant so that no mirror hostname
    appears as a literal anywhere in this file.
    """
    value = os.environ.get(ENV_MIRROR, "").strip()
    return value or None

ENV_ENDPOINT = "IDDSI_HF_ENDPOINT"
ENV_REPO = "IDDSI_HUB_REPO"
ENV_REVISION = "IDDSI_HUB_REVISION"
ENV_CACHE = "IDDSI_HUB_CACHE"

PROBE_TIMEOUT = 8.0
REF_PREFIX = "hub:"


class HubUnavailable(RuntimeError):
    """Raised when no Hub endpoint could be resolved or reached.

    Fail-closed by design: a caller that asked for release bytes gets an error,
    never a silently substituted local file.
    """


@dataclass(frozen=True)
class Endpoint:
    """A Hub endpoint plus how it was chosen."""

    url: str
    provenance: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.url} ({self.provenance})"


@dataclass(frozen=True)
class HubRef:
    """A resolved reference to one file inside one Hub repo."""

    repo: str
    revision: str
    filename: str

    @property
    def spec(self) -> str:
        return f"{REF_PREFIX}{self.repo}@{self.revision}:{self.filename}"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.spec


@dataclass
class FetchedFile:
    """One downloaded artifact with the bytes it resolved to."""

    filename: str
    path: str
    bytes: int
    sha256: str
    already_cached: bool
    snapshot_commit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "path": self.path,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "already_cached": self.already_cached,
            "snapshot_commit": self.snapshot_commit,
        }


@dataclass
class Receipt:
    """Provenance for one fetch: what was asked for and what was served."""

    repo: str
    revision: str
    endpoint: str
    endpoint_provenance: str
    resolved_commit: str | None
    requested_at_utc: str
    files: list[FetchedFile] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "revision": self.revision,
            "endpoint": self.endpoint,
            "endpoint_provenance": self.endpoint_provenance,
            "resolved_commit": self.resolved_commit,
            "requested_at_utc": self.requested_at_utc,
            "n_files": len(self.files),
            "total_bytes": sum(item.bytes for item in self.files),
            "commit_mismatch": self.commit_mismatch(),
            "all_from_cache": self.all_from_cache(),
            "files": [item.to_dict() for item in self.files],
        }

    def commit_mismatch(self) -> bool:
        """True when a served snapshot is not the commit the endpoint reported.

        A mirror can answer from its own cache; when that happens the bytes on
        disk are not the bytes the endpoint claims ``revision`` resolves to, and
        a receipt must say so rather than look identical.
        """
        if not self.resolved_commit:
            return False
        return any(
            item.snapshot_commit is not None and item.snapshot_commit != self.resolved_commit
            for item in self.files
        )

    def all_from_cache(self) -> bool:
        """True when no file was fetched: every byte was already local.

        This is the difference between "consumed the release" and "read a cache
        entry". A receipt whose endpoint was unreachable still names that
        endpoint, so without this flag the receipt would claim the release was
        pulled from a Hub that never answered. Fail closed on it: a caller that
        needs proof of consumption must check it (see ``require_network``).
        """
        return bool(self.files) and all(item.already_cached for item in self.files)

    def summary_line(self) -> str:
        commit = self.resolved_commit or "unresolved"
        line = (
            f"hub: repo={self.repo}@{self.revision} commit={commit} "
            f"endpoint={self.endpoint} provenance={self.endpoint_provenance} "
            f"files={len(self.files)} bytes={sum(item.bytes for item in self.files)}"
        )
        if self.all_from_cache():
            line += " WARNING=served-entirely-from-local-cache"
        if self.commit_mismatch():
            line += " WARNING=served-snapshot-differs-from-resolved-commit"
        return line


def _normalise(url: str) -> str:
    return url.strip().rstrip("/")


def _probe(endpoint: str, repo: str, timeout: float) -> bool:
    """Return True when ``endpoint`` answers the repo-info API for ``repo``."""
    url = f"{_normalise(endpoint)}/api/models/{repo}"
    request = urllib.request.Request(url, method="GET", headers={"User-Agent": "iddsi-open-hub-consume"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https URLs
            return response.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def resolve_endpoint(
    repo: str = RELEASE_REPO,
    *,
    explicit: str | None = None,
    timeout: float = PROBE_TIMEOUT,
    probe: bool = True,
) -> Endpoint:
    """Choose a Hub endpoint and record how it was chosen.

    ``probe=False`` skips the reachability probes and only honours explicit and
    environment settings; used by tests that must not touch the network.
    """
    if explicit:
        return Endpoint(_normalise(explicit), "explicit-argument")
    for name in (ENV_ENDPOINT, "HF_ENDPOINT"):
        value = os.environ.get(name, "").strip()
        if value:
            return Endpoint(_normalise(value), f"env:{name}")
    if not probe:
        raise HubUnavailable(
            "no Hub endpoint configured and probing disabled; set "
            f"{ENV_ENDPOINT} or pass endpoint="
        )
    if _probe(REAL_ENDPOINT, repo, timeout):
        return Endpoint(REAL_ENDPOINT, "probe:real-endpoint-reachable")
    mirror = mirror_endpoint()
    if mirror and _probe(mirror, repo, timeout):
        return Endpoint(mirror, "probe:mirror-reachable")
    raise HubUnavailable(
        "no Hugging Face endpoint is reachable from this host. Tried "
        f"{REAL_ENDPOINT} (the real endpoint, used outside CN) and "
        f"{mirror_endpoint() or 'the mirror named by ' + ENV_MIRROR} "
        f"(the operator-supplied mirror). Set {ENV_ENDPOINT} to pin one "
        "explicitly, or run where at least one of them resolves."
    )


def default_repo() -> str:
    return os.environ.get(ENV_REPO, "").strip() or RELEASE_REPO


def default_revision() -> str:
    return os.environ.get(ENV_REVISION, "").strip() or DEFAULT_REVISION


def default_cache_dir() -> str | None:
    return os.environ.get(ENV_CACHE, "").strip() or None


def is_ref(value: str) -> bool:
    """True when ``value`` is a ``hub:`` reference rather than a local path."""
    return isinstance(value, str) and value.startswith(REF_PREFIX)


def parse_ref(
    spec: str,
    *,
    repo: str | None = None,
    revision: str | None = None,
) -> HubRef:
    """Parse ``hub:<repo>@<revision>:<rfilename>`` or ``hub:<rfilename>``.

    ``rfilename`` never contains ``@`` or ``:``, so the explicit form is
    unambiguous. Anything that is not a ``hub:`` reference is rejected here
    rather than being quietly treated as a local path.
    """
    if not is_ref(spec):
        raise ValueError(f"not a hub reference (expected a leading {REF_PREFIX!r}): {spec!r}")
    body = spec[len(REF_PREFIX) :]
    if "@" in body:
        head, filename = body.split(":", 1) if ":" in body else (body, "")
        if not filename:
            raise ValueError(f"hub reference is missing ':<rfilename>': {spec!r}")
        repo_part, revision_part = head.split("@", 1)
        if not repo_part or not revision_part:
            raise ValueError(f"hub reference needs both repo and revision: {spec!r}")
        return HubRef(repo=repo_part, revision=revision_part, filename=filename)
    if not body:
        raise ValueError(f"hub reference is missing an rfilename: {spec!r}")
    return HubRef(
        repo=repo or default_repo(),
        revision=revision or default_revision(),
        filename=body,
    )


def _sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            total += len(chunk)
    return total, digest.hexdigest()


def _snapshot_commit(path: Path) -> str | None:
    """The commit SHA huggingface_hub resolved, read off the cache layout."""
    parts = path.parts
    if "snapshots" in parts:
        index = len(parts) - 1 - parts[::-1].index("snapshots")
        if index + 1 < len(parts):
            return parts[index + 1]
    return None


def _resolved_commit(repo: str, revision: str, endpoint: str) -> str | None:
    """Ask the endpoint which commit ``revision`` resolves to, for pinning."""
    try:
        from huggingface_hub import HfApi
    except ImportError:  # pragma: no cover - huggingface_hub is a declared dep
        return None
    try:
        info = HfApi(endpoint=endpoint).model_info(repo_id=repo, revision=revision)
    except Exception:  # noqa: BLE001 - a probe failure must not fail the fetch
        return None
    return getattr(info, "sha", None)


def fetch(
    filenames: Sequence[str],
    *,
    repo: str | None = None,
    revision: str | None = None,
    endpoint: str | Endpoint | None = None,
    cache_dir: str | None = None,
    probe_timeout: float = PROBE_TIMEOUT,
    require_network: bool = False,
) -> Receipt:
    """Download ``filenames`` from the Hub and return a provenance receipt.

    ``require_network=True`` re-downloads even when the cache already holds the
    file, so the named endpoint provably served the bytes. Use it wherever a
    green result has to mean "the release was consumed", not "a cache entry was
    read" — a smoke test that never touched the Hub proves nothing.

    Raises :class:`HubUnavailable` when no endpoint can be resolved, and lets
    ``huggingface_hub`` errors propagate (a missing file is an error, not an
    empty result).
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise HubUnavailable(
            "huggingface_hub is not installed; install the trainer extras with "
            "`python -m pip install -e 'trainer[hub]'` to consume the Hub release"
        ) from exc

    repo = repo or default_repo()
    revision = revision or default_revision()
    chosen = endpoint if isinstance(endpoint, Endpoint) else resolve_endpoint(
        repo, explicit=endpoint, timeout=probe_timeout
    )
    cache_dir = cache_dir or default_cache_dir()

    receipt = Receipt(
        repo=repo,
        revision=revision,
        endpoint=chosen.url,
        endpoint_provenance=chosen.provenance,
        resolved_commit=_resolved_commit(repo, revision, chosen.url),
        requested_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    for filename in filenames:
        if not filename or filename.startswith("/") or ".." in filename.split("/"):
            raise ValueError(f"unsafe rfilename: {filename!r}")
        already_cached = False if require_network else _in_cache(
            repo, filename, revision, cache_dir
        )
        target = Path(
            hf_hub_download(
                repo_id=repo,
                filename=filename,
                revision=revision,
                cache_dir=cache_dir,
                endpoint=chosen.url,
                force_download=require_network,
            )
        )
        size, checksum = _sha256(target)
        receipt.files.append(
            FetchedFile(
                filename=filename,
                path=str(target),
                bytes=size,
                sha256=checksum,
                already_cached=already_cached,
                snapshot_commit=_snapshot_commit(target),
            )
        )
    return receipt


def _in_cache(repo: str, filename: str, revision: str, cache_dir: str | None) -> bool:
    """True when the artifact was already in the local cache before fetching."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:  # pragma: no cover - huggingface_hub is a declared dep
        return False
    try:
        return try_to_load_from_cache(
            repo_id=repo, filename=filename, revision=revision, cache_dir=cache_dir
        ) is not None
    except Exception:  # noqa: BLE001 - a cache probe is best-effort bookkeeping
        return False


def write_receipt(path: str | Path, receipt: Receipt) -> Path:
    """Write ``receipt`` as JSON next to whatever consumed it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def resolve_local(
    spec: str,
    *,
    kind: str,
    repo: str | None = None,
    revision: str | None = None,
    endpoint: str | None = None,
    cache_dir: str | None = None,
    require_network: bool = False,
) -> tuple[Path, Receipt | None]:
    """Return ``(local_path, receipt)`` for a local path or a ``hub:`` ref.

    ``kind`` only appears in error messages ("predictions", "threshold").
    """
    if not is_ref(spec):
        return Path(spec).expanduser(), None
    ref = parse_ref(spec, repo=repo, revision=revision)
    receipt = fetch(
        [ref.filename],
        repo=ref.repo,
        revision=ref.revision,
        endpoint=endpoint,
        cache_dir=cache_dir,
        require_network=require_network,
    )
    return Path(receipt.files[0].path), receipt


def threshold_from_file(path: str | Path) -> float:
    """Read the validation-calibrated ``threshold`` out of a metrics JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        return float(payload["threshold"])
    except KeyError as exc:
        raise ValueError(f"{path} has no 'threshold' key; pass a number instead") from exc


def merge_receipts(receipts: Iterable[Receipt | None]) -> Receipt | None:
    """Merge several receipts into one, or None when nothing came from the Hub."""
    present = [item for item in receipts if item is not None]
    if not present:
        return None
    first = present[0]
    if len(present) == 1:
        return first
    return Receipt(
        repo=first.repo,
        revision=first.revision,
        endpoint=first.endpoint,
        endpoint_provenance=first.endpoint_provenance,
        resolved_commit=first.resolved_commit,
        requested_at_utc=first.requested_at_utc,
        files=[item for receipt in present for item in receipt.files],
    )


# --------------------------------------------------------------------------
# What the release publishes, asked of the Hub rather than assumed
# --------------------------------------------------------------------------

# Weight-file extensions. The release shape is locked: code + eval results +
# flow-test grader + the negative finding, and ops/stage_release.sh refuses to
# stage any of these. A consumer must be able to *verify* that from the Hub
# file list rather than trust this comment.
WEIGHT_SUFFIXES = (".pt", ".pth", ".safetensors", ".bin", ".ckpt")


def published_weight_files(
    repo: str | None = None,
    *,
    revision: str | None = None,
    endpoint: str | Endpoint | None = None,
    probe_timeout: float = PROBE_TIMEOUT,
) -> list[str]:
    """Return the weight files the Hub actually publishes for ``repo@revision``.

    Raises :class:`HubUnavailable` when the file list cannot be retrieved. A
    consumer that needs to state "no weights are published" must fail closed
    here: an unreachable Hub is not evidence that weights exist, and it is not
    evidence that they do not.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise HubUnavailable(
            "huggingface_hub is not installed; install the trainer extras with "
            "`python -m pip install -e 'trainer[hub]'` to consume the Hub release"
        ) from exc

    repo = repo or default_repo()
    revision = revision or default_revision()
    chosen = endpoint if isinstance(endpoint, Endpoint) else resolve_endpoint(
        repo, explicit=endpoint, timeout=probe_timeout
    )
    try:
        info = HfApi(endpoint=chosen.url).model_info(repo_id=repo, revision=revision)
    except Exception as exc:  # noqa: BLE001 - any API failure must fail closed
        raise HubUnavailable(
            f"could not list files for {repo}@{revision} via {chosen.url}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    siblings = getattr(info, "siblings", None) or []
    return sorted(
        item.rfilename
        for item in siblings
        if item.rfilename.lower().endswith(WEIGHT_SUFFIXES)
    )


# --------------------------------------------------------------------------
# The consume config — one YAML both the pipelines and the CI smoke test read
# --------------------------------------------------------------------------

# Default relative tolerance for an ``expect:`` entry given as a bare number.
DEFAULT_TOLERANCE = 1e-3


@dataclass(frozen=True)
class ConsumeConfig:
    """A parsed ``configs/consume/*.yaml`` — what to pull and what to assert."""

    path: str
    repo: str
    revision: str
    endpoint: str
    cache_dir: str
    require_network: bool
    artifacts: dict[str, str]
    expect: dict[str, Any]
    eval: dict[str, Any]

    def hub_kwargs(self) -> dict[str, Any]:
        """The keyword arguments ``resolve_local`` / ``fetch`` take."""
        return {
            "repo": self.repo,
            "revision": self.revision,
            "endpoint": self.endpoint or None,
            "cache_dir": self.cache_dir or None,
            "require_network": self.require_network,
        }

    def artifact(self, key: str) -> str:
        try:
            return self.artifacts[key]
        except KeyError as exc:
            raise KeyError(
                f"{self.path} has no artifacts.{key}; known keys: "
                f"{sorted(self.artifacts)}"
            ) from exc


def load_consume_config(path: str | Path) -> ConsumeConfig:
    """Load and validate a consume config.

    An empty ``hub.endpoint`` means "resolve it at run time" (explicit argument,
    then ``IDDSI_HF_ENDPOINT``, then ``HF_ENDPOINT``, then the reachability
    probes); it never means a local fallback.
    """
    if yaml is None:  # pragma: no cover - depends on the environment
        raise HubUnavailable(
            "PyYAML is not installed; the consume config is YAML and cannot be read"
        )
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    hub = raw.get("hub") or {}
    artifacts = raw.get("artifacts") or {}
    expect = raw.get("expect") or {}
    evaluation = raw.get("eval") or {}
    if not isinstance(hub, dict) or not isinstance(artifacts, dict):
        raise ValueError(f"{path}: 'hub' and 'artifacts' must both be mappings")
    for key, value in artifacts.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path}: artifacts.{key} must be a non-empty string")
    for key, value in expect.items():
        if isinstance(value, dict):
            if "value" not in value:
                raise ValueError(f"{path}: expect.{key} needs a 'value' key")
        elif not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{path}: expect.{key} must be a number or a mapping")
    return ConsumeConfig(
        path=str(path),
        repo=str(hub.get("repo") or "").strip() or default_repo(),
        revision=str(hub.get("revision") or "").strip() or default_revision(),
        endpoint=str(hub.get("endpoint") or "").strip(),
        cache_dir=str(hub.get("cache_dir") or "").strip(),
        require_network=bool(hub.get("require_network", False)),
        artifacts={str(k): str(v) for k, v in artifacts.items()},
        expect=dict(expect),
        eval=dict(evaluation),
    )


def _expected_entry(spec: Any) -> tuple[float, float]:
    """Normalise one ``expect:`` entry into ``(value, tolerance)``."""
    if isinstance(spec, dict):
        return float(spec["value"]), float(spec.get("tol", DEFAULT_TOLERANCE))
    return float(spec), DEFAULT_TOLERANCE


def check_expectations(
    measured: dict[str, Any],
    expect: dict[str, Any],
) -> list[str]:
    """Compare measured metrics against ``expect:``; return the mismatch list.

    ``n`` is an exact integer count (a denominator, never a tolerance). Every
    other entry is a float compared within its tolerance, which defaults to
    ``DEFAULT_TOLERANCE`` — wide enough for a 4-decimal published figure, tight
    enough that a wrong metric or a substituted artifact still fails.
    """
    failures: list[str] = []
    for key, spec in sorted(expect.items()):
        if key not in measured:
            failures.append(f"{key}: expected but not measured")
            continue
        got = measured[key]
        if key == "n":
            wanted = int(spec if not isinstance(spec, dict) else spec["value"])
            if int(got) != wanted:
                failures.append(f"n: measured {got}, expected {wanted}")
            continue
        value, tolerance = _expected_entry(spec)
        try:
            got_float = float(got)
        except (TypeError, ValueError):
            failures.append(f"{key}: measured {got!r} is not numeric")
            continue
        if abs(got_float - value) > tolerance:
            failures.append(
                f"{key}: measured {got_float:.6f}, expected {value:.6f} "
                f"(tolerance {tolerance:g})"
            )
    return failures
