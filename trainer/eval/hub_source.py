"""Resolve and download released IDDSI-Open artifacts from the Hugging Face Hub.

This module is the single implementation of "pull the release from the Hub". It
exists because the release is only genuinely consumed when our own pipelines
read the published bytes instead of a local checkout.

Endpoint selection is explicit and recorded, never silent:

1. an explicit ``endpoint=`` argument;
2. ``IDDSI_HF_ENDPOINT`` (project-specific, wins over the generic variable);
3. ``HF_ENDPOINT`` (the generic huggingface_hub variable);
4. auto-probe ``https://huggingface.co`` — the real endpoint, used outside CN;
5. auto-probe ``https://hf-mirror.com`` — the CN mirror, used on this DGX box;
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

# The public face of this release (see README.md and hf/model_README.md).
RELEASE_REPO = "lingualeap/iddsi-open-siglip2-v0"

# Default pinned release tag. ``main`` carries unreleased launch-prep commits, so
# a reproducible consumption pins a tag. Override with IDDSI_HUB_REVISION or
# --hub-revision.
DEFAULT_REVISION = "v0.7"

REAL_ENDPOINT = "https://huggingface.co"
MIRROR_ENDPOINT = "https://hf-mirror.com"

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

    def summary_line(self) -> str:
        commit = self.resolved_commit or "unresolved"
        line = (
            f"hub: repo={self.repo}@{self.revision} commit={commit} "
            f"endpoint={self.endpoint} provenance={self.endpoint_provenance} "
            f"files={len(self.files)} bytes={sum(item.bytes for item in self.files)}"
        )
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
    if _probe(MIRROR_ENDPOINT, repo, timeout):
        return Endpoint(MIRROR_ENDPOINT, "probe:hf-mirror-reachable")
    raise HubUnavailable(
        "no Hugging Face endpoint is reachable from this host. Tried "
        f"{REAL_ENDPOINT} (the real endpoint, used outside CN) and "
        f"{MIRROR_ENDPOINT} (the CN mirror). Set {ENV_ENDPOINT} to pin one "
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
) -> Receipt:
    """Download ``filenames`` from the Hub and return a provenance receipt.

    Raises :class:`HubUnavailable` when no endpoint can be resolved, and lets
    ``huggingface_hub`` errors propagate (a missing file is an error, not an
    empty result).
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise HubUnavailable(
            "huggingface_hub is not installed; install the trainer extras with "
            "`python -m pip install -e '.[hub]'` to consume the Hub release"
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
        already_cached = _in_cache(repo, filename, revision, cache_dir)
        target = Path(
            hf_hub_download(
                repo_id=repo,
                filename=filename,
                revision=revision,
                cache_dir=cache_dir,
                endpoint=chosen.url,
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
