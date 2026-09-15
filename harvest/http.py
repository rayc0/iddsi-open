from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Mapping


class HTTPError(RuntimeError):
    pass


class HTTPClient:
    """Small injectable HTTP client; tests replace it with an in-memory fake."""

    def __init__(
        self,
        *,
        user_agent: str,
        timeout: float = 30.0,
        min_interval: float = 1.0,
        max_attempts: int = 5,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        if min_interval < 0:
            raise ValueError("min_interval must be non-negative")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.user_agent = user_agent
        self.timeout = timeout
        self.min_interval = min_interval
        self.max_attempts = max_attempts
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep
        self._last_request_at: dict[str, float] = {}
        self._host_stats: dict[str, dict[str, int]] = {}

    def stats_snapshot(self) -> dict[str, dict[str, int]]:
        """Return a JSON-serializable copy of per-host request/retry counters."""
        return {
            host: dict(counters)
            for host, counters in sorted(self._host_stats.items())
        }

    def _stats_for(self, host: str) -> dict[str, int]:
        return self._host_stats.setdefault(
            host,
            {"requests": 0, "429s": 0, "retries": 0},
        )

    def _wait_for_host(self, host: str) -> None:
        last_request = self._last_request_at.get(host)
        if last_request is not None:
            remaining = self.min_interval - (self._clock() - last_request)
            if remaining > 0:
                self._sleep(remaining)
        self._last_request_at[host] = self._clock()

    @staticmethod
    def _retry_after_seconds(value: str | None) -> float | None:
        if not value:
            return None
        try:
            return max(0.0, float(value.strip()))
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(value)
            except (TypeError, ValueError, OverflowError):
                return None
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())

    def _request(self, url: str, params: Mapping[str, Any] | None = None):
        if params:
            query = urllib.parse.urlencode(params)
            url = f"{url}{'&' if '?' in url else '?'}{query}"
        request = urllib.request.Request(
            url,
            headers={"User-Agent": self.user_agent, "Accept": "*/*"},
        )
        host = urllib.parse.urlparse(url).hostname or "<unknown>"
        counters = self._stats_for(host.lower())
        for attempt in range(1, self.max_attempts + 1):
            self._wait_for_host(host.lower())
            counters["requests"] += 1
            try:
                return urllib.request.urlopen(request, timeout=self.timeout)
            except urllib.error.HTTPError as exc:
                if exc.code == 429:
                    counters["429s"] += 1
                if exc.code not in {429, 503} or attempt >= self.max_attempts:
                    exc.close()
                    raise HTTPError(
                        f"request failed for {url} after {attempt} attempt(s): {exc}"
                    ) from exc
                counters["retries"] += 1
                retry_after = self._retry_after_seconds(
                    exc.headers.get("Retry-After") if exc.headers else None
                )
                delay = retry_after if retry_after is not None else float(2**attempt)
                exc.close()
                self._sleep(delay)
            except Exception as exc:  # urllib raises several unrelated exception types
                raise HTTPError(f"request failed for {url}: {exc}") from exc
        raise AssertionError("unreachable")

    def get_bytes(
        self,
        url: str,
        params: Mapping[str, Any] | None = None,
        *,
        max_bytes: int = 25_000_000,
    ) -> bytes:
        with self._request(url, params) as response:
            declared = response.headers.get("Content-Length")
            if declared and int(declared) > max_bytes:
                raise HTTPError(f"response exceeds {max_bytes} bytes")
            chunks: list[bytes] = []
            received = 0
            while True:
                chunk = response.read(min(1024 * 1024, max_bytes - received + 1))
                if not chunk:
                    break
                chunks.append(chunk)
                received += len(chunk)
                if received > max_bytes:
                    raise HTTPError(f"response exceeds {max_bytes} bytes")
            return b"".join(chunks)

    def get_text(
        self,
        url: str,
        params: Mapping[str, Any] | None = None,
        *,
        max_bytes: int = 100_000_000,
    ) -> str:
        return self.get_bytes(url, params, max_bytes=max_bytes).decode("utf-8-sig")

    def get_json(
        self,
        url: str,
        params: Mapping[str, Any] | None = None,
        *,
        max_bytes: int = 25_000_000,
    ) -> dict[str, Any]:
        try:
            value = json.loads(self.get_bytes(url, params, max_bytes=max_bytes))
        except json.JSONDecodeError as exc:
            raise HTTPError(f"invalid JSON from {url}: {exc}") from exc
        if not isinstance(value, dict):
            raise HTTPError(f"expected a JSON object from {url}")
        return value
