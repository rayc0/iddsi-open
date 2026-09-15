from __future__ import annotations

import io
import urllib.error
import unittest
from unittest.mock import patch

from harvest.http import HTTPClient


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = io.BytesIO(body)
        self.headers: dict[str, str] = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)


class HTTPClientTests(unittest.TestCase):
    def test_429_then_success_honors_retry_after(self) -> None:
        url = "https://upload.wikimedia.org/example.jpg"
        rate_limited = urllib.error.HTTPError(
            url,
            429,
            "Too Many Requests",
            {"Retry-After": "3"},
            io.BytesIO(),
        )
        clock = FakeClock()
        client = HTTPClient(
            user_agent="test",
            min_interval=0,
            clock=clock.monotonic,
            sleep=clock.sleep,
        )

        with patch(
            "harvest.http.urllib.request.urlopen",
            side_effect=[rate_limited, FakeResponse(b"recovered")],
        ):
            self.assertEqual(client.get_bytes(url), b"recovered")

        self.assertEqual(clock.sleeps, [3.0])
        self.assertEqual(
            client.stats_snapshot()["upload.wikimedia.org"],
            {"requests": 2, "429s": 1, "retries": 1},
        )

    def test_same_host_minimum_interval_uses_injected_clock(self) -> None:
        clock = FakeClock()
        request_times: list[float] = []
        client = HTTPClient(
            user_agent="test",
            min_interval=1.0,
            clock=clock.monotonic,
            sleep=clock.sleep,
        )

        def respond(request, timeout):
            request_times.append(clock.monotonic())
            return FakeResponse(b"ok")

        with patch("harvest.http.urllib.request.urlopen", side_effect=respond):
            client.get_bytes("https://commons.wikimedia.org/first")
            client.get_bytes("https://commons.wikimedia.org/second")
            client.get_bytes("https://upload.wikimedia.org/independent")

        self.assertEqual(request_times, [0.0, 1.0, 1.0])
        self.assertEqual(clock.sleeps, [1.0])
        self.assertEqual(
            client.stats_snapshot()["commons.wikimedia.org"],
            {"requests": 2, "429s": 0, "retries": 0},
        )


if __name__ == "__main__":
    unittest.main()
