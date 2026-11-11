"""Polite HTTP for the lkml.iu.edu hypermail mirror.

Stdlib only on purpose: this is a long-running crawler, and the fewer moving
parts the better. Rate limiting is per-process and deliberately conservative —
this is a volunteer-run mirror, not an API we own.
"""
from __future__ import annotations

import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://lkml.iu.edu/hypermail/linux/kernel"
USER_AGENT = (
    "linus-pr/0.1 (corpus research; contact: nick@localhost; "
    "single-user academic dataset build)"
)
DEFAULT_DELAY = 0.35  # seconds between requests
MAX_RETRIES = 3


class FetchError(RuntimeError):
    pass


class Fetcher:
    def __init__(
        self,
        base: str = BASE,
        *,
        delay: float = DEFAULT_DELAY,
        cache_dir: Path | None = None,
        sleep=time.sleep,
    ) -> None:
        self.base = base.rstrip("/")
        self.delay = delay
        self.cache_dir = cache_dir
        self._sleep = sleep
        self._last = 0.0
        self.requests_made = 0

    def url(self, *parts: str) -> str:
        return "/".join([self.base, *[p.strip("/") for p in parts]])

    def get(self, *parts: str, retries: int = MAX_RETRIES) -> str:
        """Fetch a page, honouring the delay and an on-disk raw cache."""
        rel = "/".join(p.strip("/") for p in parts) or "index.html"
        cache = None
        if self.cache_dir is not None:
            cache = self.cache_dir / rel
            if cache.exists():
                return cache.read_text(encoding="utf-8", errors="replace")

        url = self.url(*parts)
        last_err: Exception | None = None
        for attempt in range(retries):
            self._throttle()
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_err = exc
                self._sleep(2.0 ** attempt)
                continue
            self.requests_made += 1
            if "Making sure you&#39;re not a bot" in raw[:4096]:
                raise FetchError(f"bot check from {url}")
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(raw, encoding="utf-8")
            return raw
        raise FetchError(f"failed {retries}x: {url}: {last_err}")

    def _throttle(self) -> None:
        wait = self.delay - (time.monotonic() - self._last)
        if wait > 0:
            self._sleep(wait)
        self._last = time.monotonic()
