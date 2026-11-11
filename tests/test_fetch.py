"""Fetcher tests — no real network, no real sleeping."""
from __future__ import annotations

import urllib.error

import pytest

from linuspr import fetch as fetch_mod
from linuspr.fetch import Fetcher, FetchError

BOT_PAGE = "<html><title>Making sure you&#39;re not a bot!</title></html>"


class FakeResp:
    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture()
def calls():
    return []


@pytest.fixture()
def fake_urlopen(monkeypatch, calls):
    def install(body: str, fail_times: int = 0):
        state = {"fails": fail_times}

        def _open(req, timeout=None):
            calls.append((req.full_url, req.get_header("User-agent")))
            if state["fails"] > 0:
                state["fails"] -= 1
                raise urllib.error.URLError("boom")
            return FakeResp(body.encode())

        monkeypatch.setattr(fetch_mod.urllib.request, "urlopen", _open)

    return install


def test_url_joins_base_and_parts():
    f = Fetcher("https://x/hypermail/linux/kernel")
    assert f.url("2406.0", "/index.html") == (
        "https://x/hypermail/linux/kernel/2406.0/index.html"
    )


def test_get_identifies_itself_and_hits_the_right_url(fake_urlopen, calls):
    fake_urlopen("<html>ok</html>")
    f = Fetcher("https://x/hypermail/linux/kernel", delay=0.0)
    assert f.get("2406.0", "00001.html") == "<html>ok</html>"
    url, ua = calls[0]
    assert url.endswith("/2406.0/00001.html")
    assert ua.startswith("linus-pr/")


def test_empty_parts_cache_as_index_html(fake_urlopen, tmp_path):
    fake_urlopen("<html>months</html>")
    f = Fetcher("https://x/k", delay=0.0, cache_dir=tmp_path)
    f.get("")
    assert (tmp_path / "index.html").exists()


def test_cache_short_circuits_second_request(fake_urlopen, calls, tmp_path):
    fake_urlopen("<html>ok</html>")
    f = Fetcher("https://x/k", delay=0.0, cache_dir=tmp_path)
    f.get("2406.0", "00001.html")
    f.get("2406.0", "00001.html")
    assert len(calls) == 1
    assert f.requests_made == 1


def test_retries_then_succeeds(fake_urlopen, calls, tmp_path):
    fake_urlopen("<html>ok</html>", fail_times=2)
    slept: list[float] = []
    f = Fetcher("https://x/k", delay=0.0, cache_dir=tmp_path, sleep=slept.append)
    assert f.get("a", "b.html") == "<html>ok</html>"
    assert len(calls) == 3
    assert slept == [1.0, 2.0]  # exponential backoff


def test_gives_up_after_retries(fake_urlopen, tmp_path):
    fake_urlopen("<html>ok</html>", fail_times=10)
    f = Fetcher("https://x/k", delay=0.0, cache_dir=tmp_path, sleep=lambda s: None)
    with pytest.raises(FetchError):
        f.get("a", "b.html")


def test_bot_check_raises_and_is_not_cached(fake_urlopen, tmp_path):
    fake_urlopen(BOT_PAGE)
    f = Fetcher("https://x/k", delay=0.0, cache_dir=tmp_path)
    with pytest.raises(FetchError, match="bot check"):
        f.get("2406.0", "index.html")
    assert not (tmp_path / "2406.0" / "index.html").exists()


def test_delay_throttles_between_requests(fake_urlopen, calls):
    fake_urlopen("<html>ok</html>")
    slept: list[float] = []
    f = Fetcher("https://x/k", delay=5.0, sleep=slept.append)
    f.get("a")
    f.get("b")
    assert len(calls) == 2
    assert slept and slept[-1] > 4.0  # second call had to wait out the delay
