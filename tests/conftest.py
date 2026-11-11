"""Hermetic-tier fixtures.

Rules this file encodes (house convention):
  * no network in this tier — anything that talks upstream is monkeypatched
  * corpus fixtures are *real* saved pages (tests/fixtures/), not hand-written
    HTML, so parser regressions against the live mirror are caught here
  * index_slice_2406_0.html is the first 25 rows of the real 2406.0 index plus
    every Linus row (39 entries, 14 of them Linus) — trimmed from 1.9 MB so the
    repo stays small while still exercising the real markup
"""
from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def index_html() -> str:
    return (FIXTURES / "index_slice_2406_0.html").read_text(
        encoding="utf-8", errors="replace"
    )


@pytest.fixture()
def msg_html() -> str:
    return (FIXTURES / "msg_2406_0_00001.html").read_text(
        encoding="utf-8", errors="replace"
    )
