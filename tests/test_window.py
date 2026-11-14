"""train/window-lib.sh: the maintenance-window arithmetic, checked at fixed
times with the WINDOW_NOW test clock (no waiting, no real clock)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

LIB = Path(__file__).resolve().parent.parent / "train" / "window-lib.sh"
TZ = "America/New_York"


def at(when: str, expr: str, **env) -> str:
    """Evaluate a window-lib expression as if it were `when` (local to TZ)."""
    script = (f'. "{LIB}"; export WINDOW_NOW=$(TZ={TZ} date -d "{when}" +%s); {expr}')
    e = {"WINDOW_TZ": TZ, "PATH": "/usr/bin:/bin", **env}
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=e,
                          check=False).stdout.strip()


def hm(epoch_expr: str, when: str, **env) -> str:
    return at(when, f'TZ={TZ} date -d @$({epoch_expr}) "+%a %H:%M"', **env)


@pytest.mark.parametrize("when,is_open", [
    ("2026-11-13 19:59", False), ("2026-11-13 20:00", True), ("2026-11-13 23:30", True),
    ("2026-11-14 03:00", True), ("2026-11-14 07:59", True), ("2026-11-14 08:00", False),
    ("2026-11-14 12:00", False),
])
def test_window_open(when, is_open):
    assert at(when, "window_open && echo y || echo n") == ("y" if is_open else "n")


def test_hard_stop_is_end_minus_margin_on_the_right_day():
    assert hm("window_deadline", "2026-11-13 20:00") == "Sat 07:40"   # evening: tomorrow's end
    assert hm("window_deadline", "2026-11-14 03:00") == "Sat 07:40"   # small hours: today's end
    assert hm("window_deadline", "2026-11-13 20:00", STOP_MARGIN_MIN="45") == "Sat 07:15"


def test_next_start_skips_an_open_window():
    """Launched at 06:20 for tonight, it must wait for 20:00, not start with 80 min left."""
    assert hm("window_next_start", "2026-11-13 06:20") == "Fri 20:00"
    assert hm("window_next_open", "2026-11-13 06:20") == "Fri 06:20"   # (already open)
    assert hm("window_next_start", "2026-11-13 21:00") == "Sat 20:00"


def test_need_minutes_skips_what_cannot_fit():
    out = at("2026-11-14 06:00", 'need_minutes 999 "a long run" ; echo "rc=$?"')
    assert "skipping a long run" in out and out.endswith("rc=1")
    assert at("2026-11-14 06:00", 'need_minutes 60 "short" && echo fits') == "fits"


def test_daytime_window_works_too():
    env = {"WINDOW_START": "09:00", "WINDOW_END": "17:00", "STOP_MARGIN_MIN": "10"}
    assert at("2026-11-13 12:00", "window_open && echo y || echo n", **env) == "y"
    assert at("2026-11-13 18:00", "window_open && echo y || echo n", **env) == "n"
    assert hm("window_deadline", "2026-11-13 12:00", **env) == "Fri 16:50"


def test_a_go_now_deadline_overrides_the_window():
    # 2026-11-14 07:59: GO_NOW with an 8-hour hard stop. The window's own hard
    # stop (07:40) has passed; need_minutes must use night-run's deadline, or it
    # skips everything ("-19 min left", seen for real).
    dl = "$(TZ=America/New_York date -d '2026-11-14 15:59' +%s)"
    assert at("2026-11-14 07:59", f"export PLAN_DEADLINE={dl}; window_remaining_min") == "480"
    assert at("2026-11-14 07:59", f"export PLAN_DEADLINE={dl}; need_minutes 100 x && echo fits") == "fits"
    assert "skipping" in at("2026-11-14 07:59", "need_minutes 100 x")
