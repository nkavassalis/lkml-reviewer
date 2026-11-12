"""PR reviewer: gh handling, prompt parity with training, CLI behaviour.

A fake `gh` executable on PATH imitates the real CLI's failure modes, so these
tests exercise the actual subprocess and error-mapping code. No network, no GPU.
"""
from __future__ import annotations

import ast
import json
import os
import stat
import sys
from pathlib import Path

import pytest

from linuspr import template
from linuspr.pairs import ReviewPair
from reviewer import gh
from reviewer.__main__ import main
from reviewer.prompt import DISCLAIMER, pr_input, split_by_file, strip_html_comments

ROOT = Path(__file__).resolve().parent.parent

DIFF = (
    "diff --git a/mm/slub.c b/mm/slub.c\n--- a/mm/slub.c\n+++ b/mm/slub.c\n"
    "@@ -1,3 +1,3 @@\n-\tint x;\n+\tlong x;\n"
    "diff --git a/mm/slab.h b/mm/slab.h\n--- a/mm/slab.h\n+++ b/mm/slab.h\n"
    "@@ -9 +9 @@\n-#define A 1\n+#define A 2\n"
)
PR_JSON = {"number": 7, "title": "mm: widen x", "body": "<!-- template -->Widens x.",
           "url": "https://github.com/o/r/pull/7", "author": {"login": "alice"},
           "baseRefName": "main", "headRefName": "widen", "files": [{"path": "mm/slub.c"}]}

FAKE_GH = r'''#!/usr/bin/env python3
import json, os, sys
mode = os.environ.get("FAKE_GH_MODE", "ok")
log = os.environ["FAKE_GH_LOG"]
with open(log, "a") as f:
    f.write(json.dumps({"args": sys.argv[1:], "stdin": sys.stdin.read() if "--body-file" in sys.argv else None}) + "\n")
a = sys.argv[1:]
if a[:2] == ["auth", "status"]:
    if mode == "noauth":
        print("You are not logged into any GitHub hosts. To log in, run: gh auth login", file=sys.stderr); sys.exit(1)
    print("github.com\n  ✓ Logged in to github.com account alice (keyring)"); sys.exit(0)
if a[:2] == ["pr", "view"]:
    if mode == "nopr":
        print("GraphQL: Could not resolve to a PullRequest with the number of 999. (repository.pullRequest)", file=sys.stderr); sys.exit(1)
    if mode == "norepo":
        print("GraphQL: Could not resolve to a Repository with the name 'o/nope'. (repository)", file=sys.stderr); sys.exit(1)
    if mode == "noremote":
        print("none of the git remotes configured for this repository point to a known GitHub host", file=sys.stderr); sys.exit(1)
    print(os.environ["FAKE_GH_PR"]); sys.exit(0)
if a[:2] == ["pr", "diff"]:
    print(os.environ["FAKE_GH_DIFF"], end=""); sys.exit(0)
if a[:2] == ["pr", "review"]:
    if mode == "readonly":
        print("GraphQL: Resource not accessible by personal access token (addPullRequestReview)", file=sys.stderr); sys.exit(1)
    sys.exit(0)
print("unexpected: " + " ".join(a), file=sys.stderr); sys.exit(2)
'''


@pytest.fixture
def fake_gh(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "gh"
    exe.write_text(FAKE_GH.replace("#!/usr/bin/env python3", f"#!{sys.executable}"))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "gh.log"
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_GH_LOG", str(log))
    monkeypatch.setenv("FAKE_GH_PR", json.dumps(PR_JSON))
    monkeypatch.setenv("FAKE_GH_DIFF", DIFF)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    def calls():
        return [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
    return calls


# --- gh handling --------------------------------------------------------------

def test_gh_missing_is_a_clear_error(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))  # nothing on PATH
    with pytest.raises(gh.GhError, match="not installed"):
        gh.fetch_pr(1)


def test_not_authenticated_says_how_to_fix(fake_gh, monkeypatch):
    monkeypatch.setenv("FAKE_GH_MODE", "noauth")
    with pytest.raises(gh.GhError, match="gh auth login"):
        gh.fetch_pr(7, "o/r")


def test_gh_token_skips_the_auth_probe(fake_gh, monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "x")
    gh.fetch_pr(7, "o/r")
    assert not any(c["args"][:2] == ["auth", "status"] for c in fake_gh())


@pytest.mark.parametrize("mode,needle", [
    ("nopr", "no such pull request"),
    ("norepo", "repository not found"),
    ("noremote", "pass --repo"),
])
def test_lookup_failures_are_explained(fake_gh, monkeypatch, mode, needle):
    monkeypatch.setenv("FAKE_GH_MODE", mode)
    with pytest.raises(gh.GhError, match=needle):
        gh.fetch_pr(999, "o/nope")


def test_fetch_pr_parses_metadata_and_diff(fake_gh):
    pr = gh.fetch_pr(7, "o/r")
    assert (pr.title, pr.author, pr.base, pr.files) == ("mm: widen x", "alice", "main", ["mm/slub.c"])
    assert pr.diff == DIFF
    view = next(c for c in fake_gh() if c["args"][:2] == ["pr", "view"])
    assert view["args"][2:5] == ["7", "--repo", "o/r"]


def test_post_without_write_access_is_explained(fake_gh, monkeypatch):
    monkeypatch.setenv("FAKE_GH_MODE", "readonly")
    with pytest.raises(gh.GhError, match="permission denied"):
        gh.post_review(7, "body", "o/r")


def test_post_is_a_comment_never_an_approval(fake_gh):
    gh.post_review(7, "hello", "o/r")
    call = next(c for c in fake_gh() if c["args"][:2] == ["pr", "review"])
    assert "--comment" in call["args"]
    assert "--approve" not in call["args"] and "--request-changes" not in call["args"]
    assert call["stdin"] == "hello"


# --- prompt parity with training ---------------------------------------------

def _assigned_strings(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


def test_training_script_template_matches_shared_template():
    """train/sft_lora.py runs alone in a container and keeps its own copy.
    If this fails, training and inference prompts have drifted."""
    trained = _assigned_strings(ROOT / "train" / "sft_lora.py")
    assert trained["SYSTEM"] == template.SYSTEM
    assert trained["TRUNC_MARK"] == template.TRUNC_MARK
    src = (ROOT / "train" / "sft_lora.py").read_text()
    assert "enable_thinking=False" in src and "add_generation_prompt=True" in src


def test_pr_prompt_has_the_exact_training_layout():
    user, truncated = pr_input("mm: widen x", DIFF, body="Widens x.", author="alice")
    pair = ReviewPair(mid="1", month="2401.0", subject="Re: [PATCH] mm: widen x", date="d",
                      context="", review="r", resolved_context=DIFF,
                      thread_context="[alice]:\nWidens x.")
    assert user == pair.as_sft()["input"]
    assert not truncated


def test_pr_without_description_has_no_thread_section():
    user, _ = pr_input("t", DIFF)
    assert template.THREAD_HEADER not in user and user.startswith("Subject: Re: [PATCH] t")


def test_long_diff_is_cut_at_the_tail_with_the_training_marker():
    user, truncated = pr_input("t", DIFF * 2000, max_prompt_tokens=500)
    assert truncated and user.endswith(template.TRUNC_MARK)
    assert user.startswith("Subject: Re: [PATCH] t")


def test_pr_template_comments_are_dropped():
    assert strip_html_comments("<!-- describe -->Real text<!--\nmore -->") == "Real text"


def test_split_by_file():
    assert [p for p, _ in split_by_file(DIFF)] == ["mm/slub.c", "mm/slab.h"]


# --- CLI --------------------------------------------------------------------

def test_dry_run_on_a_diff_file_needs_no_gh_and_no_gpu(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))  # no gh anywhere
    f = tmp_path / "x.patch"
    f.write_text(DIFF)
    assert main(["--diff-file", str(f), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "Subject: Re: [PATCH] x.patch" in out and "mm/slub.c, mm/slab.h" in out


def test_dry_run_on_a_pr_uses_gh(fake_gh, capsys):
    assert main(["--pr", "7", "--repo", "o/r", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "[alice]:\nWidens x." in out and "template" not in out


def test_unauthenticated_pr_exits_2_with_the_remedy(fake_gh, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_GH_MODE", "noauth")
    with pytest.raises(SystemExit) as e:
        main(["--pr", "7", "--repo", "o/r", "--dry-run"])
    assert e.value.code == 2 and "gh auth login" in capsys.readouterr().err


def test_post_requires_a_pr(tmp_path):
    f = tmp_path / "x.patch"
    f.write_text(DIFF)
    with pytest.raises(SystemExit) as e:
        main(["--diff-file", str(f), "--post", "--adapter", "a"])
    assert e.value.code == 4


def test_missing_adapter_fails_before_any_model_load(tmp_path, monkeypatch):
    monkeypatch.delenv("LKML_REVIEWER_ADAPTER", raising=False)
    f = tmp_path / "x.patch"
    f.write_text(DIFF)
    with pytest.raises(SystemExit) as e:
        main(["--diff-file", str(f)])
    assert e.value.code == 4


def test_empty_diff_is_rejected(tmp_path):
    f = tmp_path / "x.patch"
    f.write_text("")
    with pytest.raises(SystemExit) as e:
        main(["--diff-file", str(f), "--dry-run"])
    assert e.value.code == 4


def test_disclaimer_says_it_is_not_linus():
    assert "not Linus Torvalds" in DISCLAIMER


# --- doctor -------------------------------------------------------------------

def test_doctor_adapter_checks(tmp_path):
    from reviewer.doctor import FAIL, OK, WARN, check_adapter
    assert check_adapter(None).status == WARN
    assert check_adapter(str(tmp_path / "nope")).status == FAIL
    half = tmp_path / "half"
    half.mkdir()
    (half / "adapter_config.json").write_text("{}")
    assert check_adapter(str(half)).status == FAIL          # weights missing
    (half / "adapter_model.safetensors").write_bytes(b"")
    assert check_adapter(str(half)).status == OK
    assert check_adapter("someone/lkml-reviewer-lora").status == OK   # HF id


def test_doctor_reports_without_crashing_and_flags_failures(tmp_path, capsys):
    from reviewer.doctor import run
    assert run(None) == 0                       # warnings only: still exit 0
    assert "dry runs (--dry-run):       ready" in capsys.readouterr().out
    assert run(str(tmp_path / "missing")) == 1  # a FAIL is a non-zero exit


def test_cli_version_and_doctor(capsys):
    from reviewer import __version__
    assert main(["--version"]) == 0 and __version__ in capsys.readouterr().out
    assert main(["--doctor"]) in (0, 1)


def test_pyproject_entry_point_and_packages():
    import tomllib
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert cfg["project"]["scripts"]["lkml-reviewer"] == "reviewer.__main__:main"
    assert set(cfg["tool"]["setuptools"]["packages"]) >= {"reviewer", "linuspr"}
    assert cfg["project"]["dependencies"] == []      # the core stays stdlib-only
