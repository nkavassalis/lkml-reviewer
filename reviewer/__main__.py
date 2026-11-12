"""lkml-reviewer: review a GitHub PR or a local diff with the fine-tuned model.

    python -m reviewer --pr 123 --repo owner/name --adapter path/to/adapter
    python -m reviewer --local --base origin/main --adapter ...   # no GitHub needed
    python -m reviewer --diff-file my.patch --adapter ...         # no GitHub needed
    python -m reviewer --pr 123 --dry-run                          # show the prompt, no GPU

Prints the review. Posts it as a PR *comment* only with --post (after a
confirmation, or --yes), always prefixed with a disclaimer that it's an
automated style imitation. Cheap checks (gh, auth, PR, diff) run before the
model loads, so a typo never costs a model load.

Exit codes: 0 ok, 2 GitHub/gh problem, 3 model/backend problem, 4 bad input.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from . import gh
from .prompt import DISCLAIMER, changed_files, pr_input, split_by_file

EXIT_GH, EXIT_BACKEND, EXIT_INPUT = 2, 3, 4


def die(msg: str, code: int) -> None:
    print(f"lkml-reviewer: {msg}", file=sys.stderr)
    sys.exit(code)


def local_diff(base: str) -> str:
    """`git diff base...HEAD` of the current checkout."""
    try:
        r = subprocess.run(["git", "diff", f"{base}...HEAD"], capture_output=True, text=True,
                           timeout=60)
    except FileNotFoundError:
        die("git is not installed", EXIT_INPUT)
    if r.returncode != 0:
        die(f"git diff {base}...HEAD failed: {r.stderr.strip() or 'not a git repository?'}",
            EXIT_INPUT)
    return r.stdout


def gather(args) -> tuple[str, str, str, str]:
    """(title, body, author, diff) from whichever source was chosen."""
    if args.pr is not None:
        try:
            pr = gh.fetch_pr(args.pr, args.repo)
        except gh.GhError as e:
            die(str(e), EXIT_GH)
        return pr.title, pr.body, pr.author, pr.diff
    if args.diff_file:
        try:
            if args.diff_file == "-":
                diff = sys.stdin.read()
            else:
                with open(args.diff_file) as f:
                    diff = f.read()
        except OSError as e:
            die(f"can't read {args.diff_file}: {e.strerror}", EXIT_INPUT)
        return args.title or os.path.basename(args.diff_file), "", "author", diff
    return args.title or f"local changes vs {args.base}", "", "author", local_diff(args.base)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m reviewer", description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pr", type=int, help="pull request number (uses gh)")
    src.add_argument("--diff-file", help="unified diff file, or - for stdin")
    src.add_argument("--local", action="store_true", help="review `git diff BASE...HEAD`")
    ap.add_argument("--repo", help="owner/name (default: the current checkout's remote)")
    ap.add_argument("--base", default="origin/main", help="base ref for --local")
    ap.add_argument("--title", help="subject line for --diff-file / --local")
    ap.add_argument("--adapter", default=os.environ.get("LKML_REVIEWER_ADAPTER"),
                    help="LoRA adapter dir or HF repo id (or $LKML_REVIEWER_ADAPTER)")
    ap.add_argument("--backend", default="cuda", choices=["cuda", "mlx"])
    ap.add_argument("--load-in-4bit", action="store_true", help="NF4 base (~22 GB GPU)")
    ap.add_argument("--per-file", action="store_true",
                    help="review each changed file separately (big PRs get truncated otherwise)")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the prompt(s); load no model")
    ap.add_argument("--post", action="store_true", help="post the review as a PR comment")
    ap.add_argument("--yes", action="store_true", help="post without the confirmation prompt")
    args = ap.parse_args(argv)

    if args.post and args.pr is None:
        die("--post needs --pr", EXIT_INPUT)
    if not args.dry_run and not args.adapter:
        die("no adapter: pass --adapter or set LKML_REVIEWER_ADAPTER", EXIT_INPUT)

    title, body, author, diff = gather(args)
    if not diff.strip():
        die("the diff is empty: nothing to review", EXIT_INPUT)

    if args.per_file:
        units = [(path, *pr_input(f"{title} ({path})", d, body=body, author=author))
                 for path, d in split_by_file(diff)]
    else:
        units = [("", *pr_input(title, diff, body=body, author=author))]
    for path, _, truncated in units:
        if truncated:
            where = path or "this diff"
            print(f"note: {where} is longer than the model's input budget; its tail is cut"
                  + ("" if args.per_file else " (try --per-file)"), file=sys.stderr)

    if args.dry_run:
        for path, user, _ in units:
            print(f"===== prompt{f' for {path}' if path else ''} =====\n{user}\n")
        print(f"({len(units)} prompt(s), files: {', '.join(changed_files(diff)) or 'none'})")
        return 0

    from .backends import BackendError, load_backend  # noqa: PLC0415  (heavy, and last)

    try:
        backend = load_backend(args.backend, args.adapter, load_in_4bit=args.load_in_4bit,
                               max_new_tokens=args.max_new_tokens, seed=args.seed)
        reviews = backend.generate([u for _, u, _ in units])
    except BackendError as e:
        die(str(e), EXIT_BACKEND)

    parts = []
    for (path, _, _), review in zip(units, reviews, strict=True):
        parts.append(f"### `{path}`\n\n{review}" if path else review)
    text = "\n\n".join(parts)
    print(text)

    if args.post:
        if not args.yes:
            if not sys.stdin.isatty():
                die("refusing to post without confirmation on a non-interactive stdin; "
                    "add --yes", EXIT_INPUT)
            if input(f"\nPost this as a comment on PR #{args.pr}? [y/N] ").strip().lower() != "y":
                print("not posted.")
                return 0
        try:
            gh.post_review(args.pr, DISCLAIMER + text, args.repo)
        except gh.GhError as e:
            die(str(e), EXIT_GH)
        print(f"posted to PR #{args.pr}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
