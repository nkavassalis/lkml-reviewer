#!/usr/bin/env bash
# Install the lkml-reviewer CLI with pipx, picking the right extras for this
# machine, then run its doctor. Safe to re-run (it upgrades in place).
#
#   curl -fsSL https://raw.githubusercontent.com/nkavassalis/lkml-reviewer/main/scripts/install-reviewer.sh | bash
#   scripts/install-reviewer.sh                  # from a checkout (installs that checkout)
#   scripts/install-reviewer.sh --core           # no inference libraries (dry runs, gh only)
#   scripts/install-reviewer.sh --cuda           # force the CUDA extras
#   scripts/install-reviewer.sh --mlx            # force the Apple-silicon (mlx) extras
#   scripts/install-reviewer.sh --torch-index https://download.pytorch.org/whl/cu130
#   scripts/install-reviewer.sh --print          # show what it would run, do nothing
#
# It never uses sudo. If something system-level is missing it says what to run.
set -euo pipefail

REPO_URL="git+https://github.com/nkavassalis/lkml-reviewer"
mode=auto; torch_index=""; dry=0; src=""

while [ $# -gt 0 ]; do
  case "$1" in
    --core) mode=core ;;
    --cuda) mode=cuda ;;
    --mlx) mode=mlx ;;
    --torch-index) torch_index="${2:?--torch-index needs a URL}"; shift ;;
    --from) src="${2:?--from needs a path or URL}"; shift ;;
    --print) dry=1 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done

say() { printf '==> %s\n' "$*"; }
run() { if [ $dry = 1 ]; then printf '    $ %s\n' "$*"; else "$@"; fi; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

# --- python ----------------------------------------------------------------
py=$(command -v python3 || true)
[ -n "$py" ] || die "python3 not found. Install Python 3.10+ (macOS: brew install python; Ubuntu: sudo apt install python3)."
"$py" -c 'import sys; sys.exit(sys.version_info < (3, 10))' \
  || die "Python 3.10+ required, found $("$py" -V 2>&1)."

# --- pipx ------------------------------------------------------------------
if ! command -v pipx >/dev/null 2>&1; then
  case "$(uname -s)" in
    Darwin) hint="brew install pipx && pipx ensurepath" ;;
    Linux)  hint="sudo apt install pipx && pipx ensurepath   (or: $py -m pip install --user pipx)" ;;
    *)      hint="$py -m pip install --user pipx && $py -m pipx ensurepath" ;;
  esac
  die "pipx not found. Install it with:  $hint   then re-run this script."
fi

# --- where to install from -------------------------------------------------
if [ -z "$src" ]; then
  here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." 2>/dev/null && pwd || true)"
  if [ -n "$here" ] && [ -f "$here/pyproject.toml" ] && grep -q 'name = "lkml-reviewer"' "$here/pyproject.toml"; then
    src="$here"   # running from a checkout: install exactly this code
  else
    src="$REPO_URL"
  fi
fi

# --- which extras ----------------------------------------------------------
os="$(uname -s)"; arch="$(uname -m)"
if [ $mode = auto ]; then
  if [ "$os" = Linux ] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    mode=cuda
    say "NVIDIA GPU found: $(nvidia-smi -L | head -1)"
  elif [ "$os" = Darwin ] && [ "$arch" = arm64 ]; then
    mode=mlx
    say "Apple silicon: installing with the mlx backend."
  else
    mode=core
    say "No NVIDIA GPU found: installing the core CLI (dry runs and gh only)."
  fi
fi
spec="$src"
[ $mode = cuda ] && spec="lkml-reviewer[cuda] @ $src"
[ $mode = mlx ] && spec="lkml-reviewer[mlx] @ $src"
[ "$src" != "$REPO_URL" ] && [ $mode = cuda ] && spec="$src[cuda]"
[ "$src" != "$REPO_URL" ] && [ $mode = mlx ] && spec="$src[mlx]"

# --- install ---------------------------------------------------------------
say "installing: $spec"
run pipx install --force "$spec"

if [ -n "$torch_index" ]; then
  say "replacing torch with the build from $torch_index"
  run pipx runpip lkml-reviewer install --upgrade --force-reinstall --index-url "$torch_index" torch
fi

# --- gh (optional) ---------------------------------------------------------
if ! command -v gh >/dev/null 2>&1; then
  say "note: gh (GitHub CLI) isn't installed. It's only needed for --pr:"
  say "      https://cli.github.com   (--local and --diff-file work without it)"
fi

# --- check -----------------------------------------------------------------
if [ $dry = 0 ]; then
  bin="$(command -v lkml-reviewer || true)"
  [ -n "$bin" ] || bin="$(pipx environment --value PIPX_BIN_DIR 2>/dev/null)/lkml-reviewer"
  if [ ! -x "$bin" ]; then
    die "installed, but lkml-reviewer isn't on PATH. Run: pipx ensurepath  (then open a new shell)"
  fi
  say "running: lkml-reviewer --doctor"
  "$bin" --doctor || true
  cat <<'EOF'

Next:
  lkml-reviewer --pr 123 --repo owner/name --dry-run       # see the prompt (no GPU)
  lkml-reviewer --pr 123 --repo owner/name --adapter DIR   # generate a review
  lkml-reviewer --help
EOF
fi
