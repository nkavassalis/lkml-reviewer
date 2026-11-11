#!/usr/bin/env bash
# secret-scan.sh — the pre-commit / pre-push leak scan.
#
#   scripts/secret-scan.sh                 # scan staged diff (default)
#   scripts/secret-scan.sh --staged
#   scripts/secret-scan.sh --range origin/main..HEAD
#   scripts/secret-scan.sh --worktree      # scan working-tree tracked files
#   scripts/secret-scan.sh --all           # scan every tracked file (slow, thorough)
#   scripts/secret-scan.sh path/to/file ...
#
# Exit 0 = clean, 1 = hits found (inspect every hit; most are false positives,
# but the ones that aren't are a rotation-and-rewrite incident, not a delete).
#
# Patterns are the house list: credentials, private keys, LAN/internal topology,
# and real-world identifiers that must never appear in a tracked file.
set -uo pipefail

PATTERNS=(
  # --- credentials -------------------------------------------------------
  'gh[pousr]_[A-Za-z0-9]{20,}'                       # GitHub tokens
  'AKIA[0-9A-Z]{16}'                                 # AWS access key id
  'aws_secret_access_key[[:space:]]*=[[:space:]]*[A-Za-z0-9/+=]{30,}'
  '-----BEGIN [A-Z ]*PRIVATE KEY-----'
  'xox[baprs]-[A-Za-z0-9-]{10,}'                     # Slack
  'sk-[A-Za-z0-9]{24,}'                              # OpenAI-style keys
  'hf_[A-Za-z0-9]{20,}'                              # Hugging Face
  '(AIza[0-9A-Za-z_-]{30,})'                         # Google API key
  '[A-Za-z0-9_-]{32,}\.apps\.googleusercontent\.com' # Google client id
  'Authorization:[[:space:]]*(Bearer|Basic)[[:space:]]+[A-Za-z0-9._+/=-]{12,}'
  '(eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,})'  # JWT
  '(password|passwd|secret|token|api_key|apikey)[[:space:]]*[:=][[:space:]]*["'"'"'][^"'"'"' ]{8,}["'"'"']'
  # --- internal topology -------------------------------------------------
  '\b(10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|192\.168\.[0-9]{1,3}\.[0-9]{1,3}|172\.(1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3})\b'
  # Host names belong in a local watchlist (below), never here: a pattern list in
  # a public repo would publish the very names it protects.
  '/home/[a-z][a-z0-9_-]+/'
  '\bC:[\\/]+Users[\\/]'
  '\b[a-z0-9-]+\.internal\b'
  # --- infra identifiers -------------------------------------------------
  '\bE[A-Z0-9]{13}\b'                                # CloudFront distribution id
  '\bZ[A-Z0-9]{13,14}\b'                             # Route 53 hosted zone id
  '\b[0-9]{12}\.aws\.amazonaws\.com'                 # account-id URLs
  # --- model-origin canaries (AI-written code leaving the building) -----------
  # shapes an agent-authored file should never contain unasked; see
  # skills/agent-egress-defense. Generic cloud domains only — the known-bad
  # host list lives in a local-only file loaded below, never in this script.
  '\b169\.254\.169\.254\b|\b169\.254\.170\.2\b|\b100\.100\.100\.200\b|metadata\.google\.internal'
  'X-Amz-Signature|OSSAccessKeyId|x-oss-signature|q-sign-algorithm'
  '\b[a-z0-9-]+\.oss-[a-z0-9-]+\.aliyuncs\.com'      # Alibaba OSS bucket hosts
)

# Optional local-only watchlist: one ERE per line, merged into the pattern list.
# On Nick's boxes this pulls in the known-bad agent-exfil hosts; elsewhere the
# scan simply runs the static list above.
WATCHLIST="${META_WATCHLIST:-$HOME/meta/global/egress-watchlist.txt}"
# Plus a per-repo, gitignored watchlist (.secret-watchlist): this machine's host
# names, cluster addresses, account names. Same format.
REPO_WATCHLIST="$(git rev-parse --show-toplevel 2>/dev/null || pwd)/.secret-watchlist"
for wl in "$WATCHLIST" "$REPO_WATCHLIST"; do
  [ -f "$wl" ] || continue
  while IFS= read -r line; do
    line="${line%%#*}"; line="${line// /}"
    [ -n "$line" ] && PATTERNS+=("$line")
  done < "$wl"
done

mode="staged"; target=""
case "${1:-}" in
  --staged)   mode="staged"; shift ;;
  --worktree) mode="worktree"; shift ;;
  --all)      mode="all"; shift ;;
  --range)    mode="range"; target="${2:-}"; shift 2 ;;
esac

scan() {  # $1 = label, $2 = command that emits "file:line:content"-ish text on stdout
  local label="$1"; shift
  local hits
  hits=$( { "$@" 2>/dev/null || true; } | grep -nE -- "$(IFS='|'; echo "${PATTERNS[*]}")" \
          | grep -viE '(example|placeholder|changeme|change-me|put-me-in-env|<[^>]*>|\bX{3,}\b|your[-_]|dummy|redacted)' )
  if [ -n "$hits" ]; then
    echo "== $label: $(printf '%s\n' "$hits" | wc -l) possible hit(s) =="
    printf '%s\n' "$hits"
    found=1
  fi
}

found=0
SELF=':!scripts/secret-scan.sh'  # its own pattern list would match itself
case "$mode" in
  staged)   scan "staged diff"  git diff --cached -U0 ;;
  range)    scan "$target"      git diff -U0 "$target" -- . "$SELF" ;;
  worktree) scan "tracked files" git grep -nE -- "$(IFS='|'; echo "${PATTERNS[*]}")" -- . "$SELF" ;;
  all)      scan "all tracked"  git grep -nE -- "$(IFS='|'; echo "${PATTERNS[*]}")" -- . "$SELF" ;;
esac

# explicit file arguments always get scanned too
for f in "$@"; do
  [ -f "$f" ] && scan "$f" grep -nE -- "$(IFS='|'; echo "${PATTERNS[*]}")" "$f"
done

if [ "$found" = 1 ]; then
  cat >&2 <<'EOF'

secret-scan: inspect every hit above.
  Real secret committed          -> STOP, tell Nick NOW; rotate + rewrite history.
  Real internal detail committed -> remove it and reword the doc abstractly.
  False positive                 -> reword the example (placeholders: <host>, example.com,
                                    /srv/media, change-me) rather than adding a pattern skip.
EOF
  exit 1
fi
echo "secret-scan: clean"
