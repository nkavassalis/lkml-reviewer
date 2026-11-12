#!/usr/bin/env bash
# Steps 2, 4 and 5 of docs/CONVERSION.md: merge a LoRA into its BF16 base,
# quantize + convert to MLX, smoke-test it. Run on Apple silicon (or merge
# elsewhere and pass --skip-merge with an existing merged dir).
#
#   scripts/convert-mlx.sh --adapter run-3/adapter-final --out ~/models/lkml-reviewer-v3
#   scripts/convert-mlx.sh --adapter A --out O --bits 4 --base Qwen/Qwen3.8-27B
#   scripts/convert-mlx.sh --merged merged/ --out O        # skip the merge
#
# PYTHON must have torch, safetensors, huggingface_hub and mlx-lm installed
# (default: python3). Produces $OUT/merged-bf16 (deleted afterwards unless
# --keep-merged) and $OUT/mlx-<bits>bit.
set -euo pipefail

base="Qwen/Qwen3.8-27B"; adapter=""; out=""; bits=8; group=64; merged=""; keep=0
PY=${PYTHON:-python3}
here="$(cd "$(dirname "$0")/.." && pwd)"

while [ $# -gt 0 ]; do
  case "$1" in
    --base) base=$2; shift ;;
    --adapter) adapter=$2; shift ;;
    --merged) merged=$2; shift ;;
    --out) out=$2; shift ;;
    --bits) bits=$2; shift ;;
    --group) group=$2; shift ;;
    --keep-merged) keep=1 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
[ -n "$out" ] || { echo "--out is required" >&2; exit 2; }
[ -n "$adapter" ] || [ -n "$merged" ] || { echo "need --adapter (or --merged)" >&2; exit 2; }
case "$bits" in 4|6|8) ;; *) echo "--bits must be 4, 6 or 8" >&2; exit 2 ;; esac
mkdir -p "$out"
say() { printf '[%s] %s\n' "$(date +%T)" "$*"; }

"$PY" -c "import torch, safetensors, huggingface_hub, mlx_lm" 2>/dev/null \
  || { echo "$PY lacks torch / safetensors / huggingface_hub / mlx-lm" >&2; exit 2; }

free_gb() { df -Pk "$1" | awk 'NR==2 {print int($4/1048576)}'; }
need=$(( bits * 27 / 8 + 5 ))            # the quantized model, ~27B params
[ -z "$merged" ] && need=$(( need + 56 )) # plus the merged BF16 copy
if [ "$(free_gb "$out")" -lt "$need" ]; then
  echo "only $(free_gb "$out") GB free at $out; this needs ~$need GB" >&2; exit 1
fi

# --- 2. merge -----------------------------------------------------------------
if [ -z "$merged" ]; then
  merged="$out/merged-bf16"
  say "merging $adapter into $base -> $merged"
  "$PY" "$here/train/merge_lora.py" --base "$base" --adapter "$adapter" --out "$merged"
fi
[ -f "$merged/MERGE_INFO.json" ] || [ -n "${adapter}" ] || say "note: $merged has no MERGE_INFO.json"

# --- 4. quantize + convert ----------------------------------------------------
mlx="$out/mlx-${bits}bit"
say "converting to MLX ${bits}-bit (group $group) -> $mlx"
rm -rf "$mlx"
"$PY" -m mlx_lm.convert --hf-path "$merged" --mlx-path "$mlx" -q --q-bits "$bits" \
  --q-group-size "$group"
[ -f "$merged/MERGE_INFO.json" ] && cp "$merged/MERGE_INFO.json" "$mlx/"
say "converted: $(du -sh "$mlx" | cut -f1)"

# --- 5. smoke test ------------------------------------------------------------
say "smoke test"
"$PY" - "$mlx" <<'EOF'
import sys, time
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
model, tok = load(sys.argv[1])
msgs = [{"role": "user", "content": "Subject: Re: [PATCH] mm: fix\n\ndiff --git a/mm/x.c b/mm/x.c\n-\tint n;\n+\tlong n;"}]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
last, text = None, ""
for last in stream_generate(model, tok, prompt, max_tokens=120, sampler=make_sampler(temp=0.7, top_p=0.95)):
    text += last.text
print(text.strip()[:400])
print(f"\n{last.generation_tps:.1f} tok/s, peak {last.peak_memory:.1f} GB")
EOF

if [ $keep = 0 ] && [ "$merged" = "$out/merged-bf16" ]; then
  say "removing the merged BF16 copy (pass --keep-merged to keep it)"
  rm -rf "$merged"
fi
say "done: $mlx"
