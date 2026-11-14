# Post-training conversion: from a LoRA adapter to an Apple-silicon model

This is the standard last stage for every model this project trains. Target:
**a 64 GB Apple M4 Pro Mac mini, via MLX, at 8-bit.** The pipeline is
model-agnostic. Swap the base model and adapter, and re-check §1's sizing
before training anything you mean to run on the Mac.

```
PEFT LoRA adapter ──┐
                    ├─ 1. size check ─ 2. merge (BF16) ─ 3. verify merge ─ 4. quantize + convert (MLX)
BF16 base (HF)   ───┘                                                        │
                       7. publish ◄─ 6. quality gate vs the CUDA model ◄─ 5. smoke test on the Mac
```

`scripts/convert-mlx.sh` runs steps 2, 4 and 5; steps 3 and 6 are deliberate
checks you run and read.

## 1. Will it fit? (do this before training)

On Apple silicon the GPU shares memory with everything else, and macOS caps
how much it may **wire**: ~75% of RAM by default (~48 GB on 64 GB), raisable
with `sudo sysctl iogpu.wired_limit_mb=<MB>` (resets on reboot; the test Mac
has it at 57,344). Budget against the wired limit, not the RAM.

| term | formula | Qwen3.8-27B |
| --- | --- | --- |
| weights | text params × (bits/8 + 4/group) bytes. MLX stores an fp16 scale and bias per group of 64 | 26.9B × (1 + 0.0625) = **28.6 GB** at 8-bit; 15.1 GB at 4-bit |
| KV cache | full-attention layers × 2 × KV heads × head dim × 2 bytes, per token | 16 × 2 × 4 × 256 × 2 = **64 KB/token** (only 16 of 64 layers are full attention) |
| recurrent state | linear layers × V heads × K dim × V dim × 4 bytes (Gated DeltaNet; constant) | 48 × 48 × 128 × 128 × 4 = 0.15 GB |
| working buffers | prefill activations, sampler: a few GB | ~3 GB |

Qwen3.8-27B at 8-bit: **~32 GB at 4k context, ~40 GB at 128k.** Comfortable on
a 64 GB machine. A dense model of the same size without the hybrid layers
would need 4× the KV cache per token. Always read `layer_types` from the
config. Generation is bandwidth-bound: tokens/s ≈ memory bandwidth / weight
bytes (M4 Pro, ~273 GB/s: ~9.5 tok/s ceiling at 8-bit, ~18 at 4-bit).

**Also check that mlx-lm supports the architecture** (`mlx_lm/models/<model_type>.py`).
This is the gate that decides everything else; for `qwen3_5` it exists
(mlx-lm 0.32.2), including the multimodal wrapper and dropping the vision tower.

## 2. Merge the LoRA into the BF16 base

```bash
python3 train/merge_lora.py --base Qwen/Qwen3.8-27B --adapter run-3/adapter-final --out merged/
```

- **Merge before quantizing, always.** The LoRA was trained against BF16
  weights; adding it to an already-quantized base adds W's quantization error
  to every merged layer. Merge in BF16, then quantize the result once.
- **Streaming:** one safetensors shard at a time, fp32 math, cast back. Peak
  memory is a shard plus the adapter (a few GB), so it runs on CPU on the Mac
  itself or on a GPU box without stopping anything. (`peft`'s
  `merge_and_unload()` loads all ~55 GB at once.)
- It refuses adapter types it doesn't model (DoRA, bias deltas,
  fan_in_fan_out), fails unless every LoRA module merged exactly once, and
  writes `MERGE_INFO.json` (adapter SHA-256, module count, scale) next to the
  weights.

## 3. Verify the merge (on a GPU box)

Run the same few prompts through (a) base + PEFT adapter and (b) the merged
model, both BF16, and compare next-token logits. Expect a max absolute
difference at BF16 noise level (~1e-2), and identical greedy tokens for the
first few dozen steps. A wrong scale (α/r vs α/√r) or a missed module shows up
here as an obvious divergence, long before any quality metric would.

## 4. Quantize and convert to MLX

```bash
mlx_lm.convert --hf-path merged/ --mlx-path lkml-reviewer-mlx-8bit/ -q --q-bits 8 --q-group-size 64
```

mlx-lm's `qwen3_5` loader drops the vision tower and remaps
`model.language_model.*`. Keep 8-bit as the default: 4-bit halves memory and
roughly doubles speed, but the thing this adapter learned is *style*, which is
exactly what quantization erodes first. Only ship 4-bit if step 6 says it
holds up.

## 5. Smoke test on the Mac

```bash
lkml-reviewer --backend mlx --model lkml-reviewer-mlx-8bit/ --diff-file some.patch --stats
```

`--stats` prints prompt and generation tokens/s and peak memory. Peak memory
must sit well under the wired limit, with room left for the OS.

## 6. Quality gate: the Mac model against the CUDA model

Quantization is lossy, so check that it didn't lose what training bought.
Generate for the **same fixed held-out subset** (`compare.py --subset 100`)
with sampling on both platforms, then:

- `train/score.py`: proxy metrics (specificity, chatbot-isms, abuse, loops,
  length) should match the CUDA model's within noise;
- `train/judge.py --pair <cuda>,<mlx>`: blind, both orderings. The Mac model
  should tie the CUDA model, not lose to it.

## 7. Publish

Private Hugging Face repos: the PEFT adapter (for CUDA users) and the
converted MLX model (for Mac users), with `docs/REPORT.md` as the card, the
not-Linus disclaimer, and `MERGE_INFO.json`. Make a repo public only on
purpose; see the report's §8.

## Measured on the test Mac

**2026-11-12, base model only** (`mlx-community/Qwen3.8-27B-8bit`, no LoRA:
this validates the hardware path and the `lkml-reviewer` mlx backend, not our
adapter). M4 Pro, 64 GB, wired limit 56 GB, mlx 0.32.4, mlx-lm 0.32.2,
installed with `scripts/install-reviewer.sh` (pipx, `[mlx]` extra):

| | predicted (§1) | measured |
| --- | --- | --- |
| peak memory, 838-token prompt | ~32 GB at 4k | **30.2 GB** |
| generation | ~7–8 tok/s (ceiling 9.5) | **8.5 tok/s** |
| prompt processing | — | 74 tok/s |
| wall time incl. model load, 277-token review | — | 53 s |

The 8-bit 27B fits with ~26 GB of the wired limit to spare.

**Disk: the conversion peaks at ~88 GB** (merged BF16 ~55 GB plus the MLX
output ~27 GB) on top of the BF16 base in the HF cache. To make room, remove
cached models with `hf cache rm <repo>` and then `hf cache prune`, never `rm -rf`
on a `models--*` folder: recent `huggingface_hub` versions keep file data in a
shared blob store (`hub/blobs`, marked `.huggingface-shared-blobs`), so deleting
a model's folder frees nothing and leaves its data orphaned (lkml-reviewer,
2026-11-14: 29.5 GB stranded, the v4 build failed on space).

**2026-11-12, v2 adapter through the full pipeline** (`scripts/convert-mlx.sh
--adapter v2-adapter --bits 8`, on the Mac itself, from the BF16 base in the
HF cache):

| step | result |
| --- | --- |
| merge (`merge_lora.py`) | **400 / 400 modules**, scale 2.0 (α 32 / r 16), ~2 min for 18 shards, CPU only |
| convert (`mlx_lm convert -q --q-bits 8`) | 8.501 bits/weight, **27 GB** (vision tower dropped; the stock 8-bit build is 29.5 GB), ~80 s |
| smoke test | 8.5 tok/s, 28.9 GB peak |
| review of a held-out 2024 patch | 101 tok/s prompt, **8.8 tok/s** generation, **30.2 GB peak** |

Total from adapter to a working Mac model: under 4 minutes once the base is
downloaded. On the held-out DEPT patch, the merged model answers in v2's
voice (a terse, plain-text question about a disabled self-test, signed
"Linus" because v2's targets kept the sign-off), where the stock base writes
"This patch is garbage… Resend with: 1. 2. 3." The §6 quality gate (Mac vs
CUDA on the same 100 held-out patches) runs with v3.
