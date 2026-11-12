"""Merge a PEFT LoRA adapter into its base checkpoint, one shard at a time.

    python3 train/merge_lora.py --base <hf-dir-or-id> --adapter run-3/adapter-final \
        --out merged/

Why a streaming merge instead of peft's merge_and_unload(): that loads the whole
27B model (~55 GB in BF16), which doesn't fit beside anything else on a 64 GB
Mac and stalls a GPU box mid-run. Here peak memory is one shard plus the
adapter (a few GB), and it runs on CPU anywhere torch does.

    W' = W + scale * (B @ A),  scale = lora_alpha / r  (alpha / sqrt(r) for rsLoRA)

The math is done in fp32 and cast back to the base's dtype. Merging happens
*before* any quantization (docs/CONVERSION.md): applying a LoRA trained against
BF16 weights to an already-quantized base would add the quantization error of
W to every merged layer.

Refuses adapters it doesn't model (DoRA, biases, fan_in_fan_out), and fails
unless every LoRA module in the adapter was merged exactly once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

COPY_SUFFIXES = (".json", ".jinja", ".txt", ".model", ".py", ".md")
PEFT_PREFIX = "base_model.model."


def resolve_base(base: str) -> Path:
    p = Path(base).expanduser()
    if p.exists():
        return p
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    return Path(snapshot_download(base))


def lora_pairs(adapter_dir: Path) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], float]:
    """{base weight key: (A, B)} and the merge scale."""
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    if cfg.get("use_dora"):
        sys.exit("refusing: DoRA adapters need magnitude vectors this merge doesn't apply")
    if cfg.get("fan_in_fan_out"):
        sys.exit("refusing: fan_in_fan_out adapters (Conv1D layers) aren't handled")
    if cfg.get("bias", "none") != "none":
        sys.exit(f"refusing: bias={cfg['bias']!r} adapters carry bias deltas this merge ignores")
    r, alpha = int(cfg["r"]), float(cfg["lora_alpha"])
    scale = alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r
    tensors = load_file(str(adapter_dir / "adapter_model.safetensors"))
    pairs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for key, a in tensors.items():
        if ".lora_A." not in key:
            continue
        b = tensors[key.replace(".lora_A.", ".lora_B.")]
        module = key.split(".lora_A.")[0]
        if module.startswith(PEFT_PREFIX):
            module = module[len(PEFT_PREFIX):]
        pairs[f"{module}.weight"] = (a, b)
    leftovers = [k for k in tensors if ".lora_A." not in k and ".lora_B." not in k]
    if leftovers:
        sys.exit(f"refusing: adapter has non-LoRA tensors ({leftovers[:3]}...)")
    return pairs, scale


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="HF repo id or local snapshot dir")
    ap.add_argument("--adapter", required=True, help="PEFT adapter dir (adapter-final)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base, adapter, out = resolve_base(args.base), Path(args.adapter), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs, scale = lora_pairs(adapter)
    print(f"{len(pairs)} LoRA modules, scale {scale:g}", flush=True)

    index = json.loads((base / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    merged: set[str] = set()
    for i, shard in enumerate(shards, 1):
        tensors = {}
        with safe_open(str(base / shard), framework="pt") as f:
            metadata = f.metadata()
            for key in f.keys():  # noqa: SIM118 (safe_open handle, not a dict)
                w = f.get_tensor(key)
                if key in pairs:
                    a, b = pairs[key]
                    if (b.shape[0], a.shape[1]) != tuple(w.shape):
                        sys.exit(f"shape mismatch at {key}: W {tuple(w.shape)} vs "
                                 f"B@A {(b.shape[0], a.shape[1])}")
                    w = (w.float() + scale * (b.float() @ a.float())).to(w.dtype)
                    merged.add(key)
                tensors[key] = w.contiguous()
        save_file(tensors, str(out / shard), metadata=metadata or {"format": "pt"})
        print(f"  [{i}/{len(shards)}] {shard}: {sum(k in pairs for k in tensors)} merged",
              flush=True)
        del tensors

    missing = sorted(set(pairs) - merged)
    if missing:
        sys.exit(f"{len(missing)} LoRA modules found no base weight, e.g. {missing[:3]}")

    for f in base.iterdir():
        if f.suffix in COPY_SUFFIXES and not f.name.startswith("adapter_"):
            shutil.copy2(f, out / f.name)
    (out / "MERGE_INFO.json").write_text(json.dumps({
        "base": args.base, "base_dir": str(base),
        "adapter": str(adapter), "adapter_sha256": sha256(adapter / "adapter_model.safetensors"),
        "modules_merged": len(merged), "scale": scale, "formula": "W + scale * B @ A (fp32)",
    }, indent=2))
    print(f"merged {len(merged)}/{len(pairs)} modules -> {out}")


if __name__ == "__main__":
    main()
