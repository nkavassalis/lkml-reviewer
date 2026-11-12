"""`lkml-reviewer --doctor`: check everything a review needs and say how to fix
what's missing. Each check is independent: one failure never hides another.
Imports nothing heavy at module level (torch is probed, not required).
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from . import gh

OK, WARN, FAIL = "ok", "warn", "FAIL"


@dataclass
class Check:
    name: str
    status: str
    detail: str
    fix: str = ""


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def check_python() -> Check:
    v = sys.version_info
    if v < (3, 10):
        return Check("python", FAIL, f"{v.major}.{v.minor}", "Python 3.10+ is required")
    return Check("python", OK, f"{v.major}.{v.minor}.{v.micro} ({sys.executable})")


def check_gh() -> list[Check]:
    if shutil.which("gh") is None:
        return [Check("gh", WARN, "not installed",
                      "only needed for --pr: https://cli.github.com "
                      "(--local and --diff-file work without it)")]
    out = [Check("gh", OK, shutil.which("gh") or "")]
    try:
        account = gh.check_auth()
        out.append(Check("gh auth", OK, f"logged in ({account})"))
    except gh.GhError:
        out.append(Check("gh auth", WARN, "not logged in", "gh auth login  (or set GH_TOKEN)"))
    return out


def check_inference() -> list[Check]:
    missing = [m for m in ("torch", "transformers", "peft", "accelerate") if not _has(m)]
    if missing:
        return [Check("inference libs", WARN, f"missing: {', '.join(missing)}",
                      "only needed to generate reviews: re-run scripts/install-reviewer.sh "
                      "--cuda, or pipx install --force "
                      "'lkml-reviewer[cuda] @ git+https://github.com/nkavassalis/lkml-reviewer'")]
    out = [Check("inference libs", OK, "torch, transformers, peft, accelerate")]
    if not _has("fla"):
        out.append(Check("fla kernels", WARN, "flash-linear-attention missing",
                         "pip install flash-linear-attention  (else a slow fallback path)"))
    try:
        import torch
    except Exception as e:  # noqa: BLE001  (a broken torch install is a finding, not a crash)
        out.append(Check("torch", FAIL, f"import failed: {e}"))
        return out
    if not torch.cuda.is_available():
        out.append(Check("cuda", WARN, f"torch {torch.__version__} sees no CUDA GPU",
                         "install a CUDA build of torch for your GPU (see README), "
                         "or wait for the mlx backend on Apple silicon"))
        return out
    p = torch.cuda.get_device_properties(0)
    gb = p.total_memory / 1e9
    from .backends import BF16_NEEDS_GB, NF4_NEEDS_GB  # noqa: PLC0415
    if gb >= BF16_NEEDS_GB:
        out.append(Check("gpu", OK, f"{p.name}, {gb:.0f} GB: BF16 fits"))
    elif gb >= NF4_NEEDS_GB:
        out.append(Check("gpu", WARN, f"{p.name}, {gb:.0f} GB: too small for BF16",
                         "use --load-in-4bit (pip install bitsandbytes)"))
    else:
        out.append(Check("gpu", FAIL, f"{p.name}, {gb:.0f} GB", f"needs ≥{NF4_NEEDS_GB} GB"))
    return out


def check_mlx() -> list[Check]:
    """Apple silicon: mlx installed, and how much memory the GPU may wire."""
    if sys.platform != "darwin":
        return []
    if not _has("mlx_lm"):
        return [Check("mlx", WARN, "mlx-lm not installed",
                      "scripts/install-reviewer.sh --mlx  (or pipx inject lkml-reviewer mlx-lm)")]
    try:
        import mlx.core as mx
        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        gb = info["max_recommended_working_set_size"] / 1e9
    except Exception as e:  # noqa: BLE001
        return [Check("mlx", FAIL, f"mlx import failed: {e}")]
    status = OK if gb >= 35 else WARN
    return [Check("mlx", status, f"GPU may wire {gb:.0f} GB (8-bit 27B needs ~32 GB at 4k context)",
                  "" if status == OK else
                  "raise it: sudo sysctl iogpu.wired_limit_mb=<MB>  (resets on reboot)")]


def check_adapter(adapter: str | None) -> Check:
    if not adapter:
        return Check("adapter", WARN, "none given",
                     "pass --adapter DIR_OR_HF_ID or set LKML_REVIEWER_ADAPTER")
    path = Path(adapter).expanduser()
    if path.exists():
        need = ["adapter_config.json", "adapter_model.safetensors"]
        missing = [n for n in need if not (path / n).exists()]
        if missing:
            return Check("adapter", FAIL, f"{path} lacks {', '.join(missing)}",
                         "point at the adapter-final directory itself")
        return Check("adapter", OK, str(path))
    if adapter.count("/") == 1 and not adapter.startswith((".", "/", "~")):
        return Check("adapter", OK, f"{adapter} (Hugging Face id; downloaded on first use; "
                     "a private repo needs `hf auth login`)")
    return Check("adapter", FAIL, f"{adapter}: no such directory", "check the path")


def run(adapter: str | None) -> int:
    on_mac = sys.platform == "darwin"
    engine = check_mlx() if on_mac else check_inference()
    checks = [check_python(), *check_gh(), *engine] + ([] if on_mac else [check_adapter(adapter)])
    width = max(len(c.name) for c in checks)
    for c in checks:
        mark = {"ok": "✓", "warn": "!", "FAIL": "✗"}[c.status]
        print(f" {mark} {c.name:<{width}}  {c.detail}")
        if c.fix and c.status != OK:
            print(f"   {'':<{width}}  → {c.fix}")
    failed = [c for c in checks if c.status == FAIL]
    can_dry_run = check_python().status == OK
    can_review = can_dry_run and not any(
        c.status != OK for c in checks if c.name in ("inference libs", "cuda", "gpu", "adapter", "mlx"))
    print()
    print(f"  dry runs (--dry-run):       {'ready' if can_dry_run else 'not ready'}")
    print(f"  reviews on this machine:    {'ready' if can_review else 'not ready (see above)'}")
    print(f"  GitHub PRs (--pr):          "
          f"{'ready' if all(c.status == OK for c in checks if c.name.startswith('gh')) else 'not ready'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run(os.environ.get("LKML_REVIEWER_ADAPTER")))
