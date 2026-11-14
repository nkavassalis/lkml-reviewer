"""Checkpoint layout helpers for training scripts. Stdlib only, so the hermetic
tests run them without torch.

PEFT writes an adapter that isn't named "default" into a subdirectory
(`save_pretrained(d, selected_adapters=["policy"])` -> `d/policy/...`). Every
consumer here (the chain's success check, the peer copy, merge_lora.py, the Mac
conversion) expects `d/adapter_model.safetensors`, so the DPO trainer saves
through `publish_adapter`, which also makes the swap atomic: a hard stop in
the middle of a save leaves the previous checkpoint intact.
"""
from __future__ import annotations

import shutil
from pathlib import Path


def publish_adapter(saved: Path, name: str, dest: Path) -> None:
    """Move what save_pretrained wrote under `saved` (in `saved/name/` for a
    named adapter, or `saved/` itself) to `dest`, replacing it atomically."""
    src = saved / name if (saved / name / "adapter_model.safetensors").exists() else saved
    if not (src / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"no adapter_model.safetensors under {saved}")
    if src != saved:                      # pull extras (optimizer.pt, STEP) in with it
        for extra in saved.iterdir():
            if extra.name != name:
                shutil.move(str(extra), src / extra.name)
    old = dest.with_name(dest.name + ".old")
    shutil.rmtree(old, ignore_errors=True)
    if dest.exists():
        dest.rename(old)
    src.rename(dest)
    shutil.rmtree(old, ignore_errors=True)
    if src != saved:
        shutil.rmtree(saved, ignore_errors=True)


def resume_point(out: Path) -> tuple[str, Path | None, int]:
    """("done", None, 0) if out/adapter-final exists; ("resume", dir, step) if
    out/adapter-latest has a STEP file; else ("fresh", None, 0)."""
    if (out / "adapter-final" / "adapter_model.safetensors").exists():
        return "done", None, 0
    latest = out / "adapter-latest"
    if (latest / "STEP").exists() and (latest / "adapter_model.safetensors").exists():
        return "resume", latest, int((latest / "STEP").read_text().strip())
    return "fresh", None, 0
