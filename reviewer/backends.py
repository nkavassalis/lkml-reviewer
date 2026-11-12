"""Inference backends. Heavy imports happen inside the backend, never at module
import, so the gh/prompt layers (and their tests) need nothing but the stdlib.

  cuda   transformers + peft on an NVIDIA GPU (BF16, or 4-bit with --load-in-4bit)
  mlx    Apple silicon via mlx-lm, on a merged + quantized model (docs/CONVERSION.md)
"""
from __future__ import annotations

import sys

from linuspr.template import prompt_text

BASE_MODEL = "Qwen/Qwen3.8-27B"
# What a person trying the model should see. Greedy decoding loops on ~44% of
# outputs for this adapter (docs/REPORT.md), so it's never the default.
SAMPLING = {"do_sample": True, "temperature": 0.7, "top_p": 0.95, "repetition_penalty": 1.1}
BF16_NEEDS_GB = 60   # ~55 GB of weights + KV cache headroom
NF4_NEEDS_GB = 22


class BackendError(RuntimeError):
    """A setup problem with an actionable message."""


class CudaBackend:
    def __init__(self, adapter: str, *, base: str = BASE_MODEL, load_in_4bit: bool = False,
                 max_new_tokens: int = 1024, seed: int | None = None) -> None:
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForImageTextToText, AutoTokenizer
        except ImportError as e:
            raise BackendError(
                f"missing inference dependency ({e.name}): "
                "pip install -r reviewer/requirements-cuda.txt") from e
        if not torch.cuda.is_available():
            raise BackendError("no CUDA GPU visible to torch (the Mac version is the mlx backend)")
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        need = NF4_NEEDS_GB if load_in_4bit else BF16_NEEDS_GB
        if total < need:
            hint = "" if load_in_4bit else " Try --load-in-4bit (needs bitsandbytes, ~22 GB)."
            raise BackendError(f"GPU has {total:.0f} GB; this needs ~{need} GB.{hint}")
        try:
            import fla  # noqa: F401  (Gated DeltaNet kernels)
        except ImportError:
            print("warning: flash-linear-attention not installed; the hybrid attention "
                  "layers fall back to a much slower PyTorch path", file=sys.stderr)

        kwargs = {"dtype": torch.bfloat16, "device_map": {"": 0}}
        if load_in_4bit:
            try:
                import bitsandbytes  # noqa: F401
                from transformers import BitsAndBytesConfig
            except ImportError as e:
                raise BackendError("--load-in-4bit needs bitsandbytes: pip install bitsandbytes") from e
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(base, padding_side="left")
        model = AutoModelForImageTextToText.from_pretrained(base, **kwargs)
        try:
            self.model = PeftModel.from_pretrained(model, adapter)
        except (OSError, ValueError) as e:
            raise BackendError(f"could not load the LoRA adapter from {adapter!r}: {e}") from e
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.seed = seed

    def generate(self, users: list[str], *, batch: int = 4) -> list[str]:
        torch = self.torch
        out: list[str] = []
        for i in range(0, len(users), batch):
            prompts = [prompt_text(self.tok, u) for u in users[i : i + batch]]
            enc = self.tok(prompts, return_tensors="pt", padding=True,
                           add_special_tokens=False).to("cuda")
            if self.seed is not None:
                torch.manual_seed(self.seed + i)
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                          pad_token_id=self.tok.pad_token_id, **SAMPLING)
            out += self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                                         skip_special_tokens=True)
        return [t.strip() for t in out]


class MlxBackend:
    """Apple silicon via mlx-lm. Takes an already-converted MLX model: for
    lkml-reviewer that's the LoRA *merged into* the base, then quantized
    (docs/CONVERSION.md), so there is no separate adapter at inference time."""

    def __init__(self, model: str, *, max_new_tokens: int = 1024, seed: int | None = None,
                 **_ignored) -> None:
        try:
            import mlx.core as mx
            from mlx_lm import load
            from mlx_lm.sample_utils import make_logits_processors, make_sampler
        except ImportError as e:
            raise BackendError(
                f"missing inference dependency ({e.name}): "
                "pipx inject lkml-reviewer mlx-lm  (Apple silicon only)") from e
        self.mx = mx
        try:
            self.model, self.tok = load(model)
        except Exception as e:  # noqa: BLE001  (bad path, bad repo id, unsupported arch)
            raise BackendError(f"could not load the MLX model {model!r}: {e}") from e
        self.sampler = make_sampler(temp=SAMPLING["temperature"], top_p=SAMPLING["top_p"])
        self.processors = make_logits_processors(
            repetition_penalty=SAMPLING["repetition_penalty"])
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.stats: list[dict] = []

    def generate(self, users: list[str], *, batch: int = 1) -> list[str]:
        from mlx_lm import stream_generate  # noqa: PLC0415

        out = []
        for i, u in enumerate(users):
            if self.seed is not None:
                self.mx.random.seed(self.seed + i)
            prompt = prompt_text(self.tok, u)
            text, last = "", None
            for last in stream_generate(self.model, self.tok, prompt,
                                        max_tokens=self.max_new_tokens, sampler=self.sampler,
                                        logits_processors=self.processors):
                text += last.text
            if last is not None:
                self.stats.append({"prompt_tokens": last.prompt_tokens,
                                   "prompt_tps": last.prompt_tps,
                                   "generation_tokens": last.generation_tokens,
                                   "generation_tps": last.generation_tps,
                                   "peak_memory_gb": last.peak_memory})
            out.append(text.strip())
        return out


def load_backend(name: str, adapter: str | None, *, model: str | None = None, **kwargs):
    if name == "cuda":
        if not adapter:
            raise BackendError("the cuda backend needs --adapter")
        return CudaBackend(adapter, **kwargs)
    if name == "mlx":
        if not model:
            raise BackendError("the mlx backend needs --model (a converted MLX model dir or HF id)")
        return MlxBackend(model, **kwargs)
    raise BackendError(f"unknown backend {name!r} (cuda or mlx)")
