"""dpo_lora.py wraps the model in DistributedDataParallel on 2 nodes; DDP
doesn't pass PEFT methods (set_adapter, save_pretrained) through. Those calls
must go to the unwrapped `core`. Single-GPU runs never hit this: the 2-node
DPO smoke test found it (2026-11-15)."""
import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "train" / "dpo_lora.py"
PEFT_ONLY = {"set_adapter", "save_pretrained", "load_adapter", "named_parameters"}


def test_peft_methods_are_called_on_the_unwrapped_model_after_ddp():
    tree = ast.parse(SRC.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    wrap = next(n.lineno for n in ast.walk(main) if isinstance(n, ast.Attribute)
                and n.attr == "DistributedDataParallel")
    bad = [(n.lineno, n.func.attr) for n in ast.walk(main)
           if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
           and isinstance(n.func.value, ast.Name) and n.func.value.id == "model"
           and n.func.attr in PEFT_ONLY
           and (n.lineno > wrap or _in_closure(main, n))]
    assert not bad, f"PEFT calls on the DDP-wrapped `model` (use `core`): {bad}"


def _in_closure(main, node):
    """True if node sits in a nested function, which runs after the wrap."""
    return any(isinstance(f, ast.FunctionDef) and f is not main and node in ast.walk(f)
               for f in ast.walk(main))


def test_the_model_is_put_in_training_mode():
    # HF's gradient checkpointing is a no-op in eval mode (from_pretrained's
    # default): without .train() the DPO smoke test ran out of memory.
    calls = [n.func.attr for n in ast.walk(ast.parse(SRC.read_text()))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert "train" in calls and "gradient_checkpointing_enable" in calls
