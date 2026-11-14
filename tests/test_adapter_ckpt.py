"""train/adapter_ckpt.py: where the DPO trainer's checkpoints land, and resume."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
from adapter_ckpt import publish_adapter, resume_point  # noqa: E402


def peft_save(d: Path, name: str, tag: str) -> None:
    """What PEFT 0.21 writes for save_pretrained(d, selected_adapters=[name])."""
    sub = d / name if name != "default" else d
    sub.mkdir(parents=True)
    (sub / "adapter_model.safetensors").write_text(tag)
    (sub / "adapter_config.json").write_text("{}")


def test_a_named_adapter_is_published_flat_with_its_extras(tmp_path):
    tmp = tmp_path / "adapter-latest.saving"
    peft_save(tmp, "policy", "w1")
    (tmp / "optimizer.pt").write_text("opt")
    (tmp / "STEP").write_text("25")
    publish_adapter(tmp, "policy", tmp_path / "adapter-latest")
    dest = tmp_path / "adapter-latest"
    assert (dest / "adapter_model.safetensors").read_text() == "w1"
    assert (dest / "optimizer.pt").exists() and (dest / "STEP").read_text() == "25"
    assert not (dest / "policy").exists() and not tmp.exists()


def test_publishing_replaces_the_previous_checkpoint(tmp_path):
    for step, tag in ((25, "w1"), (50, "w2")):
        tmp = tmp_path / "s"
        peft_save(tmp, "policy", tag)
        (tmp / "STEP").write_text(str(step))
        publish_adapter(tmp, "policy", tmp_path / "adapter-latest")
    assert (tmp_path / "adapter-latest" / "adapter_model.safetensors").read_text() == "w2"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["adapter-latest"]


def test_a_default_adapter_saved_flat_also_works(tmp_path):
    peft_save(tmp_path / "s", "default", "w")
    publish_adapter(tmp_path / "s", "policy", tmp_path / "adapter-final")
    assert (tmp_path / "adapter-final" / "adapter_model.safetensors").read_text() == "w"


def test_resume_point(tmp_path):
    assert resume_point(tmp_path) == ("fresh", None, 0)
    latest = tmp_path / "adapter-latest"
    latest.mkdir()
    (latest / "STEP").write_text("50\n")
    assert resume_point(tmp_path) == ("fresh", None, 0)       # no weights: not a checkpoint
    (latest / "adapter_model.safetensors").write_text("w")
    assert resume_point(tmp_path) == ("resume", latest, 50)
    (tmp_path / "adapter-final").mkdir()
    (tmp_path / "adapter-final" / "adapter_model.safetensors").write_text("w")
    assert resume_point(tmp_path) == ("done", None, 0)
