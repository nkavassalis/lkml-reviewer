"""linuspr.template: thinking mode for the reasoning experiment."""
from __future__ import annotations

from linuspr.template import prompt_text, split_thinking


class FakeTok:
    def apply_chat_template(self, msgs, tokenize, add_generation_prompt, enable_thinking):
        return f"think={enable_thinking}|{msgs[-1]['content']}"


def test_thinking_is_off_unless_asked():
    assert prompt_text(FakeTok(), "x") == "think=False|x"           # what the fine-tunes were trained with
    assert prompt_text(FakeTok(), "x", thinking=True) == "think=True|x"


def test_split_thinking():
    assert split_thinking("<think>\nreason\n</think>\n\nThe review.") == ("reason", "The review.")
    assert split_thinking("reason without an opening tag</think>Review") == ("reason without an opening tag", "Review")
    assert split_thinking("<think>ran out of tokens mid-thought") == ("ran out of tokens mid-thought", None)
