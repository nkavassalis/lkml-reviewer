"""train/rubric.py: score arms with the next step's yardstick, then compare them."""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
import rubric  # noqa: E402


def compare_file(tmp: Path, n: int = 30) -> Path:
    f = tmp / "compare.jsonl"
    with open(f, "w") as out:
        for i in range(n):
            out.write(json.dumps({"eval_index": i, "input": f"patch {i}", "reference": "ref",
                                  "v3_sampled": "vague", "v4_sampled": "GOOD specific"}) + "\n")
    return f


def fake_ask(patch, reference, candidate):
    return {"score": 8.0, "false_claims": 0} if "GOOD" in candidate else {"score": 4.0, "false_claims": 1}


def test_score_is_resumable_and_paired_compares(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rubric, "ask", fake_ask)
    cmp, out = compare_file(tmp_path), tmp_path / "scores.jsonl"
    rubric.main(["score", str(cmp), "--arms", "v3_sampled,v4_sampled", "--out", str(out), "--workers", "2"])
    assert len(out.read_text().splitlines()) == 60
    rubric.main(["score", str(cmp), "--arms", "v3_sampled,v4_sampled", "--out", str(out)])
    assert len(out.read_text().splitlines()) == 60                       # nothing re-scored
    js = tmp_path / "r.json"
    rubric.main(["paired", str(cmp), str(out), "--a", "v3_sampled:v3", "--b", "v4_sampled:v4", "--json", str(js)])
    res = json.loads(js.read_text())
    assert res["a"] == "v3" and res["b"] == "v4" and res["n"] == 30
    assert res["rubric score"]["diff"][0] == pytest.approx(4.0)
    assert res["false claims"]["diff"][0] == pytest.approx(-1.0)


def test_paired_uses_only_rows_scored_for_both():
    scores = [{"eval_index": 0, "key": "a", "score": 5}, {"eval_index": 0, "key": "b", "score": 7},
              {"eval_index": 1, "key": "a", "score": 5}, {"eval_index": 1, "key": "b", "score": None}]
    res = rubric.paired(scores, "a", "b")
    assert res["n"] == 1 and res["rubric score"]["diff"][0] == pytest.approx(2.0)


def test_the_result_feeds_decide_as_the_target(tmp_path, monkeypatch):
    from decide import NEEDS_PERSON
    from decide import main as decide_main
    monkeypatch.setattr(rubric, "ask", fake_ask)
    cmp, out, js = compare_file(tmp_path), tmp_path / "s.jsonl", tmp_path / "r.json"
    rubric.main(["score", str(cmp), "--arms", "v3_sampled,v4_sampled", "--out", str(out)])
    rubric.main(["paired", str(cmp), str(out), "--a", "v3_sampled:v3", "--b", "v4_sampled:v4", "--json", str(js)])
    spec = ["--paired", f"{js}:rubric score:higher", "--target", "rubric score"]
    assert decide_main(["--rule", "v4", *spec]) == 0
    assert decide_main(["--rule", "v3", *spec]) == NEEDS_PERSON


def test_the_builtin_scorer_parses_a_judge_reply(monkeypatch):
    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            self.rfile.read(int(self.headers["Content-Length"]))
            body = json.dumps({"choices": [{"message": {"content":
                   'thinking...\n{"score": 7, "false_claims": 1, "reason": "ok"}'}}]}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(rubric, "URL", f"http://127.0.0.1:{srv.server_port}/v1/chat/completions")
    try:
        assert rubric._ask("p", "r", "c") == {"score": 7.0, "false_claims": 1, "reason": "ok"}
    finally:
        srv.shutdown()
