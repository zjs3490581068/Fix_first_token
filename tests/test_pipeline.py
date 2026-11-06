from pathlib import Path

import pytest

from first_token import core
from first_token.report import summarize
from first_token.sampling import sample_condition


def test_sampling_resume_does_not_generate_completed_shards(tmp_path):
    class Engine:
        calls = 0
        monitor = type("Monitor", (), {"peak_mib": 1})()
        def generate(self, rows, forced, offset, n):
            self.calls += 1
            return [{"problem_id": r["problem_id"], "sample_index": i,
                     "forced_token_id": forced, "output_length": 3} for r in rows for i in range(offset, offset+n)]
    engine = Engine()
    c = {"problems_per_batch": 1, "rollouts_per_batch": 16, "minimum_free_disk_gib": 0}
    m = {"fingerprint": "fp", "models": {"base": {"id": "base", "revision": "rev"}}}
    spec = core.condition("A")
    rows = [{"problem_id": "p", "problem": "x"}]
    for _ in range(2):
        sample_condition(engine, c, tmp_path, m, spec, rows, 32, tmp_path / "samples/A", 0, "test")
    assert engine.calls == 2
    assert len(list((tmp_path / "samples/A").glob("*.jsonl.gz"))) == 2


def test_incomplete_report_never_claims_pass128(tmp_path):
    data = tmp_path / "test.jsonl"
    data.write_text('{"problem_key":"p","problem":"x","gt":"1","level":1,"capability":"algebra"}\n')
    c = {"test_data": str(data), "rollouts": 128, "groups": ["A"]}
    core.atomic_json(tmp_path / "manifest.json", {"config": c, "fingerprint": "fp"})
    out = summarize(c, tmp_path)
    assert out["complete"] is False
    assert out["groups"]["A"]["scores"] is None
    with pytest.raises(RuntimeError):
        summarize(c, tmp_path, require_complete=True)


def test_full_report_and_duplicate_gate(tmp_path):
    data = tmp_path / "test.jsonl"
    data.write_text('{"problem_key":"p","problem":"x","gt":"1","level":1,"capability":"algebra"}\n')
    c = {"test_data": str(data), "rollouts": 128, "groups": ["A"]}
    core.atomic_json(tmp_path / "manifest.json", {"config": c, "fingerprint": "fp"})
    raw, graded = [], []
    for i in range(128):
        r = {"problem_id": "p", "sample_index": i, "experiment_fingerprint": "fp", "group": "A", "forced_token_id": None}
        raw.append({**r, "first_token_id": 10, "second_token_id": 11, "output_length": 2, "hit_token_cap": False, "output_text": "x"})
        graded.append({**r, "correct": i == 0, "extracted_answer": "1", "grading_status": "correct" if i == 0 else "incorrect"})
    core.write_shard(tmp_path / "samples/A/s.jsonl.gz", raw)
    core.write_shard(tmp_path / "grades/A/s.jsonl.gz", graded)
    summary = summarize(c, tmp_path, require_complete=True)
    assert summary["groups"]["A"]["scores"]["pass@128"] == 1
    assert (tmp_path / "analysis/pass_at_k.png").is_file()
    core.write_shard(tmp_path / "samples/A/duplicate.jsonl.gz", raw)
    core.write_shard(tmp_path / "grades/A/duplicate.jsonl.gz", graded)
    with pytest.raises(ValueError, match="Duplicate"):
        summarize(c, tmp_path)


def test_depth_projection_keeps_sources_and_presence_gate(tmp_path, tokenizer, monkeypatch):
    from first_token.search import propose
    from transformers import AutoTokenizer
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
    data = tmp_path / "probe.jsonl"
    data.write_text(''.join('{"problem_key":"p%d","problem":"x","gt":"unused"}\n' % i for i in range(4)))
    c = {"probe_data": str(data), "beam_problems": 4, "min_problem_fraction": .5, "nominees_per_depth": 20,
         "models": {"base": {"path": "unused-local-model"}}}
    core.atomic_json(tmp_path / "manifest.json", {"config": c, "fingerprint": "fp"})
    for i in range(4):
        depths = {"1": [{"token_ids": [2014], "probability": .4}, {"token_ids": [151643], "probability": .5}],
                  "2": [{"token_ids": [2014, 8253], "probability": .2}]}
        if i == 0:
            depths["1"].append({"token_ids": [106185], "probability": .9})
        core.atomic_json(tmp_path / f"search/beam/p{i:03d}.json", {"fingerprint": "fp", "problem_id": f"p{i}", "depths": depths})
    result = propose(c, tmp_path)
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["forced_token_id"] == 2014
    assert result["candidates"][0]["sources"] == [1, 2]
    assert result["by_depth"]["1"][0]["mean_probability"] == pytest.approx(.4)
