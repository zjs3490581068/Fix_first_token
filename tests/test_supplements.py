import copy
import sys
from types import SimpleNamespace

import pytest

from experiments.common import validate_conditions, validate_samples
from experiments.english_tokens import CONDITIONS
from experiments.two_token import CUE, PREFIX
from first_token.core import ROOT, STOP_IDS, STOPS, config, prompt_ids
from first_token.sampling import Engine


def test_english_conditions_and_leading_spaces(tokenizer):
    validate_conditions(tokenizer, CONDITIONS)
    validate_conditions(tokenizer, [("two", CUE, PREFIX)])
    for _, text, ids in CONDITIONS:
        assert text.startswith(" ") and len(ids) == 1
        assert tokenizer.encode(text.lstrip(), add_special_tokens=False) != ids


def test_two_token_generation_has_full_budget_and_full_text(tokenizer, monkeypatch):
    class Params:
        def __init__(self, **kw):
            self.__dict__.update(kw)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=Params))
    continuation = tokenizer.encode(" the answer \\boxed{2}", add_special_tokens=False)
    class LLM:
        def generate(self, inputs, params, use_tqdm):
            assert params.n == 1 and params.max_tokens == 16382
            assert params.seed == 20260819 and params.top_k == -1
            assert params.stop == STOPS and params.stop_token_ids == STOP_IDS
            assert params.temperature == .6 and params.top_p == .95
            assert inputs[0]["prompt_token_ids"] == prompt_ids(tokenizer, "Q") + PREFIX
            return [SimpleNamespace(prompt_token_ids=inputs[0]["prompt_token_ids"], outputs=[
                SimpleNamespace(index=0, token_ids=continuation, finish_reason="stop", stop_reason=151645)])]
    engine = Engine.__new__(Engine)
    engine.c = config(ROOT / "configs/two_token.json")
    engine.tok, engine.llm, engine.monitor = tokenizer, LLM(), SimpleNamespace(peak_mib=0)
    rows = engine.generate([{"problem_id": "p", "problem": "Q"}], None, 0, 1, prefix_ids=PREFIX)
    row = rows[0]
    assert row["output_token_ids"] == PREFIX + continuation
    assert row["output_text"] == CUE + " the answer \\boxed{2}"
    assert row["first_token_id"] == 2014 and row["second_token_id"] == 8253
    assert row["output_budget"] == 16384 and row["continuation_budget"] == 16382
    row.update(group="two", experiment_fingerprint="fp")
    validate_samples(rows, "fp", {("p", 0)}, PREFIX, "two", engine.c)
    for field, wrong in [("seed", 0), ("output_budget", 2048), ("forced_token_ids", [2014])]:
        broken = copy.deepcopy(rows)
        broken[0][field] = wrong
        with pytest.raises(ValueError):
            validate_samples(broken, "fp", {("p", 0)}, PREFIX, "two", engine.c)
    with pytest.raises(ValueError, match="duplicate"):
        validate_samples(rows + rows, "fp", {("p", 0)}, PREFIX, "two", engine.c)


def test_prefix_special_tokens_and_context_limit(tokenizer, monkeypatch):
    class Params:
        def __init__(self, **kw):
            self.__dict__.update(kw)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=Params))
    engine = Engine.__new__(Engine)
    engine.tok = tokenizer
    engine.c = config(ROOT / "configs/two_token.json")
    with pytest.raises(ValueError, match="special"):
        engine.generate([], None, 0, 1, prefix_ids=[2014, 151643])
    engine.c["engine"]["max_model_len"] = 2048
    with pytest.raises(ValueError, match="budget"):
        engine.generate([{"problem_id": "p", "problem": "Q"}], None, 0, 1, prefix_ids=PREFIX)


def test_supplement_grading_resume_and_duplicate_gate(tmp_path):
    from experiments import common
    from first_token.core import atomic_json, digest, file_hash, write_shard
    from first_token.preflight import source_hashes
    # Math-Verify uses process-main-thread signals for parse timeouts.
    data = tmp_path / "test.jsonl"
    data.write_text('{"problem_key":"p","problem":"Q","gt":"1"}\n')
    c = {"test_data": str(data), "seed": 20260819, "output_budget": 16384, "grading_workers": 1}
    m = {"config": c, "code": source_hashes(), "baseline": None,
         "datasets": {"test_data": {"sha256": file_hash(data)}}}
    m["fingerprint"] = digest(m)
    atomic_json(tmp_path / "manifest.json", m)
    rows = [{"group": "Solution", "problem_id": "p", "sample_index": 0, "seed": c["seed"],
             "experiment_fingerprint": m["fingerprint"], "forced_token_ids": [12478],
             "first_token_id": 12478, "output_token_ids": [12478, 1], "output_length": 2,
             "output_budget": 16384, "output_text": r" Solution \boxed{1}", "hit_token_cap": False}]
    write_shard(tmp_path / "samples/Solution/part.jsonl.gz", rows)
    conditions = [("Solution", " Solution", [12478])]
    report = common.grade_and_report(tmp_path, conditions)
    assert report["complete"] and report["groups"]["Solution"]["pass@1"] == 1
    assert common.grade_and_report(tmp_path, conditions) == report
    write_shard(tmp_path / "samples/Solution/duplicate.jsonl.gz", rows)
    with pytest.raises(ValueError, match="duplicate"):
        common.grade_and_report(tmp_path, conditions)
