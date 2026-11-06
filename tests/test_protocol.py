import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from first_token import core
from first_token.grading import answer_key, entropy, grade_answer, select_winner
from first_token.metrics import pass_at_k
from first_token.sampling import Engine


def test_prompt_matches_reference():
    assert core.BOXED == "Problem: {q}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\nSolution:"
    assert core.STOPS == ["\nProblem:", "\nProblem :"]
    assert core.BOXED.format(q="Q").endswith(".\nSolution:")


def test_token_conditions_and_chinese_split(tokenizer):
    assert tokenizer.encode("解答", add_special_tokens=False) == [core.B_TOKEN]
    assert tokenizer.encode(" To determine", add_special_tokens=False) == [core.C_TOKEN, 8253]
    text = "证明这个数学结论"
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert len(ids) > 1
    assert core.prompt_ids(tokenizer, "x", ids[0]) == core.prompt_ids(tokenizer, "x") + ids[:1]
    assert core.prompt_ids(tokenizer, "x", core.C_TOKEN)[-1] == 2014
    assert tokenizer.encode("To", add_special_tokens=False) != tokenizer.encode(" To", add_special_tokens=False)


@pytest.mark.parametrize("bad", [[2014], True, -1, 151643, 151645, 999999])
def test_reject_invalid_prefix(tokenizer, bad):
    with pytest.raises(ValueError):
        core.prompt_ids(tokenizer, "x", bad)


def test_budget_and_seeds():
    assert core.token_budget(16384, None) == 16384
    assert core.token_budget(16384, 2014) == 16383
    flat = [core.sample_seed(20260819, i) for i in range(128)]
    batched = [core.sample_seed(20260819, offset + i) for offset in range(0, 128, 16) for i in range(16)]
    assert flat == batched and len(set(flat)) == 128
    with pytest.raises(ValueError):
        core.token_budget(1, 2014)


def test_generation_uses_ids_budget_and_reconstructs_prefix(tokenizer, monkeypatch):
    class Params:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=Params))
    continuation = tokenizer.encode(" determine \\boxed{2}", add_special_tokens=False)
    class LLM:
        def generate(self, inputs, params, use_tqdm):
            assert params.max_tokens == 16383
            assert params.seed == 20260835
            assert params.top_k == -1 and params.stop_token_ids == core.STOP_IDS
            assert inputs[0]["prompt_token_ids"][-1] == 2014
            return [SimpleNamespace(prompt_token_ids=x["prompt_token_ids"], outputs=[
                SimpleNamespace(index=i, token_ids=continuation, finish_reason="stop", stop_reason=151645) for i in range(params.n)]) for x in inputs]
    engine = Engine.__new__(Engine)
    engine.c = core.config(core.ROOT / "configs/main.json")
    engine.tok, engine.llm, engine.monitor = tokenizer, LLM(), SimpleNamespace(peak_mib=0)
    records = engine.generate([{"problem_id": "q", "problem": "1+1"}], 2014, 16, 2)
    assert records[0]["output_token_ids"] == [2014] + continuation
    assert records[0]["output_text"] == " To determine \\boxed{2}"
    assert records[0]["second_token_id"] == continuation[0]
    assert [r["seed"] for r in records] == [20260835, 20260836]


@pytest.mark.parametrize("text,expected", [
    (r"x \boxed{1} revised \boxed{\frac{2}{4}}", True),
    (r"\boxed{\frac{1}{2}} incomplete \boxed{7", True),
    ("Final answer: 0.5", True),
    (r"\boxed{3}", False),
    ("no committed answer", False),
])
def test_grading(text, expected):
    assert grade_answer(text, r"\frac{1}{2}")["correct"] is expected


def test_answer_key_and_entropy():
    assert answer_key(r"\boxed{0.5}") == answer_key(r"\boxed{\frac{1}{2}}")
    assert answer_key("missing") is None
    assert entropy([None] * 16) == 0
    assert entropy([None] * 8 + [1] * 8) == 1


def test_label_free_decider_guard_and_tie():
    scores = {"none": {"none_share": 0.0},
              "x": {"none_share": 0.11, "entropy": 0.0, "forced_token_id": 1, "accuracy": 1},
              "y": {"none_share": 0.1, "entropy": 0.5, "forced_token_id": 3, "accuracy": 1},
              "z": {"none_share": 0.0, "entropy": 0.5, "forced_token_id": 2, "accuracy": 0}}
    assert select_winner(scores, 0.1)["forced_token_id"] == 2
    with pytest.raises(ValueError):
        select_winner({}, 0.1)
    assert select_winner({"none": {"none_share": 0}, "x": {"none_share": 1, "entropy": 0, "forced_token_id": 1}}, .1) is None


def test_pass_k_reference_formula():
    for c in [0, 1, 64, 127, 128]:
        for k in [1, 2, 8, 64, 128]:
            expected = 1 - math.comb(128-c, k) / math.comb(128, k) if 128-c >= k else 1
            assert pass_at_k(128, c, k) == pytest.approx(expected)
        assert pass_at_k(128, c, 1) == pytest.approx(c / 128)
    assert pass_at_k(16, 10, 128) is None


def test_durable_resume_rejects_corruption_duplicates_and_mismatch(tmp_path):
    path = tmp_path / "shard.jsonl.gz"
    rows = [{"problem_id": "p", "sample_index": i, "experiment_fingerprint": "fp"} for i in range(16)]
    core.write_shard(path, rows)
    keys = {("p", i) for i in range(16)}
    assert core.checked_records(path, "fp", keys) == rows
    with pytest.raises(FileExistsError):
        core.write_shard(path, rows)
    with pytest.raises(ValueError):
        core.checked_records(path, "wrong", keys)
    with pytest.raises(ValueError):
        core.checked_records(path, "fp", keys | {("p", 16)})
    dup = tmp_path / "duplicate.jsonl.gz"
    core.write_shard(dup, rows + rows[:1])
    with pytest.raises(ValueError):
        core.checked_records(dup, "fp")
    broken = tmp_path / "broken.jsonl.gz"
    broken.write_bytes(path.read_bytes()[:30])
    with pytest.raises((EOFError, OSError)):
        core.checked_records(broken, "fp")


def test_search_reader_excludes_labels(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text('{"problem_key":"p","problem":"x","gt":"SECRET_LABEL","solution":"S"}\n')
    assert core.problems(p) == [{"problem_id": "p", "problem": "x"}]


def test_condition_and_gpu_authorization(tmp_path):
    assert core.condition("D", 2014)["forced_token_id"] == core.condition("C")["forced_token_id"]
    with pytest.raises(ValueError):
        core.condition("D")
    with pytest.raises(ValueError):
        core.condition("F", 2014)
    c = core.config(core.ROOT / "configs/main.json")
    c["gpus"] = [0, 1, 1]
    f = tmp_path / "bad.json"
    core.atomic_json(f, c)
    with pytest.raises(ValueError):
        core.config(f)
