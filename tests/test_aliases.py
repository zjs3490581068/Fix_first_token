import pytest

from first_token.core import atomic_json, write_shard
from first_token.aliases import group_aliases, sampling_groups
from first_token.report import summarize


def setup_run(tmp_path, winner=2014):
    data = tmp_path / "test.jsonl"
    data.write_text('{"problem_key":"p","problem":"x","gt":"1","level":1,"capability":"algebra"}\n')
    c = {"test_data": str(data), "rollouts": 128, "groups": ["A", "C", "D", "E"], "grading_workers": 1}
    m = {"config": c, "fingerprint": "fp"}
    atomic_json(tmp_path / "manifest.json", m)
    atomic_json(tmp_path / "group_aliases.json", {"fingerprint": "fp", "aliases": {"D": "C"}})
    atomic_json(tmp_path / "search/winner.json", {"fingerprint": "fp", "forced_token_id": winner})
    return c, m


def test_alias_only_identical_conditions_and_no_separate_data(tmp_path):
    c, m = setup_run(tmp_path)
    aliases = group_aliases(c, tmp_path, m)
    assert sampling_groups(["A", "B", "C", "D", "E"], aliases) == ["A", "B", "C", "E"]
    assert sampling_groups(["D"], aliases) == ["C"]
    write_shard(tmp_path / "samples/D/conflict.jsonl.gz", [])
    with pytest.raises(ValueError, match="Separate D"):
        group_aliases(c, tmp_path, m)


def test_reject_alias_for_different_winner(tmp_path):
    c, m = setup_run(tmp_path, winner=5512)
    with pytest.raises(ValueError, match="conditions differ"):
        group_aliases(c, tmp_path, m)


def test_shared_report_reuses_128_samples_without_double_counting(tmp_path):
    c, m = setup_run(tmp_path)
    for group in ["A", "C", "E"]:
        raw, grades = [], []
        for i in range(128):
            row = {"group": group, "problem_id": "p", "sample_index": i, "experiment_fingerprint": "fp"}
            raw.append({**row, "forced_token_id": 2014 if group == "C" else None,
                        "first_token_id": 2014, "second_token_id": 1477, "output_length": 2,
                        "output_text": "test", "hit_token_cap": False})
            grades.append({**row, "correct": i < 32, "extracted_answer": "1", "grading_status": "correct" if i < 32 else "incorrect"})
        write_shard(tmp_path / f"samples/{group}/s.jsonl.gz", raw)
        write_shard(tmp_path / f"grades/{group}/s.jsonl.gz", grades)
    s = summarize(c, tmp_path, require_complete=True)
    assert s["complete"]
    assert s["independent_scored_samples"] == s["expected_independent_samples"] == 384
    assert s["groups"]["D"]["samples"] == 128
    assert s["groups"]["D"]["source_group"] == "C"
    assert s["groups"]["D"]["scores"] == s["groups"]["C"]["scores"]
    assert s["groups"]["D"]["scores"]["pass@1"] == .25
    from first_token.core import read_json
    comparison = read_json(tmp_path / "analysis/paired_comparisons.json")["D-C"]
    assert comparison["status"] == "shared_samples_not_an_independent_comparison"
    assert not (tmp_path / "samples/D").exists()
    assert not (tmp_path / "grades/D").exists()
    assert "not independent" in (tmp_path / "analysis/results.md").read_text()
    assert (tmp_path / "analysis/pass_at_k.png").is_file()


def test_incomplete_c_makes_d_incomplete(tmp_path):
    c, _ = setup_run(tmp_path)
    result = summarize(c, tmp_path)
    assert not result["groups"]["D"]["complete"]
    assert result["groups"]["D"]["scores"] is None


def test_grading_d_targets_c_only(tmp_path, monkeypatch):
    c, _ = setup_run(tmp_path)
    from first_token import grading
    from concurrent.futures import ThreadPoolExecutor
    monkeypatch.setattr(grading, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(grading, "grade_answer", lambda text, ref: {"correct": True, "extracted_answer": "1", "grading_status": "correct"})
    write_shard(tmp_path / "samples/C/s.jsonl.gz", [{"group": "C", "problem_id": "p", "sample_index": 0,
                "experiment_fingerprint": "fp", "output_text": "x"}])
    grading.grade_run(c, tmp_path, ["D"])
    assert (tmp_path / "grades/C/s.jsonl.gz").exists()
    assert not (tmp_path / "grades/D").exists()
