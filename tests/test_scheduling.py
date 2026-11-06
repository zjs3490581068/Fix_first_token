from collections import Counter

import pytest

from first_token import core
from first_token.aliases import freeze_aliases, group_aliases
from first_token.scheduling import prepare_sample_tasks, load_sample_tasks


def fixture_run(tmp_path):
    data = tmp_path / "test.jsonl"
    data.write_text("".join('{"problem_key":"p%d","problem":"x"}\n' % i for i in range(7)))
    c = {"test_data": str(data), "gpus": [0, 4, 3], "groups": ["A"],
         "problems_per_batch": 2, "rollouts": 5, "rollouts_per_batch": 2}
    m = {"fingerprint": "fp", "config": c}
    core.atomic_json(tmp_path / "manifest.json", m)
    return c, m


def test_resume_distributes_exact_keys_without_completed_batches(tmp_path):
    c, m = fixture_run(tmp_path)
    plan = prepare_sample_tasks(c, tmp_path, ["A"])
    jobs = [j for worker in plan["workers"] for j in worker]
    keys = [(p, i) for j in jobs for p in j["problem_ids"] for i in range(j["offset"], j["offset"] + j["n"])]
    assert len(keys) == len(set(keys)) == 35
    completed = jobs[0]
    path = tmp_path / "samples/A" / f"w{completed['artifact_worker']}_p{completed['block_index']:04d}_r{completed['offset']:03d}.jsonl.gz"
    records = [{"problem_id": p, "sample_index": i, "group": "A", "forced_token_id": None,
                "experiment_fingerprint": "fp"} for p in completed["problem_ids"]
               for i in range(completed["offset"], completed["offset"] + completed["n"])]
    core.write_shard(path, records)
    plan = prepare_sample_tasks(c, tmp_path, ["A"])
    pending = [(p, i) for worker in plan["workers"] for j in worker for p in j["problem_ids"]
               for i in range(j["offset"], j["offset"] + j["n"])]
    assert Counter(pending) + Counter((r["problem_id"], r["sample_index"]) for r in records) == Counter(keys)
    assert plan["completed_samples"] == len(records)
    assert load_sample_tasks(c, tmp_path, ["A"], 0, m) == plan["workers"][0]
    plan["workers"][0].append({"tampered": True})
    core.atomic_json(tmp_path / "task_plans/sample_A.json", plan)
    with pytest.raises(ValueError, match="mismatched"):
        load_sample_tasks(c, tmp_path, ["A"], 0, m)


def test_freezes_sharing_only_when_winner_matches_c(tmp_path):
    c = {"groups": ["C", "D"]}
    m = {"fingerprint": "fp"}
    core.atomic_json(tmp_path / "search/winner.json", {"fingerprint": "fp", "forced_token_id": 2014})
    freeze_aliases(c, tmp_path, m)
    assert group_aliases(c, tmp_path, m) == {"D": "C"}
    other = tmp_path / "other"
    core.atomic_json(other / "search/winner.json", {"fingerprint": "fp", "forced_token_id": 6771})
    freeze_aliases(c, other, m)
    assert group_aliases(c, other, m) == {}
