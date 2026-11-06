import copy

import pytest

from first_token import core
from first_token.artifacts import matched_baseline, source_manifest
from first_token.preflight import source_hashes


def source_run(root):
    root.mkdir()
    data = root / "data.jsonl"
    data.write_text('{"problem_key":"p","problem":"Q","gt":"1"}\n')
    code = root / "source" / "old.py"
    code.parent.mkdir()
    code.write_text("# archived experiment source\n")
    c = {"test_data": str(data), "seed": 20260819, "temperature": .6, "top_p": .95,
         "output_budget": 16384, "rollouts": 2, "groups": ["A"]}
    model = {"id": "base", "revision": "revision", "files": {"weight": "hash"}}
    manifest = {"config": c, "models": {"base": model}, "code": {"old.py": core.file_hash(code)},
                "datasets": {"test_data": {"path": str(data), "sha256": core.file_hash(data)}},
                "prompt": core.BOXED, "stops": core.STOPS, "stop_ids": core.STOP_IDS}
    manifest["fingerprint"] = core.digest(manifest)
    core.atomic_json(root / "manifest.json", manifest)
    core.atomic_json(root / "status.json", {"state": "complete"})
    raw, grades = [], []
    for i in range(2):
        common = {"group": "A", "problem_id": "p", "sample_index": i,
                  "experiment_fingerprint": manifest["fingerprint"]}
        raw.append({**common, "seed": c["seed"] + i, "forced_token_id": None,
                    "output_budget": 16384, "output_length": 1, "output_token_ids": [10]})
        grades.append({**common, "correct": i == 0})
    raw_path = root / "samples/A/part.jsonl.gz"
    grade_path = root / "grades/A/part.jsonl.gz"
    core.write_shard(raw_path, raw)
    core.write_shard(grade_path, grades)
    core.atomic_json(grade_path.with_name(grade_path.name + ".meta.json"),
                     {"raw_sha256": core.file_hash(raw_path), "records": 2})
    return c, model, manifest


def test_historical_source_is_independent_of_current_code(tmp_path):
    root = tmp_path / "baseline"
    c, model, m = source_run(root)
    assert source_hashes() != m["code"]
    assert source_manifest(root) == m
    baseline, identity = matched_baseline(root, c, model, m["datasets"]["test_data"]["sha256"])
    assert baseline == {"A": {"p": 1}}
    assert identity["fingerprint"] == m["fingerprint"]
    changed = copy.deepcopy(c)
    changed["temperature"] = .7
    with pytest.raises(ValueError, match="scientific"):
        matched_baseline(root, changed, model, m["datasets"]["test_data"]["sha256"])
    (root / "source/old.py").write_text("# changed archived code\n")
    with pytest.raises(ValueError, match="snapshot"):
        source_manifest(root)


def test_rejects_manifest_and_grading_tampering(tmp_path):
    root = tmp_path / "baseline"
    c, model, m = source_run(root)
    receipt = root / "grades/A/part.jsonl.gz.meta.json"
    core.atomic_json(receipt, {"raw_sha256": "bad", "records": 2})
    with pytest.raises(ValueError, match="receipt"):
        matched_baseline(root, c, model, m["datasets"]["test_data"]["sha256"])
    m["config"]["seed"] += 1
    core.atomic_json(root / "manifest.json", m)
    with pytest.raises(ValueError, match="fingerprint"):
        source_manifest(root)


def test_runner_rejects_old_output_without_mutating_status(tmp_path):
    from experiments.common import run
    from experiments.two_token import CONDITIONS
    root = tmp_path / "baseline"
    c, _, _ = source_run(root)
    before = (root / "status.json").read_bytes()
    with pytest.raises(ValueError, match="separate"):
        run(c, root, CONDITIONS, "experiments.two_token", baseline_run=root)
    with pytest.raises(ValueError, match="frozen"):
        run(c, root, CONDITIONS, "experiments.two_token")
    assert (root / "status.json").read_bytes() == before
