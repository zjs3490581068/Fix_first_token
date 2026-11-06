"""Assign immutable sample batches to a fixed, configured set of workers."""
from pathlib import Path

from .core import (atomic_json, checked_records, check_manifest, chunks, condition,
                   digest, expected_keys, problems, read_json)


def smoke_assignments(count):
    groups = [[] for _ in range(count)]
    groups[0] = ["A", "B", "C"]
    groups[min(1, count - 1)].append("E")
    return groups


def task_plan_path(run_dir, groups):
    return Path(run_dir) / "task_plans" / ("sample_" + "".join(groups) + ".json")


def prepare_sample_tasks(c, run_dir, groups):
    from .aliases import group_aliases, sampling_groups
    m = check_manifest(c, run_dir)
    groups = sampling_groups(groups, group_aliases(c, run_dir, m))
    rows = problems(c["test_data"])
    winner_path = Path(run_dir) / "search/winner.json"
    winner = read_json(winner_path)["forced_token_id"] if winner_path.exists() else None
    jobs = [[] for _ in c["gpus"]]
    done = pending = 0
    for group in groups:
        spec = condition(group, winner)
        expected_files = set()
        for shard in range(len(jobs)):
            for index, block in enumerate(chunks(rows[shard::len(jobs)], c["problems_per_batch"])):
                for offset in range(0, c["rollouts"], c["rollouts_per_batch"]):
                    n = min(c["rollouts_per_batch"], c["rollouts"] - offset)
                    name = f"w{shard}_p{index:04d}_r{offset:03d}.jsonl.gz"
                    expected_files.add(name)
                    path = Path(run_dir) / "samples" / group / name
                    if path.exists():
                        saved = checked_records(path, m["fingerprint"], expected_keys(block, offset, n))
                        if any(r["group"] != group or r["forced_token_id"] != spec["forced_token_id"] for r in saved):
                            raise ValueError("Saved condition differs from the requested task")
                        done += len(saved)
                        continue
                    jobs[pending % len(jobs)].append({"group": group, "artifact_worker": shard,
                        "block_index": index, "offset": offset, "n": n,
                        "problem_ids": [r["problem_id"] for r in block]})
                    pending += 1
        actual = {p.name for p in (Path(run_dir) / "samples" / group).glob("*.jsonl.gz")}
        if not actual <= expected_files:
            raise ValueError("Unexpected files outside the sample layout")
    plan = {"fingerprint": m["fingerprint"], "gpus": c["gpus"], "groups": groups,
            "completed_samples": done, "pending_batches": pending, "workers": jobs}
    plan["plan_digest"] = digest(plan)
    atomic_json(task_plan_path(run_dir, groups), plan)
    print(f"Sampling tasks: {done} completed samples, {pending} missing batches", flush=True)
    return plan


def load_sample_tasks(c, run_dir, groups, worker, manifest):
    plan = read_json(task_plan_path(run_dir, groups))
    if (digest({k: v for k, v in plan.items() if k != "plan_digest"}) != plan["plan_digest"]
            or plan["fingerprint"] != manifest["fingerprint"]
            or plan["gpus"] != c["gpus"] or plan["groups"] != groups):
        raise ValueError("Stale or mismatched sample task plan")
    return plan["workers"][worker]
