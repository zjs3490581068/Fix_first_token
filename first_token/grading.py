"""Cues boxed grading with explicit diagnostic statuses; bounded per-answer work."""
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .core import atomic_json, checked_records, check_manifest, file_hash, problems, read_json, write_shard
from .grading_base import _parse, extract
from math_verify import verify


def grade_answer(text, reference):
    a = extract(text)
    result = {"extracted_answer": a, "correct": False, "grading_status": "missing_answer"}
    if a is None:
        return result
    gold, pred = _parse(reference), _parse(a)
    if not gold or not pred:
        result["grading_status"] = "reference_parse_failed" if not gold else "answer_parse_failed"
        return result
    try:
        result["correct"] = bool(verify(gold, pred))
        result["grading_status"] = "correct" if result["correct"] else "incorrect"
    except Exception:
        result["grading_status"] = "verification_failed"
    return result


def answer_key(text):
    # Byte-for-byte normalization policy from cues.agree.answer_key.
    a = extract(text[-2000:])
    if a is None:
        return None
    v = _parse(a[:80])
    if not v:
        return None
    num, txt = (list(v) + [None, None])[:2]
    try:
        val = round(float(num), 6)
        return val if math.isfinite(val) else str(val)
    except Exception:
        return str(txt if txt is not None else num)


def entropy(keys):
    counts = Counter(keys)
    n = len(keys)
    return -sum((v / n) * math.log2(v / n) for v in counts.values())


def select_winner(scores, guard):
    if "none" not in scores:
        raise ValueError("Complete no-cue baseline is required")
    limit = scores["none"]["none_share"] + guard
    eligible = []
    for key, val in scores.items():
        val["eligible"] = key != "none" and val["none_share"] <= limit + 1e-12
        if val["eligible"]:
            eligible.append(val)
    return min(eligible, key=lambda x: (x["entropy"], x["forced_token_id"])) if eligible else None


def _grade_job(job):
    row, ref = job
    # Keep grading lean: raw texts/IDs remain in immutable generation shards.
    out = {k: row[k] for k in ["group", "problem_id", "sample_index", "experiment_fingerprint"]}
    out.update(grade_answer(row["output_text"], ref))
    return out


def grade_run(c, run_dir, groups=None):
    run_dir = Path(run_dir)
    m = check_manifest(c, run_dir)
    from .aliases import group_aliases, sampling_groups
    groups = sampling_groups(groups or c["groups"], group_aliases(c, run_dir, m))
    refs = {p["problem_id"]: p["answer"] for p in problems(c["test_data"], labels=True)}
    with ProcessPoolExecutor(max_workers=c["grading_workers"]) as pool:
        for group in groups:
            for raw in sorted((run_dir / "samples" / group).glob("*.jsonl.gz")):
                target = run_dir / "grades" / group / raw.name
                receipt = target.with_name(target.name + ".meta.json")
                records = checked_records(raw, m["fingerprint"])
                keys = {(r["problem_id"], r["sample_index"]) for r in records}
                raw_sha = file_hash(raw)
                if target.exists():
                    checked_records(target, m["fingerprint"], keys)
                    if not receipt.exists() or read_json(receipt)["raw_sha256"] != raw_sha:
                        raise ValueError(f"Grading receipt mismatch: {target}")
                    continue
                result = list(pool.map(_grade_job, ((r, refs[r["problem_id"]]) for r in records)))
                atomic_json(receipt, {"raw_sha256": raw_sha, "records": len(result)})
                write_shard(target, result)
                print(f"graded {group}/{raw.name}: {len(result)}", flush=True)
