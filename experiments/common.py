"""Shared runner for predefined, single-sample Base-model interventions."""
import argparse
import json
import random
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from first_token.artifacts import matched_baseline, verify_fingerprint
from first_token.core import (BOXED, ROOT, STOP_IDS, STOPS, atomic_json, checked_records,
                              chunks, config, digest, expected_keys, file_hash,
                              problems, read_json, write_shard)
from first_token.grading import _grade_job
from first_token.metrics import bootstrap
from first_token.pipeline import run_lock, timestamp
from first_token.preflight import check_gpu, dependency_versions, model_identity, source_hashes
from first_token.sampling import Engine
from first_token.runtime import worker_environment


def validate_conditions(tok, conditions):
    for name, text, ids in conditions:
        if (tok.encode(text, add_special_tokens=False) != ids
                or tok.decode(ids, clean_up_tokenization_spaces=False) != text
                or any(t in set(tok.all_special_ids) | set(STOP_IDS) for t in ids)):
            raise ValueError(f"Cue text/token IDs changed: {name}")


def validate_samples(rows, fingerprint, expected, prefix, group, c):
    keys = [(r["problem_id"], r["sample_index"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Missing, extra, or duplicate sample keys")
    for r in rows:
        if (r["experiment_fingerprint"] != fingerprint or r["group"] != group
                or r["forced_token_ids"] != prefix or r["output_token_ids"][:len(prefix)] != prefix
                or r["first_token_id"] != prefix[0] or r["sample_index"] != 0
                or r["output_length"] != len(r["output_token_ids"])
                or r["output_length"] > c["output_budget"] or r["output_budget"] != c["output_budget"]
                or r["seed"] != c["seed"]):
            raise ValueError("Condition, seed, prefix or output budget mismatch")


def worker(run_dir, index, conditions):
    m = read_json(run_dir / "manifest.json")
    verify_fingerprint(m)
    if source_hashes() != m["code"]:
        raise ValueError("Code changed after the run was frozen")
    c = m["config"]
    all_rows = problems(c["test_data"])
    if file_hash(c["test_data"]) != m["datasets"]["test_data"]["sha256"]:
        raise ValueError("Dataset changed after preflight")
    engine = None
    try:
        for ci, (name, cue, prefix) in enumerate(conditions):
            if len(conditions) == 1:
                rows = all_rows[index::len(c["gpus"])]
            elif ci % len(c["gpus"]) == index:
                rows = all_rows
            else:
                continue
            for bi, block in enumerate(chunks(rows, c["problems_per_batch"])):
                path = run_dir / "samples" / name / f"w{index}_p{bi:04d}.jsonl.gz"
                expected = expected_keys(block, 0, 1)
                if path.exists():
                    validate_samples(checked_records(path, m["fingerprint"], expected),
                                     m["fingerprint"], expected, prefix, name, c)
                    continue
                if shutil.disk_usage(run_dir).free < c["minimum_free_disk_gib"] * 1024**3:
                    raise RuntimeError("Insufficient disk headroom")
                if engine is None:
                    engine = Engine(c, "base", c["gpus"][index])
                    validate_conditions(engine.tok, conditions)
                start = time.monotonic()
                if len(prefix) == 1:
                    records = engine.generate(block, prefix[0], 0, 1)
                else:
                    records = engine.generate(block, None, 0, 1, prefix_ids=prefix)
                for r in records:
                    r.update(group=name, cue=cue, forced_token_ids=prefix, forced_token_count=len(prefix),
                             experiment_fingerprint=m["fingerprint"], model_id=m["model"]["id"],
                             model_revision=m["model"]["revision"], tokenizer_revision=m["model"]["revision"],
                             dataset_id="HuggingFaceH4/MATH-500")
                validate_samples(records, m["fingerprint"], expected, prefix, name, c)
                write_shard(path, records)
                progress = {"group": name, "gpu": c["gpus"][index], "path": str(path),
                            "samples": len(records), "seconds": time.monotonic() - start,
                            "peak_gpu_mib": engine.monitor.peak_mib, "updated_utc": timestamp()}
                atomic_json(run_dir / "progress" / f"worker{index}.json", progress)
                print(json.dumps(progress), flush=True)
    finally:
        if engine is not None:
            engine.close()


def grade_and_report(run_dir, conditions):
    m = read_json(run_dir / "manifest.json")
    c = m["config"]
    verify_fingerprint(m)
    if file_hash(c["test_data"]) != m["datasets"]["test_data"]["sha256"] or source_hashes() != m["code"]:
        raise ValueError("Code or dataset changed after the run was frozen")
    refs = {p["problem_id"]: p["answer"] for p in problems(c["test_data"], labels=True)}
    expected = {(p, 0) for p in refs}
    baseline = {}
    if m["baseline"] is not None:
        baseline, identity = matched_baseline(m["baseline"]["path"], c, m["model"],
                                             m["datasets"]["test_data"]["sha256"])
        if identity != m["baseline"]:
            raise ValueError("Baseline identity changed")
    result = {}
    with ProcessPoolExecutor(max_workers=c["grading_workers"]) as pool:
        for name, cue, prefix in conditions:
            raw_files = sorted((run_dir / "samples" / name).glob("*.jsonl.gz"))
            records = [r for p in raw_files for r in checked_records(p, m["fingerprint"])]
            validate_samples(records, m["fingerprint"], expected, prefix, name, c)
            hashes = {p.name: file_hash(p) for p in raw_files}
            target = run_dir / "grades" / f"{name}.jsonl.gz"
            receipt = target.with_name(target.name + ".meta.json")
            if target.exists():
                grades = checked_records(target, m["fingerprint"], expected)
                if read_json(receipt) != hashes:
                    raise ValueError("Raw/grade receipt mismatch")
            else:
                grades = list(pool.map(_grade_job, ((r, refs[r["problem_id"]]) for r in records)))
                atomic_json(receipt, hashes)
                write_shard(target, grades)
            if any(g["group"] != name for g in grades):
                raise ValueError("Grading group mismatch")
            correct = sum(g["correct"] for g in grades)
            comparisons = {}
            for group, old in baseline.items():
                diff = [int(g["correct"]) - old[g["problem_id"]] for g in grades]
                comparisons[group] = {"source_sample_index": 0, "baseline_pass@1": sum(old.values()) / len(refs),
                    "difference": sum(diff) / len(refs), "paired_problem_ci95": bootstrap(diff, random.Random(0))[0]}
            result[name] = {"cue": cue, "forced_token_ids": prefix, "samples": len(records),
                "correct": correct, "pass@1": correct / len(refs),
                "ci95": bootstrap([int(g["correct"]) for g in grades], random.Random(0))[0],
                "mean_output_tokens": sum(r["output_length"] for r in records) / len(refs),
                "cap_rate": sum(r["hit_token_cap"] for r in records) / len(refs),
                "grading_statuses": dict(Counter(g["grading_status"] for g in grades)),
                "matched_single_sample_comparison": comparisons}
    summary = {"complete": True, "independent_samples": len(refs) * len(conditions),
               "samples_per_problem": 1, "groups": result,
               "note": "One sample per problem. Comparisons use baseline rollout zero. Only pass@1 is estimated."}
    atomic_json(run_dir / "summary.json", summary)
    lines = ["# Single-sample MATH-500 results", "", summary["note"], "",
             "| Cue | Token IDs | Correct | pass@1 | 95% CI |", "|---|---|---:|---:|---|"]
    for name, row in result.items():
        lines.append(f"| `{row['cue']}` | {row['forced_token_ids']} | {row['correct']} / {len(refs)} | {row['pass@1']:.4f} | {row['ci95']} |")
    (run_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return summary


def run(c, run_dir, conditions, module, baseline_run=None):
    run_dir = Path(run_dir).resolve()
    if baseline_run is not None and run_dir.is_relative_to(Path(baseline_run).resolve()):
        raise ValueError("Use an output directory separate from the read-only baseline")
    if (run_dir / "manifest.json").exists():
        saved = read_json(run_dir / "manifest.json")
        verify_fingerprint(saved)
        if saved.get("module") != module or saved["config"] != c or saved["code"] != source_hashes():
            raise ValueError("Configuration/code differs from the frozen run; use a new run directory")
    with run_lock(run_dir):
        procs, logs = [], []
        try:
            atomic_json(run_dir / "status.json", {"state": "running", "stage": "preflight"})
            if c["rollouts"] != 1 or c["rollouts_per_batch"] != 1:
                raise ValueError("Supplementary experiments require exactly one sample per problem")
            if len(problems(c["test_data"])) != 500:
                raise ValueError("Expected MATH-500 with 500 unique problems")
            model = model_identity(c["models"]["base"])
            data_sha = file_hash(c["test_data"])
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(model["path"], local_files_only=True)
            validate_conditions(tok, conditions)
            from first_token.core import prompt_ids
            longest = max(len(prompt_ids(tok, p["problem"])) for p in problems(c["test_data"]))
            if longest + c["output_budget"] > c["engine"]["max_model_len"]:
                raise ValueError("Insufficient context for the full output budget")
            baseline = None
            if baseline_run is not None:
                _, baseline = matched_baseline(baseline_run, c, model, data_sha)
            payload = {"config": c, "model": model, "datasets": {"test_data": {"path": c["test_data"], "sha256": data_sha}},
                       "code": source_hashes(), "dependencies": dependency_versions(),
                       "prompt": BOXED, "stops": STOPS, "stop_ids": STOP_IDS,
                       "conditions": conditions, "baseline": baseline, "module": module}
            # JSON normalization also makes tuple/list representation stable across resume.
            payload = json.loads(json.dumps(payload))
            payload["fingerprint"] = digest(payload)
            manifest_path = run_dir / "manifest.json"
            if manifest_path.exists() and read_json(manifest_path) != payload:
                raise ValueError("Resume fingerprint mismatch")
            if shutil.disk_usage(run_dir).free < c["minimum_free_disk_gib"] * 1024**3:
                raise RuntimeError("Insufficient disk headroom")
            for gpu in c["gpus"]:
                check_gpu(c, gpu)
            atomic_json(manifest_path, payload)
            atomic_json(run_dir / "config.json", c)
            for relative in payload["code"]:
                dest = run_dir / "source" / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, dest)
            atomic_json(run_dir / "status.json", {"state": "running", "stage": "sampling"})
            (run_dir / "logs").mkdir(exist_ok=True)
            active = len(c["gpus"]) if len(conditions) == 1 else min(len(conditions), len(c["gpus"]))
            for index, gpu in enumerate(c["gpus"][:active]):
                env = worker_environment(run_dir, gpu)
                log = (run_dir / "logs" / f"gpu{gpu}.log").open("a")
                logs.append(log)
                cmd = [sys.executable, "-u", "-m", module, "--run-dir", str(run_dir), "--worker", str(index)]
                procs.append(subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT))
            while True:
                states = [p.poll() for p in procs]
                atomic_json(run_dir / "workers.json", {"workers": [{"gpu": c["gpus"][i], "pid": p.pid, "exit_code": s}
                            for i, (p, s) in enumerate(zip(procs, states))]})
                if any(s is not None and s != 0 for s in states):
                    raise RuntimeError(f"Worker failure: {states}; inspect worker logs")
                if all(s is not None for s in states):
                    break
                time.sleep(2)
            atomic_json(run_dir / "status.json", {"state": "running", "stage": "grading"})
            grade_and_report(run_dir, conditions)
            atomic_json(run_dir / "status.json", {"state": "complete", "exit_code": 0, "finished_utc": timestamp()})
        except BaseException as exc:
            atomic_json(run_dir / "status.json", {"state": "failed", "exit_code": 1, "error": str(exc),
                        "traceback": traceback.format_exc(), "updated_utc": timestamp()})
            raise
        finally:
            for p in procs:
                if p.poll() is None:
                    p.terminate()
            for p in procs:
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait(timeout=10)
            for log in logs:
                log.close()


def main(module, config_name, conditions):
    parser = argparse.ArgumentParser(description=module)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / config_name)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path)
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is None:
        run(config(args.config), args.run_dir.resolve(), conditions, module, args.baseline_run)
    else:
        worker(args.run_dir.resolve(), args.worker, conditions)
