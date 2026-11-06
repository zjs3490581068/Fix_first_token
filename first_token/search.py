"""Depth-1/2 beam proposals and label-free single-ID screening (cues.beam/agree)."""
import math
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .core import (atomic_json, checked_records, check_manifest, expected_keys,
                   problems, prompt_ids, read_json)
from .grading import answer_key, entropy, select_winner
from .preflight import check_gpu
from .sampling import Engine, sample_condition


def beam_worker(c, run_dir, gpu, worker):
    check_gpu(c, gpu)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise ValueError("Incorrect worker GPU")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = check_manifest(c, run_dir)
    rows = problems(c["probe_data"], limit=c["beam_problems"])
    tasks = [(i, p) for i, p in enumerate(rows) if i % len(c["gpus"]) == worker]
    pending = []
    for i, p in tasks:
        path = Path(run_dir) / "search/beam" / f"p{i:03d}.json"
        if path.exists():
            saved = read_json(path)
            if saved["fingerprint"] != m["fingerprint"] or saved["problem_id"] != p["problem_id"]:
                raise ValueError("Beam checkpoint mismatch")
        else:
            pending.append((i, p, path))
    if not pending:
        return
    tok = AutoTokenizer.from_pretrained(c["models"]["base"]["path"], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(c["models"]["base"]["path"], local_files_only=True,
                                                dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0").eval()
    width = c["beam_width"]
    with torch.inference_mode():
        for i, p, path in pending:
            initial = prompt_ids(tok, p["problem"])
            beams = [((), 0.0)]
            levels = {}
            for depth in [1, 2]:
                scored = []
                for prefix, log_mass in beams:
                    ids = torch.tensor([initial + list(prefix)], dtype=torch.long, device="cuda:0")
                    logits = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[0, -1].float()
                    vals, ids = torch.topk(torch.log_softmax(logits, -1), width)
                    scored.extend((prefix + (t,), log_mass + v) for t, v in zip(ids.tolist(), vals.tolist()))
                beams = sorted(scored, key=lambda pair: (-pair[1], pair[0]))[:width]
                levels[str(depth)] = [{"token_ids": list(ids), "probability": math.exp(logp)} for ids, logp in beams]
            atomic_json(path, {"fingerprint": m["fingerprint"], "problem_id": p["problem_id"], "depths": levels})
            print(f"beam gpu={gpu} problem={i} completed", flush=True)


def propose(c, run_dir):
    from transformers import AutoTokenizer
    m = check_manifest(c, run_dir)
    rows = problems(c["probe_data"], limit=c["beam_problems"])
    tok = AutoTokenizer.from_pretrained(c["models"]["base"]["path"], local_files_only=True)
    banned = set(tok.all_special_ids)
    stats = {1: defaultdict(list), 2: defaultdict(list)}
    for i, p in enumerate(rows):
        b = read_json(Path(run_dir) / "search/beam" / f"p{i:03d}.json")
        if b["fingerprint"] != m["fingerprint"] or b["problem_id"] != p["problem_id"]:
            raise ValueError("Mismatched beam artifact")
        for depth in [1, 2]:
            seen = set()
            for v in b["depths"][str(depth)]:
                key = tuple(v["token_ids"])
                if len(key) != depth or key in seen:
                    raise ValueError("Invalid beam proposal")
                seen.add(key)
                if not banned.intersection(key):
                    stats[depth][key].append(v["probability"])
    selected = {}
    merged = {}
    for depth in [1, 2]:
        eligible = [{"token_ids": list(key), "mean_probability": sum(vals) / len(rows),
                     "problem_fraction": len(vals) / len(rows)} for key, vals in stats[depth].items()
                    if len(vals) >= len(rows) * c["min_problem_fraction"]]
        selected[str(depth)] = sorted(eligible, key=lambda x: (-x["mean_probability"], x["token_ids"]))[:c["nominees_per_depth"]]
        for proposal in selected[str(depth)]:
            token = proposal["token_ids"][0]
            entry = merged.setdefault(token, {"forced_token_id": token, "text": tok.decode([token]), "sources": [], "proposals": []})
            if depth not in entry["sources"]:
                entry["sources"].append(depth)
            entry["proposals"].append({"depth": depth, **proposal})
    if not merged:
        raise RuntimeError("No non-special proposals passed the presence filter")
    result = {"fingerprint": m["fingerprint"], "by_depth": selected,
              "candidates": [merged[t] for t in sorted(merged)]}
    target = Path(run_dir) / "search/nominees.json"
    if target.exists() and read_json(target) != result:
        raise ValueError("Frozen nominees changed")
    atomic_json(target, result)
    print(f"single-ID nominees: {len(merged)}", flush=True)
    return result


def screen_specs(c, run_dir):
    m = check_manifest(c, run_dir)
    nominees = read_json(Path(run_dir) / "search/nominees.json")
    if nominees["fingerprint"] != m["fingerprint"]:
        raise ValueError("Nominees from a different run")
    return [{"group": "none", "model": "base", "forced_token_id": None}] + [
        {"group": f"token_{p['forced_token_id']}", "model": "base", "forced_token_id": p["forced_token_id"]}
        for p in nominees["candidates"]]


def screen_worker(c, run_dir, gpu, worker):
    m = check_manifest(c, run_dir)
    rows = problems(c["probe_data"], limit=c["screen_problems"])
    engine = Engine(c, "base", gpu)
    try:
        assigned = [(spec, worker) for i, spec in enumerate(screen_specs(c, run_dir))
                    if i % len(c["gpus"]) == worker]
        for spec, artifact_worker in assigned:
            sample_condition(engine, c, run_dir, m, spec, rows, c["screen_rollouts"],
                             Path(run_dir) / "search/screen" / spec["group"], worker, "MATH-train-probe", artifact_worker=artifact_worker)
    finally:
        engine.close()


def decide(c, run_dir):
    m = check_manifest(c, run_dir)
    rows = problems(c["probe_data"], limit=c["screen_problems"])
    expected = expected_keys(rows, 0, c["screen_rollouts"])
    scores = {}
    with ProcessPoolExecutor(max_workers=c["grading_workers"]) as pool:
        for spec in screen_specs(c, run_dir):
            records = []
            for path in sorted((Path(run_dir) / "search/screen" / spec["group"]).glob("*.jsonl.gz")):
                records.extend(checked_records(path, m["fingerprint"]))
            keys = [(r["problem_id"], r["sample_index"]) for r in records]
            if len(set(keys)) != len(keys) or set(keys) != expected:
                raise ValueError(f"Incomplete/duplicate screening for {spec['group']}")
            if any(r["forced_token_id"] != spec["forced_token_id"] or r["group"] != spec["group"] for r in records):
                raise ValueError("Screen condition mismatch")
            answers = list(pool.map(answer_key, (r["output_text"] for r in records)))
            by_problem = defaultdict(list)
            for record, answer in zip(records, answers, strict=True):
                by_problem[record["problem_id"]].append(answer)
            entropies = {p: entropy(vals) for p, vals in by_problem.items()}
            scores[spec["group"]] = {**spec, "entropy": sum(entropies.values()) / len(rows),
                                     "none_share": answers.count(None) / len(answers), "entropies": entropies,
                                     "records": len(records)}
    winner = select_winner(scores, c["guard"])
    atomic_json(Path(run_dir) / "search/decider.json", {"fingerprint": m["fingerprint"], "scores": scores, "winner": winner})
    if winner is None:
        raise RuntimeError("No candidate passed the missing-answer guard; D is not defined")
    nominees = read_json(Path(run_dir) / "search/nominees.json")["candidates"]
    result = {**winner, "fingerprint": m["fingerprint"], "label_free": True,
              "proposal": next(p for p in nominees if p["forced_token_id"] == winner["forced_token_id"])}
    target = Path(run_dir) / "search/winner.json"
    if target.exists() and read_json(target) != result:
        raise ValueError("Search winner changed")
    atomic_json(target, result)
    print(f"frozen D token: {winner['forced_token_id']}", flush=True)
    return result
