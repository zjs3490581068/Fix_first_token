"""Freeze local artifacts and validate data/protocol before any expensive work."""
import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .core import (BOXED, B_TOKEN, C_TOKEN, ROOT, SCHEMA, STOP_IDS, STOPS,
                   atomic_json, digest, file_hash, problems, prompt_ids, read_json)


def gpu_status():
    p = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits"], check=True, text=True, capture_output=True)
    out = {}
    for line in p.stdout.splitlines():
        i, uid, name, total, free, driver = [x.strip() for x in line.split(",")]
        out[int(i)] = {"uuid": uid, "name": name, "total_mib": int(total), "free_mib": int(free), "driver": driver}
    return out


def check_gpu(c, physical):
    if physical not in c["gpus"]:
        raise ValueError("GPU is not enabled in the configuration")
    devices = gpu_status()
    if physical not in devices:
        raise ValueError(f"GPU {physical} was not reported by nvidia-smi")
    info = devices[physical]
    if info["free_mib"] < c["minimum_free_gpu_gib"] * 1024:
        raise RuntimeError(f"GPU {physical} insufficient free memory: {info['free_mib']} MiB")
    return info


def source_hashes():
    paths = [*ROOT.glob("first_token/*.py"), *ROOT.glob("experiments/*.py"),
             *ROOT.glob("scripts/*.sh"), *ROOT.glob("configs/*")]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(paths) if p.is_file()}


def dependency_versions():
    return {x.metadata["Name"]: x.version for x in importlib.metadata.distributions() if x.metadata["Name"]}


def model_identity(spec):
    path = Path(spec["path"])
    names = set(read_json(path / "model.safetensors.index.json")["weight_map"].values())
    names.update(["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "model.safetensors.index.json"])
    files = {}
    revisions = set()
    for name in sorted(names):
        f = path / name
        if not f.is_file():
            raise FileNotFoundError(f)
        files[name] = {"bytes": f.stat().st_size, "sha256": file_hash(f)}
        metadata = path / ".cache/huggingface/download" / (name + ".metadata")
        if metadata.exists():
            revisions.add(metadata.read_text().splitlines()[0])
    if len(revisions) > 1:
        raise ValueError(f"Mixed model download revisions: {spec['id']}")
    return {**spec, "revision": next(iter(revisions), "unknown-local"), "files": files}


def preflight(c, run_dir):
    from transformers import AutoTokenizer
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(run_dir).free < c["minimum_free_disk_gib"] * 1024**3:
        raise RuntimeError("Insufficient disk headroom")
    test = problems(c["test_data"], labels=True)
    probe = problems(c["probe_data"], limit=c["beam_problems"])
    if len(test) != 500 or len(probe) != c["beam_problems"]:
        raise ValueError("Dataset size does not match protocol")
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    held = {norm(p["problem"]) for p in test}
    train = {norm(p["problem"]) for p in probe}
    if len(held) != 500 or len(train) != len(probe) or held & train:
        raise ValueError("Duplicate or overlapping search/test problems")
    toks = {k: AutoTokenizer.from_pretrained(v["path"], local_files_only=True) for k, v in c["models"].items()}
    base, rl = toks["base"], toks["rl"]
    if base.get_vocab() != rl.get_vocab():
        raise ValueError("Incompatible model token vocabularies")
    texts = [p["problem"] for p in test + probe]
    for text in texts:
        if prompt_ids(base, text) != prompt_ids(rl, text):
            raise ValueError("Tokenizer prompt encoding mismatch")
    for t in toks.values():
        if t.encode("解答", add_special_tokens=False) != [B_TOKEN] or t.encode(" To determine", add_special_tokens=False) != [C_TOKEN, 8253]:
            raise ValueError("B/C tokenizer conditions changed")
        if [t.convert_tokens_to_ids(s) for s in ["<|endoftext|>", "<|im_end|>"]] != STOP_IDS:
            raise ValueError("Stop ID mismatch")
    longest = max(len(prompt_ids(base, s)) for s in texts)
    if longest + c["output_budget"] > c["engine"]["max_model_len"]:
        raise ValueError("Prompt plus full output budget exceeds engine context")
    print("hashing local model artifacts", flush=True)
    identities = {k: model_identity(v) for k, v in c["models"].items()}
    versions = dependency_versions()
    payload = {"schema": SCHEMA, "config": c, "models": identities, "code": source_hashes(), "dependencies": versions,
               "datasets": {k: {"path": c[k], "sha256": file_hash(c[k])} for k in ["test_data", "probe_data"]},
               "prompt": BOXED, "stops": STOPS, "stop_ids": STOP_IDS,
               "runtime": {"VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_USE_DEEP_GEMM": "0"},
               "canonical_tokenizer": c["models"]["base"]["path"],
               "tokenizer_note": "Equal vocabulary and prompt encodings; use Base tokenizer for both models to unify special-token decoding.",
               "search_problem_ids": [p["problem_id"] for p in probe],
               "screen_problem_ids": [p["problem_id"] for p in probe[:c["screen_problems"]]],
               "data_audit": {"test_count": len(test), "search_overlap": 0, "max_prompt_tokens": longest,
                              "levels": dict(Counter(str(p["level"]) for p in test)), "subjects": dict(Counter(p["subject"] for p in test))},
               "rl_training_overlap": {"status": "unverified", "dataset_reported": "DAPO-Math-17k-Processed",
                                       "reason": "Checkpoint training data is not pinned to an exact artifact. Training/test overlap has not been verified."}}
    fingerprint = digest(payload)
    target = run_dir / "manifest.json"
    if target.exists():
        original = read_json(target)
        if original["fingerprint"] != fingerprint:
            raise ValueError("Frozen experiment changed; use a new run directory")
        atomic_json(run_dir / "config.json", c)
    else:
        for relative in payload["code"]:
            target_source = run_dir / "source" / relative
            target_source.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target_source)
        shutil.copy2(ROOT / "LICENSE.cues", run_dir / "source/LICENSE.cues")
        shutil.copy2(ROOT / "README.md", run_dir / "source/README.md")
        atomic_json(target, {**payload, "fingerprint": fingerprint, "created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.executable})
        atomic_json(run_dir / "config.json", c)
    atomic_json(run_dir / "preflight_status.json", {"ok": True, "gpus": {i: check_gpu(c, i) for i in c["gpus"]}, "free_disk_bytes": shutil.disk_usage(run_dir).free})
    print(json.dumps({"preflight": "passed", "fingerprint": fingerprint, "run_dir": str(run_dir)}), flush=True)
    return read_json(target)
