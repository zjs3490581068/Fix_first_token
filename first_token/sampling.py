"""Token-ID prefill and resumable vLLM sampling, adapted from cues.evaluate."""
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

from .core import (BOXED, STOP_IDS, STOPS, atomic_json, checked_records, check_manifest,
                   chunks, condition, digest, expected_keys, problems, prompt_ids,
                   read_json, sample_seed, token_budget, write_shard)
from .preflight import check_gpu


class MemoryMonitor:
    """NVML via nvidia-smi, counting only this worker and its descendants."""
    def __init__(self):
        self.peak_mib = 0
        self.error = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while not self.stop.is_set():
            try:
                pairs = subprocess.check_output(["ps", "-eo", "pid=,ppid="], text=True)
                pairs = [tuple(map(int, s.split())) for s in pairs.splitlines()]
                own = {os.getpid()}
                for _ in range(10):
                    new = own | {p for p, parent in pairs if parent in own}
                    if new == own:
                        break
                    own = new
                mem = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], text=True)
                total = 0
                for s in mem.splitlines():
                    p, size = s.split(",")
                    if int(p.strip()) in own and size.strip().isdigit():
                        total += int(size.strip())
                self.peak_mib = max(self.peak_mib, total)
            except Exception as exc:
                self.error = str(exc)
            self.stop.wait(2)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=5)


class Engine:
    def __init__(self, c, model, gpu):
        check_gpu(c, gpu)
        if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
            raise RuntimeError("Worker must have exactly its assigned physical GPU visible")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        # Keep sampling kernels consistent across experiment conditions.
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        os.environ["VLLM_USE_DEEP_GEMM"] = "0"
        from vllm import LLM
        self.monitor = MemoryMonitor()
        self.c = c
        self.model = model
        self.llm = LLM(model=c["models"][model]["path"], tokenizer=c["models"]["base"]["path"],
                       dtype="bfloat16", tensor_parallel_size=1, enforce_eager=True,
                       trust_remote_code=False, seed=c["seed"], generation_config="vllm",
                       **c["engine"])
        self.tok = self.llm.get_tokenizer()

    def generate(self, rows, forced, offset, n, budget=None, ignore_eos=False, prefix_ids=None):
        from vllm import SamplingParams
        c = self.c
        budget = c["output_budget"] if budget is None else budget
        if prefix_ids is not None and forced is not None:
            raise ValueError("Specify either a single token or a token sequence")
        prefix = ([] if forced is None else [forced]) if prefix_ids is None else list(prefix_ids)
        if any(type(t) is not int or t < 0 or t >= len(self.tok)
               or t in set(self.tok.all_special_ids) | set(STOP_IDS) for t in prefix):
            raise ValueError("Prefix contains an invalid or special token")
        params = SamplingParams(n=n, temperature=c["temperature"], top_p=c["top_p"], top_k=-1,
                                repetition_penalty=1.0, max_tokens=token_budget(budget, prefix),
                                stop=None if ignore_eos else STOPS, stop_token_ids=None if ignore_eos else STOP_IDS,
                                ignore_eos=ignore_eos, skip_special_tokens=False,
                                seed=sample_seed(c["seed"], offset))
        inputs = [{"prompt_token_ids": prompt_ids(self.tok, r["problem"]) + prefix} for r in rows]
        if any(len(p["prompt_token_ids"]) + params.max_tokens > c["engine"]["max_model_len"] for p in inputs):
            raise ValueError("Generation would silently lose output budget")
        outputs = self.llm.generate(inputs, params, use_tqdm=False)
        result = []
        for problem, out in zip(rows, outputs, strict=True):
            if list(out.prompt_token_ids) != inputs[len(result)]["prompt_token_ids"]:
                raise AssertionError("Engine changed prompt token IDs")
            if sorted(o.index for o in out.outputs) != list(range(n)):
                raise AssertionError("Missing samples")
            samples = []
            for cand in sorted(out.outputs, key=lambda x: x.index):
                ids = prefix + list(cand.token_ids)
                if len(ids) > budget or (forced is not None and ids[0] != forced):
                    raise AssertionError("Single-token intervention/budget violated")
                text_ids = ids[:-1] if ids and ids[-1] in STOP_IDS and not ignore_eos else ids
                text = self.tok.decode(text_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                if not ignore_eos:
                    positions = [text.find(s) for s in STOPS if s in text]
                    if positions:
                        text = text[:min(positions)]
                samples.append({"problem_id": problem["problem_id"], "sample_index": offset + cand.index,
                                "seed": sample_seed(c["seed"], offset + cand.index),
                                "forced_token_id": forced, "first_token_id": ids[0] if ids else None,
                                "second_token_id": ids[1] if len(ids) > 1 else None,
                                "output_text": text, "output_token_ids": ids, "output_length": len(ids),
                                "finish_reason": cand.finish_reason, "stop_reason": cand.stop_reason,
                                "hit_token_cap": cand.finish_reason == "length", "output_budget": budget,
                                "prompt_hash": digest({"text": BOXED.format(q=problem["problem"]),
                                                       "token_ids": prompt_ids(self.tok, problem["problem"])}),
                                "input_ids_hash": digest(prompt_ids(self.tok, problem["problem"]) + prefix),
                                "forced_token_ids": prefix, "forced_token_count": len(prefix),
                                "continuation_budget": budget - len(prefix)})
            result.append(samples)
        if self.monitor.peak_mib > c.get("maximum_worker_gpu_gib", 41) * 1024:
            raise RuntimeError(f"Worker exceeded configured memory budget: {self.monitor.peak_mib} MiB")
        return [r for rr in result for r in rr]

    def close(self):
        self.monitor.close()
        self.llm.llm_engine.engine_core.shutdown(timeout=120)


def sample_condition(engine, c, run_dir, manifest, spec, rows, rollouts, directory, worker, dataset_id,
                     artifact_worker=None, block_index_base=0, rollout_start=0, rollout_end=None):
    directory = Path(directory)
    artifact_worker = worker if artifact_worker is None else artifact_worker
    model = manifest["models"][spec["model"]]
    end = rollouts if rollout_end is None else rollout_end
    if not 0 <= rollout_start < end <= rollouts:
        raise ValueError("Invalid rollout interval")
    for block_index, block in enumerate(chunks(rows, c["problems_per_batch"]), start=block_index_base):
        for offset in range(rollout_start, end, c["rollouts_per_batch"]):
            n = min(c["rollouts_per_batch"], end - offset)
            path = directory / f"w{artifact_worker}_p{block_index:04d}_r{offset:03d}.jsonl.gz"
            keys = expected_keys(block, offset, n)
            if path.exists():
                existing = checked_records(path, manifest["fingerprint"], keys)
                if any(r["forced_token_id"] != spec["forced_token_id"] or r["group"] != spec["group"] for r in existing):
                    raise ValueError("Saved condition differs from requested condition")
                continue
            if shutil.disk_usage(Path(run_dir)).free < c["minimum_free_disk_gib"] * 1024**3:
                raise RuntimeError("Low disk: stopped before next shard")
            start = time.monotonic()
            records = engine.generate(block, spec["forced_token_id"], offset, n)
            for r in records:
                r.update(group=spec["group"], model_id=model["id"], model_revision=model["revision"],
                         tokenizer_revision=manifest["models"]["base"]["revision"],
                         dataset_id=dataset_id, experiment_fingerprint=manifest["fingerprint"])
            write_shard(path, records)
            elapsed = time.monotonic() - start
            status = {"path": str(path), "records": len(records), "seconds": elapsed,
                      "tokens": sum(r["output_length"] for r in records), "peak_gpu_mib": engine.monitor.peak_mib,
                      "compressed_bytes": path.stat().st_size, "updated_unix": time.time()}
            atomic_json(Path(run_dir) / "progress" / f"worker{worker}.json", status)
            print(json.dumps(status), flush=True)


def sample_worker(c, run_dir, gpu, worker, groups):
    m = check_manifest(c, run_dir)
    from .aliases import group_aliases, sampling_groups
    groups = sampling_groups(groups, group_aliases(c, run_dir, m))
    winner_path = Path(run_dir) / "search/winner.json"
    winner = read_json(winner_path)["forced_token_id"] if winner_path.exists() else None
    from .scheduling import load_sample_tasks
    tasks = load_sample_tasks(c, run_dir, groups, worker, m)
    by_id = {r["problem_id"]: r for r in problems(c["test_data"])}
    engine = None
    try:
        for task in tasks:
            group = task["group"]
            spec = condition(group, winner)
            if engine is None or engine.model != spec["model"]:
                if engine is not None:
                    engine.close()
                    engine = None
                engine = Engine(c, spec["model"], gpu)
            sample_condition(engine, c, run_dir, m, spec, [by_id[p] for p in task["problem_ids"]], c["rollouts"],
                             Path(run_dir) / "samples" / group, worker, "HuggingFaceH4/MATH-500",
                             artifact_worker=task["artifact_worker"], block_index_base=task["block_index"],
                             rollout_start=task["offset"], rollout_end=task["offset"] + task["n"])
    finally:
        if engine is not None:
            engine.close()


def smoke_worker(c, run_dir, gpu, worker, groups, long_check=False):
    m = check_manifest(c, run_dir)
    rows = problems(c["probe_data"], limit=2)
    winner_path = Path(run_dir) / "search/winner.json"
    winner = read_json(winner_path)["forced_token_id"] if winner_path.exists() else None
    report = []
    engine = None
    try:
        for group in groups:
            spec = condition(group, winner)
            if engine is None or engine.model != spec["model"]:
                if engine:
                    engine.close()
                    engine = None
                engine = Engine(c, spec["model"], gpu)
            start = time.monotonic()
            records = engine.generate(rows, spec["forced_token_id"], 0, 2)
            for r in records:
                r.update(group=group, experiment_fingerprint=m["fingerprint"])
            target = Path(run_dir) / "smoke" / f"{group}_w{worker}.jsonl.gz"
            if not target.exists():
                write_shard(target, records)
            report.append({"group": group, "samples": len(records), "seconds": time.monotonic() - start,
                           "tokens": sum(r["output_length"] for r in records),
                           "compressed_bytes": target.stat().st_size, "peak_gpu_mib": engine.monitor.peak_mib})
        if long_check:
            # Diagnostic only: disable stopping and demand >2048 generated tokens.
            records = engine.generate(rows[:1], None, 10000, 1, budget=2050, ignore_eos=True)
            if records[0]["output_length"] != 2050 or records[0]["finish_reason"] != "length":
                raise AssertionError("Hidden 2048-token cap detected")
            report.append({"long_budget_check": "passed", "generated_tokens": 2050})
        atomic_json(Path(run_dir) / "smoke" / f"report_w{worker}_{''.join(groups)}.json", report)
    finally:
        if engine:
            engine.close()
