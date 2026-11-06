"""GPU subprocess orchestration, fail-fast stage gates, and tmux-safe resume."""
import fcntl
import json
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .core import ROOT, atomic_json, check_manifest, read_json
from .preflight import preflight
from .runtime import worker_environment


def timestamp():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def run_lock(run_dir):
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    with (Path(run_dir) / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("A coordinator already owns this run")
        yield


def workers(c, run_dir, mode, groups_by_worker=None, long_check=False):
    run_dir = Path(run_dir)
    procs = []
    logs = []
    if mode == "sample":
        from .scheduling import prepare_sample_tasks
        for groups in dict.fromkeys(tuple(g) for g in groups_by_worker if g):
            prepare_sample_tasks(c, run_dir, list(groups))
    try:
        for w, gpu in enumerate(c["gpus"]):
            groups = groups_by_worker[w] if groups_by_worker is not None else []
            if groups_by_worker is not None and not groups:
                continue
            env = worker_environment(run_dir, gpu)
            log_dir = run_dir / "logs"
            log_dir.mkdir(exist_ok=True)
            log = (log_dir / f"{mode}_{''.join(groups)}_gpu{gpu}.log").open("a", buffering=1)
            logs.append(log)
            cmd = [sys.executable, "-u", "-m", "first_token", "_worker", "--config", str(run_dir / "config.json"),
                   "--run-dir", str(run_dir), "--mode", mode, "--gpu", str(gpu), "--worker", str(w)]
            if groups:
                cmd += ["--groups", *groups]
            if long_check and w == 0:
                cmd.append("--long-check")
            print(json.dumps({"launch": cmd, "log": log.name}), flush=True)
            procs.append((w, subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)))
        while True:
            states = {w: p.poll() for w, p in procs}
            atomic_json(run_dir / "workers.json", {"mode": mode, "updated_utc": timestamp(),
                                                   "workers": [{"worker": w, "gpu": c["gpus"][w], "pid": p.pid, "exit_code": states[w]} for w, p in procs]})
            failed = [w for w, code in states.items() if code is not None and code != 0]
            if failed:
                raise RuntimeError(f"{mode} workers failed: {failed}; inspect logs")
            if all(v is not None for v in states.values()):
                break
            time.sleep(2)
    finally:
        # Only children launched by this coordinator; never touch other users' jobs.
        for _, p in procs:
            if p.poll() is None:
                p.terminate()
        for _, p in procs:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=10)
        for log in logs:
            log.close()


def resource_gate(c, run_dir):
    reports = []
    for p in (Path(run_dir) / "smoke").glob("report_*.json"):
        reports.extend(v for v in read_json(p) if "samples" in v)
    if not {"A", "B", "C", "E"} <= {r["group"] for r in reports}:
        raise RuntimeError("Missing initial smoke conditions")
    if max(r["peak_gpu_mib"] for r in reports) > c.get("maximum_worker_gpu_gib", 41) * 1024:
        raise RuntimeError("Smoke exceeded memory allowance")
    per_token = max(r["compressed_bytes"] / max(1, r["tokens"]) for r in reports)
    maximum_records = 500 * c["rollouts"] * len(c["groups"]) + (2 * c["nominees_per_depth"] + 1) * c["screen_problems"] * c["screen_rollouts"]
    estimate = per_token * maximum_records * c["output_budget"] * 1.5
    free = shutil.disk_usage(run_dir).free
    status = {"smoke_reports": reports, "estimated_worst_budget_raw_bytes_with_margin": estimate,
              "free_disk_bytes": free, "minimum_reserve_bytes": c["minimum_free_disk_gib"] * 1024**3,
              "note": "Storage projection uses observed compressed bytes/token, full token budget, all candidate slots, and 1.5x margin. Throughput is a small-sample observation, not a runtime guarantee."}
    atomic_json(Path(run_dir) / "resource_gate.json", status)
    if free < estimate + status["minimum_reserve_bytes"]:
        raise RuntimeError("Estimated full run would exhaust disk headroom")


def search_stages(c, run_dir, stage):
    from .search import propose, decide
    stage("beam", lambda: workers(c, run_dir, "beam"))
    stage("nominees", lambda: propose(c, run_dir))
    stage("screen", lambda: workers(c, run_dir, "screen"))
    stage("decide", lambda: decide(c, run_dir))


def run(c, run_dir, search_only=False):
    run_dir = Path(run_dir)
    if (run_dir / "manifest.json").exists():
        check_manifest(c, run_dir)
    with run_lock(run_dir):
        try:
            m = preflight(c, run_dir)

            def stage(name, action):
                receipt = run_dir / "stages" / f"{name}.json"
                if receipt.exists():
                    if read_json(receipt)["fingerprint"] != m["fingerprint"]:
                        raise ValueError("Stage fingerprint mismatch")
                    print(f"resume: completed stage {name}", flush=True)
                    return
                atomic_json(run_dir / "status.json", {"state": "running", "stage": name, "updated_utc": timestamp()})
                print(f"START {name}", flush=True)
                action()
                atomic_json(receipt, {"fingerprint": m["fingerprint"], "completed_utc": timestamp()})
                print(f"DONE {name}", flush=True)

            # Base smoke on 0, GRPO smoke on 4. With one GPU, run sequentially.
            from .scheduling import smoke_assignments
            smoke_groups = smoke_assignments(len(c["gpus"]))
            stage("initial_smoke", lambda: workers(c, run_dir, "smoke", smoke_groups, long_check=True))
            stage("resource_gate", lambda: resource_gate(c, run_dir))
            search_stages(c, run_dir, stage)
            from .aliases import freeze_aliases
            freeze_aliases(c, run_dir, m)
            if not search_only:
                d_groups = [["D"]] + [[] for _ in c["gpus"][1:]]
                stage("D_smoke", lambda: workers(c, run_dir, "smoke", d_groups))
                from .aliases import group_aliases, sampling_groups
                independent = sampling_groups(c["groups"], group_aliases(c, run_dir, m))
                base_groups = [g for g in independent if g in "ABCD"]
                rl_groups = [g for g in independent if g == "E"]
                if base_groups:
                    stage("base_sampling", lambda: workers(c, run_dir, "sample", [base_groups for _ in c["gpus"]]))
                if rl_groups:
                    stage("rl_sampling", lambda: workers(c, run_dir, "sample", [rl_groups for _ in c["gpus"]]))
                from .grading import grade_run
                from .report import summarize
                stage("grading", lambda: grade_run(c, run_dir))
                # Always rerun final completeness check, including on resumed runs.
                summarize(c, run_dir, require_complete=True)
            atomic_json(run_dir / "status.json", {"state": "search_complete" if search_only else "complete", "updated_utc": timestamp(), "exit_code": 0})
        except BaseException as exc:
            previous = read_json(run_dir / "status.json") if (run_dir / "status.json").exists() else {}
            atomic_json(run_dir / "status.json", {"state": "failed", "stage": previous.get("stage", "preflight"),
                                                  "error": str(exc), "traceback": traceback.format_exc(), "updated_utc": timestamp(), "exit_code": 1})
            raise
