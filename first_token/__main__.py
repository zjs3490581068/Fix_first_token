import argparse
from pathlib import Path

from .core import ROOT, config


def main():
    ap = argparse.ArgumentParser(description="Single-token MATH-500 experiments; local models only")
    ap.add_argument("command", choices=["preflight", "search", "sample", "grade", "summarize", "run", "_worker"])
    ap.add_argument("--config", type=Path, default=ROOT / "configs/main.json")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--groups", nargs="+", choices=list("ABCDE"))
    ap.add_argument("--mode", choices=["beam", "screen", "smoke", "sample"])
    ap.add_argument("--gpu", type=int)
    ap.add_argument("--worker", type=int)
    ap.add_argument("--long-check", action="store_true")
    args = ap.parse_args()
    c = config(args.config)
    if args.command == "preflight":
        from .preflight import preflight
        preflight(c, args.run_dir)
    elif args.command in ["run", "search"]:
        from .pipeline import run
        run(c, args.run_dir, search_only=args.command == "search")
    elif args.command == "sample":
        from .pipeline import workers, run_lock
        with run_lock(args.run_dir):
            workers(c, args.run_dir, "sample", [args.groups or c["groups"] for _ in c["gpus"]])
    elif args.command == "grade":
        from .grading import grade_run
        from .pipeline import run_lock
        with run_lock(args.run_dir):
            grade_run(c, args.run_dir, args.groups)
    elif args.command == "summarize":
        from .report import summarize
        summarize(c, args.run_dir)
    else:
        if args.worker is None or args.gpu not in c["gpus"] or c["gpus"][args.worker] != args.gpu:
            ap.error("Worker/GPU assignment invalid")
        if args.mode == "beam":
            from .search import beam_worker
            beam_worker(c, args.run_dir, args.gpu, args.worker)
        elif args.mode == "screen":
            from .search import screen_worker
            screen_worker(c, args.run_dir, args.gpu, args.worker)
        elif args.mode == "smoke":
            from .sampling import smoke_worker
            smoke_worker(c, args.run_dir, args.gpu, args.worker, args.groups or [], args.long_check)
        elif args.mode == "sample":
            from .sampling import sample_worker
            sample_worker(c, args.run_dir, args.gpu, args.worker, args.groups or c["groups"])
        else:
            ap.error("--mode required for _worker")


if __name__ == "__main__":
    main()
