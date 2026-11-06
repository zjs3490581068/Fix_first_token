"""Per-worker offline environments and run-local caches."""
import os
from pathlib import Path


def worker_environment(run_dir, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    root = Path(run_dir).resolve() / ".runtime" / f"gpu{gpu}"
    for variable, subdir in [("TRITON_CACHE_DIR", "triton"), ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
                             ("VLLM_CACHE_ROOT", "vllm"), ("TMPDIR", "tmp")]:
        path = root / subdir
        path.mkdir(parents=True, exist_ok=True)
        env[variable] = str(path)
    return env
