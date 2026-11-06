from pathlib import Path
import subprocess
import os
import sys

import pytest

from first_token.core import ROOT, atomic_json, config
from first_token.runtime import worker_environment


def test_default_paths_are_relative_to_the_repository():
    c = config(ROOT / "configs/main.json")
    assert Path(c["test_data"]) == ROOT / "data/math500.jsonl"
    assert Path(c["models"]["base"]["path"]) == ROOT / "models/Qwen3-4B-Base"
    assert Path(c["output_root"]) == ROOT / "outputs"


def test_cli_works_with_a_minimal_environment():
    result = subprocess.run([sys.executable, "-m", "first_token", "--help"], cwd=ROOT,
                            env={"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"},
                            check=True, text=True, capture_output=True)
    assert "preflight" in result.stdout and "summarize" in result.stdout


def test_inherited_paths_keep_their_declaring_directory(tmp_path):
    child = tmp_path / "nested" / "local.json"
    atomic_json(child, {"extends": str(ROOT / "configs/main.json"), "gpus": [1, 7],
                       "models": {"base": {"path": "./weights"}}})
    c = config(child)
    assert c["gpus"] == [1, 7]
    assert Path(c["test_data"]) == ROOT / "data/math500.jsonl"
    assert Path(c["models"]["base"]["path"]) == child.parent / "weights"
    assert c["models"]["base"]["id"] == "Qwen/Qwen3-4B-Base"


@pytest.mark.parametrize("gpus", [[-1], [True], [0, 0], []])
def test_gpu_configuration_rejects_invalid_indices(tmp_path, gpus):
    path = tmp_path / "bad.json"
    atomic_json(path, {"extends": str(ROOT / "configs/main.json"), "gpus": gpus})
    with pytest.raises(ValueError, match="GPU indices"):
        config(path)


def test_worker_caches_are_contained_in_the_run(tmp_path):
    env = worker_environment(tmp_path, 7)
    assert env["CUDA_VISIBLE_DEVICES"] == "7"
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
    for key in ["TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "VLLM_CACHE_ROOT", "TMPDIR"]:
        assert Path(env[key]).is_relative_to(tmp_path)
        assert Path(env[key]).is_dir()


@pytest.mark.parametrize("script", ["launch_tmux.sh", "run_pipeline.sh"])
def test_shell_help_requires_no_private_environment(script):
    result = subprocess.run(["bash", str(ROOT / "scripts" / script), "--help"],
                            check=True, text=True, capture_output=True)
    assert "Usage:" in result.stdout
