from pathlib import Path

import pytest

from first_token.core import ROOT, config


def pytest_addoption(parser):
    parser.addoption("--tokenizer-path", default=None, help="Local Qwen3-4B-Base tokenizer for encoding tests")


@pytest.fixture(scope="session")
def tokenizer(request):
    path = request.config.getoption("--tokenizer-path") or config(ROOT / "configs/main.json")["models"]["base"]["path"]
    if not Path(path).is_dir():
        pytest.skip("Token-encoding tests require the configured local Base tokenizer")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, local_files_only=True)
