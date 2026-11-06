import pytest

from first_token.core import ROOT, atomic_json, config


def test_configs_preserve_scientific_defaults():
    main = config(ROOT / "configs/main.json")
    assert main["seed"] == 20260819 and main["output_budget"] == 16384
    assert main["temperature"] == .6 and main["top_p"] == .95
    assert main["rollouts"] == 128 and main["rollouts_per_batch"] == 16
    for name in ["english_tokens", "two_token"]:
        extra = config(ROOT / f"configs/{name}.json")
        assert extra["rollouts"] == extra["rollouts_per_batch"] == 1
        assert {k: v for k, v in extra.items() if k not in {"rollouts", "rollouts_per_batch"}} == {
            k: v for k, v in main.items() if k not in {"rollouts", "rollouts_per_batch"}}


def test_config_rejects_cycle_and_undefined_paths(tmp_path):
    atomic_json(tmp_path / "a.json", {"extends": "b.json"})
    atomic_json(tmp_path / "b.json", {"extends": "a.json"})
    with pytest.raises(ValueError, match="Circular"):
        config(tmp_path / "a.json")
    values = config(ROOT / "configs/main.json")
    values["test_data"] = "$FT_UNDEFINED_FOR_TEST/data.jsonl"
    atomic_json(tmp_path / "bad.json", values)
    with pytest.raises(ValueError, match="Undefined"):
        config(tmp_path / "bad.json")


def test_resume_rejects_config_and_code_change(tmp_path, monkeypatch):
    from first_token import core, preflight
    atomic_json(tmp_path / "manifest.json", {"config": {"seed": 1}, "code": {"a": "hash"}})
    monkeypatch.setattr(preflight, "source_hashes", lambda: {"a": "hash"})
    assert core.check_manifest({"seed": 1}, tmp_path)["config"] == {"seed": 1}
    with pytest.raises(ValueError, match="frozen"):
        core.check_manifest({"seed": 2}, tmp_path)
    monkeypatch.setattr(preflight, "source_hashes", lambda: {"a": "changed"})
    with pytest.raises(ValueError, match="frozen"):
        core.check_manifest({"seed": 1}, tmp_path)
