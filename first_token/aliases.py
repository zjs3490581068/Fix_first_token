"""Explicit reuse of identical C/D conditions, never extra independent samples."""
from pathlib import Path

from .core import condition, read_json


def freeze_aliases(c, run_dir, manifest):
    from .core import atomic_json
    winner = read_json(Path(run_dir) / "search/winner.json")
    if winner["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Winner identity mismatch")
    if {"C", "D"} <= set(c["groups"]) and winner["forced_token_id"] == condition("C")["forced_token_id"]:
        path = Path(run_dir) / "group_aliases.json"
        value = {"fingerprint": manifest["fingerprint"], "aliases": {"D": "C"}}
        if path.exists() and read_json(path) != value:
            raise ValueError("Frozen sharing policy changed")
        atomic_json(path, value)
        group_aliases(c, run_dir, manifest)


def group_aliases(c, run_dir, manifest):
    path = Path(run_dir) / "group_aliases.json"
    if not path.exists():
        return {}
    data = read_json(path)
    aliases = data["aliases"]
    if data["fingerprint"] != manifest["fingerprint"] or aliases != {"D": "C"}:
        raise ValueError("Invalid group-alias identity or unsupported mapping")
    if not {"C", "D"} <= set(c["groups"]):
        raise ValueError("Both C and D must be configured")
    winner = read_json(Path(run_dir) / "search/winner.json")
    if winner["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Winner identity mismatch")
    a, b = condition("C"), condition("D", winner["forced_token_id"])
    if (a["model"], a["forced_token_id"]) != (b["model"], b["forced_token_id"]):
        raise ValueError("C/D conditions differ; results cannot be shared")
    # A second physical dataset would be ambiguous; no silent merging/discarding.
    for kind in ["samples", "grades"]:
        if any((Path(run_dir) / kind / "D").glob("*.jsonl.gz")):
            raise ValueError("Separate D artifacts exist; cannot alias them silently")
    return aliases


def sampling_groups(groups, aliases):
    return list(dict.fromkeys(aliases.get(g, g) for g in groups))
