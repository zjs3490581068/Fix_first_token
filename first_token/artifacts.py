"""Read completed source runs without importing or executing their code."""
from pathlib import Path

from .core import (BOXED, STOP_IDS, STOPS, checked_records, digest, file_hash,
                   condition, problems, read_json)


def verify_fingerprint(manifest):
    payload = {k: v for k, v in manifest.items() if k not in {"fingerprint", "created_utc", "python"}}
    if digest(payload) != manifest["fingerprint"]:
        raise ValueError("Manifest fingerprint does not match its contents")


def verify_snapshot(root, hashes):
    root = Path(root).resolve()
    for name, sha in hashes.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or file_hash(path) != sha:
            raise ValueError(f"Source snapshot checksum mismatch: {name}")


def source_manifest(run_dir):
    run_dir = Path(run_dir)
    if read_json(run_dir / "status.json")["state"] != "complete":
        raise ValueError("Source run is incomplete")
    manifest = read_json(run_dir / "manifest.json")
    verify_fingerprint(manifest)
    if (manifest["prompt"] != BOXED or manifest["stops"] != STOPS
            or manifest["stop_ids"] != STOP_IDS):
        raise ValueError("Source prompt/stopping protocol differs")
    verify_snapshot(run_dir / "source", manifest["code"])
    # Older completed runs can carry an additional source snapshot. It is read-only provenance.
    audit_path = run_dir / "runtime_revision.json"
    if audit_path.exists():
        audit = read_json(audit_path)
        old = {k: v for k, v in manifest["config"].items() if k != "gpus"}
        new = {k: v for k, v in audit["config"].items() if k != "gpus"}
        if (audit["base_fingerprint"] != manifest["fingerprint"]
                or audit["prior_code"] != manifest["code"] or old != new):
            raise ValueError("Source provenance/configuration mismatch")
        snapshot = run_dir / "runtime_revisions" / audit["revision_id"]
        verify_snapshot(snapshot, audit["code"])
        if "group_aliases_sha256" in audit:
            if file_hash(run_dir / "group_aliases.json") != audit["group_aliases_sha256"]:
                raise ValueError("Source sharing policy checksum mismatch")
    data = manifest["datasets"]["test_data"]
    if file_hash(data["path"]) != data["sha256"]:
        raise ValueError("Source test dataset changed")
    return manifest


def matched_baseline(run_dir, c, model, dataset_sha):
    """Validated rollout zero from each available main-experiment baseline."""
    run_dir = Path(run_dir)
    manifest = source_manifest(run_dir)
    for key in ["seed", "temperature", "top_p", "output_budget"]:
        if manifest["config"][key] != c[key]:
            raise ValueError(f"Baseline scientific parameter differs: {key}")
    original_model = manifest["models"]["base"]
    if (original_model["id"] != model["id"] or original_model["revision"] != model["revision"]
            or original_model["files"] != model["files"]
            or manifest["datasets"]["test_data"]["sha256"] != dataset_sha):
        raise ValueError("Baseline model or dataset identity differs")
    refs = {p["problem_id"] for p in problems(c["test_data"])}
    result = {}
    for group in [g for g in ["A", "B", "C", "E"] if g in manifest["config"]["groups"]]:
        selected, seen = {}, set()
        for raw in sorted((run_dir / "samples" / group).glob("*.jsonl.gz")):
            rows = checked_records(raw, manifest["fingerprint"])
            keys = {(r["problem_id"], r["sample_index"]) for r in rows}
            if keys & seen:
                raise ValueError("Duplicate baseline samples")
            seen |= keys
            target = run_dir / "grades" / group / raw.name
            grades = checked_records(target, manifest["fingerprint"], keys)
            receipt = read_json(target.with_name(target.name + ".meta.json"))
            if receipt["raw_sha256"] != file_hash(raw) or receipt["records"] != len(rows):
                raise ValueError("Baseline raw/grade receipt mismatch")
            by_key = {(r["problem_id"], r["sample_index"]): r for r in rows}
            for grade in grades:
                key = grade["problem_id"], grade["sample_index"]
                record = by_key[key]
                if record["group"] != group or grade["group"] != group:
                    raise ValueError("Baseline group mismatch")
                forced = condition(group)["forced_token_id"]
                if (record["forced_token_id"] != forced or record["seed"] != c["seed"] + key[1]
                        or record["output_budget"] != c["output_budget"]
                        or record["output_length"] != len(record["output_token_ids"])
                        or record["output_length"] > c["output_budget"]
                        or (forced is not None and record["output_token_ids"][:1] != [forced])):
                    raise ValueError("Baseline condition, seed or output budget mismatch")
                if key[1] != 0:
                    continue
                selected[key[0]] = int(grade["correct"])
        expected = {(p, i) for p in refs for i in range(manifest["config"]["rollouts"])}
        if seen != expected or set(selected) != refs:
            raise ValueError("Baseline coverage is incomplete or out of protocol")
        result[group] = selected
    return result, {"path": str(run_dir.resolve()), "fingerprint": manifest["fingerprint"],
                    "manifest_sha256": file_hash(run_dir / "manifest.json")}
