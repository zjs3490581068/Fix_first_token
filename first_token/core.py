"""Protocol and durable, checked artifacts. No GPU dependencies in this module."""
import gzip
import hashlib
import json
import os
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOXED = "Problem: {q}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\nSolution:"
STOPS = ["\nProblem:", "\nProblem :"]
STOP_IDS = [151643, 151645]
B_TOKEN, C_TOKEN = 106185, 2014
SCHEMA = 1


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
    with tmp.open("w") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_shard(path, records):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite a completed shard: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=3) as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    with tmp.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def config(path):
    path = Path(path).resolve()
    c = _config_values(path, set())
    if (not c["gpus"] or any(type(g) is not int or g < 0 for g in c["gpus"])
            or len(set(c["gpus"])) != len(c["gpus"])):
        raise ValueError("GPU indices must be unique nonnegative integers")
    if not c["groups"] or len(set(c["groups"])) != len(c["groups"]) or not set(c["groups"]) <= set("ABCDE"):
        raise ValueError("Invalid groups")
    for k in ["rollouts", "rollouts_per_batch", "problems_per_batch", "output_budget"]:
        if c[k] < 1:
            raise ValueError(k)
    return c


def _config_values(path, seen):
    if path in seen:
        raise ValueError("Circular configuration inheritance")
    seen.add(path)
    values = read_json(path)
    for key in ["test_data", "probe_data", "output_root"]:
        if key in values:
            values[key] = _resolve_path(values[key], path.parent)
    for spec in values.get("models", {}).values():
        if "path" in spec:
            spec["path"] = _resolve_path(spec["path"], path.parent)
    parent = values.pop("extends", None)
    if parent is None:
        return values
    result = _config_values((path.parent / parent).resolve(), seen)
    return _merge(result, values)


def _resolve_path(value, directory):
    value = os.path.expandvars(value)
    if "$" in value:
        raise ValueError(f"Undefined environment variable in path: {value}")
    return str((directory / Path(value).expanduser()).resolve())


def _merge(base, overrides):
    result = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def problems(path, labels=False, limit=None):
    rows = []
    for r in jsonl(path):
        # The search/generation interface never receives gt/answer/solution.
        item = {"problem_id": r["problem_key"], "problem": r["problem"]}
        if labels:
            item.update(answer=r["gt"], level=r.get("level"), subject=r.get("capability"))
        rows.append(item)
        if limit and len(rows) >= limit:
            break
    if len({r["problem_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate problem IDs")
    return rows


def prompt_ids(tok, problem, forced=None):
    ids = tok.encode(BOXED.format(q=problem), add_special_tokens=False)
    if forced is not None:
        if type(forced) is not int or forced < 0 or forced >= len(tok) or forced in set(tok.all_special_ids) | set(STOP_IDS):
            raise ValueError("The forced prefix must be exactly one non-special token ID")
        return ids + [forced]
    return ids


def token_budget(total, forced):
    count = len(forced) if isinstance(forced, (list, tuple)) else int(forced is not None)
    n = total - count
    if n < 1:
        raise ValueError("No continuation budget")
    return n


def sample_seed(base, index):
    return base + index


def condition(group, winner=None):
    if group not in "ABCDE" or len(group) != 1:
        raise ValueError(group)
    forced = {"A": None, "B": B_TOKEN, "C": C_TOKEN, "D": winner, "E": None}[group]
    if group == "D" and forced is None:
        raise ValueError("D requires a frozen search winner")
    return {"group": group, "model": "rl" if group == "E" else "base", "forced_token_id": forced}


def expected_keys(rows, offset, n):
    return {(r["problem_id"], i) for r in rows for i in range(offset, offset + n)}


def checked_records(path, fingerprint, expected=None):
    rows = list(jsonl(path))
    keys = [(r["problem_id"], r["sample_index"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(f"Duplicate sample keys: {path}")
    if any(r["experiment_fingerprint"] != fingerprint for r in rows):
        raise ValueError(f"Experiment mismatch: {path}")
    if expected is not None and set(keys) != expected:
        raise ValueError(f"Incomplete or unexpected samples: {path}")
    return rows


def chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def load_manifest(run_dir):
    return read_json(Path(run_dir) / "manifest.json")


def check_manifest(c, run_dir):
    from .preflight import source_hashes
    m = load_manifest(run_dir)
    current_code = source_hashes() if "code" in m else None
    if m["config"] != c or (current_code is not None and m["code"] != current_code):
        raise ValueError("Configuration/code differs from the frozen run; use a new run directory")
    return m
