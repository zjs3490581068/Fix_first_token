import copy
from collections import Counter

import pytest

from experiments.extract_judge_samples import QUOTAS, anonymize, fixed_selection, validate_public


def selection_plan():
    rows = []
    i = 0
    for level, count in QUOTAS.items():
        for _ in range(count):
            i += 1
            rows.append(f"| {i} | {level} | `question_{i}` | {i + 20} |")
    return "\n".join(rows)


def test_fixed_selection_and_paired_anonymization():
    plan = selection_plan()
    selected = fixed_selection(plan)
    tasks = anonymize(selected)
    assert tasks == anonymize(selected)
    assert Counter(t["source_group"] for t in tasks) == {"A": 50, "B": 50, "C": 50}
    assert Counter(t["level"] for t in selected) == QUOTAS
    for s in selected:
        matches = [t for t in tasks if t["problem_id"] == s["problem_id"]]
        assert len(matches) == 3 and all(t["sample_index"] == s["sample_index"] for t in matches)
    with pytest.raises(ValueError):
        fixed_selection(plan.replace("`question_1` | 21", "`question_1` | 128"))


def test_blind_input_retains_complete_text_and_rejects_leakage():
    tasks = anonymize(fixed_selection(selection_plan()))
    originals, refs, public = {}, {}, []
    text = "解答 " + "Mixed text 中文\n" * 5000 + "完整末尾"
    for task in tasks:
        key = task["source_group"], task["problem_id"], task["sample_index"]
        originals[key] = {"record": {"output_text": text}}
        refs[task["problem_id"]] = {"problem": "Q", "answer": "1", "solution": "S"}
        public.append({"judge_id": task["judge_id"], "problem": "Q", "reference_answer": "1",
                       "reference_solution": "S", "model_response": text})
    validate_public(public, tasks, originals, refs)
    leaked = copy.deepcopy(public)
    leaked[0]["source_group"] = "B"
    with pytest.raises(ValueError):
        validate_public(leaked, tasks, originals, refs)
    shortened = copy.deepcopy(public)
    shortened[0]["model_response"] = text[:2000]
    with pytest.raises(ValueError):
        validate_public(shortened, tasks, originals, refs)
