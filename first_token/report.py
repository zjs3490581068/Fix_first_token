"""Streaming completeness checks, pass@k, behavioral summaries, and review cases."""
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

from .core import atomic_json, checked_records, check_manifest, condition, expected_keys, problems, read_json
from .metrics import bootstrap, cell_scores

KS = (1, 2, 4, 8, 16, 32, 64, 128)


def summarize(c, run_dir, require_complete=False):
    run_dir = Path(run_dir)
    m = check_manifest(c, run_dir)
    from .aliases import group_aliases
    aliases = group_aliases(c, run_dir, m)
    problems_list = problems(c["test_data"], labels=True)
    by_id = {p["problem_id"]: p for p in problems_list}
    expected = expected_keys(problems_list, 0, c["rollouts"])
    out = run_dir / "analysis"
    out.mkdir(exist_ok=True)
    winner_path = run_dir / "search/winner.json"
    winner = read_json(winner_path)["forced_token_id"] if winner_path.exists() else None
    result, counts, cases = {}, {}, {}
    for group in c["groups"]:
        source_group = aliases.get(group, group)
        seen = set()
        correct = defaultdict(dict)
        first, lengths, statuses = Counter(), Counter(), Counter()
        capped = 0
        for raw in sorted((run_dir / "samples" / source_group).glob("*.jsonl.gz")):
            grade_path = run_dir / "grades" / source_group / raw.name
            if not grade_path.exists():
                continue
            rows = checked_records(raw, m["fingerprint"])
            keys = {(r["problem_id"], r["sample_index"]) for r in rows}
            grades = {(r["problem_id"], r["sample_index"]): r for r in checked_records(grade_path, m["fingerprint"], keys)}
            if keys & seen or not keys <= expected:
                raise ValueError(f"Duplicate or out-of-protocol sample in group {group}")
            seen |= keys
            spec = condition(group, winner)
            for r in rows:
                if r["group"] != source_group or r["forced_token_id"] != spec["forced_token_id"]:
                    raise ValueError("Wrong group/forced ID in sample")
                p, i = r["problem_id"], r["sample_index"]
                g = grades[(p, i)]
                if g["group"] != source_group:
                    raise ValueError("Grading source group mismatch")
                correct[p][i] = int(g["correct"])
                first[r["first_token_id"]] += 1
                lengths[r["output_length"]] += 1
                statuses[g["grading_status"]] += 1
                capped += int(r["hit_token_cap"])
                case_key = (group, p, bool(g["correct"]))
                if case_key not in cases:
                    cases[case_key] = {"group": group, "source_group": source_group, "problem_id": p, "sample_index": i,
                                       "forced_token_id": r["forced_token_id"], "correct": g["correct"],
                                       "extracted_answer": g["extracted_answer"], "raw_shard": str(raw),
                                       "opening_excerpt": r["output_text"][:1500], "ending_excerpt": r["output_text"][-1500:]}
        complete = seen == expected
        series = {p: [v[i] for i in sorted(v)] for p, v in correct.items()}
        scores = cell_scores(series, ks=KS[1:]) if complete else None
        stratified = {}
        if complete:
            for field in ["level", "subject"]:
                strata = defaultdict(list)
                for p, vals in series.items():
                    strata[str(by_id[p][field])].append(sum(vals) / len(vals))
                stratified[field] = {k: {"n_problems": len(v), "pass@1": sum(v) / len(v),
                                         "ci95": bootstrap(v, random.Random(0))[0]} for k, v in strata.items()}
        result[group] = {"complete": complete, "samples": len(seen), "expected_samples": len(expected),
                         "source_group": source_group, "shared_samples": group in aliases,
                         "scored_problems": len(correct), "scores": scores, "stratified": stratified,
                         "mean_output_tokens": sum(k * v for k, v in lengths.items()) / len(seen) if seen else None,
                         "cap_rate": capped / len(seen) if seen else None, "grading_statuses": dict(statuses),
                         "first_token_counts": {str(k): v for k, v in first.most_common()}}
        counts[group] = {p: sum(v) for p, v in series.items()}
    complete = all(v["complete"] for v in result.values())
    summary = {"fingerprint": m["fingerprint"], "complete": complete, "groups": result,
               "group_aliases": aliases,
               "expected_independent_samples": len(expected) * len({aliases.get(g, g) for g in c["groups"]}),
               "independent_scored_samples": sum(v["samples"] for g, v in result.items() if g not in aliases),
               "D_token_id": winner, "D_equals_B": winner == 106185, "D_equals_C": winner == 2014,
               "conditional_distribution_note": "A/E conditioned on naturally generating D's first token; D is an intervention. Report sample counts; these are behavioral associations, not a full causal mechanism."}
    atomic_json(out / "summary.json", summary)
    if not complete:
        print("Analysis incomplete: see analysis/summary.json", flush=True)
        if require_complete:
            raise RuntimeError("Formal completeness gate failed")
        return summary
    with (out / "per_problem.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["problem_id", "level", "subject", *[f"{g}_correct" for g in c["groups"]]])
        for p in problems_list:
            writer.writerow([p["problem_id"], p["level"], p["subject"], *[counts[g][p["problem_id"]] for g in c["groups"]]])
    comparisons = {}
    for left, right in [("A", "B"), ("A", "C"), ("A", "D"), ("A", "E"), ("C", "D")]:
        if left in counts and right in counts:
            if aliases.get(left, left) == aliases.get(right, right):
                comparisons[f"{right}-{left}"] = {"status": "shared_samples_not_an_independent_comparison", "source_group": aliases.get(left, left)}
                continue
            diff = [(counts[right][p] - counts[left][p]) / c["rollouts"] for p in by_id]
            comparisons[f"{right}-{left}"] = {"pass@1_difference": sum(diff) / len(diff), "paired_problem_ci95": bootstrap(diff, random.Random(0))[0]}
    atomic_json(out / "paired_comparisons.json", comparisons)
    if {"A", "D", "E"} <= counts.keys():
        chosen = {}
        rules = {"A_wrong_D_correct": lambda p: counts["D"][p] > counts["A"][p],
                 "A_correct_D_wrong": lambda p: counts["A"][p] > counts["D"][p],
                 "D_E_both_correct": lambda p: counts["D"][p] > 0 and counts["E"][p] > 0}
        for name, predicate in rules.items():
            candidates = [p for p in by_id if predicate(p)]
            candidates.sort(key=lambda p: (-abs(counts["D"][p] - counts["A"][p]), p))
            chosen[name] = [{"problem": by_id[p], "correct_counts": {g: counts[g][p] for g in "ADE"},
                             "examples": [v for (g, pk, verdict), v in cases.items() if pk == p and g in "ADE"]}
                            for p in candidates[:5]]
        atomic_json(out / "case_candidates.json", {"selection_rule": "Largest absolute A/D correct-count differences, ties by problem ID; includes both gain/loss directions. Excerpts only; inspect linked raw shards for full reasoning. No automated logic-quality claim.", "cases": chosen})
    _plots(c, result, counts, out)
    lines = ["# Formal MATH-500 results", "", "Each reported condition has 500 × 128 graded samples; aliases reuse their source samples.",
             f"Independent generated samples: {summary['expected_independent_samples']}.",
             "C and D share the same 128 samples per problem; they are not independent replications and are never pooled into 256 samples." if aliases else "No shared groups.", "",
             "| Group | pass@1 | 95% CI | pass@128 | Cap rate |", "|---|---:|---|---:|---:|"]
    for g, v in result.items():
        s = v["scores"]
        lines.append(f"| {g} | {s['pass@1']:.4f} | {s['pass@1_ci95']} | {s['pass@128']:.4f} | {v['cap_rate']:.4f} |")
    lines += ["", f"D token: {winner}. D=C: {winner == 2014}; D=B: {winner == 106185}.", "",
              "GRPO training/test overlap remains unverified. pass@128 is finite-budget empirical coverage. Cases require manual review; token-distribution similarities do not establish the full mechanism of RL."]
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print("Formal A–E completeness and report gate passed", flush=True)
    return summary


def _plots(c, results, counts, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for group, item in results.items():
        if item.get("shared_samples"):
            continue
        labels = [g for g, v in results.items() if v.get("source_group", g) == group]
        label = " / ".join(labels) + (" (shared)" if len(labels) > 1 else "")
        ax.plot(KS, [item["scores"][f"pass@{k}"] for k in KS], marker="o", label=label)
    ax.set(xscale="log", xlabel="k", ylabel="pass@k", ylim=(0, 1), title="MATH-500 (128 samples/problem)")
    ax.set_xticks(KS, labels=list(map(str, KS)))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "pass_at_k.png", dpi=180)
    fig.savefig(out / "pass_at_k.pdf")
    plt.close(fig)
    if {"A", "E"} <= results.keys():
        tokens = set()
        for group in ["A", "E"]:
            tokens.update(list(results[group]["first_token_counts"])[:8])
        tokens = sorted(tokens, key=lambda t: -sum(results[g]["first_token_counts"].get(t, 0) for g in ["A", "E"]))
        fig, ax = plt.subplots(figsize=(9, 4))
        for i, group in enumerate(["A", "E"]):
            freq = [results[group]["first_token_counts"].get(t, 0) / results[group]["samples"] for t in tokens]
            ax.bar([x + (i - .5) * .4 for x in range(len(tokens))], freq, width=.4, label=group)
        ax.set_xticks(range(len(tokens)), tokens, rotation=45)
        ax.set(xlabel="First token ID", ylabel="Observed frequency", title="Base / GRPO first-token distribution")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "first_token_distribution.png", dpi=180)
        fig.savefig(out / "first_token_distribution.pdf")
        plt.close(fig)
