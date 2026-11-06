"""pass@k and problem-level bootstrap, adapted from cues.metrics.

Input: {problem_id: [0/1, ...]} in sample-index order. Bootstrap uses 1000 resamples
with random.Random(0); pass@k uses 1 - C(n-c,k)/C(n,k) (Chen et al., 2021).
"""
import math
import random

BOOT = 1000
SEED = 0
KS = (8, 16)


def pass_at_k(n: int, c: int, k: int):
    """Unbiased pass@k for one problem with n rollouts and c correct; None when n < k."""
    if n < k:
        return None
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def mean(xs):
    return sum(xs) / len(xs)


def sd(xs):
    """Sample SD (n-1); 0 for a single value."""
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def bootstrap(per_problem, rng, boot=BOOT):
    """(ci95, sd) of the mean over `boot` resamples of the problems, with replacement, using `rng`."""
    n = len(per_problem)
    means = sorted(mean(rng.choices(per_problem, k=n)) for _ in range(boot))
    return [round(means[int(0.025 * boot)], 4), round(means[int(0.975 * boot) - 1], 4)], round(sd(means), 4)


def per_problem_pass1(correct_by_problem):
    return [mean(v) for v in correct_by_problem.values()]


def per_problem_pass_at_k(correct_by_problem, k):
    vals = []
    for v in correct_by_problem.values():
        p = pass_at_k(len(v), sum(v), k)
        if p is None:
            return None
        vals.append(p)
    return vals


def by_rollout_index(correct_by_problem):
    """Accuracy of rollout i over the problems that have it, for every index present, in index order."""
    idx = sorted({i for v in correct_by_problem.values() for i in range(len(v))})
    return [round(mean([v[i] for v in correct_by_problem.values() if i < len(v)]), 4) for i in idx]


def cell_scores(correct_by_problem, ks=KS, seed=SEED):
    """Every score for one cell, keys as written in the cell summaries."""
    if not correct_by_problem:
        return {}
    rng = random.Random(seed)
    per1 = per_problem_pass1(correct_by_problem)
    ci, s = bootstrap(per1, rng)
    idx = by_rollout_index(correct_by_problem)
    out = {"pass@1": round(mean(per1), 4), "pass@1_ci95": ci, "pass@1_sd_boot": s,
           "pass@1_by_rollout_index": idx, "pass@1_sd_rollouts": round(sd(idx), 4)}
    for k in ks:
        perk = per_problem_pass_at_k(correct_by_problem, k)
        out[f"pass@{k}"] = None if perk is None else round(mean(perk), 4)
        if perk is not None:
            out[f"pass@{k}_ci95"], out[f"pass@{k}_sd_boot"] = bootstrap(perk, rng)
    return out
