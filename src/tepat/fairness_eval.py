"""Evaluation-only summaries of regenerated synthetic populations, not entity ranks."""
from __future__ import annotations

import pandas as pd

SEEDS = [20261007, 20261101, 20261102, 20261103, 20261104]
BRANCH_ROLES = {"A": "tuning_only", "B": "held_out", "C": "shifted"}
COUNTS = ["n_total", "n_selected", "n_violators", "n_honest", "n_honest_selected"]
RATES = {"pick_rate": ("n_selected", "n_total"),
         "violation_rate": ("n_violators", "n_total"),
         "honest_flag_rate": ("n_honest_selected", "n_honest")}
GROUP_KEYS = ["branch", "ranking", "group", "value"]


def aggregate_groups(rows: pd.DataFrame, expected_seeds: list[int] | None = None) -> pd.DataFrame:
    """Unweighted per-seed rate mean/min/max, plus sums/ranges of counts.

    Missing rows stay missing. A present row with a zero denominator contributes
    counts, but never a measured zero to an undefined rate. Pooled ratios are
    separately named and count repeated synthetic populations, not unique firms.
    """
    expected = SEEDS if expected_seeds is None else expected_seeds
    if len(set(expected)) != len(expected) or not expected:
        raise ValueError("expected seeds must be nonempty and unique")
    if rows.duplicated(["seed"] + GROUP_KEYS).any():
        raise ValueError("duplicate seed/group rows")
    if not rows.seed.isin(expected).all():
        raise ValueError("unexpected seed in group rows")
    summaries = []
    for keys, part in rows.groupby(GROUP_KEYS, observed=True, sort=True):
        present = set(part.seed)
        out = dict(zip(GROUP_KEYS, keys))
        out.update(branch_role=BRANCH_ROLES[out["branch"]], n_seeds_expected=len(expected),
                   n_seeds_present=len(present), n_nonempty_seeds=int(part.n_total.gt(0).sum()),
                   seeds_present=";".join(str(s) for s in expected if s in present),
                   seeds_missing=";".join(str(s) for s in expected if s not in present),
                   small_n_seeds=int(part.small_n.sum()))
        for col in COUNTS:
            out.update({f"{col}_sum": int(part[col].sum()), f"{col}_min": int(part[col].min()),
                        f"{col}_max": int(part[col].max())})
        for rate, (numerator, denominator) in RATES.items():
            values = part[rate].dropna()
            out.update({f"{rate}_mean": values.mean(), f"{rate}_min": values.min(),
                        f"{rate}_max": values.max(), f"{rate}_n_seeds": len(values),
                        f"{rate}_undefined_seeds": int(part[rate].isna().sum()),
                        f"{rate}_pooled": part[numerator].sum() / part[denominator].sum()
                        if part[denominator].sum() else None})
        summaries.append(out)
    return pd.DataFrame(summaries)
