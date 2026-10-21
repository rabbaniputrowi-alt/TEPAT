"""Benchmark: compare rankings under a fixed inspection-hour budget.

This is the ONLY module (besides the generator) that reads truth/. It never feeds truth into scoring
for the branch being evaluated; branch A's sampled labels stand in for past inspection outcomes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RANDOM_REPS = 50


def select_by_budget(order: list[str], hours: pd.Series, budget: float, preselected: list[str] | None = None) -> list[str]:
    picked, used = list(preselected or []), float(hours.loc[preselected].sum()) if preselected else 0.0
    seen = set(picked)
    for eid in order:
        if eid in seen:
            continue
        h = float(hours.loc[eid])
        if used + h > budget:
            continue
        picked.append(eid)
        seen.add(eid)
        used += h
        if used >= budget - 2:
            break
    return picked


def stratified_random(F: pd.DataFrame, hours: pd.Series, budget: float, rng) -> list[str]:
    """I-56 exploration arm: random picks spread across size bands in proportion to their count."""
    picks, used = [], 0.0
    ids = F.index.to_numpy().copy()
    rng.shuffle(ids)
    by_band = {b: [i for i in ids if F.at[i, "band_obs"] == b] for b in F.band_obs.unique()}
    total = sum(len(v) for v in by_band.values())
    for b, lst in by_band.items():
        quota = budget * len(lst) / total
        q_used = 0.0
        for i in lst:
            h = float(hours.loc[i])
            if q_used + h > quota:
                break
            picks.append(i)
            q_used += h
        used += q_used
    return picks


def evaluate(picked: list[str], T: pd.DataFrame, hours: pd.Series) -> dict:
    p = T.loc[picked]
    viol = T[T.is_violator]
    h_used = float(hours.loc[picked].sum())
    rp = float(p.loc[p.is_violator, "recoverable_rp_true"].sum())
    out = dict(inspected=len(picked), hours=h_used, hits=int(p.is_violator.sum()),
               precision=float(p.is_violator.mean()) if len(p) else 0.0,
               recall=float(p.is_violator.sum() / max(1, len(viol))),
               rp_recovered=rp, rp_per_hour=rp / max(1.0, h_used),
               rp_share=rp / max(1.0, float(viol.recoverable_rp_true.sum())))
    var_recall = {}
    for v in sorted({x for s in viol.variants for x in s.split(",") if x}):
        tv = viol[viol.variants.str.contains(v)]
        var_recall[v] = float(tv.index.isin(picked).mean())
    la_fp = {}
    honest = T[~T.is_violator]
    for c in sorted({x for s in honest.lookalikes for x in s.split(",") if x}):
        tc = honest[honest.lookalikes.str.contains(c)]
        la_fp[c] = float(tc.index.isin(picked).mean())
    out["recall_by_variant"], out["lookalike_flag_rate"] = var_recall, la_fp
    out["honest_flag_rate"] = float(honest.index.isin(picked).mean())
    return out


def topk(order: list[str], T: pd.DataFrame, fracs=(0.05, 0.10, 0.20)) -> dict:
    n = len(order)
    return {f"top{int(f * 100)}_precision": float(T.loc[order[: max(1, int(n * f))], "is_violator"].mean()) for f in fracs}


def fairness(picked: list[str], F: pd.DataFrame, T: pd.DataFrame, *,
             registry: pd.DataFrame | None = None,
             group_values: dict[str, list[str]] | None = None, min_group_n: int = 30) -> pd.DataFrame:
    """Group diagnostics; rates always carry their population denominators."""
    for name, frame in (("features", F), ("truth", T)):
        if not frame.index.is_unique or frame.index.hasnans:
            raise ValueError(f"{name} employer keys must be unique and non-null")
    if len(F) != len(T) or not F.index.isin(T.index).all():
        raise ValueError("features and truth must have exactly the same employer keys")
    selected = pd.Index(picked)
    if not selected.is_unique or selected.hasnans or not selected.isin(F.index).all():
        raise ValueError("selected employer keys must be unique, non-null and present in features")
    if not pd.api.types.is_bool_dtype(T.is_violator) or T.is_violator.isna().any():
        raise ValueError("is_violator must contain complete boolean labels")
    groups = ["sector", "band_obs", "kabkota"]
    df = F[["sector", "band_obs"]].copy()
    if registry is not None:
        if (not registry.index.is_unique or registry.index.hasnans or len(registry) != len(F)
                or not F.index.isin(registry.index).all()):
            raise ValueError("registry must have exactly the same unique employer keys as features")
    df["kabkota"] = F["kabkota"] if "kabkota" in F else registry["kabkota"] if registry is not None else None
    df = df.join(T[["is_violator"]], validate="one_to_one")
    df["picked"] = df.index.isin(picked)
    rows = []
    for col in groups:
        categorical = list(df[col].cat.categories) if isinstance(df[col].dtype, pd.CategoricalDtype) else []
        labels = df[col].astype(object).where(df[col].notna() & df[col].astype(str).str.strip().ne(""), "__missing__")
        values = list(dict.fromkeys((group_values or {}).get(col, []) + categorical + list(labels.unique())))
        for value in sorted(values):
            part = df[labels == value]
            n_total = len(part)
            n_honest = int((~part.is_violator).sum())
            n_selected = int(part.picked.sum())
            n_violators = int(part.is_violator.sum())
            n_honest_selected = int((part.picked & ~part.is_violator).sum())
            rows.append(dict(group=col, value=value, n=n_total, n_total=n_total,
                             n_selected=n_selected, n_violators=n_violators, n_honest=n_honest,
                             n_honest_selected=n_honest_selected,
                             violation_rate=n_violators / n_total if n_total else None,
                             pick_rate=n_selected / n_total if n_total else None,
                             honest_flag_rate=n_honest_selected / n_honest if n_honest else None,
                             small_n=n_total < min_group_n or n_honest < min_group_n,
                             warning=";".join(x for x, yes in [("empty_group", n_total == 0),
                                                               ("missing_label", value == "__missing__"),
                                                               ("no_honest", n_honest == 0),
                                                               ("small_n_total", n_total < min_group_n),
                                                               ("small_n_honest", n_honest < min_group_n)] if yes)))
    columns = ["group", "value", "n", "n_total", "n_selected", "n_violators", "n_honest", "n_honest_selected",
               "violation_rate", "pick_rate", "honest_flag_rate", "small_n", "warning"]
    return pd.DataFrame(rows, columns=columns)


def run(F: pd.DataFrame, T: pd.DataFrame, rankings: dict[str, pd.DataFrame], cfg: dict, seed: int,
        final: str = "tepat_v2_1") -> dict:
    hours = F["hours"]
    budget = cfg["inspection"]["budget_frac"] * float(hours.sum())
    res: dict = {"budget_hours": budget, "employers": len(F), "violators": int(T.is_violator.sum()),
                 "prevalence": float(T.is_violator.mean()),
                 "total_recoverable_rp": float(T.loc[T.is_violator, "recoverable_rp_true"].sum()), "rankings": {}}
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(RANDOM_REPS):
        order = list(rng.permutation(F.index.to_numpy()))
        reps.append({**evaluate(select_by_budget(order, hours, budget), T, hours), **topk(order, T)})
    keys = [k for k in reps[0] if isinstance(reps[0][k], (int, float))]
    rnd = {k: float(np.mean([r[k] for r in reps])) for k in keys}
    rnd["recall_by_variant"] = {v: float(np.mean([r["recall_by_variant"].get(v, 0) for r in reps])) for v in reps[0]["recall_by_variant"]}
    rnd["lookalike_flag_rate"] = {c: float(np.mean([r["lookalike_flag_rate"].get(c, 0) for r in reps])) for c in reps[0]["lookalike_flag_rate"]}
    res["rankings"]["random"] = rnd
    picks = {}
    for name, R in rankings.items():
        order = list(R.index)
        picked = select_by_budget(order, hours, budget)
        picks[name] = picked
        res["rankings"][name] = {**evaluate(picked, T, hours), **topk(order, T)}
    # final TEPAT version with an 8% random exploration arm (I-56)
    if final in rankings:
        arm_budget = cfg["inspection"]["random_arm_frac"] * budget
        arm = stratified_random(F, hours, arm_budget, rng)
        picked = select_by_budget(list(rankings[final].index), hours, budget, preselected=arm)
        ev = evaluate(picked, T, hours)
        ev["random_arm_size"] = len(arm)
        ev["random_arm_violation_rate"] = float(T.loc[arm, "is_violator"].mean()) if arm else 0.0
        res["rankings"][f"{final}_with_random_arm"] = ev
        R2 = rankings[final]
        top = R2.loc[picks[final]]
        tv = top.join(T[["is_violator", "recoverable_rp_true"]])
        tv = tv[tv.is_violator & (tv.est_rp > 0)]
        inside = (tv.recoverable_rp_true >= tv.rp_low) & (tv.recoverable_rp_true <= tv.rp_high)
        res["rp_range_coverage"] = float(inside.mean()) if len(tv) else None
        res["fairness"] = fairness(picks[final], F, T).to_dict(orient="records")
        res["final_ranking"] = final
    return res


def summary_table(res: dict) -> pd.DataFrame:
    rows = []
    for name, r in res["rankings"].items():
        rows.append(dict(ranking=name, inspected=round(r["inspected"]), hits=round(r["hits"], 1),
                         precision=round(r["precision"], 3), recall=round(r["recall"], 3),
                         rp_recovered_juta=round(r["rp_recovered"] / 1e6, 1), rp_per_hour_juta=round(r["rp_per_hour"] / 1e6, 3),
                         rp_share=round(r["rp_share"], 3), honest_flag_rate=round(r["honest_flag_rate"], 3),
                         top10_precision=round(r.get("top10_precision", float("nan")), 3)))
    return pd.DataFrame(rows)
