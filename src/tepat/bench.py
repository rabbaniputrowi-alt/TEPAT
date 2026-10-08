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


def fairness(picked: list[str], F: pd.DataFrame, T: pd.DataFrame) -> pd.DataFrame:
    df = F[["sector", "band_obs"]].join(T[["is_violator"]])
    df["picked"] = df.index.isin(picked)
    rows = []
    for col in ("sector", "band_obs"):
        g = df.groupby(col)
        rows.append(pd.DataFrame({"group": col, "value": g.size().index, "n": g.size().values,
                                  "violation_rate": g.is_violator.mean().values, "pick_rate": g.picked.mean().values,
                                  "honest_flag_rate": g.apply(lambda x: (x.picked & ~x.is_violator).sum() / max(1, (~x.is_violator).sum())).values}))
    return pd.concat(rows, ignore_index=True)


def run(F: pd.DataFrame, T: pd.DataFrame, rankings: dict[str, pd.DataFrame], cfg: dict, seed: int) -> dict:
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
    # TEPAT v2 with an 8% random exploration arm (I-56)
    if "tepat_v2" in rankings:
        arm_budget = cfg["inspection"]["random_arm_frac"] * budget
        arm = stratified_random(F, hours, arm_budget, rng)
        picked = select_by_budget(list(rankings["tepat_v2"].index), hours, budget, preselected=arm)
        ev = evaluate(picked, T, hours)
        ev["random_arm_size"] = len(arm)
        ev["random_arm_violation_rate"] = float(T.loc[arm, "is_violator"].mean()) if arm else 0.0
        res["rankings"]["tepat_v2_with_random_arm"] = ev
        R2 = rankings["tepat_v2"]
        top = R2.loc[picks["tepat_v2"]]
        tv = top.join(T[["is_violator", "recoverable_rp_true"]])
        tv = tv[tv.is_violator & (tv.est_rp > 0)]
        inside = (tv.recoverable_rp_true >= tv.rp_low) & (tv.recoverable_rp_true <= tv.rp_high)
        res["rp_range_coverage_tepat_v2"] = float(inside.mean()) if len(tv) else None
        res["fairness_tepat_v2"] = fairness(picks["tepat_v2"], F, T).to_dict(orient="records")
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
