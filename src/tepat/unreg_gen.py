"""Synthetic world for employers that never registered with BPJS Kesehatan (mode 1, variants 1e and 1f).

Reads the already-generated branch (registry NIBs, KS membership, UMK) and adds:
  observed/oss_registry.parquet         every NIB holder in the area: registered employers + unregistered candidates
  observed/tk_unregistered.parquet      BPJS TK worker snapshot for TK-registered businesses absent from the KS registry
  truth/unregistered_truth.parquet      who is obliged, true workers, true Rp gap (only the benchmark reads this)

It uses its own RNG stream and never rewrites the main branch files, so the main benchmark is unchanged.
All companies, people and NIKs are synthetic (NIK prefix 99).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import rules as R
from .generator import company_name, name_variant, rand_name, typo

ROOT = Path(__file__).resolve().parents[2]
UCFG = ROOT / "config" / "unregistered.yaml"
BANDS = {"2-9": (2, 9), "10-49": (10, 49), "50-249": (50, 249)}
SNAP_DAY = R.SCORING_DAY - 15


def load_ucfg(path: str | Path = UCFG) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _band_of(n: float) -> str:
    return "2-9" if n < 10 else "10-49" if n < 50 else "50-249" if n < 250 else "250+"


def _umkm_of(n: float) -> str:
    return {"2-9": "micro", "10-49": "small", "50-249": "medium", "250+": "large"}[_band_of(max(n, 1))]


def _draw_size(rng, shares: dict) -> int:
    keys = list(shares)
    p = np.array([shares[k] for k in keys], float)
    lo, hi = BANDS[keys[int(rng.choice(len(keys), p=p / p.sum()))]]
    return int(min(max(int(np.exp(rng.uniform(np.log(lo), np.log(hi + 1)))), lo), hi))


def generate_unregistered(branch_dir: str | Path, branch: str, seed: int, ucfg: dict | None = None) -> dict:
    u = ucfg or load_ucfg()
    bd = Path(branch_dir)
    obs, tru = bd / "observed", bd / "truth"
    import json
    b = json.loads((bd / "config.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(int(seed) * 31 + int(u["seed_offset"]) + ord(branch))

    reg = pd.read_parquet(obs / "employer_registry.parquet")
    ks = pd.read_parquet(obs / "ks_membership.parquet")
    umk = {k["name"]: float(k["umk"][str(R.year_of_day(R.SCORING_DAY))] if str(R.year_of_day(R.SCORING_DAY)) in k["umk"]
                            else k["umk"][R.year_of_day(R.SCORING_DAY)]) for k in b["kabkota"]}
    active = ks[ks.ppu_end.isna() | (ks.ppu_end >= R.SCORING_DAY)]
    covered_pool = active[["nik", "name", "employer_id"]].drop_duplicates("nik").reset_index(drop=True)

    kinds = dict(u["kinds"])
    scale = float(u["branches"].get(branch, {}).get("lookalike_scale", 1.0))
    for k in kinds:
        if k.startswith("C"):
            kinds[k] *= scale
    kk = list(kinds)
    kp = np.array([kinds[k] for k in kk], float)
    kp /= kp.sum()

    codes = list(b["sectors"])
    sw = np.array([b["sectors"][c]["weight"] for c in codes], float)
    sw /= sw.sum()
    kabs = [k["name"] for k in b["kabkota"]]
    used_nibs = set(reg.nib.astype(str))
    seen_nik = set(ks.nik.astype(str))

    def new_nib() -> str:
        while True:
            n = "".join(str(d) for d in rng.integers(0, 10, 13))
            if n not in used_nibs:
                used_nibs.add(n)
                return n

    def new_nik() -> str:
        while True:
            n = "99" + str(int(rng.integers(0, 10 ** 14))).zfill(14)
            if n not in seen_nik:
                seen_nik.add(n)
                return n

    oss_rows, tk_rows, truth_rows = [], [], []
    for i in range(int(u["n_candidates"])):
        kind = str(rng.choice(kk, p=kp))
        sec = str(rng.choice(codes, p=sw))
        kab = kabs[int(rng.integers(len(kabs)))]
        nib = new_nib()
        mu, sig = b["sectors"][sec]["wage_mu"], b["sectors"][sec]["wage_sigma"]
        has_npwp = bool(rng.random() < u["npwp_rate"][kind])
        workers, tk_workers, gap_workers = 0, [], 0
        wages = []

        if kind in ("U1_tk_only", "U2_oss_only"):
            workers = _draw_size(rng, u["violator_size_bands"])
            m = np.exp(rng.normal(mu, sig, workers))
            wages = list(umk[kab] * m)
            gap_mask = np.ones(workers, bool)
            if kind == "U1_tk_only":
                n_tk = max(2, int(round(workers * rng.uniform(*u["tk_coverage"]))))
                n_tk = min(n_tk, workers)
                moon = float(rng.uniform(*u["moonlighter_share"]))
                for j in range(n_tk):
                    if rng.random() < moon and len(covered_pool):
                        p = covered_pool.iloc[int(rng.integers(len(covered_pool)))]
                        nik, name = str(p.nik), str(p["name"])
                        gap_mask[j] = False                       # already PPU elsewhere: no gap for this worker
                    else:
                        nik, name = new_nik(), rand_name(rng)
                    tk_workers.append((nik, name, wages[j]))
            declared = max(1, int(round(workers * np.exp(rng.normal(0, u["oss_sigma"])))))
            base = np.minimum(np.array(wages), R.WAGE_CAP)
            gap_month = float((base * R.KS_RATE)[gap_mask].sum())
            gap_workers = int(gap_mask.sum())
        elif kind == "C15_exempt_micro":
            workers = 1
            declared = int(rng.integers(2, 4)) if rng.random() < u["exempt_overstate"] else 1
            if rng.random() < u["exempt_tk_rate"]:
                tk_workers.append((new_nik(), rand_name(rng), umk[kab] * float(np.exp(rng.normal(mu, sig)))))
            gap_month = 0.0
        elif kind == "C16_dormant":
            workers = 0
            declared = int(rng.integers(u["dormant_declared"][0], u["dormant_declared"][1] + 1))
            gap_month = 0.0
        elif kind == "C18_closed_after_snapshot":
            # closed after the last quarterly TK snapshot: TK still lists its former workers, nothing left to recover
            n_tk = _draw_size(rng, u["violator_size_bands"])
            for _ in range(n_tk):
                tk_workers.append((new_nik(), rand_name(rng), umk[kab] * float(np.exp(rng.normal(mu, sig)))))
            declared = max(1, int(round(n_tk * np.exp(rng.normal(0, u["oss_sigma"])))))
            gap_month = 0.0
        else:  # C17: subsidiary whose workers are PPU under a related registered employer
            workers = _draw_size(rng, u["violator_size_bands"])
            stale = float(rng.uniform(*u["group_stale_share"]))
            idx = rng.choice(len(covered_pool), size=min(workers, len(covered_pool)), replace=False)
            for j in idx:
                p = covered_pool.iloc[int(j)]
                nik = str(p.nik) if rng.random() >= stale else new_nik()   # stale: left the parent, TK not updated
                tk_workers.append((nik, str(p["name"]), umk[kab] * float(np.exp(rng.normal(mu, sig)))))
            declared = max(1, int(round(workers * np.exp(rng.normal(0, u["oss_sigma"])))))
            gap_month = 0.0

        name = company_name(rng, _umkm_of(declared))
        oss_rows.append(dict(nib=nib, name=name, sector=sec, kabkota=kab, skala_usaha=_umkm_of(declared),
                             oss_declared_headcount=declared, has_npwp=has_npwp,
                             nib_issued_day=-int(rng.integers(30, 2500))))
        for nik, wname, wage in tk_workers:
            obs_nik = typo(nik, rng) if rng.random() < u["nik_typo"] else nik
            obs_name = name_variant(wname, rng) if rng.random() < u["name_variant"] else wname
            tk_rows.append(dict(tk_employer_nib=nib, snap_day=SNAP_DAY, nik=obs_nik, name=obs_name,
                                wage_reported_tk=round(float(wage), -3),
                                tk_start=-int(rng.integers(30, 900))))
        is_v = kind in ("U1_tk_only", "U2_oss_only")
        truth_rows.append(dict(nib=nib, kind=kind, is_violator=is_v,
                               variants=("1e" if kind == "U1_tk_only" else "1f" if kind == "U2_oss_only" else ""),
                               lookalikes=("" if is_v else kind.split("_")[0]),
                               true_workers=workers, gap_workers=gap_workers, has_npwp_true=has_npwp,
                               gap_rp_month_true=gap_month,
                               recoverable_rp_true=gap_month * float(b["inspection"]["lookback_months"])))

    # registered employers are NIB holders too: the pass must find the unregistered ones among all of them
    reg_rows = reg.assign(has_npwp=True, nib_issued_day=reg.registered_since - 30)[
        ["nib", "name", "sector", "kabkota", "skala_usaha", "oss_declared_headcount", "has_npwp", "nib_issued_day"]]
    oss = pd.concat([reg_rows, pd.DataFrame(oss_rows)], ignore_index=True)
    oss = oss.sample(frac=1.0, random_state=int(rng.integers(2 ** 31))).reset_index(drop=True)
    oss.to_parquet(obs / "oss_registry.parquet", index=False)
    pd.DataFrame(tk_rows, columns=["tk_employer_nib", "snap_day", "nik", "name", "wage_reported_tk", "tk_start"]).to_parquet(
        obs / "tk_unregistered.parquet", index=False)
    T = pd.DataFrame(truth_rows)
    T.to_parquet(tru / "unregistered_truth.parquet", index=False)
    return dict(candidates=len(T), violators=int(T.is_violator.sum()), tk_rows=len(tk_rows),
                by_kind=T.kind.value_counts().to_dict())
