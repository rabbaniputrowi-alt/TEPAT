"""TEPAT scorer. Reads observed/ ONLY (never truth/, never tepat.generator).

Pipeline: entity matching across the two BPJS -> per-employer signals (rules + peer models)
-> Rp estimate (iuran gap, care liability, PHK cover) -> P(violation) from a model trained on
past inspection outcomes -> priority = expected Rp recovered per inspection hour -> reasons + action.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Hamming
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import rules as R

D = R.SCORING_DAY
RECENT = 120                      # a TK worker counts as registered if a KS spell overlaps the last 120 days
TABLES = ["employer_registry", "ks_membership", "tk_exchange", "billing", "care_events", "ks_members",
          "segment_moves", "fktp", "umk", "inspection_cost"]
BANDS = [("2-9", 2, 9), ("10-49", 10, 49), ("50-249", 50, 249), ("250+", 250, 10 ** 9)]

V1_FEATURES = ["f_tk_unreg", "f_tk_unreg_ratio", "f_oss_ratio", "f_bunch_z", "f_level_z", "f_below_umk",
               "f_all_at_umk", "f_spread_z", "f_wage2c", "f_wage2c_ratio", "f_late_ks", "f_partial_ks", "f_arrears", "f_lag_share",
               "f_lag_med", "f_no_ks_joiners", "f_onoff", "f_group_move", "f_rp_v1"]
V2_FEATURES = V1_FEATURES + ["f_piggy", "f_selective", "f_care_susp", "f_clinic", "f_rws", "f_rws_excess",
                             "f_phk_cycle", "f_late_tk", "f_decline", "f_rp_v2"]


def load_observed(branch_dir: str | Path) -> dict[str, pd.DataFrame]:
    d = Path(branch_dir) / "observed"
    return {t: pd.read_parquet(d / f"{t}.parquet") for t in TABLES}


def _band(n: float) -> str:
    for name, lo, hi in BANDS:
        if n <= hi:
            return name
    return "250+"


def _iuran(wage, umk, umkm):
    floor = np.where(np.isin(umkm, R.SMALL_CLASSES), 0.0, umk)
    return R.KS_RATE * np.minimum(np.maximum(wage, floor), R.WAGE_CAP)


def _robust_z(x: pd.Series, groups: pd.Series, invert=False) -> pd.Series:
    med = x.groupby(groups).transform("median")
    mad = (x - med).abs().groupby(groups).transform("median")
    z = (x - med) / (1.4826 * mad + 0.05)
    return -z if invert else z


# --------------------------------------------------------------------------- matching

def match_tk_to_ks(tk: pd.DataFrame, ks: pd.DataFrame) -> pd.DataFrame:
    """Return tk rows with `matched_nik` (KS NIK) or None. Exact NIK first, then fuzzy NIK+name
    within the same employer (handles typos and name variants, look-alike C6)."""
    from rapidfuzz.process import cdist

    tk = tk.reset_index(drop=True).copy()
    ks_set = ks[["employer_id", "nik", "name"]].drop_duplicates(["employer_id", "nik"])
    exact = tk.merge(ks_set[["employer_id", "nik"]].assign(_hit=1), on=["employer_id", "nik"], how="left")
    tk["matched_nik"] = np.where(exact["_hit"].eq(1).to_numpy(), tk["nik"].to_numpy(), None)
    un = tk[tk["matched_nik"].isna()]
    if not len(un):
        return tk
    matched = tk[tk.matched_nik.notna()].groupby("employer_id").matched_nik.apply(set).to_dict()
    cand = {e: g for e, g in ks_set.groupby("employer_id")}
    for eid, rows in un.groupby("employer_id"):
        c = cand.get(eid)
        if c is None:
            continue
        c = c[~c.nik.isin(matched.get(eid, set()))]
        if not len(c):
            continue
        sim_nik = cdist(rows.nik.tolist(), c.nik.tolist(), scorer=Hamming.normalized_similarity)
        sim_name = cdist(rows["name"].tolist(), c["name"].tolist(), scorer=fuzz.token_sort_ratio)
        ok = (sim_nik >= 0.93) & (sim_name >= 80)
        used = set()
        cn = c.nik.to_numpy()
        for i, idx in enumerate(rows.index):
            js = [j for j in np.flatnonzero(ok[i]) if j not in used]
            if js:
                used.add(js[0])
                tk.at[idx, "matched_nik"] = cn[js[0]]
    return tk


# --------------------------------------------------------------------------- features

def build_features(obs: dict[str, pd.DataFrame], lookback: int = 12) -> pd.DataFrame:
    reg = obs["employer_registry"].set_index("employer_id")
    umk = {(r.kabkota, int(r.year)): float(r.umk_rp) for r in obs["umk"].itertuples()}
    hours = dict(zip(obs["inspection_cost"].size_band, obs["inspection_cost"].hours_est))
    ks, tk, bill = obs["ks_membership"].copy(), obs["tk_exchange"], obs["billing"]
    care, mem, moves, fktp = obs["care_events"], obs["ks_members"], obs["segment_moves"], obs["fktp"]
    ks["ppu_end_f"] = ks["ppu_end"].fillna(10 ** 9)
    F = pd.DataFrame(index=reg.index)
    F["sector"], F["kabkota"], F["umkm"] = reg["sector"], reg["kabkota"], reg["skala_usaha"]
    F["umk_now"] = [umk[(k, 2026)] for k in F["kabkota"]]

    ks_now = ks[(ks.ppu_start <= D) & (ks.ppu_end_f > D)]
    ks_recent = ks[(ks.ppu_start <= D) & (ks.ppu_end_f >= D - RECENT)]
    tk_reg = tk[(tk.program != "BPU") & (tk.employer_id.isin(F.index))]
    tk7 = tk_reg[tk_reg.quarter == 7]
    F["ks_n"] = ks_now.groupby("employer_id").size().reindex(F.index, fill_value=0)
    F["tk_n"] = tk7[tk7.program == "regular"].groupby("employer_id").size().reindex(F.index, fill_value=0)
    F["tk_project_n"] = tk7[tk7.program == "project"].groupby("employer_id").size().reindex(F.index, fill_value=0)
    F["oss"] = reg["oss_declared_headcount"]
    F["size_obs"] = F[["ks_n", "tk_n", "oss"]].max(axis=1)
    F["band_obs"] = F["size_obs"].map(_band)
    F["hours"] = F["band_obs"].map(hours)

    # --- mode 1: headcount mismatch after fuzzy matching (TEPAT) and naive exact join (rules baseline)
    m7 = match_tk_to_ks(tk7[tk7.program == "regular"], ks_recent)
    un7 = m7[m7.matched_nik.isna()].copy()
    un7["umk"] = [umk[(F.at[e, "kabkota"], 2026)] for e in un7.employer_id]
    un7["iuran"] = _iuran(un7.wage_reported_tk.to_numpy(), un7.umk.to_numpy(), F.loc[un7.employer_id, "umkm"].to_numpy())
    F["tk_unreg_n"] = un7.groupby("employer_id").size().reindex(F.index, fill_value=0)
    F["rp_headcount"] = un7.groupby("employer_id")["iuran"].sum().reindex(F.index, fill_value=0.0)
    naive = tk7.merge(ks_now[["employer_id", "nik"]].assign(_h=1), on=["employer_id", "nik"], how="left")
    F["naive_unmatched_n"] = naive[naive._h.isna()].groupby("employer_id").size().reindex(F.index, fill_value=0)
    dep = set(mem.loc[mem.segment == "DEPENDENT", "nik"])
    F["piggy_n"] = un7[un7.nik.isin(dep)].groupby("employer_id").size().reindex(F.index, fill_value=0)

    # --- mode 2: wages (peer model on reported wage / UMK)
    kn = ks_now.copy()
    kn["umk"] = [umk[(F.at[e, "kabkota"], int(y))] for e, y in zip(kn.employer_id, kn.wage_year)]
    kn["ratio"] = kn.wage_reported / kn.umk
    kn["at"] = (kn.ratio - 1).abs() < 0.005
    kn["below"] = kn.ratio < 0.995
    kn["lr"] = np.log(kn.ratio)
    g = kn.groupby("employer_id")
    F["share_at_umk"] = g["at"].mean().reindex(F.index)
    F["share_below_umk"] = g["below"].mean().reindex(F.index)
    F["mean_lr"] = g["lr"].mean().reindex(F.index)
    F["wage_sd"] = g["lr"].std().reindex(F.index).fillna(0.0)
    peer = F["sector"] + "|" + F["band_obs"]
    F["bunch_z"] = _robust_z(F["share_at_umk"].fillna(0), peer).where(F.ks_n >= 3, 0.0)
    F["level_z"] = _robust_z(F["mean_lr"].fillna(0), peer, invert=True).where(F.ks_n >= 3, 0.0)
    F["spread_z"] = _robust_z(F["wage_sd"], peer, invert=True).where(F.ks_n >= 5, 0.0)
    F["peer_med_lr"] = F["mean_lr"].groupby(peer).transform("median")
    # 2a fingerprint: everyone exactly at UMK, zero spread (honest UMK payers, C2, still have some above)
    F["all_at_umk"] = ((F.share_at_umk.fillna(0) >= 0.999) & (F.ks_n >= 8)).astype(float)
    pairs = m7[m7.matched_nik.notna()].merge(
        ks_now[["employer_id", "nik", "wage_reported"]].rename(columns={"nik": "matched_nik"}),
        on=["employer_id", "matched_nik"], how="inner")
    pairs["umk"] = [umk[(F.at[e, "kabkota"], 2026)] for e in pairs.employer_id]
    um = F.loc[pairs.employer_id, "umkm"].to_numpy()
    pairs["diff"] = (_iuran(pairs.wage_reported_tk.to_numpy(), pairs.umk.to_numpy(), um)
                     - _iuran(pairs.wage_reported.to_numpy(), pairs.umk.to_numpy(), um))
    gt = pairs[pairs.wage_reported_tk > pairs.wage_reported * 1.02]
    F["wage2c_n"] = gt.groupby("employer_id").size().reindex(F.index, fill_value=0)
    F["rp_wage2c"] = gt.groupby("employer_id")["diff"].sum().reindex(F.index, fill_value=0.0)
    F["wage2c_ratio"] = (F.wage2c_n / pairs.groupby("employer_id").size().reindex(F.index, fill_value=0).clip(lower=1))
    flag_strong = (F.all_at_umk > 0) & (F.wage2c_ratio < 0.2)
    flag_weak = ((F.level_z > 2.5) | ((F.bunch_z > 2.5) & (F.spread_z > 2.0))) & (F.wage2c_ratio < 0.2) & ~flag_strong
    # Peer wage level: sector x size band when the group has >=5 employers, else the whole sector.
    # Employers that look like 2a themselves are excluded from the peer median.
    ok = (F.ks_n >= 3) & (F.all_at_umk == 0)
    lr = F["mean_lr"].where(ok)
    grp_med, grp_n = lr.groupby(peer).transform("median"), lr.groupby(peer).transform("count")
    sec_med = lr.groupby(F["sector"]).transform("median")
    F["peer_lr"] = np.where(grp_n >= 5, grp_med, sec_med)
    uplift = np.maximum(0.05, np.exp(pd.Series(F.peer_lr, index=F.index).fillna(0) - F.mean_lr.fillna(0)) - 1)
    base = F.ks_n * R.KS_RATE * F.umk_now * uplift
    F["rp_wage_peer"] = np.where(flag_strong, 0.8 * base, np.where(flag_weak, 0.4 * base, 0.0))
    F["below_umk_flag"] = ((F.share_below_umk.fillna(0) > 0) & F.umkm.map(R.below_umk_is_violation)).astype(float)

    # --- mode 3: payments per agency (last 12 months)
    b12 = bill[bill.month >= 12].copy()
    b12["late"] = b12.days_late > 0
    b12["partial"] = (b12.billed_rp > 0) & (b12.paid_rp < b12.billed_rp * 0.995)
    b12["short"] = (b12.billed_rp - b12.paid_rp).clip(lower=0)
    for ag in ("KS", "TK"):
        x = b12[(b12.agency == ag) & (b12.billed_rp > 0)].groupby("employer_id")
        F[f"late_{ag.lower()}"] = x.late.mean().reindex(F.index, fill_value=0.0)
        F[f"partial_{ag.lower()}"] = x.partial.mean().reindex(F.index, fill_value=0.0)
    F["rp_arrears"] = b12[b12.agency == "KS"].groupby("employer_id")["short"].sum().reindex(F.index, fill_value=0.0) / 12
    bad_ks = F.late_ks + F.partial_ks
    bad_tk = F.late_tk + F.partial_tk
    F["selective"] = np.where(bad_tk < 0.15, np.maximum(0.0, bad_ks - bad_tk), 0.0)
    n359 = ks[(ks.ppu_start <= 359) & (ks.ppu_end_f > 359)].groupby("employer_id").size().reindex(F.index, fill_value=0)
    F["decline"] = np.where(n359 > 0, 1 - F.ks_n / n359.clip(lower=1), 0.0)
    susp = care[care.coverage_status == "suspended"].groupby("employer_id")
    F["care_susp_n"] = susp.size().reindex(F.index, fill_value=0)
    F["rp_care_liability"] = susp.cost_rp.sum().reindex(F.index, fill_value=0.0)

    # --- stale TK exchange file: recent KS registrations missing from the latest TK snapshot.
    # A quarter-old file makes recent leavers look "still at TK"; shift those employers' snapshots back 90 days.
    rec = ks[(ks.ppu_start >= D - 80) & (ks.ppu_start <= D - 15)][["employer_id", "nik"]].drop_duplicates()
    rec = rec.merge(tk7[["employer_id", "nik"]].drop_duplicates().assign(_t=1), on=["employer_id", "nik"], how="left")
    rec["miss"] = rec._t.isna()
    st = rec.groupby("employer_id").miss.agg(["mean", "size"])
    F["tk_stale"] = ((st["mean"] >= 0.6) & (st["size"] >= 2)).reindex(F.index, fill_value=False).astype(bool)
    tk_reg = tk_reg.assign(eff_day=np.where(tk_reg.employer_id.map(F.tk_stale).fillna(False).to_numpy(),
                                            tk_reg.snap_day - 90, tk_reg.snap_day))

    # --- mode 4: group segment moves while still on the TK roster; company-clinic check
    tk_any = tk_reg[["employer_id", "nik", "eff_day"]]
    mv = moves.merge(tk_any.rename(columns={"employer_id": "from_employer_id"}), on=["from_employer_id", "nik"])
    mv = mv[mv.eff_day > mv.day + 30].drop_duplicates(["from_employer_id", "nik"])
    F["group_move_n"] = mv.groupby("from_employer_id").size().reindex(F.index, fill_value=0)
    own = fktp[fktp.owner_employer_id.notna()].set_index("fktp_id")
    cm = mem[mem.fktp_id.isin(own.index) & mem.segment.isin(["PBPU", "PBI"])].copy()
    cm["employer_id"] = own.loc[cm.fktp_id, "owner_employer_id"].to_numpy()
    cm["open"] = own.loc[cm.fktp_id, "open_to_public"].to_numpy()
    tk_recent = tk_reg[tk_reg.quarter >= 4][["employer_id", "nik"]].drop_duplicates().assign(_t=1)
    cm = cm.merge(tk_recent, on=["employer_id", "nik"], how="left")
    F["clinic_tk_n"] = cm[cm._t.eq(1)].groupby("employer_id").size().reindex(F.index, fill_value=0)
    F["clinic_closed_pbpu_n"] = cm[(~cm.open.astype(bool)) & cm._t.isna()].groupby("employer_id").size().reindex(F.index, fill_value=0)
    F["clinic_score"] = F.clinic_tk_n + 0.5 * F.clinic_closed_pbpu_n

    # --- mode 5: hire lag, on-off-on / early deregistration, register-when-sick, PHK cycling
    tk_first = tk_reg.groupby(["employer_id", "nik"]).tk_start.min().rename("tk_start").reset_index()
    ks_first = ks.groupby(["employer_id", "nik"]).ppu_start.min().rename("ks_start").reset_index()
    hires = tk_first[tk_first.tk_start >= 0].merge(ks_first, on=["employer_id", "nik"], how="left")
    hires["lag"] = hires.ks_start - hires.tk_start
    reg_h = hires[hires.lag.notna()]
    F["lag_share"] = reg_h.groupby("employer_id").lag.apply(lambda s: float(np.mean(s > R.LEGAL_REG_LAG_DAYS))).reindex(F.index, fill_value=0.0)
    F["lag_med"] = reg_h.groupby("employer_id").lag.median().reindex(F.index, fill_value=0.0).clip(-30, 365)
    recent_h = reg_h[reg_h.tk_start >= 360].copy()
    recent_h["excess_m"] = (recent_h.lag - R.LEGAL_REG_LAG_DAYS).clip(lower=0) / 30
    F["rp_lag"] = (recent_h.groupby("employer_id").excess_m.sum().reindex(F.index, fill_value=0.0)
                   * R.KS_RATE * F.umk_now / 12)
    tkj = tk_first[tk_first.tk_start >= 360].groupby("employer_id").size().reindex(F.index, fill_value=0)
    ksj = ks_first[ks_first.ks_start >= 360].groupby("employer_id").size().reindex(F.index, fill_value=0)
    F["no_ks_joiners"] = ((tkj >= 3) & (ksj == 0)).astype(float)

    snaps = tk_reg[["employer_id", "nik", "eff_day"]]
    ksg = ks.sort_values("ppu_start").groupby(["employer_id", "nik"])
    sp = ks.sort_values(["employer_id", "nik", "ppu_start"]).copy()
    sp["next_start"] = ksg.ppu_start.shift(-1).reindex(sp.index)
    ended = sp[(sp.ppu_end.notna()) & (sp.ppu_end <= D - 30)]
    chk = ended.merge(snaps, on=["employer_id", "nik"])
    chk = chk[(chk.eff_day >= chk.ppu_end + 30) & (chk.next_start.isna() | (chk.eff_day < chk.next_start))]
    still = chk.drop_duplicates(["employer_id", "nik", "ppu_start"])
    nonphk = still[still.end_reason != "PHK"]
    F["onoff_n"] = nonphk.groupby("employer_id").size().reindex(F.index, fill_value=0)
    nonphk = nonphk.assign(gap_m=((nonphk.next_start.fillna(D) - nonphk.ppu_end.clip(lower=360)).clip(lower=0) / 30))
    F["rp_onoff"] = nonphk.groupby("employer_id").gap_m.sum().reindex(F.index, fill_value=0.0) * R.KS_RATE * F.umk_now / 12
    phk = sp[(sp.end_reason == "PHK") & sp.ppu_end.notna()]
    phk_re = phk[phk.next_start.notna() & (phk.next_start - phk.ppu_end <= R.PHK_FREE_DAYS)]
    phk_tk = still[still.end_reason == "PHK"]
    cyc = pd.concat([phk_re[["employer_id", "nik", "ppu_end", "next_start"]],
                     phk_tk[["employer_id", "nik", "ppu_end", "next_start"]]]).drop_duplicates(["employer_id", "nik"])
    cyc["gap_m"] = ((cyc.next_start.fillna(D).clip(upper=D) - cyc.ppu_end.clip(lower=360)).clip(lower=0) / 30)
    F["phk_cycle_n"] = cyc.groupby("employer_id").size().reindex(F.index, fill_value=0)
    F["rp_phk_gap"] = cyc.groupby("employer_id").gap_m.sum().reindex(F.index, fill_value=0.0) * R.KS_RATE * F.umk_now / 12
    cc = care[care.coverage_status == "phk_free"].merge(cyc, on=["employer_id", "nik"])
    cc = cc[(cc.day >= cc.ppu_end) & (cc.day < cc.ppu_end + R.PHK_FREE_DAYS)]
    F["rp_phk_cover"] = cc.groupby("employer_id").cost_rp.sum().reindex(F.index, fill_value=0.0)

    inp = care[care.setting == "inpatient"][["nik", "employer_id", "day", "cost_rp"]]
    sp_new = sp[sp.ppu_start >= 0].merge(tk_first, on=["employer_id", "nik"])
    sp_new = sp_new[sp_new.tk_start <= sp_new.ppu_start - 60]
    rws = sp_new.merge(inp, on=["employer_id", "nik"])
    rws = rws[(rws.day >= rws.ppu_start) & (rws.day <= rws.ppu_start + 14)].drop_duplicates(["employer_id", "nik"])
    F["rws_n"] = rws.groupby("employer_id").size().reindex(F.index, fill_value=0)
    F["rws_claim_rp"] = rws.groupby("employer_id").cost_rp.sum().reindex(F.index, fill_value=0.0)
    F["rws_lead_days"] = (rws.ppu_start - rws.tk_start).groupby(rws.employer_id).median().reindex(F.index, fill_value=0.0)
    base14 = 0.004 * 14 / 30 * 1.5
    F["rws_excess"] = F.rws_n - sp_new.groupby("employer_id").size().reindex(F.index, fill_value=0) * base14

    # --- Rp estimates
    F["rp_clinic"] = F.clinic_closed_pbpu_n * R.KS_RATE * F.umk_now
    F["rp_group_move"] = F.group_move_n * R.KS_RATE * F.umk_now
    v1_parts = ["rp_headcount", "rp_wage2c", "rp_wage_peer", "rp_arrears", "rp_lag", "rp_onoff", "rp_group_move"]
    F["rp_month_v1"] = F[v1_parts].sum(axis=1)
    F["rp_month_v2"] = F["rp_month_v1"] + F["rp_clinic"] + F["rp_phk_gap"]
    F["est_recoverable_v1"] = F.rp_month_v1 * lookback
    F["est_recoverable_v2"] = F.rp_month_v2 * lookback + F.rp_care_liability + F.rp_phk_cover

    # --- model inputs
    l1 = np.log1p
    F["f_tk_unreg"], F["f_tk_unreg_ratio"] = l1(F.tk_unreg_n), F.tk_unreg_n / F.tk_n.clip(lower=1)
    F["f_oss_ratio"] = np.log((F.oss + 1) / (F[["ks_n", "tk_n"]].max(axis=1) + 1))
    F["f_bunch_z"], F["f_level_z"] = F.bunch_z.clip(-5, 10), F.level_z.clip(-5, 10)
    F["f_all_at_umk"], F["f_spread_z"] = F.all_at_umk, F.spread_z.clip(-5, 10)
    F["f_below_umk"], F["f_wage2c"], F["f_wage2c_ratio"] = F.below_umk_flag, l1(F.wage2c_n), F.wage2c_ratio
    F["f_late_ks"], F["f_partial_ks"], F["f_arrears"] = F.late_ks, F.partial_ks, l1(F.rp_arrears)
    F["f_lag_share"], F["f_lag_med"] = F.lag_share, F.lag_med / 30
    F["f_no_ks_joiners"], F["f_onoff"], F["f_group_move"] = F.no_ks_joiners, l1(F.onoff_n), l1(F.group_move_n)
    F["f_rp_v1"], F["f_rp_v2"] = l1(F.rp_month_v1), l1(F.rp_month_v2)
    F["f_piggy"], F["f_selective"], F["f_care_susp"] = l1(F.piggy_n), F.selective, l1(F.care_susp_n)
    F["f_clinic"], F["f_rws"], F["f_rws_excess"] = l1(F.clinic_score), l1(F.rws_n), F.rws_excess.clip(-2, 20)
    F["f_phk_cycle"], F["f_late_tk"], F["f_decline"] = l1(F.phk_cycle_n), F.late_tk, F.decline.clip(-1, 1)
    return F


# --------------------------------------------------------------------------- model + ranking

class TepatModel:
    """Combines signals into P(violation). Trained on past inspection outcomes (branch A sample)."""

    def __init__(self, features: list[str]):
        self.features = features
        self.clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000))
        self.prior_by_band: dict[str, float] = {}
        self.range_q: tuple[float, float] = (0.5, 2.0)

    def fit(self, F: pd.DataFrame, y: pd.Series, recoverable: pd.Series, est_col: str):
        X = F.loc[y.index, self.features].fillna(0.0)
        self.clf.fit(X, y.astype(int))
        viol = y[y].index
        self.prior_by_band = recoverable.loc[viol].groupby(F.loc[viol, "band_obs"]).median().to_dict()
        est = F.loc[viol, est_col]
        ok = est > 0
        ratio = recoverable.loc[viol][ok] / est[ok]
        if len(ratio) >= 10:
            self.range_q = (float(ratio.quantile(0.10)), float(ratio.quantile(0.90)))
        return self

    def predict(self, F: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(F[self.features].fillna(0.0))[:, 1]


def rank(F: pd.DataFrame, model: TepatModel, est_col: str, tag: str) -> pd.DataFrame:
    out = pd.DataFrame(index=F.index)
    p = model.predict(F)
    prior = F["band_obs"].map(model.prior_by_band).fillna(0.0)
    est = F[est_col]
    out["p_violation"] = p
    out["est_rp"] = est
    out["expected_rp"] = p * np.maximum(est, 0.3 * prior)
    out["priority"] = out.expected_rp / F["hours"]
    out["rp_low"], out["rp_high"] = est * model.range_q[0], est * model.range_q[1]
    out["ranking"] = tag
    return out.sort_values("priority", ascending=False)


def rules_only(F: pd.DataFrame) -> pd.DataFrame:
    """Honest strong baseline (Synthetic Data Spec §7): count of plain rule hits, tie-break by size.
    Rules: TK headcount > KS headcount; wage < UMK (medium/large only, E05c); KS iuran overdue."""
    tk_total = F.tk_n + F.tk_project_n
    hits = ((tk_total > F.ks_n).astype(int) + F.below_umk_flag.astype(int)
            + ((F.late_ks > 0.3) | (F.partial_ks > 0)).astype(int))
    out = pd.DataFrame({"priority": hits * 1e6 + F.size_obs, "rule_hits": hits}, index=F.index)
    out["ranking"] = "rules_only"
    return out.sort_values("priority", ascending=False)


def biggest_first(F: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"priority": F.size_obs.astype(float), "ranking": "biggest_first"}, index=F.index).sort_values("priority", ascending=False)


# --------------------------------------------------------------------------- explanations

def _rp(x: float) -> str:
    return "Rp" + f"{x:,.0f}".replace(",", ".")


def reasons(r: pd.Series) -> list[tuple[float, str, str]]:
    """(weight, text, legal basis) in plain Indonesian, strongest first."""
    out = []
    if r.rws_n > 0:
        out.append((1e12 + r.rws_claim_rp, f"{int(r.rws_n)} pekerja baru didaftarkan ke BPJS Kesehatan ≤14 hari sebelum rawat inap, "
                    f"padahal BPJS TK mencatat mereka bekerja sejak ±{int(r.rws_lead_days // 30)} bulan sebelumnya",
                    "Perpres 82/2018 Ps. 13; PP 86/2013 Ps. 3(4)"))
    if r.rp_care_liability > 0:
        out.append((1e11 + r.rp_care_liability, f"{_rp(r.rp_care_liability)} biaya perawatan {int(r.care_susp_n)} kali layanan saat kepesertaan "
                    "pekerja nonaktif akibat tunggakan iuran: menjadi tanggung jawab pemberi kerja", "Perpres 82/2018 Ps. 42(2) jo. Perpres 59/2024"))
    if r.selective > 0.25:
        out.append((r.rp_arrears * 12 + 1e6, f"Membayar BPJS Ketenagakerjaan tepat waktu, tetapi iuran BPJS Kesehatan terlambat/kurang "
                    f"pada {round((r.late_ks + r.partial_ks) * 12)} dari 12 bulan terakhir", "UU 24/2011 Ps. 19 & 55"))
    if r.piggy_n > 0:
        out.append((r.piggy_n * 1e6 + 1e5, f"{int(r.piggy_n)} pekerja aktif di BPJS TK hanya tercatat sebagai tanggungan (suami/istri) "
                    "peserta lain di BPJS Kesehatan", "Perpres 82/2018 Ps. 14(1)"))
    if r.tk_unreg_n > 0:
        out.append((r.rp_headcount * 12, f"{int(r.tk_unreg_n)} dari {int(r.tk_n)} pekerja di BPJS Ketenagakerjaan tidak terdaftar di "
                    "BPJS Kesehatan (setelah pencocokan NIK dan nama)", "UU 24/2011 Ps. 15; Perpres 82/2018 Ps. 13(1)"))
    if r.phk_cycle_n > 0:
        out.append((r.rp_phk_cover + 1e6 * r.phk_cycle_n, f"{int(r.phk_cycle_n)} pekerja dilaporkan PHK di BPJS Kesehatan tetapi tetap aktif di "
                    "BPJS TK atau didaftarkan ulang ≤6 bulan", "Perpres 59/2024 Ps. 27"))
    if r.clinic_score > 0:
        out.append((r.rp_clinic * 12 + 1e6 * r.clinic_tk_n, f"{int(r.clinic_tk_n + r.clinic_closed_pbpu_n)} peserta mandiri/PBI terdaftar di klinik "
                    "milik perusahaan ini" + (f"; {int(r.clinic_tk_n)} di antaranya tercatat sebagai pekerja di BPJS TK" if r.clinic_tk_n else ""),
                    "Perpres 82/2018 Ps. 13(1)"))
    if r.wage2c_n > 0:
        out.append((r.rp_wage2c * 12, f"{int(r.wage2c_n)} pekerja: upah yang dilaporkan ke BPJS TK lebih tinggi dari ke BPJS Kesehatan",
                    "PP 86/2013 Ps. 3(2)b"))
    if r.rp_wage_peer > 0:
        out.append((r.rp_wage_peer * 12, f"{r.share_at_umk:.0%} upah dilaporkan tepat di UMK atau di bawah pola sektor sejenis "
                    "(sektor, skala usaha)", "PP 86/2013 Ps. 3(2)b"))
    if r.lag_share > 0.3 and r.lag_med > R.LEGAL_REG_LAG_DAYS:
        out.append((r.rp_lag * 12, f"Median keterlambatan pendaftaran pekerja baru {int(r.lag_med)} hari (batas 7 hari kerja)",
                    "PP 86/2013 Ps. 3(4)"))
    if r.no_ks_joiners:
        out.append((5e6, "Tidak ada pendaftaran pekerja baru ke BPJS Kesehatan selama 12 bulan, padahal BPJS TK mencatat pekerja baru",
                    "PP 86/2013 Ps. 3(4)"))
    if r.onoff_n > 0:
        out.append((r.rp_onoff * 12, f"{int(r.onoff_n)} pekerja dinonaktifkan di BPJS Kesehatan padahal masih tercatat bekerja di BPJS TK",
                    "PP 86/2013 Ps. 3(3)"))
    if r.group_move_n > 0:
        out.append((r.rp_group_move * 12, f"{int(r.group_move_n)} pekerja pindah ke segmen mandiri/PBI padahal masih tercatat bekerja",
                    "Perpres 82/2018 Ps. 13"))
    if r.rp_arrears > 0 and r.selective <= 0.25:
        out.append((r.rp_arrears * 12, f"Tunggakan iuran BPJS Kesehatan rata-rata {_rp(r.rp_arrears)}/bulan", "Perpres 82/2018 Ps. 42"))
    return sorted(out, key=lambda t: -t[0])


def action(r: pd.Series, tier: str) -> tuple[str, str]:
    cant_pay = (r.late_ks + r.partial_ks > 0.3) and (r.late_tk + r.partial_tk > 0.3) and r.decline > 0.1
    pay_label = ("Kemungkinan tidak mampu bayar: tawarkan cicilan, bukan penindakan" if cant_pay else
                 "Indikasi tidak mau bayar (bayar BPJS TK, tidak bayar BPJS Kes): pertimbangkan Ps. 55 UU 24/2011" if r.selective > 0.25 else "")
    strong = r.rws_n > 0 or r.rp_care_liability > 0 or r.selective > 0.25 or r.clinic_tk_n > 0
    if tier == "Tinggi":
        act = "Pemeriksaan lapangan" if strong else "Pemeriksaan bersama BPJS Ketenagakerjaan"
    elif tier == "Sedang":
        act = "Surat kepatuhan (kesempatan koreksi mandiri)"
    else:
        act = "Jalur hijau (prioritas pemeriksaan rendah)" if not reasons(r) else "Pantau"
    return act, pay_label


def brief(r: pd.Series) -> list[str]:
    asks = []
    if r.tk_unreg_n or r.piggy_n or r.clinic_score:
        asks.append("Daftar pekerja aktif (slip gaji, absensi) untuk dicocokkan dengan data kepesertaan")
    if r.rws_n or r.lag_share > 0.3 or r.no_ks_joiners:
        asks.append("Tanggal mulai kerja pekerja baru (kontrak/SK pengangkatan) dan bukti pendaftaran")
    if r.wage2c_n or r.rp_wage_peer:
        asks.append("Rekap upah per pekerja (upah pokok + tunjangan tetap) 12 bulan terakhir")
    if r.rp_arrears or r.selective > 0.25 or r.rp_care_liability:
        asks.append("Bukti pemotongan iuran 1% dari gaji pekerja dan bukti setor ke BPJS Kesehatan")
    if r.phk_cycle_n or r.onoff_n:
        asks.append("Dokumen PHK (bukti lapor Disnaker/perjanjian bersama) dan kontrak kerja lanjutan")
    return asks or ["Tidak ada permintaan dokumen khusus"]


def explain(F: pd.DataFrame, ranked: pd.DataFrame) -> pd.DataFrame:
    out = ranked.copy()
    n = len(out)
    pos = np.arange(n)
    out["rank"] = pos + 1
    out["tier"] = np.where(pos < 0.10 * n, "Tinggi", np.where(pos < 0.30 * n, "Sedang", "Rendah"))
    rs, acts, pays, briefs = [], [], [], []
    for eid, row in out.iterrows():
        r = F.loc[eid]
        rr = reasons(r)
        rs.append(" | ".join(t for _, t, _ in rr[:3]))
        a, pl = action(r, row.tier)
        acts.append(a)
        pays.append(pl)
        briefs.append(" | ".join(brief(r)))
    out["top_reasons"], out["action"], out["pay_label"], out["brief"] = rs, acts, pays, briefs
    return out
