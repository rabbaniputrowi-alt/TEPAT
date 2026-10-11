"""TEPAT registration pass: find businesses that should be registered with BPJS Kesehatan but aren't.

Reads observed data only (the same wall as scorer.py; tests/test_unregistered.py enforces it):
  oss_registry        NIB holders in the area (OSS), with declared headcount and NPWP
  employer_registry   employers already registered with BPJS Kesehatan
  tk_unregistered     BPJS TK worker snapshot for TK-registered businesses not in the KS registry (PKS exchange)
  ks_membership       to see whether a TK worker is already PPU under another employer

Legal rule (E21): a business with an NIB, an NPWP and at least 2 workers must register.
Priority = P(obliged and unregistered) x estimated Rp / inspection hours, as in the main queue.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import rules as R

FEATURES = ["on_tk", "log_tk_uncov", "cov_share", "log_oss_n", "has_npwp", "oss_ge2", "rule_obliged", "nib_age_y"]
HOURS = {"micro": 3, "small": 6, "medium": 12, "large": 24}


def load(branch_dir: str | Path) -> dict[str, pd.DataFrame]:
    o = Path(branch_dir) / "observed"
    names = ["oss_registry", "employer_registry", "tk_unregistered", "ks_membership", "umk", "inspection_cost"]
    return {n: pd.read_parquet(o / f"{n}.parquet") for n in names}


def build_features(obs: dict[str, pd.DataFrame], lookback: int = 12) -> pd.DataFrame:
    oss, reg = obs["oss_registry"], obs["employer_registry"]
    cand = oss[~oss.nib.astype(str).isin(set(reg.nib.astype(str)))].copy()
    cand["nib"] = cand.nib.astype(str)
    F = cand.set_index("nib")

    ks = obs["ks_membership"]
    active = set(ks.loc[ks.ppu_end.isna() | (ks.ppu_end >= R.SCORING_DAY), "nik"].astype(str))
    tk = obs["tk_unregistered"].copy()
    tk["covered"] = tk.nik.astype(str).isin(active)
    g = tk.groupby(tk.tk_employer_nib.astype(str))
    F["tk_n"] = g.size().reindex(F.index, fill_value=0)
    F["tk_cov"] = g.covered.sum().reindex(F.index, fill_value=0)
    F["tk_uncov"] = F.tk_n - F.tk_cov
    F["on_tk"] = (F.tk_n > 0).astype(float)
    F["cov_share"] = np.where(F.tk_n > 0, F.tk_cov / F.tk_n.clip(lower=1), 0.0)
    F["log_tk_uncov"] = np.log1p(F.tk_uncov)
    F["log_oss_n"] = np.log1p(F.oss_declared_headcount)
    F["has_npwp"] = F.has_npwp.astype(float)
    F["oss_ge2"] = (F.oss_declared_headcount >= 2).astype(float)
    F["est_workers"] = np.where(F.on_tk > 0, F.tk_uncov, F.oss_declared_headcount).astype(float)
    F["rule_obliged"] = ((F.has_npwp > 0) & (F.est_workers >= 2)).astype(float)
    F["nib_age_y"] = (R.SCORING_DAY - F.nib_issued_day) / 365.0

    u = obs["umk"]
    yr = R.year_of_day(R.SCORING_DAY)
    umk_now = u[u.year == yr].set_index("kabkota").umk_rp
    F["umk_now"] = F.kabkota.map(umk_now).astype(float)
    # Wage unknown for an unregistered business: UMK is the conservative proxy (labelled as such in the reasons)
    F["est_rp"] = F.est_workers * R.KS_RATE * F.umk_now * lookback
    F["hours"] = F.skala_usaha.map(HOURS).fillna(3).astype(float)
    return F


class RegistrationModel:
    """P(obliged and unregistered), trained on branch A's sampled outcomes like the main model."""

    def __init__(self):
        self.clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000))

    def fit(self, F: pd.DataFrame, y: pd.Series):
        self.clf.fit(F.loc[y.index, FEATURES].fillna(0.0), y.astype(int))
        return self

    def predict(self, F: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(F[FEATURES].fillna(0.0))[:, 1]


def rank(F: pd.DataFrame, model: RegistrationModel) -> pd.DataFrame:
    out = pd.DataFrame(index=F.index)
    out["p_unregistered"] = model.predict(F)
    out["est_rp"] = F.est_rp
    out["priority"] = out.p_unregistered * F.est_rp.clip(lower=0) / F.hours
    out["ranking"] = "tepat_registration"
    return out.sort_values("priority", ascending=False)


def rules_only(F: pd.DataFrame) -> pd.DataFrame:
    """Strong baseline: the legal rule from both sources (TK workers or OSS headcount, plus NPWP), biggest first."""
    size = np.maximum(F.tk_n, F.oss_declared_headcount)
    hit = ((F.has_npwp > 0) & (size >= 2)).astype(int)
    out = pd.DataFrame({"priority": hit * 1e6 + size, "ranking": "rules_only"}, index=F.index)
    return out.sort_values("priority", ascending=False)


def oss_only(F: pd.DataFrame) -> pd.DataFrame:
    """What a plain OSS/NIB match gives: NIB + NPWP + declared >=2, biggest declared first."""
    hit = ((F.has_npwp > 0) & (F.oss_declared_headcount >= 2)).astype(int)
    out = pd.DataFrame({"priority": hit * 1e6 + F.oss_declared_headcount, "ranking": "oss_only"}, index=F.index)
    return out.sort_values("priority", ascending=False)


def _rp(x: float) -> str:
    return "Rp" + f"{x:,.0f}".replace(",", ".")


def reasons(r: pd.Series) -> list[str]:
    out = []
    if r.on_tk > 0:
        out.append(f"Terdaftar di BPJS Ketenagakerjaan dengan {int(r.tk_n)} pekerja aktif, tetapi belum terdaftar "
                   "sebagai pemberi kerja di BPJS Kesehatan")
        if r.tk_cov > 0:
            out.append(f"{int(r.tk_cov)} dari {int(r.tk_n)} pekerja itu sudah menjadi peserta PPU lewat pemberi kerja lain")
    if r.has_npwp > 0 and r.oss_declared_headcount >= 2:
        out.append(f"OSS: memiliki NIB dan NPWP, menyatakan {int(r.oss_declared_headcount)} pekerja (wajib mendaftar bila minimal 2 pekerja)")
    elif r.has_npwp <= 0:
        out.append("OSS: belum tercatat memiliki NPWP")
    else:
        out.append(f"OSS: menyatakan {int(r.oss_declared_headcount)} pekerja (kurang dari 2)")
    return out


def action(r: pd.Series) -> str:
    if r.rule_obliged <= 0:
        return "Tidak wajib menurut data saat ini (kurang dari 2 pekerja atau tanpa NPWP)"
    if r.on_tk > 0:
        return "Pemeriksaan bersama BPJS Ketenagakerjaan"
    return "Surat ajakan pendaftaran (kesempatan mendaftar mandiri)"


def explain(F: pd.DataFrame, ranked: pd.DataFrame) -> pd.DataFrame:
    out = ranked.join(F[["name", "sector", "kabkota", "skala_usaha", "oss_declared_headcount", "has_npwp",
                         "tk_n", "tk_cov", "est_workers", "hours"]])
    out["rank"] = np.arange(len(out)) + 1
    out["reasons"] = [" | ".join(reasons(F.loc[i])) for i in out.index]
    out["action"] = [action(F.loc[i]) for i in out.index]
    out["legal_basis"] = "UU 24/2011 Ps. 15; Perpres 82/2018 Ps. 13 ayat (1); PP 86/2013 Ps. 5 s.d. 10"
    return out
