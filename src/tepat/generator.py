"""Synthetic BPJS branch generator.

Writes two worlds per branch:
  truth/     real payroll, planted violations, true Rp gaps (only the benchmark reads this)
  observed/  only what BPJS Kesehatan would plausibly hold or receive via the TK exchange

The scorer (tepat.scorer) must never import this module. tests/test_wall.py enforces that.
All people and companies are synthetic. NIKs start with "99", which is not a real province code.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import rules as R

FIRST = ("Agus Budi Siti Dewi Rina Andi Eko Sri Wahyu Nur Fitri Hendra Dian Rizky Putri Ahmad "
         "Muhammad Yusuf Indah Lestari Bayu Ratna Joko Teguh Ayu Fajar Novi Arif Sari Dimas Galih "
         "Wulan Hadi Rudi Tri Yuli Asep Ujang Made Ketut Nyoman Wayan Irfan Reza Taufik Anisa Rahma "
         "Lukman Bambang Slamet").split()
LAST = ("Santoso Wijaya Pratama Saputra Hidayat Kurniawan Setiawan Lestari Rahayu Susanto Nugroho "
        "Wibowo Purnomo Siregar Nasution Simanjuntak Harahap Sitompul Pangaribuan Hutapea Gunawan "
        "Halim Utomo Sulistyo Firmansyah Ramadhan Maulana Syahputra Permana Hakim Fauzi Anggraini "
        "Puspita Handayani Mulyani Kusuma Prasetyo Sutrisno Hartono Suryadi").split()
CO_WORDS = ("Sinar Maju Jaya Abadi Makmur Sentosa Karya Mandiri Bumi Nusantara Prima Sejahtera "
            "Cahaya Mulia Harapan Baru Tunas Gemilang Lestari Agung").split()


# --------------------------------------------------------------------------- config

def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def branch_config(cfg: dict, branch: str) -> dict:
    b = copy.deepcopy(cfg)
    over = cfg["branches"][branch]
    b["branch"] = branch
    b["seed"] = int(cfg["seed"]) * 10 + int(over.get("seed_offset", 0))
    b["violations"]["held_back"] = list(over.get("held_back", []))
    if "sector_weights" in over:
        for k, v in over["sector_weights"].items():
            b["sectors"][k]["weight"] = v
    scale = float(over.get("lookalike_scale", 1.0))
    for k in b["lookalikes"]:
        b["lookalikes"][k] = min(0.95, b["lookalikes"][k] * scale)
    return b


# --------------------------------------------------------------------------- entities

@dataclass
class Worker:
    wid: int
    eid: str
    nik: str
    name: str
    age: int
    start: int
    end: int | None
    end_reason: str | None
    m: float                     # true wage = UMK * m
    etype: str
    obligated: bool = True       # must be registered at BPJS Kesehatan by this employer
    ks_m: float = 0.0            # reported multiplier to BPJS Kesehatan
    tk_m: float = 0.0            # reported multiplier to BPJS Ketenagakerjaan
    in_tk: bool = True
    tk_start: int = 0
    tk_end: int | None = None
    tk_program: str = "regular"
    tk_nik: str = ""
    tk_name: str = ""
    ks: list = field(default_factory=list)       # [[start, end|None, end_reason]]
    tags: set = field(default_factory=set)
    phk_free: list = field(default_factory=list)  # [(a, b)] free-cover intervals after PHK
    forced_event: int | None = None
    forced_cost_mult: float = 1.0
    seg_move: tuple | None = None                 # (day, to_segment)
    fktp: str | None = None


class NikGen:
    def __init__(self, rng: np.random.Generator):
        self.rng, self.seen = rng, set()

    def new(self) -> str:
        while True:
            n = "99" + str(int(self.rng.integers(0, 10 ** 14))).zfill(14)
            if n not in self.seen:
                self.seen.add(n)
                return n


def typo(nik: str, rng) -> str:
    i = int(rng.integers(2, 16))
    d = str((int(nik[i]) + int(rng.integers(1, 10))) % 10)
    return nik[:i] + d + nik[i + 1:]


def name_variant(name: str, rng) -> str:
    if name.startswith("Muhammad "):
        return name.replace("Muhammad ", rng.choice(["M. ", "Moh. "]), 1)
    opts = []
    if "i" in name:
        opts.append(name.replace("i", "y", 1))
    if "y" in name:
        opts.append(name.replace("y", "i", 1))
    opts.append(name[:-1])
    return opts[int(rng.integers(len(opts)))]


def rand_name(rng) -> str:
    return f"{FIRST[int(rng.integers(len(FIRST)))]} {LAST[int(rng.integers(len(LAST)))]}"


def _pk(seq, rng):
    return seq[int(rng.integers(len(seq)))]


def company_name(rng, umkm: str) -> str:
    a, b = rng.choice(CO_WORDS, 2, replace=False)
    return f"{'CV' if umkm == 'micro' else 'PT'} {a} {b}"


# --------------------------------------------------------------------------- employers

def make_employers(b: dict, rng) -> list[dict]:
    codes = list(b["sectors"])
    sw = np.array([b["sectors"][c]["weight"] for c in codes], float)
    sw /= sw.sum()
    bands = b["size_bands"]
    bw = np.array([x["share"] for x in bands], float)
    bw /= bw.sum()
    out = []
    for i in range(int(b["n_employers"])):
        sec = str(rng.choice(codes, p=sw))
        band = bands[int(rng.choice(len(bands), p=bw))]
        h0 = int(np.exp(rng.uniform(np.log(band["lo"]), np.log(band["hi"] + 1))))
        h0 = int(min(max(h0, band["lo"]), band["hi"]))
        out.append(dict(
            eid=f"PK{i + 1:05d}", name=company_name(rng, band["umkm"]),
            nib="".join(str(d) for d in rng.integers(0, 10, 13)),
            sector=sec, kab=b["kabkota"][int(rng.integers(len(b["kabkota"])))]["name"],
            band=band["name"], umkm=band["umkm"], hours=band["hours"], band_lo=band["lo"],
            h0=h0, registered_since=-int(rng.integers(60, 3000)),
            variants=[], lookalikes=[], params={},
            growth=float(rng.uniform(-0.005, 0.01)), shock=None,
            tk_lag=bool(rng.random() < b["noise"]["tk_lag_share"]),
            clinic=None, clinic_open=False, pay_ks="honest", pay_tk="honest",
            sloppy=bool(rng.random() < b["realism"]["sloppy_payer_share"]),
            slow_admin=bool(rng.random() < b["realism"]["slow_admin_share"]),
        ))
    return out


def _pick(rng, shares: dict, exclude=()) -> str | None:
    keys = [k for k in shares if k not in exclude]
    if not keys:
        return None
    p = np.array([shares[k] for k in keys], float)
    return str(rng.choice(keys, p=p / p.sum()))


def assign_profiles(emps: list[dict], b: dict, rng) -> None:
    v, la = b["violations"], b["lookalikes"]
    held = set(v["held_back"])
    for e in emps:
        if rng.random() < v["prevalence"]:
            modes = [_pick(rng, v["mode_shares"])]
            if rng.random() < v["second_mode_prob"]:
                m2 = _pick(rng, v["mode_shares"], exclude=modes)
                if m2:
                    modes.append(m2)
            for mode in modes:
                var = _pick(rng, v["variant_shares"][mode], exclude=held)
                if var is None:
                    continue
                e["variants"].append(var)
                e["params"][var] = _variant_params(var, rng)
            if not e["variants"]:  # every variant held back -> fall back to an allowed one
                var = _pick(rng, v["variant_shares"]["1"], exclude=held)
                e["variants"].append(var)
                e["params"][var] = _variant_params(var, rng)
            for var in e["variants"]:
                if var in ("3a", "3b", "3c"):
                    sub = var if var != "3c" else str(rng.choice(["3a", "3b"]))
                    e["pay_ks"] = sub
                    e["pay_tk"] = "honest" if var == "3c" else (sub if rng.random() < 0.5 else "honest")
                if var == "4b":
                    e["clinic"], e["clinic_open"] = f"KLN-{e['eid']}", False
        else:
            L = e["lookalikes"]
            if e["slow_admin"]:
                L.append("C13")
            if e["sloppy"]:
                L.append("C14")
            if e["sector"] == "F" and rng.random() < la["C1_construction_project"]:
                L.append("C1")
            if e["sector"] in ("C", "G", "I") and rng.random() < la["C2_umk_payer"]:
                L.append("C2")
            if rng.random() < (la["C3_seasonal_A"] if e["sector"] == "A" else la["C3_seasonal_other"]):
                L.append("C3")
                e["pay_ks"] = e["pay_tk"] = "seasonal"
            if rng.random() < la["C4_layoff"]:
                L.append("C4")
                e["shock"] = int(rng.integers(4, 21))
            if rng.random() < la["C5_fast_growth"]:
                L.append("C5")
                e["growth"] = 0.05
            p_open = {"50-249": la["C10_open_clinic_medium"], "250+": la["C10_open_clinic_large"]}.get(e["band"], 0)
            if rng.random() < p_open:
                L.append("C10")
                e["clinic"], e["clinic_open"] = f"KLN-{e['eid']}", True
            if rng.random() < la["C11_distress"]:
                L.append("C11")
                e["pay_ks"] = e["pay_tk"] = "distress"
                e["growth"] = -0.02
            if e["sector"] in ("C", "G", "I", "A"):
                p12 = {"micro": la["C12_below_umk_micro"], "small": la["C12_below_umk_small"]}.get(e["umkm"], 0)
                if rng.random() < p12:
                    L.append("C12")


def _variant_params(var: str, rng) -> dict:
    u = lambda a, b: float(rng.uniform(a, b))  # noqa: E731
    return {
        "1a": lambda: {"r": u(0.30, 0.85)}, "1b": lambda: {"r": u(0.30, 0.85)},
        "1c": lambda: {"m0": int(rng.integers(4, 13))}, "1d": lambda: {"frac": u(0.3, 0.6)},
        "2a": lambda: {}, "2b": lambda: {"s": u(0.55, 0.9)}, "2c": lambda: {"s": u(0.55, 0.9)},
        "3a": lambda: {}, "3b": lambda: {}, "3c": lambda: {},
        "4a": lambda: {"t": int(rng.integers(6, 21)), "frac": u(0.05, 0.4)},
        "4b": lambda: {"frac": u(0.2, 0.5)},
        "5a": lambda: {"lag": int(rng.integers(2, 7)), "extreme": bool(rng.random() < 0.1),
                       "t0": int(rng.integers(6, 12))},
        "5b": lambda: {"gap": int(rng.integers(1, 5))},
        "5c": lambda: {"f": u(0.3, 0.7)}, "5d": lambda: {"frac": u(0.2, 0.4)},
    }[var]()


# --------------------------------------------------------------------------- workforce

def simulate_workforce(e: dict, b: dict, rng, nikgen: NikGen, wid0: int) -> list[Worker]:
    sec = b["sectors"][e["sector"]]
    ws: list[Worker] = []

    def draw_m() -> float:
        if "C2" in e["lookalikes"] and rng.random() < 0.8:
            return 1.0
        if "C12" in e["lookalikes"] and rng.random() < 0.7:
            return float(rng.uniform(0.70, 0.95))
        return max(1.0, float(np.exp(rng.normal(sec["wage_mu"], sec["wage_sigma"]))))

    def draw_etype() -> str:
        r = rng.random()
        if e["sector"] == "F":
            return "project" if r < 0.5 else ("contract" if r < 0.7 else "permanent")
        return "contract" if r < 0.3 else "permanent"

    def hire(day: int) -> Worker:
        w = Worker(wid=wid0 + len(ws), eid=e["eid"], nik=nikgen.new(), name=rand_name(rng),
                   age=int(rng.integers(19, 56)), start=day, end=None, end_reason=None,
                   m=draw_m(), etype=draw_etype())
        if "C1" in e["lookalikes"] and w.etype == "project":
            w.obligated, w.tk_program = False, "project"
        ws.append(w)
        return w

    for _ in range(e["h0"]):
        hire(-int(rng.integers(30, 2500)))
    target = float(e["h0"])
    for mo in range(R.HORIZON_MONTHS):
        act = [w for w in ws if w.end is None]
        u1, u2, dd = rng.random(len(act)), rng.random(len(act)), rng.integers(0, 30, len(act))
        for w, a1, a2, d1 in zip(act, u1, u2, dd):
            if a1 < sec["turnover"]:
                w.end = mo * 30 + int(d1)
                w.end_reason = "resign" if a2 < 0.6 else ("contract_end" if a2 < 0.9 else "PHK")
        if e["shock"] == mo:
            still = [w for w in ws if w.end is None]
            k = int(len(still) * rng.uniform(0.2, 0.4))
            for i in rng.choice(len(still), size=k, replace=False):
                w = still[int(i)]
                w.end, w.end_reason = mo * 30 + int(rng.integers(0, 30)), "PHK"
                w.tags.add("layoff")
            target = len(still) - k
        target = max(target * (1 + e["growth"]), e["band_lo"])
        n_act = sum(1 for w in ws if w.end is None)
        for _ in range(max(0, int(round(target)) - n_act)):
            hire(mo * 30 + int(rng.integers(0, 30)))
    return ws


def _employed(w: Worker, d: int) -> bool:
    return w.start <= d and (w.end is None or w.end > d)


def _ks_active(w: Worker, d: int) -> bool:
    return any(s <= d and (en is None or en > d) for s, en, _ in w.ks)


def _tk_active(w: Worker, d: int) -> bool:
    return w.in_tk and w.tk_start <= d and (w.tk_end is None or w.tk_end > d)


def apply_observed(e: dict, ws: list[Worker], b: dict, rng) -> None:
    """Turn true employment into what BPJS would see, then plant each violation."""
    nz, rz = b["noise"], b["realism"]
    small = e["umkm"] in R.SMALL_CLASSES
    floor = (lambda x: x) if small else (lambda x: max(1.0, x))
    for w in ws:
        w.ks_m = w.tk_m = w.m
        if rng.random() < rz["wage_desync_prob"]:   # raise reached the TK report before the KS one
            w.tk_m = w.m * float(rng.uniform(1.03, 1.10))
        w.tk_start = w.start + int(rng.integers(0, 11))
        w.tk_end = None if w.end is None else w.end + int(rng.integers(0, 11))
        w.tk_nik = typo(w.nik, rng) if rng.random() < nz["nik_typo"] else w.nik
        w.tk_name = name_variant(w.name, rng) if rng.random() < nz["name_variant"] else w.name
        if w.obligated:
            if e["slow_admin"]:
                lag = int(rng.integers(5, 41))
            else:
                lag = int(rng.integers(0, 8)) if rng.random() > 0.03 else int(rng.integers(8, 21))
            w.ks = [[w.start + lag, None if w.end is None else w.end + int(rng.integers(0, 8)), w.end_reason]]
            if w.etype == "contract" and rng.random() < rz["admin_rereg_prob"]:
                s0, e0, r0 = w.ks[0]
                S = max(s0, 0) + int(rng.integers(90, 300))
                G = int(rng.integers(10, 46))
                if S + G < R.HORIZON_DAYS and (e0 is None or S + G < e0):
                    w.ks = [[s0, S, "contract_end"], [S + G, e0, r0]]
                    w.tags.add("admin_gap")

    P = e["params"]
    for var in e["variants"]:
        p = P[var]
        if var in ("1a", "1b"):
            for w in ws:
                if w.obligated and rng.random() > p["r"]:
                    w.ks = []
                    w.tags.add("unreg")
                    if var == "1b":
                        w.in_tk = False
        elif var == "1c":
            for w in ws:
                if w.obligated and w.start >= p["m0"] * 30:
                    w.ks = []
                    w.tags.add("unreg_new")
        elif var == "1d":
            for w in ws:
                if w.obligated and rng.random() < p["frac"]:
                    w.ks = []
                    w.tags.add("piggyback")
        elif var == "2a":
            for w in ws:
                if w.m > 1.0:
                    w.ks_m = w.tk_m = 1.0
                    w.tags.add("wage_umk")
        elif var == "2b":
            for w in ws:
                w.ks_m = w.tk_m = floor(p["s"] * w.m)
                w.tags.add("wage_scaled")
        elif var == "2c":
            for w in ws:
                w.ks_m = floor(p["s"] * w.m)
                w.tags.add("wage_ks_only")
        elif var == "4a":
            T = p["t"] * 30 + int(rng.integers(0, 30))
            cand = [w for w in ws if w.ks and _ks_active(w, T) and _employed(w, T)
                    and (w.end is None or w.end > T + 60)]
            k = int(len(cand) * p["frac"])
            for i in (rng.choice(len(cand), size=k, replace=False) if k else []):
                w = cand[int(i)]
                w.ks[-1][1], w.ks[-1][2] = T + int(rng.integers(0, 10)), "resign"
                w.seg_move = (T + int(rng.integers(0, 60)), "PBPU" if rng.random() < 0.7 else "PBI")
                w.tags.add("moved")
        elif var == "4b":
            for w in ws:
                if w.obligated and rng.random() < p["frac"]:
                    w.ks = []
                    w.in_tk = bool(rng.random() < 0.5)
                    w.fktp = e["clinic"]
                    w.tags.add("clinic_pbpu")
        elif var == "5a":
            for w in ws:
                if w.start < 0 or not w.ks:
                    continue
                if p["extreme"] and w.start >= p["t0"] * 30:
                    w.ks = []
                    w.tags.add("no_ks_joiners")
                    continue
                w.ks[0][0] = w.start + p["lag"] * 30 + int(rng.integers(0, 30))
                w.tags.add("late_reg")
                if w.ks[0][1] is not None and w.ks[0][0] >= w.ks[0][1]:
                    w.ks = []
        elif var == "5b":
            for w in ws:
                if not w.ks:
                    continue
                s0, e0, r0 = w.ks[0]
                if e0 is not None and e0 - p["gap"] * 30 > s0 + 30:
                    w.ks[0][1] = e0 - p["gap"] * 30
                    w.tags.add("early_dereg")
                elif w.etype == "contract" and rng.random() < 0.5:
                    S = max(s0, 0) + int(rng.integers(90, 300))
                    G = p["gap"] * 30
                    if S + G < R.HORIZON_DAYS and (e0 is None or S + G < e0):
                        w.ks = [[s0, S, "contract_end"], [S + G, e0, r0]]
                        w.tags.add("onoff")
        elif var == "5c":
            for w in ws:
                if w.start < 0 or not w.ks or rng.random() >= p["f"]:
                    continue
                orig_end, orig_reason = w.ks[0][1], w.ks[0][2]
                w.ks = []
                w.tags.add("held_back")
                if rng.random() < 0.5:
                    ev = w.start + int(rng.integers(60, 241))
                    if ev < R.HORIZON_DAYS and (w.end is None or ev < w.end):
                        w.forced_event, w.forced_cost_mult = ev, 2.0
                        w.ks = [[ev - int(rng.integers(0, 15)), orig_end, orig_reason]]
                        w.tags.add("reg_when_sick")
        elif var == "5d":
            for w in ws:
                if not w.ks or rng.random() >= p["frac"]:
                    continue
                s0, e0, r0 = w.ks[-1]
                lo = max(s0 + 90, 60)
                if lo >= 600:
                    continue
                Pd = int(rng.integers(lo, 600))
                if not _employed(w, Pd) or (e0 is not None and e0 <= Pd):
                    continue
                G = int(rng.integers(30, 181))
                w.ks[-1] = [s0, Pd, "PHK"]
                if w.end is None or w.end > Pd + G:
                    w.ks.append([Pd + G, e0, r0])
                w.phk_free.append((Pd, Pd + min(G, R.PHK_FREE_DAYS)))
                w.tags.add("phk_cycle")

    # Lawful free cover after a genuine PHK exit (E44), and C4/C7 look-alike behaviour.
    honest = not e["variants"]
    for w in ws:
        if w.end_reason == "PHK" and w.ks and "phk_cycle" not in w.tags and w.ks[-1][1] is not None:
            E = w.ks[-1][1]
            w.phk_free.append((E, E + R.PHK_FREE_DAYS))
        if "layoff" in w.tags and rng.random() < 0.6 and w.end is not None:
            w.seg_move = (w.end + int(rng.integers(0, 30)), "PBPU" if rng.random() < 0.7 else "PBI")
        if (honest and w.start >= 0 and w.ks and rng.random() < b["lookalikes"]["C7_newhire_ill"]):
            ev = w.ks[0][0] + int(rng.integers(0, 31))
            if ev < R.HORIZON_DAYS and _employed(w, ev):
                w.forced_event = ev
                w.tags.add("C7_newhire_ill")


# --------------------------------------------------------------------------- money

def _pay(behavior: str, billed: float, mo: int, sector: str, rng, sloppy: float = 0.0) -> tuple[float, int, int | None]:
    """Returns (paid_rp by horizon end, last payment day, settle_day or None if never settled)."""
    due = mo * 30 + R.DUE_DAY_OF_MONTH
    on_time = mo * 30 + int(rng.integers(0, R.DUE_DAY_OF_MONTH + 1))
    if billed <= 0:
        return 0.0, on_time, on_time
    if behavior == "seasonal" and (mo % 12) in (3, 4, 9):
        d = mo * 30 + int(rng.integers(12, 28))
        return billed, d, d
    if behavior == "3a" and rng.random() < 0.7:
        share = float(rng.uniform(0.4, 0.9))
        if rng.random() < 0.4:
            d = due + int(rng.integers(30, 91))
            return (billed, d, d) if d < R.HORIZON_DAYS else (billed * share, on_time, None)
        return billed * share, on_time, None
    if behavior == "3b" and rng.random() < 0.7:
        d = due + int(rng.integers(30, 121))
        return (billed, d, d) if d < R.HORIZON_DAYS else (0.0, on_time, None)
    if behavior == "distress":
        r = rng.random()
        if r < 0.5:
            d = due + int(rng.integers(20, 91))
            return (billed, d, d) if d < R.HORIZON_DAYS else (0.0, on_time, None)
        if r < 0.75:
            d = due + int(rng.integers(30, 91))
            if rng.random() < 0.6 and d < R.HORIZON_DAYS:
                return billed, d, d
            return billed * float(rng.uniform(0.5, 0.9)), on_time, None
    if sloppy and rng.random() < sloppy:
        d = due + int(rng.integers(1, 26))
        return (billed, d, d) if d < R.HORIZON_DAYS else (billed, on_time, on_time)
    return billed, on_time, on_time


def _umk_table(b: dict) -> dict:
    return {k["name"]: {int(y): float(v) for y, v in k["umk"].items()} for k in b["kabkota"]}


BIG = 10 ** 9
MID_DAYS = np.arange(R.HORIZON_MONTHS) * 30 + 14


def _arrays(ws: list[Worker]):
    """Vectorised worker x month activity matrices (employment, KS coverage, TK presence)."""
    n = len(ws)
    start = np.array([w.start for w in ws])
    end = np.array([BIG if w.end is None else w.end for w in ws])
    emp = (start[:, None] <= MID_DAYS) & (end[:, None] > MID_DAYS)
    ks = np.zeros((n, R.HORIZON_MONTHS), dtype=bool)
    for i, w in enumerate(ws):
        for s, en, _ in w.ks:
            ks[i] |= (s <= MID_DAYS) & ((BIG if en is None else en) > MID_DAYS)
    in_tk = np.array([w.in_tk for w in ws])
    tks = np.array([w.tk_start for w in ws])
    tke = np.array([BIG if w.tk_end is None else w.tk_end for w in ws])
    tk = in_tk[:, None] & (tks[:, None] <= MID_DAYS) & (tke[:, None] > MID_DAYS)
    return emp, ks, tk


def _iuran_vec(wage: np.ndarray, umk: np.ndarray, umkm: str) -> np.ndarray:
    floor = np.zeros_like(umk) if umkm in R.SMALL_CLASSES else umk
    return R.KS_RATE * np.minimum(np.maximum(wage, floor), R.WAGE_CAP)


def billing_and_suspension(e, ws, b, umk, rng):
    rows, susp = [], np.zeros(R.HORIZON_DAYS, dtype=bool)
    emp, ks, tk = _arrays(ws)
    u = np.array([umk[e["kab"]][R.year_of_month(mo)] for mo in range(R.HORIZON_MONTHS)])
    ks_m = np.array([w.ks_m for w in ws])[:, None]
    tk_m = np.array([w.tk_m for w in ws])[:, None]
    ks_bill = (_iuran_vec(ks_m * u, u[None, :].repeat(len(ws), 0), e["umkm"]) * ks).sum(0)
    tk_bill = (b["tk_rate"] * tk_m * u * tk).sum(0)
    for mo in range(R.HORIZON_MONTHS):
        for agency, billed, beh in (("KS", float(ks_bill[mo]), e["pay_ks"]), ("TK", float(tk_bill[mo]), e["pay_tk"])):
            paid, pday, settle = _pay(beh, billed, mo, e["sector"], rng,
                                      b["realism"]["sloppy_late_prob"] if e["sloppy"] else 0.0)
            rows.append(dict(employer_id=e["eid"], agency=agency, month=mo, billed_rp=round(billed),
                             paid_rp=round(paid), paid_day=pday,
                             days_late=max(0, pday - (mo * 30 + R.DUE_DAY_OF_MONTH))))
            if agency == "KS" and billed > 0:
                month_end = mo * 30 + 29
                if settle is None:
                    susp[month_end + 1:] = True
                elif settle > month_end:
                    susp[month_end + 1:min(settle, R.HORIZON_DAYS)] = True
    return rows, susp


def care_events(e, ws, b, susp, rng, ev_counter):
    c = b["care"]
    rows = []

    def status(day, phk):
        return "phk_free" if phk else ("suspended" if susp[min(day, R.HORIZON_DAYS - 1)] else "active")

    def add(w, day, setting, cost, phk=False):
        ev_counter[0] += 1
        rows.append(dict(event_id=f"SEP{ev_counter[0]:08d}", nik=w.nik, employer_id=e["eid"], day=int(day),
                         setting=setting, cost_rp=round(cost), coverage_status=status(day, phk)))

    for w in ws:
        if not w.ks and not w.phk_free:
            continue
        pin = c["inpatient_monthly_base"] + c["inpatient_age_slope"] * max(0, w.age - 25)
        first = min((s for s, _, _ in w.ks), default=None)
        intervals = [(max(0, s), min(R.HORIZON_DAYS, R.HORIZON_DAYS if en is None else en), False) for s, en, _ in w.ks]
        intervals += [(max(0, a), min(R.HORIZON_DAYS, bb), True) for a, bb in w.phk_free]
        for a, bb, phk in intervals:
            if bb <= a:
                continue
            ca = np.arange(a, bb, 30)
            cb = np.minimum(ca + 30, bb)
            frac = (cb - ca) / 30
            bump = np.where((first is not None and first >= 0) & (ca - (first or 0) < 60), c["newjob_bump"], 1.0)
            hit_in = rng.random(len(ca)) < pin * frac * bump
            hit_out = rng.random(len(ca)) < c["outpatient_monthly"] * frac
            for j in np.flatnonzero(hit_in):
                cost = float(np.exp(rng.normal(np.log(c["inpatient_cost_median"]), c["inpatient_cost_sigma"])))
                add(w, int(rng.integers(ca[j], cb[j])), "inpatient", cost, phk)
            for j in np.flatnonzero(hit_out):
                add(w, int(rng.integers(ca[j], cb[j])), "outpatient", c["outpatient_cost"], phk)
        if w.forced_event is not None:
            cost = float(np.exp(rng.normal(np.log(c["inpatient_cost_median"]), c["inpatient_cost_sigma"])))
            add(w, w.forced_event, "inpatient", cost * w.forced_cost_mult)
    return rows


def truth_money(e, ws, b, umk, bill_rows, care_rows):
    emp, ks, _ = _arrays(ws)
    u = np.array([umk[e["kab"]][R.year_of_month(mo)] for mo in range(R.HORIZON_MONTHS)])
    U = u[None, :].repeat(len(ws), 0)
    m = np.array([w.m for w in ws])[:, None]
    ks_m = np.array([w.ks_m for w in ws])[:, None]
    oblig = np.array([w.obligated for w in ws])[:, None]
    due = _iuran_vec(m * u, U, e["umkm"])
    billed = _iuran_vec(ks_m * u, U, e["umkm"]) * ks
    gap = np.where(emp & oblig, np.maximum(0.0, due - billed), 0.0).sum(0)
    gaps = gap[12:]
    ks_bill = [r for r in bill_rows if r["agency"] == "KS" and r["month"] >= 12]
    shortfall = sum(r["billed_rp"] - r["paid_rp"] for r in ks_bill) / 12
    phk_niks = {w.nik for w in ws if "phk_cycle" in w.tags}
    rws_niks = {w.nik: w.forced_event for w in ws if "reg_when_sick" in w.tags}
    care_susp = sum(r["cost_rp"] for r in care_rows if r["coverage_status"] == "suspended")
    phk_cover = sum(r["cost_rp"] for r in care_rows if r["coverage_status"] == "phk_free" and r["nik"] in phk_niks)
    late_claim = sum(r["cost_rp"] for r in care_rows if r["nik"] in rws_niks and r["day"] == rws_niks[r["nik"]])
    return dict(gap_rp_month_true=float(np.mean(gaps)), payment_shortfall_month_true=float(shortfall),
                care_liability_true=float(care_susp), phk_cover_true=float(phk_cover),
                late_reg_claim_true=float(late_claim))


# --------------------------------------------------------------------------- main

def generate_branch(cfg: dict, branch: str, out_dir: Path) -> dict:
    b = branch_config(cfg, branch)
    rng = np.random.default_rng(b["seed"])
    nikgen = NikGen(rng)
    umk = _umk_table(b)
    emps = make_employers(b, rng)
    assign_profiles(emps, b, rng)
    lookback = b["inspection"]["lookback_months"]

    all_w: dict[str, list[Worker]] = {}
    wid = 0
    for e in emps:
        ws = simulate_workforce(e, b, rng, nikgen, wid)
        wid += len(ws)
        apply_observed(e, ws, b, rng)
        all_w[e["eid"]] = ws

    # FKTP registry: public FKTP per kab/kota + company clinics (E45)
    fktp_rows = []
    for k in b["kabkota"]:
        for j in range(12):
            fktp_rows.append(dict(fktp_id=f"FKTP-{k['name'][-4:].upper()}-{j:02d}", kabkota=k["name"],
                                  type=["puskesmas", "klinik", "dokter"][j % 3], owner_employer_id=None,
                                  open_to_public=True))
    for e in emps:
        if e["clinic"]:
            fktp_rows.append(dict(fktp_id=e["clinic"], kabkota=e["kab"], type="klinik_perusahaan",
                                  owner_employer_id=e["eid"], open_to_public=e["clinic_open"]))
    public_fktp = {k["name"]: [r["fktp_id"] for r in fktp_rows if r["kabkota"] == k["name"] and r["owner_employer_id"] is None]
                   for k in b["kabkota"]}

    ks_rows, tk_rows, bill_rows_all, care_rows_all, mem_rows, move_rows = [], [], [], [], [], []
    truth_rows, label_rows, payroll_rows = [], [], []
    ev_counter = [0]
    snaps = [(q, (3 * q + 3) * 30 - 1) for q in range(8)]
    ppu_pool = []

    for e in emps:
        ws = all_w[e["eid"]]
        bill_rows, susp = billing_and_suspension(e, ws, b, umk, rng)
        crows = care_events(e, ws, b, susp, rng, ev_counter)
        bill_rows_all += bill_rows
        care_rows_all += crows
        for w in ws:
            for s, en, reason in w.ks:
                yr = R.year_of_day(min(en if en is not None else R.SCORING_DAY, R.SCORING_DAY))
                u = umk[e["kab"]][yr]
                ks_rows.append(dict(employer_id=e["eid"], nik=w.nik, name=w.name, ppu_start=int(s),
                                    ppu_end=None if en is None else int(en),
                                    end_reason=reason if en is not None else None,
                                    wage_reported=round(u * w.ks_m), wage_year=yr))
            if w.in_tk:
                for q, sd in snaps:
                    ref = sd - 90 if e["tk_lag"] else sd
                    if ref >= 0 and _tk_active(w, ref):
                        u = umk[e["kab"]][R.year_of_day(ref)]
                        tk_rows.append(dict(employer_id=e["eid"], quarter=q, snap_day=sd, nik=w.tk_nik,
                                            name=w.tk_name, wage_reported_tk=round(u * w.tk_m),
                                            program=w.tk_program, tk_start=int(w.tk_start)))
            active_end = w.ks and _ks_active(w, R.SCORING_DAY)
            if active_end:
                fk = e["clinic"] if (e["clinic"] and rng.random() < 0.5) else _pk(public_fktp[e["kab"]], rng)
                mem_rows.append(dict(nik=w.nik, segment="PPU", employer_id=e["eid"], dependent_of=None, fktp_id=fk))
                ppu_pool.append((w.nik, e["eid"], e["kab"]))
                if rng.random() < b["noise"]["dependent_spouse_share"]:
                    dn = nikgen.new()
                    mem_rows.append(dict(nik=dn, segment="DEPENDENT", employer_id=None, dependent_of=w.nik, fktp_id=fk))
                    if rng.random() < b["noise"]["dependent_self_employed"]:  # C8
                        tk_rows.append(dict(employer_id="BPU", quarter=7, snap_day=snaps[-1][1], nik=dn,
                                            name=rand_name(rng), wage_reported_tk=round(umk[e["kab"]][2026]),
                                            program="BPU", tk_start=int(rng.integers(-900, 600))))
            elif w.seg_move and w.seg_move[0] <= R.SCORING_DAY:
                move_rows.append(dict(nik=w.nik, from_segment="PPU", to_segment=w.seg_move[1], day=int(w.seg_move[0]),
                                      from_employer_id=e["eid"]))
                mem_rows.append(dict(nik=w.nik, segment=w.seg_move[1], employer_id=None, dependent_of=None,
                                     fktp_id=_pk(public_fktp[e["kab"]], rng)))
            elif "clinic_pbpu" in w.tags and _employed(w, R.SCORING_DAY):
                mem_rows.append(dict(nik=w.nik, segment="PBPU", employer_id=None, dependent_of=None, fktp_id=w.fktp))
            payroll_rows.append(dict(worker_id=w.wid, employer_id=e["eid"], nik=w.nik, start=w.start, end=w.end,
                                     end_reason=w.end_reason, wage_multiplier=round(w.m, 4), employment_type=w.etype,
                                     ks_obligated=w.obligated, tags=",".join(sorted(w.tags))))
        if e["clinic"] and e["clinic_open"]:  # C10: genuine residents enrolled at an open company clinic
            for _ in range(int(rng.integers(10, 61))):
                mem_rows.append(dict(nik=nikgen.new(), segment="PBPU", employer_id=None, dependent_of=None,
                                     fktp_id=e["clinic"]))
        money = truth_money(e, ws, b, umk, bill_rows, crows)
        violator = bool(e["variants"])
        recoverable = ((money["gap_rp_month_true"] + money["payment_shortfall_month_true"]) * lookback
                       + money["care_liability_true"] + money["phk_cover_true"]) if violator else 0.0
        truth_rows.append(dict(employer_id=e["eid"], sector=e["sector"], kabkota=e["kab"], size_band=e["band"],
                               umkm=e["umkm"], is_violator=violator, variants=",".join(e["variants"]),
                               lookalikes=",".join(e["lookalikes"]), true_headcount_end=sum(_employed(w, R.SCORING_DAY) for w in ws),
                               recoverable_rp_true=float(recoverable), **money))
        for var in e["variants"]:
            label_rows.append(dict(employer_id=e["eid"], mode=var[0], variant=var, params=json.dumps(e["params"][var])))

    # 1d piggyback: unregistered worker shows up at KS as the dependant of a PPU spouse at ANOTHER employer
    for e in emps:
        for w in all_w[e["eid"]]:
            if "piggyback" in w.tags and _employed(w, R.SCORING_DAY):
                while True:
                    sp = ppu_pool[int(rng.integers(len(ppu_pool)))]
                    if sp[1] != e["eid"]:
                        break
                mem_rows.append(dict(nik=w.nik, segment="DEPENDENT", employer_id=None, dependent_of=sp[0],
                                     fktp_id=_pk(public_fktp[e["kab"]], rng)))

    registry = []
    for e in emps:
        oss = max(1, int(round(e["h0"] * np.exp(rng.normal(0, b["noise"]["oss_sigma"])))))
        registry.append(dict(employer_id=e["eid"], name=e["name"], nib=e["nib"], sector=e["sector"],
                             kabkota=e["kab"], skala_usaha=e["umkm"], oss_declared_headcount=oss,
                             registered_since=e["registered_since"]))
    umk_rows = [dict(kabkota=k, year=y, umk_rp=v) for k, yv in umk.items() for y, v in yv.items()]
    cost_rows = [dict(size_band=x["name"], hours_est=x["hours"]) for x in b["size_bands"]]

    obs_dir, tr_dir = out_dir / "observed", out_dir / "truth"
    obs_dir.mkdir(parents=True, exist_ok=True)
    tr_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        obs_dir / "employer_registry.parquet": registry, obs_dir / "ks_membership.parquet": ks_rows,
        obs_dir / "tk_exchange.parquet": tk_rows, obs_dir / "billing.parquet": bill_rows_all,
        obs_dir / "care_events.parquet": care_rows_all, obs_dir / "ks_members.parquet": mem_rows,
        obs_dir / "segment_moves.parquet": move_rows, obs_dir / "fktp.parquet": fktp_rows,
        obs_dir / "umk.parquet": umk_rows, obs_dir / "inspection_cost.parquet": cost_rows,
        tr_dir / "employers_truth.parquet": truth_rows, tr_dir / "labels.parquet": label_rows,
        tr_dir / "payroll_truth.parquet": payroll_rows,
    }
    for path, rows in tables.items():
        pd.DataFrame(rows).to_parquet(path, index=False)
    (out_dir / "config.json").write_text(json.dumps(b, indent=2, default=str), encoding="utf-8")
    t = pd.DataFrame(truth_rows)
    return dict(branch=branch, employers=len(emps), workers=wid, violators=int(t.is_violator.sum()),
                ks_rows=len(ks_rows), tk_rows=len(tk_rows), care_events=len(care_rows_all))
