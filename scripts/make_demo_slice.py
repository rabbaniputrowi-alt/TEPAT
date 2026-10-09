"""Build the committed demo slice in demo/ from a full local run (data/ + results/).

    uv run tepat all                          # full run first (gitignored output)
    uv run python scripts/make_demo_slice.py  # writes demo/<branch>/ + demo/manifest.json

Why a slice: data/<branch>/ (about 13 MB per branch) and results/*.parquet are gitignored, so a fresh
clone or Streamlit Community Cloud would have nothing to show. The slice keeps every view faithful:

* Employer-level files keep all employers, frame-equal to the full run (re-compressed with zstd):
  employer_registry, billing, fktp, umk, inspection_cost, truth/employers_truth, ranked, features.
  The queue, case, benchmark and fairness views therefore show exactly what the full run shows.
* Worker-level files (ks_membership, tk_exchange, ks_members, care_events, segment_moves) keep only
  workers linked to DEMO_EMPLOYERS employers per branch. The cut is closed under NIK: a kept worker
  keeps every row, at every employer, so a lookup by NIK in "Cek kepesertaan saya" returns the same
  answer as on the full data. Workers of other employers are simply absent.
* truth/payroll_truth and truth/labels are not used by the app and are left out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tepat import rules as R  # noqa: E402

BRANCHES = ("A", "B", "C")
DEMO_EMPLOYERS = 200
TOP_QUEUE = 40          # first cases a reviewer opens from the queue
SEED = 20261009
FULL_OBSERVED = ("employer_registry", "billing", "fktp", "umk", "inspection_cost")
WORKER_TABLES = ("ks_membership", "tk_exchange", "ks_members", "care_events", "segment_moves")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, compression="zstd")   # about 30% smaller than the default snappy


def scenario_employers(obs: dict[str, pd.DataFrame], ranked: pd.DataFrame) -> dict[str, str]:
    """Best-ranked employer for each "Cek kepesertaan saya" scenario the full run has, keyed by the app's label.

    Mirrors demo_niks() in app/streamlit_app.py. Not every branch has every scenario (only C plants
    workers registered solely as a dependant), so the slice records which ones exist and
    tests/test_demo.py checks the app offers exactly those.
    """
    ks, tk, mem, bill = obs["ks_membership"], obs["tk_exchange"], obs["ks_members"], obs["billing"]
    tk7 = tk[(tk.quarter == 7) & (tk.program == "regular")]          # final quarterly exchange file, as in the app
    active = ks[ks.ppu_end.isna() | (ks.ppu_end > R.SCORING_DAY)]
    last = bill[(bill.agency == "KS") & (bill.month == R.HORIZON_MONTHS - 1)]
    unpaid = set(last.loc[last.paid_rp < last.billed_rp * 0.995, "employer_id"])
    paid = set(last.loc[last.paid_rp >= last.billed_rp, "employer_id"])
    scenarios = {
        "Tidak didaftarkan pemberi kerja": tk7.loc[~tk7.nik.isin(ks.nik) & ~tk7.nik.isin(mem.nik), "employer_id"],
        "Iuran tidak dibayar pemberi kerja": active.loc[active.employer_id.isin(unpaid), "employer_id"],
        "Hanya tercatat sebagai tanggungan": tk7.loc[tk7.nik.isin(mem.loc[mem.segment == "DEPENDENT", "nik"])
                                                     & ~tk7.nik.isin(ks.nik), "employer_id"],
        "Semua beres": active.loc[active.employer_id.isin(paid), "employer_id"],
    }
    picks = {}
    for label, ids in scenarios.items():
        ids = [e for e in set(ids) if e in ranked.index]
        if ids:
            picks[label] = str(min(ids, key=lambda e: ranked.at[e, "rank"]))
    return picks


def pick_employers(ranked: pd.DataFrame, features: pd.DataFrame, must: list[str], rng: np.random.Generator) -> list[str]:
    """Top of the inspection queue, the whole installment queue, one employer per worker scenario,
    then a size-stratified random fill."""
    order = ranked.sort_values("rank")
    chosen = list(order.index[order.queue == "inspeksi"][:TOP_QUEUE])
    chosen += [e for e in order.index[order.queue == "cicilan"] if e not in chosen]
    chosen += [e for e in must if e not in chosen]
    rest = [e for e in order.index if e not in set(chosen)]
    need = DEMO_EMPLOYERS - len(chosen)
    bands = features.loc[rest, "band_obs"]
    for band, ids in bands.groupby(bands).groups.items():
        quota = round(need * len(ids) / len(rest))
        chosen += list(rng.choice(sorted(ids), size=min(quota, len(ids)), replace=False))
    leftover = sorted(set(rest) - set(chosen))
    if len(chosen) < DEMO_EMPLOYERS:
        chosen += list(rng.choice(leftover, size=DEMO_EMPLOYERS - len(chosen), replace=False))
    return sorted(str(e) for e in chosen[:DEMO_EMPLOYERS])


def build_branch(branch: str, src_data: Path, src_results: Path, out: Path, rng: np.random.Generator) -> dict:
    src, dst = src_data / branch, out / branch
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "observed").mkdir(parents=True)
    (dst / "truth").mkdir()
    ranked_src, features_src = src_results / f"ranked_{branch}.parquet", src_results / f"features_{branch}.parquet"
    ranked, features = pd.read_parquet(ranked_src), pd.read_parquet(features_src)
    obs = {t: pd.read_parquet(src / "observed" / f"{t}.parquet") for t in WORKER_TABLES + ("billing",)}
    scenarios = scenario_employers(obs, ranked)
    employers = pick_employers(ranked, features, list(scenarios.values()), rng)
    del obs["billing"]

    # Employer-level: all employers, same frames (re-written with zstd, checked equal below).
    whole = {src / "observed" / f"{t}.parquet": dst / "observed" / f"{t}.parquet" for t in FULL_OBSERVED}
    whole[src / "truth" / "employers_truth.parquet"] = dst / "truth" / "employers_truth.parquet"
    whole[ranked_src] = dst / "ranked.parquet"
    whole[features_src] = dst / "features.parquet"
    for a, b in whole.items():
        write(pd.read_parquet(a), b)
        pd.testing.assert_frame_equal(pd.read_parquet(a), pd.read_parquet(b))
    shutil.copyfile(src / "config.json", dst / "config.json")

    # Worker-level: every NIK linked to a demo employer, plus their registered dependants; all rows kept.
    emp = set(employers)
    niks = set()
    for t in ("ks_membership", "tk_exchange", "ks_members"):
        niks |= set(obs[t].loc[obs[t].employer_id.isin(emp), "nik"])
    mem = obs["ks_members"]
    niks |= set(mem.loc[mem.dependent_of.isin(niks), "nik"])
    rows = {}
    for t, df in obs.items():
        cut = df[df.nik.isin(niks)].reset_index(drop=True)
        write(cut, dst / "observed" / f"{t}.parquet")
        rows[t] = [len(cut), len(df)]
    return {"employers_with_worker_detail": employers, "workers": len(niks), "rows_kept_of_total": rows,
            "employers_total": len(ranked), "worker_scenarios": scenarios}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "demo"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    manifest = {"built_by": "scripts/make_demo_slice.py", "seed": SEED, "demo_employers": DEMO_EMPLOYERS,
                "top_queue": TOP_QUEUE, "branches": {}, "files": {}}
    for b in BRANCHES:
        manifest["branches"][b] = build_branch(b, Path(a.data), Path(a.results), out, rng)
        info = manifest["branches"][b]
        print(f"{b}: {len(info['employers_with_worker_detail'])} employers with worker detail, {info['workers']:,} workers")
    for f in sorted(p for p in out.rglob("*") if p.is_file() and p.suffix in (".parquet", ".json") and p.name != "manifest.json"):
        manifest["files"][f.relative_to(out).as_posix()] = {"bytes": f.stat().st_size, "sha256": sha256(f)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    total = sum(v["bytes"] for v in manifest["files"].values())
    print(f"demo/: {len(manifest['files'])} files, {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
