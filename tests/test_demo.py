"""The committed demo slice: what a fresh clone and Streamlit Community Cloud run on."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"
DEMO = ROOT / "demo"
MANIFEST = json.loads((DEMO / "manifest.json").read_text(encoding="utf-8"))
VIEWS = ["Daftar prioritas", "Kasus perusahaan", "Tindak lanjut", "Usaha belum terdaftar", "Benchmark", "Keadilan & jalur acak",
         "Cek kepesertaan saya"]
EMPLOYER_LEVEL = ["employer_registry", "billing", "fktp", "umk", "inspection_cost"]
WORKER_LEVEL = ["ks_membership", "tk_exchange", "ks_members", "care_events", "segment_moves"]
FULL_RUN = (ROOT / "data" / "B" / "observed").is_dir() and (ROOT / "results" / "ranked_B.parquet").is_file()


def test_manifest_matches_the_committed_files():
    on_disk = {p.relative_to(DEMO).as_posix() for p in DEMO.rglob("*")
               if p.is_file() and p.suffix in (".parquet", ".json") and p.name != "manifest.json"}
    assert on_disk == set(MANIFEST["files"])
    for rel, meta in MANIFEST["files"].items():
        assert hashlib.sha256((DEMO / rel).read_bytes()).hexdigest() == meta["sha256"], f"{rel} changed; rebuild the slice"


def test_slice_stays_small_and_leaves_out_payroll_and_labels():
    assert sum(m["bytes"] for m in MANIFEST["files"].values()) < 10_000_000
    assert not any("payroll_truth" in f or f.endswith("labels.parquet") for f in MANIFEST["files"])
    for b in ("A", "B", "C"):
        assert len(MANIFEST["branches"][b]["employers_with_worker_detail"]) == MANIFEST["demo_employers"]


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not installed")
def test_cloud_requirements_match_the_lockfile():
    """Community Cloud reads app/requirements.txt (the entrypoint folder wins over the root uv.lock)."""
    cmd = ["uv", "export", "--frozen", "--no-dev", "--no-hashes", "--no-emit-project", "--format", "requirements-txt"]
    exported = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True).stdout
    committed = (ROOT / "app" / "requirements.txt").read_text(encoding="utf-8")
    assert exported.splitlines() == committed.splitlines(), "Regenerate app/requirements.txt with: " + " ".join(cmd)


@pytest.mark.parametrize("branch", ["B", "C", "A"])
@pytest.mark.parametrize("view", VIEWS)
def test_each_view_renders_on_the_demo_slice(monkeypatch, branch, view):
    monkeypatch.setenv("TEPAT_DEMO", "1")
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    app.selectbox(key="branch").set_value(branch).run()
    app.sidebar.radio(key="view").set_value(view).run()
    assert not app.exception
    assert app.title[0].value == view


@pytest.mark.parametrize("branch", ["B", "C", "A"])
def test_every_worker_scenario_resolves_on_the_demo_slice(monkeypatch, branch):
    """The slice offers the same worker scenarios as the full run (recorded by the builder)."""
    monkeypatch.setenv("TEPAT_DEMO", "1")
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    app.selectbox(key="branch").set_value(branch).run()
    app.sidebar.radio(key="view").set_value("Cek kepesertaan saya").run()
    scenarios = app.radio(key="worker_scenario").options
    assert set(scenarios) == set(MANIFEST["branches"][branch]["worker_scenarios"]), scenarios
    assert len(scenarios) >= 3
    assert any("200 perusahaan contoh" in c.value for c in app.caption), "Say plainly that worker data is a slice."
    for s in scenarios:
        app.radio(key="worker_scenario").set_value(s).run()
        assert not app.exception
        assert not any("NIK tidak ditemukan" in i.value for i in app.info), s


@pytest.mark.skipif(not FULL_RUN, reason="needs the full local run (uv run tepat all)")
@pytest.mark.parametrize("branch", ["A", "B", "C"])
def test_slice_is_faithful_to_the_full_run(branch):
    full, demo = ROOT / "data" / branch, DEMO / branch
    for t in EMPLOYER_LEVEL:
        pd.testing.assert_frame_equal(pd.read_parquet(full / "observed" / f"{t}.parquet"),
                                      pd.read_parquet(demo / "observed" / f"{t}.parquet"))
    pd.testing.assert_frame_equal(pd.read_parquet(full / "truth" / "employers_truth.parquet"),
                                  pd.read_parquet(demo / "truth" / "employers_truth.parquet"))
    for name in ("ranked", "features"):
        pd.testing.assert_frame_equal(pd.read_parquet(ROOT / "results" / f"{name}_{branch}.parquet"),
                                      pd.read_parquet(demo / f"{name}.parquet"))
    cut = {t: pd.read_parquet(demo / "observed" / f"{t}.parquet") for t in WORKER_LEVEL}
    niks = set().union(*(set(df.nik) for df in cut.values()))
    for t in WORKER_LEVEL:   # closed under NIK: a kept worker keeps every row in every table
        f = pd.read_parquet(full / "observed" / f"{t}.parquet")
        pd.testing.assert_frame_equal(f[f.nik.isin(niks)].reset_index(drop=True), cut[t])
    employers = set(MANIFEST["branches"][branch]["employers_with_worker_detail"])
    tk = pd.read_parquet(full / "observed" / "tk_exchange.parquet")
    assert set(tk.loc[tk.employer_id.isin(employers), "nik"]) <= niks, "every worker of a demo employer is kept"
