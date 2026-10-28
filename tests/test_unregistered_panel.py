"""The isolated registration workspace on the committed synthetic demo slice."""
from pathlib import Path
import sys

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import unregistered_panel


def test_budget_selection_uses_ranked_prefix_not_set_order():
    U = pd.read_parquet(ROOT / "demo/B/unreg_ranked.parquet")
    budget = float(U.hours.iloc[:3].sum())
    selected = unregistered_panel.budget_selection(U, budget)
    assert list(selected.index) == list(U.index[:3])
    assert selected.expected_rp.tolist() == (U.p_unregistered.iloc[:3] * U.est_rp.iloc[:3]).tolist()


@pytest.mark.parametrize("branch", ["A", "B", "C"])
def test_render_real_data_and_preserves_native_widgets(branch):
    results_dir = ROOT / ("missing_results" if branch == "B" else "results")
    script = f'''
from pathlib import Path
import pandas as pd
from unregistered_panel import render
ROOT = Path(r"{ROOT}")
U = pd.read_parquet(ROOT / "demo/{branch}/unreg_ranked.parquet")
UT = pd.read_parquet(ROOT / "demo/{branch}/truth/unregistered_truth.parquet").set_index("nib")
def jt(x): return "Rp" + str(round(x / 1e6, 1)) + " jt"
def rp(x): return "Rp" + str(round(x))
def ribu(x): return str(x)
def angka(x, d=1): return str(round(x, d))
def kali(x): return str(round(x, 1)) + "x"
def pct_range(a,b,c): return str(round(a, 2))
def load_csv(path): return pd.read_csv(path)
render(branch="{branch}", U=U, UT=UT, sektor={{"I":"Akomodasi", "C":"Manufaktur"}},
       skala={{"medium":"Menengah"}}, results_dir=Path(r"{results_dir}"), jt=jt,
       rp=rp, ribu=ribu, angka=angka, kali=kali, pct_range=pct_range, load_csv=load_csv)
'''
    app = AppTest.from_string(script, default_timeout=120).run()
    assert not app.exception
    assert len(app.dataframe[0].value) == 800
    assert app.slider(key="unreg_budget")
    assert app.selectbox(key="unreg_pick")
    app.button(key="unreg_preset_5").click().run()
    assert not app.exception
    assert app.slider(key="unreg_budget").value <= int(pd.read_parquet(ROOT / f"demo/{branch}/unreg_ranked.parquet").hours.sum() * 0.05)
    app.button(key="unreg_card_2").click().run()
    assert not app.exception
    assert app.selectbox(key="unreg_pick").value == str(pd.read_parquet(ROOT / f"demo/{branch}/unreg_ranked.parquet").index[1])
    labels = [m.label for m in app.metric]
    assert "Perkiraan iuran 12 bulan" in labels
    assert "Usaha dalam anggaran" in labels
    assert "Perkiraan Rp per jam" in labels
    assert "Pemeriksaan bersama TK" in labels
    if branch == "A":
        assert any("Cabang A dipakai untuk melatih" in w.value for w in app.warning)
        assert "Dibanding aturan hukum" not in labels
    else:
        assert "Dibanding aturan hukum" in labels
        assert any("Keunggulan tipis" in c.value for c in app.caption)
    nib = str(pd.read_parquet(ROOT / f"demo/{branch}/unreg_ranked.parquet").index[5])
    app.selectbox(key="unreg_pick").set_value(nib).run()
    assert not app.exception
    assert any(h.value == pd.read_parquet(ROOT / f"demo/{branch}/unreg_ranked.parquet").iloc[5]["name"] for h in app.subheader)


def test_html_escaping_never_interprets_business_names():
    assert unregistered_panel._html('<img src=x onerror="alert(1)">') == '&lt;img src=x onerror=&quot;alert(1)&quot;&gt;'


def test_source_has_no_forbidden_glyphs_or_truth_ranking():
    text = (ROOT / "app/unregistered_panel.py").read_text(encoding="utf-8")
    assert not any(g in text for g in ("\u2014", "\u2013", "\u00b7", "\u2026"))
    assert "UT.loc[nib]" in text
