"""Registration pass (variants 1e/1f): same truth/observed wall as the main scorer, plus the legal rule."""
from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from tepat import registration as RG
from tepat.generator import generate_branch, load_config
from tepat.unreg_gen import generate_unregistered, load_ucfg

SRC = Path(__file__).resolve().parents[1] / "src" / "tepat"
CFG = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


def _imports(path: Path) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            out.add((node.module or ""))
    return out


def test_registration_never_imports_generators_or_bench():
    imps = _imports(SRC / "registration.py")
    assert not any(x in i for i in imps for x in ("generator", "unreg_gen", "bench")), imps


def test_registration_never_mentions_truth_dir():
    tree = ast.parse((SRC / "registration.py").read_text(encoding="utf-8"))
    docs = {id(n.body[0].value) for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body
            and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    lits = [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs]
    assert not [s for s in lits if "truth" in s]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    cfg = load_config(CFG)
    cfg["n_employers"] = 120
    out = tmp_path_factory.mktemp("tinyu") / "A"
    generate_branch(cfg, "A", out)
    before = {p.name: p.stat().st_mtime_ns for p in (out / "observed").glob("*.parquet")}
    u = load_ucfg()
    u["n_candidates"] = 150
    generate_unregistered(out, "A", 1, u)
    return out, before


def test_main_branch_files_untouched(tiny):
    out, before = tiny
    after = {p.name: p.stat().st_mtime_ns for p in (out / "observed").glob("*.parquet") if p.name in before}
    assert after == before


def test_new_observed_tables_have_no_labels(tiny):
    out, _ = tiny
    banned = {"is_violator", "kind", "variants", "lookalikes", "true_workers", "recoverable_rp_true", "gap_workers"}
    for t in ("oss_registry", "tk_unregistered"):
        cols = set(pd.read_parquet(out / "observed" / f"{t}.parquet").columns)
        assert not cols & banned, (t, cols & banned)


def test_candidates_are_exactly_the_unregistered(tiny):
    out, _ = tiny
    F = RG.build_features(RG.load(out))
    T = pd.read_parquet(out / "truth" / "unregistered_truth.parquet")
    reg = set(pd.read_parquet(out / "observed" / "employer_registry.parquet").nib.astype(str))
    assert set(F.index) == set(T.nib.astype(str))
    assert not set(F.index) & reg


def test_legal_rule_needs_npwp_and_two_workers(tiny):
    out, _ = tiny
    F = RG.build_features(RG.load(out))
    ok = (F.has_npwp > 0) & (F.est_workers >= 2)
    assert (F.rule_obliged.astype(bool) == ok).all()
    for i in F.index[~ok][:20]:
        assert RG.action(F.loc[i]).startswith("Tidak wajib")


def test_glyph_free_reasons(tiny):
    out, _ = tiny
    F = RG.build_features(RG.load(out))
    text = " ".join(" ".join(RG.reasons(F.loc[i])) + RG.action(F.loc[i]) for i in F.index[:60])
    assert not any(g in text for g in ("\u2014", "\u2013", "\u00b7", "\u2026"))
