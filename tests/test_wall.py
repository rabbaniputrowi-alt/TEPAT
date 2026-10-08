"""Guards the truth/observed wall and the legal rules that TEPAT encodes on purpose."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import pytest

from tepat import rules as R
from tepat import scorer
from tepat.generator import generate_branch, load_config

SRC = Path(__file__).resolve().parents[1] / "src" / "tepat"
CFG = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            out.add(("." * node.level) + (node.module or ""))
    return out


def test_scorer_never_imports_generator_or_bench():
    imps = _imports(SRC / "scorer.py")
    assert not any("generator" in i or "bench" in i for i in imps), imps


def test_scorer_never_mentions_truth_dir():
    tree = ast.parse((SRC / "scorer.py").read_text(encoding="utf-8"))
    docs = {id(n.body[0].value) for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body
            and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    lits = [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs]
    assert not [s for s in lits if "truth" in s], "scorer.py must not reference truth/ in code"


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    cfg = load_config(CFG)
    cfg["n_employers"] = 120
    out = tmp_path_factory.mktemp("tiny") / "A"
    generate_branch(cfg, "A", out)
    return out


def test_observed_has_no_label_columns(tiny):
    banned = {"is_violator", "variants", "lookalikes", "tags", "wage_multiplier", "recoverable_rp_true"}
    for f in (tiny / "observed").glob("*.parquet"):
        cols = set(pd.read_parquet(f).columns)
        assert not cols & banned, (f.name, cols & banned)


def test_features_build_from_observed_only(tiny):
    F = scorer.build_features(scorer.load_observed(tiny))
    assert len(F) == 120
    assert F.index.is_unique


def test_niks_are_synthetic(tiny):
    ks = pd.read_parquet(tiny / "observed" / "ks_membership.parquet")
    assert ks.nik.str.startswith("99").all()          # 99 is not a real province code
    assert ks.nik.str.len().eq(16).all()


def test_config_is_saved_with_data(tiny):
    assert json.loads((tiny / "config.json").read_text(encoding="utf-8"))["branch"] == "A"


# ---- legal rules (Evidence Bank references in rules.py)

def test_iuran_floor_applies_only_to_medium_and_large():
    umk = 2_600_000
    assert R.iuran_ks(2_000_000, umk, "micro") == pytest.approx(100_000)     # E05c: no UMK floor
    assert R.iuran_ks(2_000_000, umk, "small") == pytest.approx(100_000)
    assert R.iuran_ks(2_000_000, umk, "medium") == pytest.approx(130_000)    # floored at UMK
    assert not R.below_umk_is_violation("micro")
    assert R.below_umk_is_violation("large")


def test_wage_cap():
    assert R.iuran_ks(50_000_000, 2_600_000, "large") == pytest.approx(600_000)  # 5% of Rp12M


def test_calendar():
    assert R.year_of_month(0) == 2024 and R.year_of_month(3) == 2025 and R.year_of_month(15) == 2026
    assert R.HORIZON_DAYS == 720 and R.SCORING_DAY == 719
