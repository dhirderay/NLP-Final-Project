"""Phase 1 vs Phase 2B replication.

For each (model x language x category x condition) cell with items from
both phases, reports per-phase accuracy with bootstrap CIs and a 95%
unpaired-bootstrap CI on the phase2b - phase1 difference. The two phases
draw from disjoint item pools (no shared triplets), so resampling is
independent rather than paired.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.bootstrap import bootstrap_mean_ci, DEFAULT_N, DEFAULT_SEED


PHASE1_PREFIXES = ("hi_kin_", "hi_age_", "hi_for_", "hi_vga_", "es_ser_", "es_for_", "es_gen_")
PHASE2B_PREFIXES = ("hi_kin2_", "hi_age2_", "hi_for2_", "es_ser2_", "es_for2_", "es_gen2_")


def _phase(triplet_id: str) -> str:
    if triplet_id.startswith(PHASE2B_PREFIXES):
        return "phase2b"
    if triplet_id.startswith(PHASE1_PREFIXES):
        return "phase1"
    return "other"


def replication_table(
    per_triplet: pd.DataFrame,
    *,
    multi_models: Iterable[str],
) -> pd.DataFrame:
    df = per_triplet.copy()
    df["phase"] = df["triplet_id"].apply(_phase)

    rows = []
    multi_models = list(multi_models)
    for (model, lang, cat, cond), sub in df.groupby(["model", "language", "category", "condition"]):
        p1 = sub[sub["phase"] == "phase1"]["correct"].to_numpy(dtype=float)
        p2 = sub[sub["phase"] == "phase2b"]["correct"].to_numpy(dtype=float)
        if len(p1) == 0 or len(p2) == 0:
            continue
        ci_p1 = bootstrap_mean_ci(p1, n=DEFAULT_N, seed=DEFAULT_SEED)
        ci_p2 = bootstrap_mean_ci(p2, n=DEFAULT_N, seed=DEFAULT_SEED)
        rng = np.random.default_rng(DEFAULT_SEED + 7)
        n_p1, n_p2 = len(p1), len(p2)
        idx1 = rng.integers(0, n_p1, size=(DEFAULT_N, n_p1))
        idx2 = rng.integers(0, n_p2, size=(DEFAULT_N, n_p2))
        diffs = p2[idx2].mean(axis=1) - p1[idx1].mean(axis=1)
        diff_lo, diff_hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
        rows.append(
            {
                "model": model,
                "language": lang,
                "category": cat,
                "condition": cond,
                "phase1_n": int(n_p1),
                "phase1_acc": ci_p1.point,
                "phase1_lo": ci_p1.lo,
                "phase1_hi": ci_p1.hi,
                "phase2b_n": int(n_p2),
                "phase2b_acc": ci_p2.point,
                "phase2b_lo": ci_p2.lo,
                "phase2b_hi": ci_p2.hi,
                "diff_phase2b_minus_phase1": ci_p2.point - ci_p1.point,
                "diff_lo": diff_lo,
                "diff_hi": diff_hi,
                "consistent": bool(diff_lo <= 0 <= diff_hi),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "language", "category", "condition"]).reset_index(drop=True)
