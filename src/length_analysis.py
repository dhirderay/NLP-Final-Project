"""Length-confound analysis.

Whitespace-tokenises anchor/near/far, then correlates
|len(anchor) - len(far)| (in tokens) with the cosine gap. Reports
Pearson and Spearman per (model, condition).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def _tokens(s: str) -> list[str]:
    return [t for t in re.split(r"\s+", str(s).strip()) if t]


def length_table(per_triplet: pd.DataFrame, triplets_json_path: str | Path) -> pd.DataFrame:
    """Attach token-length features to per-triplet rows.

    Length is measured in the field the row actually used: native rows
    use the source-language sentences, english_pivot rows use the _en
    versions.
    """
    raw = json.loads(Path(triplets_json_path).read_text(encoding="utf-8"))
    by_id: dict[str, dict] = {t["id"]: t for t in raw["triplets"]}

    def lens(row):
        t = by_id.get(row["triplet_id"])
        if t is None:
            return pd.Series({"len_anchor": np.nan, "len_near": np.nan, "len_far": np.nan, "len_diff_anchor_far": np.nan})
        if row["condition"] == "english_pivot":
            a, n, f = t.get("anchor_en", ""), t.get("near_en", ""), t.get("far_en", "")
        else:
            a, n, f = t.get("anchor", ""), t.get("near", ""), t.get("far", "")
        return pd.Series(
            {
                "len_anchor": len(_tokens(a)),
                "len_near": len(_tokens(n)),
                "len_far": len(_tokens(f)),
                "len_diff_anchor_far": abs(len(_tokens(a)) - len(_tokens(f))),
            }
        )

    augmented = per_triplet.copy()
    augmented = pd.concat([augmented, augmented.apply(lens, axis=1)], axis=1)
    return augmented


def correlations(augmented: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, condition), sub in augmented.groupby(["model", "condition"]):
        x = sub["len_diff_anchor_far"].to_numpy(dtype=float)
        y = sub["cos_gap"].to_numpy(dtype=float)
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 5:
            continue
        x, y = x[mask], y[mask]
        try:
            pr = stats.pearsonr(x, y)
            sr = stats.spearmanr(x, y)
            rows.append(
                {
                    "model": model,
                    "condition": condition,
                    "n": int(mask.sum()),
                    "pearson_r": float(pr.statistic) if hasattr(pr, "statistic") else float(pr[0]),
                    "pearson_p": float(pr.pvalue) if hasattr(pr, "pvalue") else float(pr[1]),
                    "spearman_rho": float(sr.statistic) if hasattr(sr, "statistic") else float(sr[0]),
                    "spearman_p": float(sr.pvalue) if hasattr(sr, "pvalue") else float(sr[1]),
                }
            )
        except Exception:
            continue
    return pd.DataFrame(rows).sort_values(["condition", "model"]).reset_index(drop=True)
