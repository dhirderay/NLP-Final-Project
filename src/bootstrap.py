"""Bootstrap CIs for triplet accuracy and paired comparisons.

Defaults (10k resamples, percentile method, seed 20260501) match the
pre-registration. Paired comparisons resample triplet IDs jointly so
the per-item correlation between the two conditions is preserved in
the resampling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_N = 10_000
DEFAULT_SEED = 20260501


@dataclass
class CIResult:
    point: float
    lo: float
    hi: float
    width: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.point, self.lo, self.hi, self.width


def _percentile_ci(samples: np.ndarray, alpha: float) -> tuple[float, float]:
    return float(np.percentile(samples, 100 * (alpha / 2))), float(np.percentile(samples, 100 * (1 - alpha / 2)))


def bootstrap_mean_ci(values: np.ndarray, *, alpha: float = 0.05, n: int = DEFAULT_N, seed: int = DEFAULT_SEED) -> CIResult:
    """Mean of `values` with percentile bootstrap CI."""
    rng = np.random.default_rng(seed)
    if len(values) == 0:
        return CIResult(float("nan"), float("nan"), float("nan"), float("nan"))
    idx = rng.integers(0, len(values), size=(n, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = _percentile_ci(means, alpha)
    return CIResult(float(values.mean()), lo, hi, hi - lo)


def paired_bootstrap_diff_ci(
    a_values: np.ndarray,
    b_values: np.ndarray,
    *,
    alpha: float = 0.05,
    n: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
) -> CIResult:
    """CI for mean(a) - mean(b) with paired resampling.

    Caller must align the two arrays on item index (a_values[i] and
    b_values[i] are the per-item statistic for the same triplet) so
    that resampling preserves the within-item correlation.
    """
    if len(a_values) != len(b_values):
        raise ValueError("a and b must have the same length")
    if len(a_values) == 0:
        return CIResult(float("nan"), float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_items = len(a_values)
    idx = rng.integers(0, n_items, size=(n, n_items))
    diffs = (a_values[idx].mean(axis=1) - b_values[idx].mean(axis=1))
    lo, hi = _percentile_ci(diffs, alpha)
    return CIResult(float(a_values.mean() - b_values.mean()), lo, hi, hi - lo)


def add_accuracy_cis(
    per_triplet: pd.DataFrame,
    by: list[str],
    *,
    alpha: float = 0.05,
    n: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Group `per_triplet` by `by` and attach 95% CIs on triplet accuracy."""
    rows = []
    for keys, sub in per_triplet.groupby(by):
        ci = bootstrap_mean_ci(sub["correct"].to_numpy(dtype=float), alpha=alpha, n=n, seed=seed)
        gap_ci = bootstrap_mean_ci(sub["cos_gap"].to_numpy(dtype=float), alpha=alpha, n=n, seed=seed)
        if not isinstance(keys, tuple):
            keys = (keys,)
        rows.append(
            {
                **{k: v for k, v in zip(by, keys)},
                "triplet_accuracy": ci.point,
                "triplet_accuracy_lo": ci.lo,
                "triplet_accuracy_hi": ci.hi,
                "mean_cosine_gap": gap_ci.point,
                "mean_cosine_gap_lo": gap_ci.lo,
                "mean_cosine_gap_hi": gap_ci.hi,
                "n_items": len(sub),
            }
        )
    return pd.DataFrame(rows).sort_values(by).reset_index(drop=True)


def paired_headline(
    per_triplet: pd.DataFrame,
    *,
    multilingual_models: list[str],
    pivot_model: str = "MiniLM-en",
    alpha_bonferroni: float = 0.01,
    alpha_uncorrected: float = 0.05,
    n: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Per-category paired bootstrap of (best_multi - pivot)."""
    rows = []
    for (lang, cat), sub in per_triplet.groupby(["language", "category"]):
        flattening = sub["flattening"].mode().iloc[0]
        piv = sub[(sub["model"] == pivot_model) & (sub["condition"] == "english_pivot")]
        if len(piv) == 0:
            continue
        best_model = None
        best_acc = -1.0
        for m in multilingual_models:
            mm = sub[(sub["model"] == m) & (sub["condition"] == "native")]
            if len(mm) == 0:
                continue
            acc = mm["correct"].mean()
            if acc > best_acc:
                best_acc = acc
                best_model = m
        if best_model is None:
            continue
        mm = sub[(sub["model"] == best_model) & (sub["condition"] == "native")]
        ids = sorted(set(piv["triplet_id"]).intersection(mm["triplet_id"]))
        if not ids:
            continue
        piv_acc = piv.set_index("triplet_id").loc[ids, "correct"].to_numpy(dtype=float)
        mm_acc = mm.set_index("triplet_id").loc[ids, "correct"].to_numpy(dtype=float)
        ci99 = paired_bootstrap_diff_ci(mm_acc, piv_acc, alpha=alpha_bonferroni, n=n, seed=seed)
        ci95 = paired_bootstrap_diff_ci(mm_acc, piv_acc, alpha=alpha_uncorrected, n=n, seed=seed)
        rows.append(
            {
                "language": lang,
                "category": cat,
                "flattening_type": flattening,
                "best_multilingual_model": best_model,
                "best_multilingual_acc": float(mm_acc.mean()),
                "english_pivot_acc": float(piv_acc.mean()),
                "gap": ci95.point,
                "gap_ci95_lo": ci95.lo,
                "gap_ci95_hi": ci95.hi,
                "gap_ci99_lo": ci99.lo,
                "gap_ci99_hi": ci99.hi,
                "headline_significant": bool(ci99.lo > 0 or ci99.hi < 0),
                "n_items": len(ids),
            }
        )
    return pd.DataFrame(rows).sort_values(["flattening_type", "language", "category"]).reset_index(drop=True)
