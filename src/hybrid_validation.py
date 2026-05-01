"""MultiBLiMP pair-distinction metric and triplet-vs-pair cross-validation.

The pair metric is

    pair_distinction(item) = 1 - cos(grammatical_emb, ungrammatical_emb)

A model that ignores the agreement violation has pair_distinction near 0;
a model that's sensitive to it scores higher. We also report a binary
pair_pass = cos(gram, ungram) < threshold, where threshold is the median
cosine over 200 random non-paired sentences from the same pool, frozen
per model.

cross_validate_rankings() uses Spearman rho between Phase-1 triplet
accuracy and mean_pair_distinction on verb_gender_agreement to test
whether the two evaluation frameworks induce the same model ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from src.embed import EmbedModel


@dataclass(frozen=True)
class HybridResult:
    model: str
    n_items: int
    mean_pair_distinction: float
    threshold_cos: float
    pair_pass_rate: float


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", a, b)


def calibrate_threshold(model: EmbedModel, pool: list[str], *, n_pairs: int = 200, seed: int = 20260501) -> float:
    rng = np.random.default_rng(seed)
    sents = list(set(pool))
    if len(sents) < 4:
        return 0.5
    embs = model.encode(sents)
    n = min(n_pairs, len(sents) // 2)
    idx_a = rng.integers(0, len(sents), size=n)
    idx_b = rng.integers(0, len(sents), size=n)
    keep = idx_a != idx_b
    a, b = embs[idx_a[keep]], embs[idx_b[keep]]
    cos_vals = _cos(a, b)
    return float(np.median(cos_vals))


def evaluate_multiblimp(
    items: list[dict],
    model: EmbedModel,
    *,
    threshold: float,
) -> tuple[pd.DataFrame, HybridResult]:
    grams = [it["grammatical"] for it in items]
    ungrams = [it["ungrammatical"] for it in items]
    g = model.encode(grams)
    u = model.encode(ungrams)
    cos_vals = _cos(g, u)
    pair_dist = 1.0 - cos_vals
    pair_pass = (cos_vals < threshold).astype(int)

    rows = []
    for it, c, pd_, pp in zip(items, cos_vals, pair_dist, pair_pass):
        rows.append(
            {
                "triplet_id": it["id"],
                "language": it.get("language", "hindi"),
                "category": "verb_gender_agreement",
                "phenomenon": it.get("phenomenon", ""),
                "feature_vals": it.get("feature_vals", ""),
                "model": model.spec.key,
                "cos_grammatical_ungrammatical": float(c),
                "pair_distinction": float(pd_),
                "pair_pass": int(pp),
            }
        )
    df = pd.DataFrame(rows)
    summary = HybridResult(
        model=model.spec.key,
        n_items=len(items),
        mean_pair_distinction=float(pair_dist.mean()),
        threshold_cos=threshold,
        pair_pass_rate=float(pair_pass.mean()),
    )
    return df, summary


def cross_validate_rankings(
    pair_summary: list[HybridResult],
    triplet_per_item: pd.DataFrame,
    *,
    triplet_category: str = "verb_gender_agreement",
    multi_keys: Iterable[str] = ("LaBSE", "MiniLM-multi", "E5-multi", "E5-large", "bge-m3"),
) -> dict:
    """Spearman rho between triplet accuracy and pair-distinction per model."""
    multi_keys = list(multi_keys)
    multi_set = set(multi_keys)
    sub = triplet_per_item[
        (triplet_per_item["category"] == triplet_category)
        & (triplet_per_item["condition"] == "native")
        & (triplet_per_item["model"].isin(multi_set))
    ]
    triplet_acc = sub.groupby("model")["correct"].mean().to_dict()
    pair_pass = {r.model: r.pair_pass_rate for r in pair_summary}
    pair_dist = {r.model: r.mean_pair_distinction for r in pair_summary}

    rows = []
    for m in multi_keys:
        if m in triplet_acc and m in pair_pass:
            rows.append(
                {
                    "model": m,
                    "triplet_accuracy_phase1": triplet_acc[m],
                    "pair_pass_rate_multiblimp": pair_pass[m],
                    "mean_pair_distinction_multiblimp": pair_dist[m],
                }
            )
    df = pd.DataFrame(rows)
    # The binary pair_pass rate is degenerate for our items (every
    # encoder scores ~0.99 cos on these one-suffix-different pairs), so
    # we rank-correlate against the continuous pair_distinction instead.
    if len(df) >= 2 and df["mean_pair_distinction_multiblimp"].nunique() >= 2:
        rho = float(stats.spearmanr(df["triplet_accuracy_phase1"], df["mean_pair_distinction_multiblimp"]).statistic)
    else:
        rho = float("nan")
    return {"per_model": df, "spearman_rho": rho}
