"""Triplet accuracy and cosine-gap evaluation."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd

from .embed import EmbedModel
from .flattening_labels import flattening_label
from .load_data import Triplet


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    # Inputs are already L2-normalised in EmbedModel.encode.
    return float(np.dot(a, b))


def evaluate_triplets(
    model: EmbedModel,
    triplets: List[Triplet],
    *,
    text_field: str,
    model_key: str,
    condition: str,
) -> pd.DataFrame:
    """Score every triplet with one model under one text-field.

    text_field="native" reads anchor/near/far; "english" reads the _en
    variants. condition is a free-form label that ends up in the output
    rows ("native" or "english_pivot" in the existing pipeline).
    """
    if text_field not in {"native", "english"}:
        raise ValueError(text_field)

    anchors, nears, fars = [], [], []
    for t in triplets:
        if text_field == "native":
            a, n, f = t.native()
        else:
            a, n, f = t.english()
        anchors.append(a)
        nears.append(n)
        fars.append(f)

    anchor_emb = model.encode(anchors)
    near_emb = model.encode(nears)
    far_emb = model.encode(fars)

    rows = []
    for i, t in enumerate(triplets):
        a, n, f = anchor_emb[i], near_emb[i], far_emb[i]
        sim_an = _cos(a, n)
        sim_af = _cos(a, f)
        gap = sim_an - sim_af
        correct = int(sim_an > sim_af)
        rows.append(
            {
                "triplet_id": t.id,
                "language": t.language,
                "category": t.category,
                "flattening": flattening_label(t.id, t.category),
                "model": model_key,
                "condition": condition,
                "sim_anchor_near": sim_an,
                "sim_anchor_far": sim_af,
                "cos_gap": gap,
                "correct": correct,
            }
        )
    return pd.DataFrame(rows)


def aggregate(per_triplet: pd.DataFrame, by: Iterable[str]) -> pd.DataFrame:
    by = list(by)
    grouped = per_triplet.groupby(by, dropna=False)
    out = grouped.agg(
        triplet_accuracy=("correct", "mean"),
        mean_cosine_gap=("cos_gap", "mean"),
        std_cosine_gap=("cos_gap", "std"),
        n_items=("correct", "size"),
    ).reset_index()
    out["std_cosine_gap"] = out["std_cosine_gap"].fillna(0.0)
    return out
