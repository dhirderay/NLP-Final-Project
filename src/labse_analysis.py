"""LaBSE-style cross-lingual-collapse analysis on ser_vs_estar.

The hypothesis: a translation-equivalence training objective squeezes
intra-language polysemous forms (anchor_es, far_es) onto the same
English embedding because anchor_en is roughly equal to far_en under
default translation. We measure that directly via cross-lingual cosines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.embed import EmbedModel


def labse_ser_estar_table(per_triplet: pd.DataFrame) -> pd.DataFrame:
    """Per-item cosines across multilingual models on ser_vs_estar."""
    sub = per_triplet[
        (per_triplet["category"] == "ser_vs_estar")
        & (per_triplet["condition"] == "native")
    ]
    pivot = sub.pivot_table(
        index=["triplet_id", "language", "category", "flattening"],
        columns="model",
        values=["sim_anchor_near", "sim_anchor_far", "cos_gap"],
        aggfunc="first",
    )
    pivot.columns = [f"{a}__{b}" for a, b in pivot.columns]
    return pivot.reset_index()


def cross_lingual_collapse(
    triplets: list[dict],
    model: EmbedModel,
) -> pd.DataFrame:
    """Embed Spanish and English forms of anchor/far on ser/estar items.

    The output cosines tell us whether the model is collapsing the
    Spanish anchor and the Spanish far onto the same English vector
    (cos(anchor_es, far_en) and cos(far_es, anchor_en)) and whether the
    cross-lingual alignment is symmetric (cos(anchor_es, anchor_en) vs
    cos(far_es, far_en)).
    """
    items = [t for t in triplets if t.get("category") == "ser_vs_estar"]
    if not items:
        return pd.DataFrame()
    anchors = [t["anchor"] for t in items]
    fars = [t["far"] for t in items]
    anchors_en = [t["anchor_en"] for t in items]
    fars_en = [t["far_en"] for t in items]
    a_es = model.encode(anchors)
    f_es = model.encode(fars)
    a_en = model.encode(anchors_en)
    f_en = model.encode(fars_en)

    def _cos(x, y):
        return np.einsum("ij,ij->i", x, y)

    rows = []
    for i, t in enumerate(items):
        rows.append(
            {
                "triplet_id": t["id"],
                "model": model.spec.key,
                "anchor_es_to_anchor_en": float(_cos(a_es[i:i+1], a_en[i:i+1])[0]),
                "far_es_to_far_en": float(_cos(f_es[i:i+1], f_en[i:i+1])[0]),
                "anchor_es_to_far_en": float(_cos(a_es[i:i+1], f_en[i:i+1])[0]),
                "far_es_to_anchor_en": float(_cos(f_es[i:i+1], a_en[i:i+1])[0]),
                "anchor_es_to_far_es": float(_cos(a_es[i:i+1], f_es[i:i+1])[0]),
                "anchor_en_to_far_en": float(_cos(a_en[i:i+1], f_en[i:i+1])[0]),
                "flattening": t.get("flattening_intent", "?"),
            }
        )
    return pd.DataFrame(rows)
