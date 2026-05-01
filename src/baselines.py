"""TF-IDF surface-form baseline.

The English-translation pivot baseline uses evaluate_triplets() with
the English-only model on the _en fields, so it lives in run_all.py
rather than here.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .flattening_labels import flattening_label
from .load_data import Triplet


def tfidf_eval(
    triplets: List[Triplet],
    *,
    text_field: str,
    condition_label: str,
) -> pd.DataFrame:
    """Score triplets with a character-n-gram TF-IDF cosine.

    Character n-grams (3-5) work across both Devanagari and Latin
    scripts without language-specific tokenisers, and give us a fair
    "surface form only" baseline: semantic encoders that beat this are
    doing more than character overlap.
    """
    if text_field not in {"native", "english"}:
        raise ValueError(text_field)

    sentences: List[str] = []
    for t in triplets:
        a, n, f = t.native() if text_field == "native" else t.english()
        sentences.extend([a, n, f])

    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
        min_df=1,
    )
    X = vec.fit_transform(sentences).toarray()
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = X / norms

    rows = []
    for i, t in enumerate(triplets):
        a, n, f = X[3 * i], X[3 * i + 1], X[3 * i + 2]
        sim_an = float(np.dot(a, n))
        sim_af = float(np.dot(a, f))
        rows.append(
            {
                "triplet_id": t.id,
                "language": t.language,
                "category": t.category,
                "flattening": flattening_label(t.id, t.category),
                "model": "TF-IDF",
                "condition": condition_label,
                "sim_anchor_near": sim_an,
                "sim_anchor_far": sim_af,
                "cos_gap": sim_an - sim_af,
                "correct": int(sim_an > sim_af),
            }
        )
    return pd.DataFrame(rows)
