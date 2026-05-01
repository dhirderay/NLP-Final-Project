"""Sentence-transformer model wrappers with a uniform encode() API.

Handles per-model input formatting quirks (E5 wants a 'query: ' prefix;
bge-m3 doesn't) so callers can swap models without thinking about it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    multilingual: bool
    e5_style: bool = False
    description: str = ""


MODEL_SPECS: list[ModelSpec] = [
    ModelSpec(
        key="LaBSE",
        hf_id="sentence-transformers/LaBSE",
        multilingual=True,
        description="Explicit cross-lingual alignment (109 languages).",
    ),
    ModelSpec(
        key="MiniLM-multi",
        hf_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        multilingual=True,
        description="Distilled multilingual paraphrase model.",
    ),
    ModelSpec(
        key="E5-multi",
        hf_id="intfloat/multilingual-e5-small",
        multilingual=True,
        e5_style=True,
        description="Contrastive multilingual; requires 'query: ' prefix.",
    ),
    ModelSpec(
        key="E5-large",
        hf_id="intfloat/multilingual-e5-large",
        multilingual=True,
        e5_style=True,
        description="Larger E5; same prefix convention as E5-multi.",
    ),
    ModelSpec(
        key="bge-m3",
        hf_id="BAAI/bge-m3",
        multilingual=True,
        e5_style=False,
        description="BAAI multilingual; no input prefix required.",
    ),
    ModelSpec(
        key="MiniLM-en",
        hf_id="sentence-transformers/all-MiniLM-L6-v2",
        multilingual=False,
        description="English-only; used for the translation-pivot baseline.",
    ),
]


class EmbedModel:
    """Thin wrapper around SentenceTransformer that respects model quirks."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self._model = SentenceTransformer(spec.hf_id)

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.spec.e5_style:
            texts = [f"query: {t}" for t in texts]
        embs = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embs


def get_spec(key: str) -> ModelSpec:
    for s in MODEL_SPECS:
        if s.key == key:
            return s
    raise KeyError(key)
