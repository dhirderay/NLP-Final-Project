"""Load and validate the triplets.json file."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


REQUIRED_KEYS = (
    "id",
    "language",
    "category",
    "anchor",
    "near",
    "far",
    "anchor_en",
    "near_en",
    "far_en",
)


@dataclass(frozen=True)
class Triplet:
    id: str
    language: str
    category: str
    anchor: str
    near: str
    far: str
    anchor_en: str
    near_en: str
    far_en: str
    notes: str = ""

    def native(self) -> tuple[str, str, str]:
        return self.anchor, self.near, self.far

    def english(self) -> tuple[str, str, str]:
        return self.anchor_en, self.near_en, self.far_en


def load_triplets(path: str | Path = "triplets.json") -> List[Triplet]:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw["triplets"]
    triplets: List[Triplet] = []
    seen_ids: set[str] = set()
    for it in items:
        for k in REQUIRED_KEYS:
            if k not in it:
                raise ValueError(f"Triplet missing key {k}: {it.get('id', '?')}")
        if it["id"] in seen_ids:
            raise ValueError(f"Duplicate triplet id: {it['id']}")
        seen_ids.add(it["id"])
        triplets.append(
            Triplet(
                id=it["id"],
                language=it["language"],
                category=it["category"],
                anchor=it["anchor"],
                near=it["near"],
                far=it["far"],
                anchor_en=it["anchor_en"],
                near_en=it["near_en"],
                far_en=it["far_en"],
                notes=it.get("notes", ""),
            )
        )
    return triplets


def filter_triplets(
    triplets: Iterable[Triplet],
    language: str | None = None,
    category: str | None = None,
    ids: Iterable[str] | None = None,
) -> List[Triplet]:
    out = list(triplets)
    if language is not None:
        out = [t for t in out if t.language == language]
    if category is not None:
        out = [t for t in out if t.category == category]
    if ids is not None:
        ids_set = set(ids)
        out = [t for t in out if t.id in ids_set]
    return out
