"""Flattening labels per item.

Three buckets:
  strong          English cannot express the distinction without a gloss.
  partial         English can express it, but a default translation drops it.
  non_flattening  English does distinguish; included as sanity checks.

The lookup is by category, with explicit per-item overrides for the
ser/estar non-flattening exceptions (es_ser_012/013/014: aburrido,
listo, malo - English distinguishes boring/bored, smart/ready, bad/sick).
"""

from __future__ import annotations

NON_FLATTENING_IDS = {"es_ser_012", "es_ser_013", "es_ser_014"}

STRONG_CATEGORIES = {
    "kinship_paternal_maternal",
    "verb_gender_agreement",
    "formality_tv",
    "formality",
    "gender_agreement_adjectives",
}

PARTIAL_CATEGORIES = {
    "kinship_relative_age",
    "ser_vs_estar",
}


def flattening_label(triplet_id: str, category: str) -> str:
    if triplet_id in NON_FLATTENING_IDS:
        return "non_flattening"
    if category in STRONG_CATEGORIES:
        return "strong"
    if category in PARTIAL_CATEGORIES:
        return "partial"
    raise ValueError(f"Unknown category for flattening label: {category}")
