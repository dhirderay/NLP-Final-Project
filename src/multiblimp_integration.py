"""Pull Hindi gender-agreement items from MultiBLiMP and convert to our schema.

Result is written to data/multiblimp_hindi_gender.json with both the
sampled items and a 200-sentence pool used to calibrate per-model
pair-pass thresholds in src.hybrid_validation.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "multiblimp_hindi_gender.json"


def fetch_hindi_rows() -> list[dict[str, Any]]:
    path = hf_hub_download(
        repo_id="jumelet/multiblimp",
        filename="hin/data.tsv",
        repo_type="dataset",
    )
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def filter_gender(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only subject-verb and subject-predicate gender-agreement items."""
    return [r for r in rows if r["phenomenon"] in {"SV-G", "SP-G"}]


def sample_n(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    if len(rows) <= n:
        return list(rows)
    return rng.sample(rows, n)


def to_record(row: dict[str, Any], idx: int) -> dict[str, Any]:
    return {
        "id": f"mbp_hi_{idx:03d}",
        "language": "hindi",
        "category": "verb_gender_agreement",
        "source": "multiblimp",
        "phenomenon": row["phenomenon"],
        "grammatical": row["sen"],
        "ungrammatical": row["wrong_sen"],
        "feature": "gender",
        "feature_value_grammatical": row["grammatical_feature"],
        "feature_value_ungrammatical": row["ungrammatical_feature"],
        "verb_gram": row.get("verb", ""),
        "verb_ungram": row.get("swap_head", ""),
        "child": row.get("child", ""),
        "head": row.get("head", ""),
        "feature_vals": row["feature_vals"],
    }


def build_threshold_pool(rows: list[dict[str, Any]], n: int, seed: int) -> list[str]:
    """Sample n unrelated sentences for per-model pair-pass threshold calibration."""
    rng = random.Random(seed + 1)
    sens = list({r["sen"] for r in rows})
    rng.shuffle(sens)
    return sens[:n]


def main() -> None:
    rows = fetch_hindi_rows()
    print(f"Fetched {len(rows)} Hindi MultiBLiMP rows")
    gender_rows = filter_gender(rows)
    print(f"Gender-agreement rows: {len(gender_rows)}")

    sampled = sample_n(gender_rows, n=100, seed=20260501)
    print(f"Sampled {len(sampled)} for evaluation")

    items = [to_record(r, i + 1) for i, r in enumerate(sampled)]

    threshold_pool = build_threshold_pool(rows, n=200, seed=20260501)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "source": "jumelet/multiblimp (Jumelet et al. 2025)",
                    "language": "hindi",
                    "category": "verb_gender_agreement",
                    "phenomena": ["SV-G", "SP-G"],
                    "n_items": len(items),
                    "seed": 20260501,
                    "threshold_pool_size": len(threshold_pool),
                },
                "items": items,
                "threshold_pool": threshold_pool,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
