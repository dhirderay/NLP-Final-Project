"""Phase 1 entry point.

Usage:
    python run_all.py             full eval on triplets.json
    python run_all.py --smoke     5-triplet sanity check
    python run_all.py --no-plots  skip figure generation
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from src.baselines import tfidf_eval
from src.embed import MODEL_SPECS, EmbedModel
from src.evaluate import aggregate, evaluate_triplets
from src.load_data import Triplet, load_triplets

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def _select_smoke_triplets(triplets: list[Triplet]) -> list[Triplet]:
    """Pick a representative 5-triplet smoke set spanning all categories."""
    wanted_ids = [
        "hi_kin_001",
        "hi_age_001",
        "hi_for_001",
        "es_ser_001",
        "es_ser_012",
    ]
    by_id = {t.id: t for t in triplets}
    out = [by_id[i] for i in wanted_ids if i in by_id]
    if len(out) < 3:
        out = triplets[:5]
    return out


def _run_one_model(
    spec,
    triplets: list[Triplet],
) -> pd.DataFrame:
    """Score all triplets with one model, including pivot condition where applicable."""
    log = logging.getLogger(__name__)
    log.info("Loading model %s (%s)", spec.key, spec.hf_id)
    t0 = time.time()
    model = EmbedModel(spec)
    log.info("  loaded in %.1fs", time.time() - t0)

    frames: list[pd.DataFrame] = []
    if spec.multilingual:
        frames.append(evaluate_triplets(
            model, triplets, text_field="native",
            model_key=spec.key, condition="native",
        ))
        # Also run on the English translations: useful for figure 2,
        # not the headline.
        frames.append(evaluate_triplets(
            model, triplets, text_field="english",
            model_key=spec.key, condition="english_pivot",
        ))
    else:
        frames.append(evaluate_triplets(
            model, triplets, text_field="english",
            model_key=spec.key, condition="english_pivot",
        ))

    return pd.concat(frames, ignore_index=True)


def run_all(
    triplets: list[Triplet],
    *,
    write_outputs: bool = True,
) -> dict:
    log = logging.getLogger(__name__)
    log.info("Running on %d triplets", len(triplets))

    all_frames: list[pd.DataFrame] = []
    for spec in MODEL_SPECS:
        all_frames.append(_run_one_model(spec, triplets))

    log.info("Running TF-IDF baselines")
    all_frames.append(
        tfidf_eval(triplets, text_field="native", condition_label="native")
    )
    all_frames.append(
        tfidf_eval(triplets, text_field="english", condition_label="english_pivot")
    )

    per_triplet = pd.concat(all_frames, ignore_index=True)

    summary = aggregate(per_triplet, by=["model", "condition", "language", "category"])
    summary_flat = aggregate(
        per_triplet, by=["model", "condition", "flattening"]
    )

    headline = build_headline(per_triplet)

    if write_outputs:
        RESULTS.mkdir(exist_ok=True, parents=True)
        per_triplet.to_csv(RESULTS / "per_triplet.csv", index=False)
        summary.to_csv(RESULTS / "summary.csv", index=False)
        summary_flat.to_csv(RESULTS / "summary_by_flattening.csv", index=False)
        headline.to_csv(RESULTS / "headline.csv", index=False)
        log.info("Wrote results CSVs to %s", RESULTS)

    return {
        "per_triplet": per_triplet,
        "summary": summary,
        "summary_flat": summary_flat,
        "headline": headline,
    }


def build_headline(per_triplet: pd.DataFrame) -> pd.DataFrame:
    """Per-category best-multilingual vs English-pivot table."""
    multi_keys = [s.key for s in MODEL_SPECS if s.multilingual]
    native = per_triplet[
        (per_triplet["model"].isin(multi_keys))
        & (per_triplet["condition"] == "native")
    ]
    pivot = per_triplet[
        (per_triplet["model"] == "MiniLM-en")
        & (per_triplet["condition"] == "english_pivot")
    ]

    rows = []
    cat_keys = (
        per_triplet[["language", "category", "flattening"]]
        .drop_duplicates()
        .sort_values(["language", "category"])
    )
    # Flattening can vary inside a category (es_ser_012/013/014 are
    # non_flattening exceptions inside ser_vs_estar), so we report the
    # category-level mode here.
    for (lang, cat), sub in cat_keys.groupby(["language", "category"]):
        flattening_dominant = (
            per_triplet[(per_triplet["language"] == lang) & (per_triplet["category"] == cat)]
            ["flattening"].mode().iloc[0]
        )
        n_items = int(
            per_triplet[(per_triplet["language"] == lang) & (per_triplet["category"] == cat)]
            ["triplet_id"].nunique()
        )

        nat_cat = native[(native["language"] == lang) & (native["category"] == cat)]
        per_model_acc = nat_cat.groupby("model")["correct"].mean()
        if per_model_acc.empty:
            best_model = ""
            best_acc = float("nan")
        else:
            best_model = per_model_acc.idxmax()
            best_acc = float(per_model_acc.max())

        piv_cat = pivot[(pivot["language"] == lang) & (pivot["category"] == cat)]
        pivot_acc = float(piv_cat["correct"].mean()) if len(piv_cat) else float("nan")

        rows.append(
            {
                "language": lang,
                "category": cat,
                "flattening_type": flattening_dominant,
                "best_multilingual_model": best_model,
                "best_multilingual_acc": best_acc,
                "english_pivot_acc": pivot_acc,
                "gap": best_acc - pivot_acc,
                "n_items": n_items,
            }
        )
    return pd.DataFrame(rows).sort_values(["flattening_type", "language", "category"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="run on 5 triplets")
    parser.add_argument("--no-plots", action="store_true", help="skip figures")
    parser.add_argument(
        "--triplets", type=str, default="triplets.json", help="path to triplets.json"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("run_all")

    triplets = load_triplets(args.triplets)
    log.info("Loaded %d triplets", len(triplets))

    if args.smoke:
        triplets = _select_smoke_triplets(triplets)
        log.info("Smoke test: %d triplets", len(triplets))

    out = run_all(triplets, write_outputs=not args.smoke)

    if args.smoke:
        cols = [
            "triplet_id",
            "language",
            "category",
            "model",
            "condition",
            "sim_anchor_near",
            "sim_anchor_far",
            "cos_gap",
            "correct",
        ]
        with pd.option_context(
            "display.max_rows", 200, "display.width", 200, "display.float_format", "{:.3f}".format
        ):
            print(out["per_triplet"][cols].to_string(index=False))
            print()
            print("Aggregate by model x condition:")
            agg = (
                out["per_triplet"]
                .groupby(["model", "condition"])["correct"]
                .agg(["mean", "size"])
                .reset_index()
            )
            print(agg.to_string(index=False))
        return 0

    if not args.no_plots:
        # Local import: matplotlib is slow to load and we don't want to
        # pay the cost on --no-plots runs.
        from src.plots import build_all_figures

        FIGURES.mkdir(exist_ok=True, parents=True)
        build_all_figures(out, FIGURES)

    print("\n=== Headline (native vs. English pivot) ===")
    with pd.option_context("display.max_rows", 50, "display.width", 200, "display.float_format", "{:.3f}".format):
        print(out["headline"].to_string(index=False))
    print("\n=== Accuracy by model x condition x flattening ===")
    with pd.option_context("display.max_rows", 200, "display.width", 200, "display.float_format", "{:.3f}".format):
        print(out["summary_flat"].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
