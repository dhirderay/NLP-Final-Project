"""Phase 2B full analysis pipeline.

Builds the validated combined dataset (Phase 1 hand-authored + Phase 2B
LLM-generated and dual-LLM-validated + MultiBLiMP), runs every encoder
on the triplets, runs the MultiBLiMP pair metric, and writes all
summaries and figures to results/ and figures/.

The dataset stage reads only from the validation snapshots; the
embedding stage reads only the validated combined dataset. The two
stages don't see each other's intermediate state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import pandas as pd

from src.baselines import tfidf_eval
from src.bootstrap import add_accuracy_cis, paired_headline
from src.embed import MODEL_SPECS, EmbedModel
from src.evaluate import evaluate_triplets
from src.hybrid_validation import (
    calibrate_threshold,
    cross_validate_rankings,
    evaluate_multiblimp,
)
from src.labse_analysis import cross_lingual_collapse, labse_ser_estar_table
from src.length_analysis import correlations as length_corr
from src.length_analysis import length_table
from src.load_data import Triplet, load_triplets
from src.qualitative import select_qualitative
from src.replication_check import replication_table

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
DATA = ROOT / "data"


def build_validated_dataset(
    *,
    phase1_path: Path,
    generated_path: Path,
    primary_log_path: Path,
    dual_log_path: Path | None,
    out_path: Path,
) -> dict:
    """Combine Phase 1 with validated Phase 2B items.

    A Phase 2B item is included only if it
      - survived the auto-filter (the input file is the post-filter snapshot),
      - was tagged 'auto_validated' by the primary LLM validator, and
      - if it was sampled into the dual-LLM proxy review, was also
        tagged 'auto_validated' by the dual validator.

    Anything either validator flagged is dropped, even if the dual review
    only saw 15% of the auto-validated pool. The output dataset is hashed
    in results/dataset_manifest.json.
    """
    log = logging.getLogger("dataset")
    p1 = json.loads(phase1_path.read_text(encoding="utf-8"))["triplets"]
    log.info("Phase 1: %d triplets", len(p1))

    p2 = json.loads(generated_path.read_text(encoding="utf-8"))["items"]
    log.info("Phase 2B (post auto-filter): %d items", len(p2))

    primary = pd.read_csv(primary_log_path)
    primary_pass = set(primary[primary["overall"] == "auto_validated"]["triplet_id"])
    log.info("LLM-validated (primary): %d items", len(primary_pass))

    dual_pass = None
    dual_reviewed = set()
    if dual_log_path is not None and dual_log_path.exists():
        dual = pd.read_csv(dual_log_path)
        dual_reviewed = set(dual["triplet_id"])
        dual_pass = set(dual[dual["overall"] == "auto_validated"]["triplet_id"])
        log.info("Dual-LLM reviewed: %d (passed: %d)", len(dual_reviewed), len(dual_pass))

    p2_validated = []
    for it in p2:
        if it["id"] not in primary_pass:
            continue
        if it["id"] in dual_reviewed and (dual_pass is not None and it["id"] not in dual_pass):
            continue
        it = dict(it)
        it["validation_status"] = "validated"
        # The triplet loader doesn't require these Phase-2B-only fields,
        # but we preserve them for downstream traceability.
        for k in ("source", "flattening_intent", "dialect", "validation_status"):
            it.setdefault(k, "")
        p2_validated.append(it)
    log.info("Phase 2B post-LLM validation: %d items", len(p2_validated))

    combined = p1 + p2_validated
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"triplets": combined, "n": len(combined)}
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    sha = hashlib.sha256(blob).hexdigest()
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Wrote combined dataset (%d items, sha256=%s) -> %s", len(combined), sha[:12], out_path)

    manifest = {
        "phase1_n": len(p1),
        "phase2b_generated_n": len(p2),
        "phase2b_validated_n": len(p2_validated),
        "combined_n": len(combined),
        "sha256": sha,
        "primary_validator": "gpt-4.1",
        "dual_validator": "claude-sonnet-4-6",
    }
    (RESULTS / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def run_embedding_pass(
    triplets: list[Triplet],
    *,
    skip_models: set[str] | None = None,
) -> pd.DataFrame:
    log = logging.getLogger("embed")
    skip_models = skip_models or set()
    frames: list[pd.DataFrame] = []
    for spec in MODEL_SPECS:
        if spec.key in skip_models:
            continue
        log.info("Loading %s ...", spec.key)
        t0 = time.time()
        model = EmbedModel(spec)
        log.info("  loaded in %.1fs", time.time() - t0)
        if spec.multilingual:
            df_native = evaluate_triplets(model, triplets, text_field="native", model_key=spec.key, condition="native")
            df_en = evaluate_triplets(model, triplets, text_field="english", model_key=spec.key, condition="english_pivot")
            frames.extend([df_native, df_en])
        else:
            frames.append(evaluate_triplets(model, triplets, text_field="english", model_key=spec.key, condition="english_pivot"))
    frames.append(tfidf_eval(triplets, text_field="native", condition_label="native"))
    frames.append(tfidf_eval(triplets, text_field="english", condition_label="english_pivot"))
    return pd.concat(frames, ignore_index=True)


def run_hybrid_pass(
    multiblimp_path: Path,
    triplet_per_item: pd.DataFrame,
    *,
    skip_models: set[str] | None = None,
) -> dict:
    log = logging.getLogger("hybrid")
    skip_models = skip_models or set()
    raw = json.loads(multiblimp_path.read_text(encoding="utf-8"))
    items = raw["items"]
    pool = raw.get("threshold_pool", [it["grammatical"] for it in items])

    multi_specs = [s for s in MODEL_SPECS if s.multilingual and s.key not in skip_models]

    per_item_frames: list[pd.DataFrame] = []
    summaries = []
    for spec in multi_specs:
        log.info("Hybrid: %s", spec.key)
        model = EmbedModel(spec)
        threshold = calibrate_threshold(model, pool)
        df, summary = evaluate_multiblimp(items, model, threshold=threshold)
        per_item_frames.append(df)
        summaries.append(summary)
    per_item = pd.concat(per_item_frames, ignore_index=True)
    cv = cross_validate_rankings(summaries, triplet_per_item)
    per_item.to_csv(RESULTS / "multiblimp_per_item.csv", index=False)
    cv["per_model"].to_csv(RESULTS / "hybrid_framework_validation.csv", index=False)
    (RESULTS / "hybrid_summary.json").write_text(
        json.dumps(
            {
                "n_items": len(items),
                "spearman_rho": cv["spearman_rho"],
                "per_model_summary": [
                    {
                        "model": s.model,
                        "n_items": s.n_items,
                        "mean_pair_distinction": s.mean_pair_distinction,
                        "threshold_cos": s.threshold_cos,
                        "pair_pass_rate": s.pair_pass_rate,
                    }
                    for s in summaries
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"per_item": per_item, "summaries": summaries, "cross_validation": cv}


def run_length_analysis(per_triplet: pd.DataFrame, dataset_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    aug = length_table(per_triplet, dataset_path)
    corr = length_corr(aug)
    aug.to_csv(RESULTS / "per_triplet_with_length.csv", index=False)
    corr.to_csv(RESULTS / "length_analysis.csv", index=False)
    return aug, corr


def run_labse_deep_dive(
    per_triplet: pd.DataFrame,
    triplets: list[dict],
    *,
    skip_models: set[str] | None = None,
) -> dict:
    log = logging.getLogger("labse")
    skip_models = skip_models or set()
    pivot_table = labse_ser_estar_table(per_triplet)
    pivot_table.to_csv(RESULTS / "labse_ser_estar_pivot.csv", index=False)

    multi_specs = [s for s in MODEL_SPECS if s.multilingual and s.key not in skip_models]
    frames = []
    for spec in multi_specs:
        log.info("Cross-lingual collapse: %s", spec.key)
        m = EmbedModel(spec)
        frames.append(cross_lingual_collapse(triplets, m))
    cross = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    cross.to_csv(RESULTS / "labse_cross_lingual.csv", index=False)
    return {"per_item": pivot_table, "cross": cross}


def cross_language_comparison(per_triplet: pd.DataFrame) -> pd.DataFrame:
    """Hindi vs Spanish accuracy per (model, flattening), pivot for MiniLM-en."""
    rows = []
    for (model, lang, flat), sub in per_triplet.groupby(["model", "language", "flattening"]):
        if model in {"MiniLM-en"}:
            cond = "english_pivot"
        else:
            cond = "native"
        sub2 = sub[sub["condition"] == cond]
        if len(sub2) == 0:
            continue
        rows.append(
            {
                "model": model,
                "language": lang,
                "flattening": flat,
                "n_items": len(sub2),
                "triplet_accuracy": float(sub2["correct"].mean()),
                "mean_cosine_gap": float(sub2["cos_gap"].mean()),
            }
        )
    df = pd.DataFrame(rows).sort_values(["flattening", "language", "model"]).reset_index(drop=True)
    df.to_csv(RESULTS / "cross_lang_comparison.csv", index=False)
    return df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1", default="triplets.json")
    parser.add_argument("--generated", default="data/generated_triplets_filtered.json")
    parser.add_argument("--primary-log", default="data/llm_validation_log.csv")
    parser.add_argument("--dual-log", default="data/dual_validation_log.csv")
    parser.add_argument("--multiblimp", default="data/multiblimp_hindi_gender.json")
    parser.add_argument("--combined", default="data/triplets_phase2b.json")
    parser.add_argument("--skip", default="", help="comma-separated model keys to skip")
    parser.add_argument("--no-hybrid", action="store_true")
    parser.add_argument("--no-labse-cross", action="store_true", help="skip cross-lingual collapse pass (saves time)")
    parser.add_argument("--bootstrap-n", type=int, default=10000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("phase2b")

    skip_models = {s for s in args.skip.split(",") if s.strip()}
    RESULTS.mkdir(exist_ok=True, parents=True)
    FIGURES.mkdir(exist_ok=True, parents=True)

    manifest = build_validated_dataset(
        phase1_path=Path(args.phase1),
        generated_path=Path(args.generated),
        primary_log_path=Path(args.primary_log),
        dual_log_path=Path(args.dual_log) if args.dual_log else None,
        out_path=Path(args.combined),
    )

    triplets = load_triplets(args.combined)
    log.info("Running embedding pass on %d triplets", len(triplets))
    per_triplet = run_embedding_pass(triplets, skip_models=skip_models)
    per_triplet.to_csv(RESULTS / "per_triplet.csv", index=False)
    log.info("Wrote per_triplet.csv (%d rows)", len(per_triplet))

    summary = add_accuracy_cis(
        per_triplet,
        by=["model", "condition", "language", "category"],
        n=args.bootstrap_n,
    )
    summary.to_csv(RESULTS / "summary.csv", index=False)

    summary_flat = add_accuracy_cis(
        per_triplet,
        by=["model", "condition", "flattening"],
        n=args.bootstrap_n,
    )
    summary_flat.to_csv(RESULTS / "summary_by_flattening.csv", index=False)

    multi_keys = [s.key for s in MODEL_SPECS if s.multilingual and s.key not in skip_models]
    headline = paired_headline(
        per_triplet,
        multilingual_models=multi_keys,
        n=args.bootstrap_n,
    )
    headline.to_csv(RESULTS / "headline.csv", index=False)

    if not args.no_hybrid and Path(args.multiblimp).exists():
        run_hybrid_pass(Path(args.multiblimp), per_triplet, skip_models=skip_models)

    run_length_analysis(per_triplet, Path(args.combined))

    qual = select_qualitative(per_triplet, Path(args.combined))
    qual.to_csv(RESULTS / "qualitative_examples.csv", index=False)

    triplets_dicts = json.loads(Path(args.combined).read_text(encoding="utf-8"))["triplets"]
    if not args.no_labse_cross:
        run_labse_deep_dive(per_triplet, triplets_dicts, skip_models=skip_models)
    else:
        labse_pivot = labse_ser_estar_table(per_triplet)
        labse_pivot.to_csv(RESULTS / "labse_ser_estar_pivot.csv", index=False)

    rep = replication_table(per_triplet, multi_models=multi_keys + ["MiniLM-en", "TF-IDF"])
    rep.to_csv(RESULTS / "phase1_phase2b_replication.csv", index=False)

    cross_language_comparison(per_triplet)

    (RESULTS / "bootstrap_config.json").write_text(
        json.dumps(
            {
                "n_resamples": args.bootstrap_n,
                "seed": 20260501,
                "method": "percentile",
                "alpha_uncorrected": 0.05,
                "alpha_bonferroni_per_test": 0.01,
                "n_strong_categories_corrected": 5,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    log.info("Phase 2B run complete. Manifest: %s", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
