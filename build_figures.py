"""Regenerate all figures from cached CSVs without re-running embeddings."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.plots import build_all_figures
from src.plots_phase2b import (
    fig1_with_cis,
    fig5_length_confound,
    fig6_labse_ser_estar,
    fig7_scale,
    fig8_cross_lang,
    fig9_hybrid,
)

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True, parents=True)


def main():
    per_triplet = pd.read_csv(RESULTS / "per_triplet.csv")
    headline = pd.read_csv(RESULTS / "headline.csv")
    summary = pd.read_csv(RESULTS / "summary.csv")
    summary_flat = pd.read_csv(RESULTS / "summary_by_flattening.csv")

    out = {
        "per_triplet": per_triplet,
        "headline": headline,
        "summary": summary,
        "summary_flat": summary_flat,
    }
    build_all_figures(out, FIGURES)

    # Overwrite the basic fig1 with the bootstrap-CI variant.
    fig1_with_cis(headline, FIGURES / "fig1_headline.png")

    aug = pd.read_csv(RESULTS / "per_triplet_with_length.csv")
    corr = pd.read_csv(RESULTS / "length_analysis.csv")
    fig5_length_confound(aug, corr, FIGURES / "fig5_length_confound.png")

    cross_path = RESULTS / "labse_cross_lingual.csv"
    if cross_path.exists():
        cross = pd.read_csv(cross_path)
        fig6_labse_ser_estar(cross, FIGURES / "fig6_labse_ser_estar.png")

    fig7_scale(per_triplet, FIGURES / "fig7_scale.png")

    cross_lang = pd.read_csv(RESULTS / "cross_lang_comparison.csv")
    fig8_cross_lang(cross_lang, FIGURES / "fig8_cross_lang.png")

    hybrid = pd.read_csv(RESULTS / "hybrid_framework_validation.csv")
    mbp_per = pd.read_csv(RESULTS / "multiblimp_per_item.csv")
    fig9_hybrid(hybrid, mbp_per, FIGURES / "fig9_hybrid.png")

    print("All figures written to", FIGURES)


if __name__ == "__main__":
    main()
