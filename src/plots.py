"""Phase 1 figures.

Four PNGs at 150 DPI:
  fig1_headline.png         best multilingual vs English pivot, per category
  fig2_per_model.png        per-model accuracy across categories
  fig3_strong_vs_partial.png  by-flattening summary
  fig4_cosine_gap.png       boxplot of cos(a,n) - cos(a,f) per model x category
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .embed import MODEL_SPECS

CATEGORY_ORDER = [
    ("hindi", "kinship_paternal_maternal"),
    ("hindi", "verb_gender_agreement"),
    ("hindi", "formality_tv"),
    ("hindi", "kinship_relative_age"),
    ("spanish", "formality"),
    ("spanish", "gender_agreement_adjectives"),
    ("spanish", "ser_vs_estar"),
]

MULTI_MODELS = [s.key for s in MODEL_SPECS if s.multilingual]
ALL_MODEL_ORDER = [s.key for s in MODEL_SPECS] + ["TF-IDF"]

COLOR_STRONG = "#1f77b4"
COLOR_PARTIAL = "#ff7f0e"
COLOR_NONFLAT = "#2ca02c"
COLOR_PIVOT = "#d62728"


def _category_label(language: str, category: str) -> str:
    return f"{language[:3]}.{category}"


def fig1_headline(headline: pd.DataFrame, per_triplet: pd.DataFrame, out_path: Path) -> None:
    """Best multilingual encoder vs English pivot, per category."""
    rows = []
    for lang, cat in CATEGORY_ORDER:
        r = headline[(headline["language"] == lang) & (headline["category"] == cat)]
        if len(r):
            rows.append(r.iloc[0])
    df = pd.DataFrame(rows)

    labels = [_category_label(r["language"], r["category"]) for _, r in df.iterrows()]
    multi = df["best_multilingual_acc"].to_numpy()
    pivot = df["english_pivot_acc"].to_numpy()
    flat_types = df["flattening_type"].to_list()

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 5.5))
    multi_colors = [
        COLOR_STRONG if f == "strong" else (COLOR_PARTIAL if f == "partial" else COLOR_NONFLAT)
        for f in flat_types
    ]
    bars1 = ax.bar(x - width / 2, multi, width, label="Best multilingual encoder (native)", color=multi_colors, edgecolor="black")
    bars2 = ax.bar(x + width / 2, pivot, width, label="English-MiniLM (translation pivot)", color=COLOR_PIVOT, edgecolor="black")

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="chance")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Triplet accuracy")
    ax.set_title("Figure 1 — Flattening effect: best multilingual encoder vs. English-translation pivot")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    for b, v in zip(bars1, multi):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    for b, v in zip(bars2, pivot):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=COLOR_STRONG, edgecolor="black", label="strong-flattening (multilingual)"),
        Patch(facecolor=COLOR_PARTIAL, edgecolor="black", label="partial-flattening (multilingual)"),
        Patch(facecolor=COLOR_PIVOT, edgecolor="black", label="English pivot"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig2_per_model(per_triplet: pd.DataFrame, out_path: Path) -> None:
    """Per-model accuracy by category."""
    rows = []
    for lang, cat in CATEGORY_ORDER:
        sub = per_triplet[(per_triplet["language"] == lang) & (per_triplet["category"] == cat)]
        for model in ALL_MODEL_ORDER:
            cond = "english_pivot" if model == "MiniLM-en" else "native"
            ss = sub[(sub["model"] == model) & (sub["condition"] == cond)]
            if len(ss) == 0:
                ss = sub[(sub["model"] == model)]
            rows.append(
                {
                    "category": _category_label(lang, cat),
                    "model": model,
                    "accuracy": float(ss["correct"].mean()) if len(ss) else float("nan"),
                }
            )
    df = pd.DataFrame(rows)

    cats = [_category_label(l, c) for l, c in CATEGORY_ORDER]
    n_models = len(ALL_MODEL_ORDER)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(cats))
    palette = plt.cm.tab10(np.linspace(0, 1, n_models))
    for i, model in enumerate(ALL_MODEL_ORDER):
        ys = [df[(df["category"] == c) & (df["model"] == model)]["accuracy"].iloc[0] for c in cats]
        ax.bar(x + (i - n_models / 2) * width + width / 2, ys, width, label=model, color=palette[i], edgecolor="black", linewidth=0.4)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=20, ha="right")
    ax.set_ylabel("Triplet accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Figure 2 — Per-model accuracy by category (multilingual = native; MiniLM-en = pivot)")
    ax.legend(loc="lower right", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig3_strong_vs_partial(per_triplet: pd.DataFrame, out_path: Path) -> None:
    """Strong (left) vs partial (right) flattening accuracy by model."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, flat_label in zip(axes, ["strong", "partial"]):
        sub = per_triplet[per_triplet["flattening"] == flat_label]
        # native for multilingual + TF-IDF, english_pivot for MiniLM-en.
        rows = []
        for model in ALL_MODEL_ORDER:
            cond = "english_pivot" if model == "MiniLM-en" else "native"
            ss = sub[(sub["model"] == model) & (sub["condition"] == cond)]
            rows.append({"model": model, "accuracy": float(ss["correct"].mean()) if len(ss) else float("nan"), "n": len(ss)})
        df = pd.DataFrame(rows)
        x = np.arange(len(df))
        colors = ["#1f77b4", "#1f77b4", "#1f77b4", COLOR_PIVOT, "gray"]
        ax.bar(x, df["accuracy"], color=colors, edgecolor="black")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(df["model"], rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{flat_label}-flattening (n={int(df['n'].sum() / max(1, len(df)))} per model)")
        for i, (a, n) in enumerate(zip(df["accuracy"], df["n"])):
            if not np.isnan(a):
                ax.text(i, a + 0.02, f"{a:.2f}", ha="center", fontsize=9)

    axes[0].set_ylabel("Triplet accuracy")
    fig.suptitle("Figure 3 — Strong vs. partial flattening accuracy by model", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig4_cosine_gap(per_triplet: pd.DataFrame, out_path: Path) -> None:
    """Boxplot of cos(a,near) - cos(a,far) per model x category."""
    cats = [_category_label(l, c) for l, c in CATEGORY_ORDER]
    fig, ax = plt.subplots(figsize=(13, 6))

    data = []
    positions = []
    colors = []
    xtick_positions = []
    xtick_labels = []
    palette = plt.cm.tab10(np.linspace(0, 1, len(ALL_MODEL_ORDER)))

    pos = 0
    group_width = len(ALL_MODEL_ORDER) + 1
    for ci, (lang, cat) in enumerate(CATEGORY_ORDER):
        for mi, model in enumerate(ALL_MODEL_ORDER):
            cond = "english_pivot" if model == "MiniLM-en" else "native"
            ss = per_triplet[
                (per_triplet["language"] == lang)
                & (per_triplet["category"] == cat)
                & (per_triplet["model"] == model)
                & (per_triplet["condition"] == cond)
            ]
            data.append(ss["cos_gap"].to_numpy())
            positions.append(ci * group_width + mi)
            colors.append(palette[mi])
        xtick_positions.append(ci * group_width + (len(ALL_MODEL_ORDER) - 1) / 2)
        xtick_labels.append(_category_label(lang, cat))

    bp = ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, rotation=20, ha="right")
    ax.set_ylabel("cos(anchor, near) - cos(anchor, far)")
    ax.set_title("Figure 4 - Cosine-gap distribution per model x category (positive = correct)")

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=palette[i], edgecolor="black", label=ALL_MODEL_ORDER[i]) for i in range(len(ALL_MODEL_ORDER))]
    ax.legend(handles=handles, loc="upper right", ncol=2, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_all_figures(out: dict, fig_dir: Path) -> None:
    fig_dir.mkdir(exist_ok=True, parents=True)
    fig1_headline(out["headline"], out["per_triplet"], fig_dir / "fig1_headline.png")
    fig2_per_model(out["per_triplet"], fig_dir / "fig2_per_model.png")
    fig3_strong_vs_partial(out["per_triplet"], fig_dir / "fig3_strong_vs_partial.png")
    fig4_cosine_gap(out["per_triplet"], fig_dir / "fig4_cosine_gap.png")
