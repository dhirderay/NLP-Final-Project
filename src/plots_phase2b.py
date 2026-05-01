"""Phase 2B figures.

  fig1_with_cis        headline with 99% Bonferroni-corrected paired CIs
  fig5_length_confound scatter of cos_gap vs anchor/far length diff
  fig6_labse_ser_estar within-language collapse + cross-lingual alignment
  fig7_scale           accuracy vs approximate model size
  fig8_cross_lang      Hindi vs Spanish accuracy by flattening
  fig9_hybrid          MultiBLiMP pair-metric vs triplet-metric per model
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CATEGORY_ORDER = [
    ("hindi", "kinship_paternal_maternal"),
    ("hindi", "verb_gender_agreement"),
    ("hindi", "formality_tv"),
    ("hindi", "kinship_relative_age"),
    ("spanish", "formality"),
    ("spanish", "gender_agreement_adjectives"),
    ("spanish", "ser_vs_estar"),
]

ALL_MODEL_ORDER_DEFAULT = ["LaBSE", "MiniLM-multi", "E5-multi", "E5-large", "bge-m3", "MiniLM-en", "TF-IDF"]


def _category_label(language: str, category: str) -> str:
    return f"{language[:3]}.{category}"


def fig1_with_cis(headline: pd.DataFrame, out_path: Path) -> None:
    """Headline figure: best multilingual encoder vs English pivot, with 99% CIs."""
    rows_in_order = []
    for lang, cat in CATEGORY_ORDER:
        r = headline[(headline["language"] == lang) & (headline["category"] == cat)]
        if len(r):
            rows_in_order.append(r.iloc[0])
    if not rows_in_order:
        return
    df = pd.DataFrame(rows_in_order)

    labels = [_category_label(r["language"], r["category"]) for _, r in df.iterrows()]
    multi = df["best_multilingual_acc"].to_numpy()
    pivot = df["english_pivot_acc"].to_numpy()
    flat = df["flattening_type"].to_list()
    gap_lo = df["gap_ci99_lo"].to_numpy()
    gap_hi = df["gap_ci99_hi"].to_numpy()
    sig = df["headline_significant"].to_list()

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 5.5))
    multi_colors = [
        "#1f77b4" if f == "strong" else ("#ff7f0e" if f == "partial" else "#2ca02c")
        for f in flat
    ]
    ax.bar(x - width / 2, multi, width, color=multi_colors, edgecolor="black", label="Best multilingual encoder")
    ax.bar(x + width / 2, pivot, width, color="#d62728", edgecolor="black", label="English-MiniLM (pivot)")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)

    for xi, (m, lo, hi, s) in enumerate(zip(multi, gap_lo, gap_hi, sig)):
        if not (np.isnan(lo) or np.isnan(hi)):
            mark = "*" if s else ""
            ax.text(
                xi - width / 2,
                m + 0.04,
                f"{mark}\n[{lo:+.2f},{hi:+.2f}]",
                ha="center",
                fontsize=7,
                color="black",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Triplet accuracy")
    ax.set_title("Figure 1 - best multilingual encoder vs English pivot, with 99% paired-bootstrap CIs on gap (* = headline-significant)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig5_length_confound(per_triplet_with_length: pd.DataFrame, corr_table: pd.DataFrame, out_path: Path) -> None:
    """Scatter of cos_gap vs len_diff_anchor_far per model (native condition only)."""
    df = per_triplet_with_length[per_triplet_with_length["condition"] == "native"]
    models = sorted(df["model"].unique())
    n_models = len(models)
    cols = 3
    rows = int(np.ceil(n_models / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.2 * rows), sharex=True, sharey=True)
    axes = axes.flatten()

    palette = plt.cm.tab10(np.linspace(0, 1, n_models))
    for ax, model, color in zip(axes, models, palette):
        sub = df[df["model"] == model]
        ax.scatter(
            sub["len_diff_anchor_far"], sub["cos_gap"],
            alpha=0.35, s=14, color=color, edgecolor="black", linewidths=0.2,
        )
        ax.axhline(0.0, color="gray", linewidth=0.8)
        ax.set_title(model, fontsize=10)
        rec = corr_table[(corr_table["model"] == model) & (corr_table["condition"] == "native")]
        if len(rec):
            r = rec.iloc[0]
            ax.text(
                0.97, 0.05,
                f"Pearson r={r['pearson_r']:.2f} (p={r['pearson_p']:.2g})",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            )
    for ax in axes[len(models):]:
        ax.set_visible(False)
    for ax in axes:
        ax.set_xlabel("|len(anchor) - len(far)| tokens", fontsize=9)
        ax.set_ylabel("cos(a,n) - cos(a,f)", fontsize=9)
    fig.suptitle("Figure 5 - cosine gap vs |anchor - far| token-length difference (native condition)", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig6_labse_ser_estar(cross_lingual: pd.DataFrame, out_path: Path) -> None:
    """ser/estar cross-lingual collapse, two panels.

    Panel (a): cos(anchor_es, far_es) per model -- higher means the
    encoder treats the two copulas as the same vector.
    Panel (b): cos(anchor_es, anchor_en) vs cos(far_es, far_en) -- shows
    how the cross-lingual translation-equivalence training pulls the
    Spanish forms toward their (effectively identical) English forms.
    """
    if cross_lingual.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    models = sorted(cross_lingual["model"].unique())
    palette = plt.cm.tab10(np.linspace(0, 1, len(models)))
    data_a = [cross_lingual[cross_lingual["model"] == m]["anchor_es_to_far_es"].to_numpy() for m in models]
    bp = axes[0].boxplot(data_a, patch_artist=True, labels=models, showfliers=False)
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].axhline(1.0, color="black", linewidth=0.6)
    axes[0].set_ylabel("cos(anchor_es, far_es)")
    axes[0].set_title("(a) Within-language similarity on ser/estar items.\nHigher = anchor and far collapse despite copula flip.")
    axes[0].tick_params(axis="x", rotation=20)

    align_anchor = [cross_lingual[cross_lingual["model"] == m]["anchor_es_to_anchor_en"].to_numpy() for m in models]
    align_far = [cross_lingual[cross_lingual["model"] == m]["far_es_to_far_en"].to_numpy() for m in models]
    width = 0.36
    pos = np.arange(len(models))
    bp1 = axes[1].boxplot(align_anchor, positions=pos - width / 2, widths=width, patch_artist=True, showfliers=False)
    bp2 = axes[1].boxplot(align_far, positions=pos + width / 2, widths=width, patch_artist=True, showfliers=False)
    for patch in bp1["boxes"]:
        patch.set_facecolor("#1f77b4")
        patch.set_alpha(0.7)
    for patch in bp2["boxes"]:
        patch.set_facecolor("#d62728")
        patch.set_alpha(0.7)
    axes[1].set_xticks(pos)
    axes[1].set_xticklabels(models, rotation=20, ha="right")
    axes[1].set_ylabel("Spanish - English cosine")
    axes[1].set_title("(b) Cross-lingual alignment: cos(anchor_es, anchor_en) vs cos(far_es, far_en).")
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(facecolor="#1f77b4", alpha=0.7, label="anchor"), Patch(facecolor="#d62728", alpha=0.7, label="far")], loc="lower right")

    fig.suptitle("Figure 6 - ser/estar within-language collapse + cross-lingual alignment", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig7_scale(per_triplet: pd.DataFrame, out_path: Path) -> None:
    """Triplet accuracy vs approximate model size, by flattening type.

    Sizes are pulled from the public model cards (rounded). TF-IDF is
    not a parametric model and is omitted.
    """
    sizes = {
        "MiniLM-multi": 117e6,
        "E5-multi": 118e6,
        "LaBSE": 470e6,
        "E5-large": 560e6,
        "bge-m3": 567e6,
        "MiniLM-en": 22e6,
        "TF-IDF": 0,
    }

    rows = []
    for m in sizes:
        if m == "TF-IDF":
            continue
        cond = "english_pivot" if m == "MiniLM-en" else "native"
        sub = per_triplet[(per_triplet["model"] == m) & (per_triplet["condition"] == cond)]
        if len(sub) == 0:
            continue
        for flat in ("strong", "partial", "non_flattening"):
            ss = sub[sub["flattening"] == flat]
            if len(ss) == 0:
                continue
            rows.append({
                "model": m,
                "size": sizes[m],
                "flattening": flat,
                "accuracy": float(ss["correct"].mean()),
                "n": int(len(ss)),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 5.5))
    color_map = {"strong": "#1f77b4", "partial": "#ff7f0e", "non_flattening": "#2ca02c"}
    for flat, sub in df.groupby("flattening"):
        sub = sub.sort_values("size")
        ax.plot(sub["size"] / 1e6, sub["accuracy"], "o-", color=color_map[flat], label=flat, linewidth=1.5, markersize=8)
        for _, r in sub.iterrows():
            ax.annotate(r["model"], (r["size"] / 1e6, r["accuracy"]), textcoords="offset points", xytext=(0, 6), fontsize=8, ha="center")

    e5_pts = df[df["model"].isin(["E5-multi", "E5-large"]) & (df["flattening"] == "strong")].sort_values("size")
    if len(e5_pts) == 2:
        ax.plot(e5_pts["size"] / 1e6, e5_pts["accuracy"], "k--", alpha=0.4, linewidth=1.2, label="E5 family scaling")

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Approximate parameter count (M, log scale)")
    ax.set_ylabel("Triplet accuracy")
    ax.set_title("Figure 7 - model scale vs accuracy by flattening type")
    ax.legend(loc="lower right")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig8_cross_lang(cross_lang: pd.DataFrame, out_path: Path) -> None:
    """Per-model accuracy on Hindi vs Spanish, by flattening type."""
    if cross_lang.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    flats = ["strong", "partial"]
    for ax, flat in zip(axes, flats):
        sub = cross_lang[cross_lang["flattening"] == flat]
        if sub.empty:
            ax.set_visible(False)
            continue
        models = sorted(sub["model"].unique())
        x = np.arange(len(models))
        width = 0.36
        hi = [sub[(sub["model"] == m) & (sub["language"] == "hindi")]["triplet_accuracy"].iloc[0] if len(sub[(sub["model"] == m) & (sub["language"] == "hindi")]) else np.nan for m in models]
        es = [sub[(sub["model"] == m) & (sub["language"] == "spanish")]["triplet_accuracy"].iloc[0] if len(sub[(sub["model"] == m) & (sub["language"] == "spanish")]) else np.nan for m in models]
        ax.bar(x - width / 2, hi, width, color="#1f77b4", edgecolor="black", label="Hindi")
        ax.bar(x + width / 2, es, width, color="#ff7f0e", edgecolor="black", label="Spanish")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_title(f"{flat}-flattening")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("Triplet accuracy")
    axes[-1].legend(loc="upper right")
    fig.suptitle("Figure 8 - cross-language accuracy by flattening type", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig9_hybrid(hybrid: pd.DataFrame, multiblimp_per_item: pd.DataFrame, out_path: Path) -> None:
    """Side-by-side comparison of triplet accuracy and MultiBLiMP pair metrics."""
    if hybrid.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    models = list(hybrid["model"])
    x = np.arange(len(models))
    width = 0.36
    axes[0].bar(x - width / 2, hybrid["triplet_accuracy_phase1"], width, label="Triplet acc (Phase 1, native)", color="#1f77b4", edgecolor="black")
    axes[0].bar(x + width / 2, hybrid["pair_pass_rate_multiblimp"], width, label="MultiBLiMP pair-pass rate", color="#ff7f0e", edgecolor="black")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=20, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Score")
    axes[0].axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    axes[0].legend(loc="best")
    axes[0].set_title("(a) Per-model: triplet vs pair metrics")

    if not multiblimp_per_item.empty:
        models2 = list(multiblimp_per_item["model"].unique())
        data = [multiblimp_per_item[multiblimp_per_item["model"] == m]["pair_distinction"].to_numpy() for m in models2]
        bp = axes[1].boxplot(data, labels=models2, patch_artist=True, showfliers=False)
        palette = plt.cm.tab10(np.linspace(0, 1, len(models2)))
        for patch, color in zip(bp["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1].axhline(0.0, color="black", linewidth=0.6)
        axes[1].set_ylabel("pair_distinction = 1 - cos(gram, ungram)")
        axes[1].set_title("(b) MultiBLiMP pair_distinction by model")
        axes[1].tick_params(axis="x", rotation=20)

    fig.suptitle("Figure 9 - hybrid framework cross-validation (MultiBLiMP pair-metric vs triplet metric)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
