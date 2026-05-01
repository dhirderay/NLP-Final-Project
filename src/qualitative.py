"""Deterministic qualitative-example selection.

Six rules, locked in the pre-registration so the choice of examples
isn't a post-hoc cherry-pick:
  1. textbook_win              — highest cos_gap on kinship_paternal_maternal, multilingual.
  2. morphological_failure     — lowest cos_gap on verb_gender_agreement, best multilingual.
  3. labse_specific_failure    — LaBSE wrong, MiniLM-multi and E5-multi correct.
  4. cross_model_disagreement  — max stdev of `correct` across the three multilingual models.
  5. partial_flattening_surprise — highest cos_gap on kinship_relative_age, multilingual.
  6. predicted_partial_win     — pivot correct, all multilingual wrong on ser_vs_estar.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

MULTI = ["LaBSE", "MiniLM-multi", "E5-multi"]


def _by_id(triplets_json_path: str | Path) -> dict[str, dict]:
    raw = json.loads(Path(triplets_json_path).read_text(encoding="utf-8"))
    return {t["id"]: t for t in raw["triplets"]}


def select_qualitative(per_triplet: pd.DataFrame, triplets_json_path: str | Path) -> pd.DataFrame:
    by_id = _by_id(triplets_json_path)
    rows: list[dict] = []

    def add(label: str, triplet_id: str, model: str, condition: str, sub_row=None):
        t = by_id.get(triplet_id)
        if t is None:
            return
        if sub_row is None:
            sub_row = per_triplet[
                (per_triplet["triplet_id"] == triplet_id)
                & (per_triplet["model"] == model)
                & (per_triplet["condition"] == condition)
            ].iloc[0]
        rows.append(
            {
                "selection_rule": label,
                "triplet_id": triplet_id,
                "language": t["language"],
                "category": t["category"],
                "anchor": t["anchor"],
                "near": t["near"],
                "far": t["far"],
                "anchor_en": t.get("anchor_en", ""),
                "near_en": t.get("near_en", ""),
                "far_en": t.get("far_en", ""),
                "model": model,
                "condition": condition,
                "sim_anchor_near": float(sub_row["sim_anchor_near"]),
                "sim_anchor_far": float(sub_row["sim_anchor_far"]),
                "cos_gap": float(sub_row["cos_gap"]),
                "correct": int(sub_row["correct"]),
            }
        )

    cand = per_triplet[
        (per_triplet["category"] == "kinship_paternal_maternal")
        & (per_triplet["model"].isin(MULTI))
        & (per_triplet["condition"] == "native")
    ]
    if len(cand):
        r = cand.sort_values("cos_gap", ascending=False).iloc[0]
        add("textbook_win", r["triplet_id"], r["model"], r["condition"], r)

    cand = per_triplet[
        (per_triplet["category"] == "verb_gender_agreement")
        & (per_triplet["model"].isin(MULTI))
        & (per_triplet["condition"] == "native")
    ]
    if len(cand):
        per_model_acc = cand.groupby("model")["correct"].mean()
        if len(per_model_acc):
            best = per_model_acc.idxmax()
            r = cand[cand["model"] == best].sort_values("cos_gap").iloc[0]
            add("morphological_failure", r["triplet_id"], r["model"], r["condition"], r)

    nat_multi = per_triplet[
        (per_triplet["model"].isin(MULTI)) & (per_triplet["condition"] == "native")
    ]
    pivot = nat_multi.pivot_table(
        index="triplet_id", columns="model", values="correct", aggfunc="first"
    ).reset_index()
    if all(m in pivot.columns for m in MULTI):
        labse_only = pivot[
            (pivot["LaBSE"] == 0)
            & (pivot["MiniLM-multi"] == 1)
            & (pivot["E5-multi"] == 1)
        ]
        if len(labse_only):
            tid = labse_only.iloc[0]["triplet_id"]
            add("labse_specific_failure", tid, "LaBSE", "native")

    if all(m in pivot.columns for m in MULTI):
        pivot["_std"] = pivot[MULTI].std(axis=1, ddof=0)
        top = pivot.sort_values("_std", ascending=False).iloc[0]
        add("cross_model_disagreement", top["triplet_id"], "E5-multi", "native")

    cand = per_triplet[
        (per_triplet["category"] == "kinship_relative_age")
        & (per_triplet["model"].isin(MULTI))
        & (per_triplet["condition"] == "native")
    ]
    if len(cand):
        r = cand.sort_values("cos_gap", ascending=False).iloc[0]
        add("partial_flattening_surprise", r["triplet_id"], r["model"], r["condition"], r)

    cand = per_triplet[
        (per_triplet["category"] == "ser_vs_estar")
    ]
    if len(cand):
        pv = cand[(cand["model"] == "MiniLM-en") & (cand["condition"] == "english_pivot")]
        ml = cand[(cand["model"].isin(MULTI)) & (cand["condition"] == "native")]
        if len(pv) and len(ml):
            ml_pivot = ml.pivot_table(index="triplet_id", columns="model", values="correct", aggfunc="first").reset_index()
            multi_cols = [c for c in MULTI if c in ml_pivot.columns]
            if multi_cols:
                ml_pivot["multi_max"] = ml_pivot[multi_cols].max(axis=1)
                pv_correct = set(pv[pv["correct"] == 1]["triplet_id"])
                ml_zero = set(ml_pivot[ml_pivot["multi_max"] == 0]["triplet_id"])
                candidates = sorted(pv_correct & ml_zero)
                if candidates:
                    add("predicted_partial_win", candidates[0], "MiniLM-en", "english_pivot")

    return pd.DataFrame(rows)
