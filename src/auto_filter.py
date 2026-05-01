"""Automated quality filter for generated triplets.

Reject items that:
  - have any required field missing or empty
  - are malformed (unclosed brackets, inverted-question without closing ?, etc.)
  - have anchor identical to near (degenerate)
  - have anchor and far differing by more than 3 tokens (not minimal)
  - have a near/far token-length gap > 4
  - duplicate any Phase 1 anchor/near/far or any prior Phase 2B item
  - fail the gender-neutral noun whitelist (Spanish gender_agreement only)
  - claim non_flattening but have identical anchor_en and far_en
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

from src.category_specs import SPANISH_GENDER_NEUTRAL_NOUNS

REQUIRED_FIELDS = (
    "id", "language", "category",
    "anchor", "near", "far",
    "anchor_en", "near_en", "far_en",
)

WHITELIST_LOWER = {n.lower() for n in SPANISH_GENDER_NEUTRAL_NOUNS}


def _toks(s: str) -> list[str]:
    s = unicodedata.normalize("NFC", s)
    return [t for t in re.split(r"\s+|[।.,;:!?¡¿]", s) if t]


def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", s).strip().lower()


def _well_formed(text: str) -> bool:
    if not text or not text.strip():
        return False
    if text.endswith(("...", "…", "..")):
        return False
    for o, c in [("(", ")"), ("[", "]"), ("{", "}")]:
        if text.count(o) != text.count(c):
            return False
    # Spanish ¿…? and ¡…!: only require balance if the inverted opener is
    # present, since Hindi/English questions use a bare ? without one.
    if text.count("¿") > 0 and text.count("¿") != text.count("?"):
        return False
    if text.count("¡") > 0 and text.count("¡") != text.count("!"):
        return False
    return True


def filter_items(
    items: list[dict],
    *,
    phase1_items: list[dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Apply the quality rules and return (kept, rejected).

    Rejected items get a `rejection_reason` field listing the rule(s)
    they violated, semicolon-separated.
    """
    phase1_items = phase1_items or []
    seen_anchors: set[str] = set()
    for p1 in phase1_items:
        seen_anchors.add(_norm(p1.get("anchor", "")))
        seen_anchors.add(_norm(p1.get("near", "")))
        seen_anchors.add(_norm(p1.get("far", "")))

    kept: list[dict] = []
    rejected: list[dict] = []
    for it in items:
        reasons: list[str] = []

        for f in REQUIRED_FIELDS:
            if f not in it or not (str(it[f]).strip() if it[f] is not None else ""):
                reasons.append(f"missing_{f}")
        if reasons:
            it_copy = dict(it)
            it_copy["rejection_reason"] = ";".join(reasons)
            rejected.append(it_copy)
            continue

        anchor = it["anchor"]
        near = it["near"]
        far = it["far"]
        anchor_en = it.get("anchor_en", "")
        near_en = it.get("near_en", "")
        far_en = it.get("far_en", "")

        for label, txt in [("anchor", anchor), ("near", near), ("far", far),
                            ("anchor_en", anchor_en), ("near_en", near_en), ("far_en", far_en)]:
            if not _well_formed(txt):
                reasons.append(f"malformed_{label}")

        if _norm(anchor) == _norm(near):
            reasons.append("anchor_eq_near")

        a_toks = _toks(anchor)
        n_toks = _toks(near)
        f_toks = _toks(far)
        # Token symmetric-difference is a cheap proxy for edit distance:
        # one swapped token contributes two elements (one removed, one
        # added), so 3 swaps -> sym-diff of 6.
        diff_tokens = len(set(a_toks).symmetric_difference(set(f_toks)))
        if diff_tokens > 6:
            reasons.append(f"anchor_far_diff_{diff_tokens}_tokens")

        if abs(len(n_toks) - len(f_toks)) > 4:
            reasons.append(f"near_far_len_diff_{abs(len(n_toks) - len(f_toks))}")

        for label, txt in [("anchor", anchor), ("near", near), ("far", far)]:
            if _norm(txt) in seen_anchors:
                reasons.append(f"duplicate_{label}")
                break

        if it["category"] == "gender_agreement_adjectives":
            tokens = set()
            for s in (anchor, near, far):
                for t in _toks(s):
                    tokens.add(t.lower())
            if not (tokens & WHITELIST_LOWER):
                reasons.append("no_whitelisted_noun")

        if it.get("flattening_intent", "").lower() == "non_flattening":
            if _norm(anchor_en) == _norm(far_en):
                reasons.append("nonflat_identical_en_anchor_far")

        if reasons:
            it_copy = dict(it)
            it_copy["rejection_reason"] = ";".join(reasons)
            rejected.append(it_copy)
        else:
            kept.append(it)
            for s in (anchor, near, far):
                seen_anchors.add(_norm(s))

    return kept, rejected


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    src_path = root / "data" / "generated_triplets.json"
    if not src_path.exists():
        print(f"No generated triplets at {src_path} — run generate_triplets.py first.")
        return
    raw = json.loads(src_path.read_text(encoding="utf-8"))
    items = raw["items"]

    p1 = json.loads((root / "triplets.json").read_text(encoding="utf-8"))["triplets"]

    kept, rejected = filter_items(items, phase1_items=p1)

    out_kept = root / "data" / "generated_triplets_filtered.json"
    out_kept.write_text(json.dumps({"items": kept, "n": len(kept)}, ensure_ascii=False, indent=2), encoding="utf-8")

    out_log = root / "data" / "auto_filter_log.csv"
    if rejected:
        rows = [{"id": r["id"], "category": r.get("category", ""), "rejection_reason": r["rejection_reason"]} for r in rejected]
        pd.DataFrame(rows).to_csv(out_log, index=False)
    else:
        pd.DataFrame(columns=["id", "category", "rejection_reason"]).to_csv(out_log, index=False)

    pct = 100.0 * len(rejected) / max(1, len(items))
    print(f"Auto-filter: kept {len(kept)} / {len(items)} (rejected {len(rejected)} = {pct:.1f}%)")
    print(f"  kept    -> {out_kept}")
    print(f"  log     -> {out_log}")

    if rejected:
        df = pd.DataFrame(rejected)
        print("\nRejection counts by category x top reason:")
        print(df.groupby(["category", "rejection_reason"]).size().sort_values(ascending=False).head(20).to_string())


if __name__ == "__main__":
    main()
