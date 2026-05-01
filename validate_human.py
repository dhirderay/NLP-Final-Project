"""Minimal CLI for the human stratified-review pass.

Walks the same sample composition as validate_dual.py (all primary-flagged
items plus the 15% stratified random sample of primary-passed items),
collecting three yes/no answers per item: contrast preserved, naturalness,
translation accuracy. Output is appended to data/human_validation_log.csv
so you can stop and resume.

    python validate_human.py [--limit N] [--reviewer NAME]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

from validate_dual import select_sample

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

SEED = 20260501


def y_n(prompt: str) -> str:
    while True:
        ans = input(prompt + " (y/n/skip) > ").strip().lower()
        if ans in ("y", "yes"):
            return "y"
        if ans in ("n", "no"):
            return "n"
        if ans in ("s", "skip", ""):
            return "skip"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/generated_triplets_filtered.json")
    parser.add_argument("--primary-log", default="data/llm_validation_log.csv")
    parser.add_argument("--output", default="data/human_validation_log.csv")
    parser.add_argument("--sample-pct", type=float, default=0.15)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--reviewer", default="")
    args = parser.parse_args()

    items = json.loads(Path(args.input).read_text(encoding="utf-8"))["items"]
    primary = pd.read_csv(args.primary_log)
    sample, _ = select_sample(items, primary, sample_pct=args.sample_pct, seed=SEED)

    out_path = Path(args.output)
    done_ids: set[str] = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done_ids.add(row["triplet_id"])

    queue = [it for it in sample if it["id"] not in done_ids]
    if args.limit:
        queue = queue[: args.limit]

    print(f"\n=== Human review: {len(queue)} item(s) remaining ({len(done_ids)} already reviewed) ===\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()
    with out_path.open("a", encoding="utf-8", newline="") as f:
        fieldnames = [
            "triplet_id", "category", "language", "reviewer",
            "contrast_preserved", "naturalness", "translation_accuracy", "passes",
            "comment",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()

        for it in queue:
            print("---")
            print(f"id: {it['id']}  category: {it['category']}  language: {it['language']}")
            print(f"  flattening_intent: {it.get('flattening_intent','?')}  dialect: {it.get('dialect','n/a')}")
            print(f"  anchor:    {it['anchor']}")
            print(f"  near:      {it['near']}")
            print(f"  far:       {it['far']}")
            print(f"  anchor_en: {it['anchor_en']}")
            print(f"  near_en:   {it['near_en']}")
            print(f"  far_en:    {it['far_en']}")
            print()
            try:
                a = y_n("1) Contrast preserved: 'near' preserves contrast, 'far' flips it?")
                b = y_n("2) Naturalness: sounds natural to a native speaker?")
                c = y_n("3) Translation accuracy: _en fields correct without inappropriate leakage?")
            except (KeyboardInterrupt, EOFError):
                print("\n[interrupted]")
                return 0
            comment = input("Comment (optional, enter to skip) > ").strip()
            passes = "y" if (a == "y" and b == "y" and c == "y") else ("skip" if "skip" in (a, b, c) else "n")
            w.writerow({
                "triplet_id": it["id"],
                "category": it["category"],
                "language": it["language"],
                "reviewer": args.reviewer,
                "contrast_preserved": a,
                "naturalness": b,
                "translation_accuracy": c,
                "passes": passes,
                "comment": comment,
            })
            f.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
