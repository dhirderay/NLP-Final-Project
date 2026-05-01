"""Dual-LLM cross-check, used as a proxy for human spot-check.

Reads the primary-validator log, builds the same sample composition a
human panel would have reviewed (all primary-flagged items plus a 15%
stratified random sample of primary-passed items), and runs each
through a second LLM in a different family. The thresholds match the
pre-registered human-review gates; the substitution is disclosed in
the methods section. Output goes to data/dual_validation_log.csv and
data/human_llm_disagreement.csv.

Primary: gpt-4.1
Dual:    claude-sonnet-4-6
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from anthropic import AsyncAnthropic

from src.category_specs import CATEGORY_SPECS

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

DUAL_MODEL = "claude-sonnet-4-6"
DEFAULT_CONCURRENCY = 3
SEED = 20260501


VALIDATOR_SYSTEM = (
    "You are a careful linguistic data validator for a multilingual NLP probe. "
    "For each triplet you receive, output a single JSON object containing the same "
    "five checks (language, minimal_pair, contrast_feature, translation, naturalness) "
    "and an `overall` field. Each check must include a `reasoning` string before its `verdict`. "
    "Verdicts: 'pass', 'fail', or 'flag' (only for naturalness). Be strict but fair. "
    "Output ONLY the JSON object, no markdown fence."
)


def build_prompt(item: dict) -> str:
    cat = item["category"]
    spec = CATEGORY_SPECS.get(cat)
    spec_block = ""
    if spec is not None:
        spec_block = (
            f"CATEGORY DESCRIPTION: {spec.contrast_feature_description}\n"
            f"CONTRAST CHECK QUESTION: {spec.validator_check_question}\n"
            f"FLATTENING INTENT: {item.get('flattening_intent', spec.flattening_intent)}\n"
        )
    item_for_prompt = {
        k: item.get(k)
        for k in (
            "id", "category", "language",
            "anchor", "near", "far",
            "anchor_en", "near_en", "far_en",
            "flattening_intent", "dialect", "notes",
        )
        if k in item
    }
    return (
        f"{spec_block}\nINPUT:\n{json.dumps(item_for_prompt, ensure_ascii=False, indent=2)}\n\n"
        "Output a single JSON object: "
        '{"checks":{"language":{"reasoning":"...","verdict":"pass|fail"},'
        '"minimal_pair":{"reasoning":"...","verdict":"pass|fail"},'
        '"contrast_feature":{"reasoning":"...","verdict":"pass|fail"},'
        '"translation":{"reasoning":"...","verdict":"pass|fail"},'
        '"naturalness":{"reasoning":"...","verdict":"pass|flag"}},'
        '"overall":"auto_validated|needs_human_review|flagged_for_review"}'
    )


async def validate_one(client: AsyncAnthropic, item: dict, log: logging.Logger) -> dict:
    prompt = build_prompt(item)
    for attempt in range(4):
        try:
            r = await client.messages.create(
                model=DUAL_MODEL,
                max_tokens=1200,
                temperature=0.0,
                system=VALIDATOR_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            text = r.content[0].text
            start, end = text.find("{"), text.rfind("}")
            parsed = json.loads(text[start : end + 1] if start != -1 and end != -1 else text)
            return {
                "triplet_id": item["id"],
                "category": item.get("category"),
                "validator_model": DUAL_MODEL,
                "checks": parsed.get("checks", {}),
                "overall": parsed.get("overall", "needs_human_review"),
            }
        except Exception as e:
            log.warning("dual %s attempt=%d failed: %s", item.get("id"), attempt, e)
            await asyncio.sleep(1.5 ** attempt)
    return {"triplet_id": item["id"], "category": item.get("category"), "validator_model": DUAL_MODEL, "checks": {}, "overall": "validator_error"}


def select_sample(filtered_items: list[dict], llm_log: pd.DataFrame, *, sample_pct: float, seed: int) -> tuple[list[dict], dict[str, str]]:
    """Build the sample composition: all flagged + stratified random of passed."""
    by_id = {it["id"]: it for it in filtered_items}
    flagged_ids = set(llm_log[llm_log["overall"].isin(["needs_human_review", "flagged_for_review"])]["triplet_id"])
    passed_ids = set(llm_log[llm_log["overall"] == "auto_validated"]["triplet_id"])

    rng = random.Random(seed)
    sampled_passed: set[str] = set()
    for cat, sub in llm_log[llm_log["triplet_id"].isin(passed_ids)].groupby("category"):
        ids = list(sub["triplet_id"])
        # Floor of 10 per cell satisfies the pre-registration's
        # "at least 10 reviewed per cell" stratification rule.
        n = max(int(round(sample_pct * len(ids))), min(10, len(ids)))
        picked = rng.sample(ids, k=min(n, len(ids)))
        sampled_passed.update(picked)

    pool_ids = sorted(flagged_ids | sampled_passed)
    sample = [by_id[i] for i in pool_ids if i in by_id]
    primary_overall = dict(zip(llm_log["triplet_id"], llm_log["overall"]))
    return sample, primary_overall


def write_log(results: list[dict], path: Path) -> None:
    rows = []
    for r in results:
        row: dict[str, Any] = {
            "triplet_id": r["triplet_id"],
            "category": r["category"],
            "validator_model": r["validator_model"],
            "overall": r["overall"],
        }
        for check in ("language", "minimal_pair", "contrast_feature", "translation", "naturalness"):
            ck = r["checks"].get(check, {}) if isinstance(r["checks"], dict) else {}
            row[f"{check}_verdict"] = ck.get("verdict", "")
            row[f"{check}_reasoning"] = ck.get("reasoning", "")
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        pd.DataFrame().to_csv(path, index=False)
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


async def main_async(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("dual")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.error("ANTHROPIC_API_KEY not set")
        sys.exit(1)

    items = json.loads(Path(args.input).read_text(encoding="utf-8"))["items"]
    primary = pd.read_csv(args.primary_log)
    sample, primary_overall = select_sample(items, primary, sample_pct=args.sample_pct, seed=SEED)
    log.info("Selected %d items for dual-LLM review (flagged + %.0f%% sample of passed)", len(sample), 100 * args.sample_pct)

    client = AsyncAnthropic(api_key=api_key)
    sem = asyncio.Semaphore(args.concurrency)

    async def one(it):
        async with sem:
            return await validate_one(client, it, log)

    t0 = time.time()
    tasks = [asyncio.create_task(one(it)) for it in sample]
    results = await asyncio.gather(*tasks)
    log.info("Dual validation: %d items in %.1fs", len(results), time.time() - t0)

    write_log(results, Path(args.output))

    rows = []
    for r in results:
        primary_ov = primary_overall.get(r["triplet_id"], "")
        dual_ov = r["overall"]
        rows.append(
            {
                "triplet_id": r["triplet_id"],
                "category": r["category"],
                "primary_overall": primary_ov,
                "dual_overall": dual_ov,
                "agree": int((primary_ov == "auto_validated") == (dual_ov == "auto_validated")),
            }
        )
    disagreement = pd.DataFrame(rows)
    disagreement.to_csv(DATA / "human_llm_disagreement.csv", index=False)

    # The human-spot-check gate (pre-registration §4.3) is applied here
    # with the dual validator standing in for the human panel.
    # "Rejection" = dual verdict != 'auto_validated'.
    n_total = len(disagreement)
    n_rejected = int((disagreement["dual_overall"] != "auto_validated").sum())
    rejection_rate = n_rejected / max(1, n_total)
    agreement = float(disagreement["agree"].mean()) if n_total else 1.0

    summary = {
        "dual_validator_model": DUAL_MODEL,
        "primary_validator_model": "gpt-4.1",
        "n_reviewed": n_total,
        "n_rejected_by_dual": n_rejected,
        "rejection_rate_proxy": rejection_rate,
        "agreement_rate_with_primary": agreement,
        "gate_pass_le_8pct": rejection_rate <= 0.08,
        "gate_pass_agreement_ge_85pct": agreement >= 0.85,
    }
    (DATA / "dual_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/generated_triplets_filtered.json")
    parser.add_argument("--primary-log", default="data/llm_validation_log.csv")
    parser.add_argument("--output", default="data/dual_validation_log.csv")
    parser.add_argument("--sample-pct", type=float, default=0.15)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    args = parser.parse_args()
    asyncio.run(main_async(args))
