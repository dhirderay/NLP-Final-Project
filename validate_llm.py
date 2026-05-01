"""Async LLM validator (primary, gpt-4.1).

Five atomic per-item checks: language identification, minimal-pair
structure, contrast-feature correctness, English translation accuracy,
naturalness. The validator is gpt-4.1 (different family from the
Claude generator). The prompt forces a JSON object whose `reasoning`
field comes before `verdict` -- a small but reliable trick to keep the
model from snap-judging.

Two CLI modes:
  calibrate  run against Phase 1 + a small set of deliberately broken items
  validate   run against the auto-filtered Phase 2B items
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from src.category_specs import CATEGORY_SPECS

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

DEFAULT_MODEL = "gpt-4.1"
DEFAULT_CONCURRENCY = 24


VALIDATOR_SYSTEM = """You are a careful linguistic data validator for a multilingual NLP probe.
You evaluate triplets that test whether a multilingual sentence embedding preserves a source-language semantic distinction that English does not (or only partially) preserve.

For each triplet you receive, output a single JSON object with five checks plus an overall verdict. For each check, output a 'reasoning' field BEFORE the 'verdict' field. Verdicts must be one of: 'pass', 'fail', or 'flag' (only used for the naturalness check). Be strict but fair.

The overall field must be:
  - 'auto_validated' if all of (language, minimal_pair, contrast_feature, translation) pass
  - 'needs_human_review' if any of those four fails
  - 'flagged_for_review' if all four pass but naturalness is flagged

Output ONLY the JSON object, no markdown fence, no commentary."""


CALIBRATION_FEW_SHOT = """A clearly-passing example:
INPUT:
{
  "id": "demo_pass",
  "category": "kinship_paternal_maternal",
  "language": "hindi",
  "anchor": "मेरे चाचा कल आएंगे",
  "near": "मेरे चाचा कल पहुंचेंगे",
  "far": "मेरे मामा कल आएंगे",
  "anchor_en": "My uncle will come tomorrow",
  "near_en": "My uncle will arrive tomorrow",
  "far_en": "My uncle will come tomorrow",
  "flattening_intent": "strong"
}
OUTPUT:
{"checks":{"language":{"reasoning":"All Hindi/English fields match claimed languages.","verdict":"pass"},"minimal_pair":{"reasoning":"Anchor↔near differ in verb (ayenge/pahuncenge), preserving chacha. Anchor↔far swap chacha→mama only.","verdict":"pass"},"contrast_feature":{"reasoning":"Far swaps chacha (paternal uncle) → mama (maternal uncle); anchor and near keep chacha.","verdict":"pass"},"translation":{"reasoning":"English collapses chacha and mama to 'uncle'; anchor_en == far_en, no leakage.","verdict":"pass"},"naturalness":{"reasoning":"Standard Hindi.","verdict":"pass"}},"overall":"auto_validated"}

A clearly-failing example (translation leakage):
INPUT:
{
  "id": "demo_fail",
  "category": "kinship_paternal_maternal",
  "language": "hindi",
  "anchor": "मेरे चाचा कल आएंगे",
  "near": "मेरे चाचा कल पहुंचेंगे",
  "far": "मेरे मामा कल आएंगे",
  "anchor_en": "My paternal uncle will come tomorrow",
  "near_en": "My paternal uncle will arrive tomorrow",
  "far_en": "My maternal uncle will come tomorrow",
  "flattening_intent": "strong"
}
OUTPUT:
{"checks":{"language":{"reasoning":"All language tags correct.","verdict":"pass"},"minimal_pair":{"reasoning":"Source-side structure is fine.","verdict":"pass"},"contrast_feature":{"reasoning":"Source-side chacha→mama swap is correct.","verdict":"pass"},"translation":{"reasoning":"anchor_en and far_en are disambiguated as 'paternal' and 'maternal' uncle, leaking the strong-flattening contrast into English.","verdict":"fail"},"naturalness":{"reasoning":"Source sentences are natural.","verdict":"pass"}},"overall":"needs_human_review"}
"""


def build_prompt(item: dict) -> str:
    cat = item["category"]
    spec = CATEGORY_SPECS.get(cat)
    spec_block = ""
    if spec is not None:
        spec_block = (
            f"CATEGORY DESCRIPTION: {spec.contrast_feature_description}\n"
            f"CONTRAST CHECK QUESTION (for check 3): {spec.validator_check_question}\n"
            f"FLATTENING INTENT (declared): {item.get('flattening_intent', spec.flattening_intent)}\n"
        )

    item_for_prompt = {
        k: item.get(k)
        for k in (
            "id", "category", "language",
            "anchor", "near", "far",
            "anchor_en", "near_en", "far_en",
            "flattening_intent", "dialect", "notes", "source",
        )
        if k in item
    }

    return (
        f"{CALIBRATION_FEW_SHOT}\n"
        "Now validate this item. Output the JSON object only.\n\n"
        f"{spec_block}\n"
        "INPUT:\n"
        f"{json.dumps(item_for_prompt, ensure_ascii=False, indent=2)}\n"
    )


async def validate_one(
    client: AsyncOpenAI,
    item: dict,
    model: str,
    log: logging.Logger,
) -> dict:
    prompt = build_prompt(item)
    for attempt in range(4):
        try:
            r = await client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": VALIDATOR_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            text = r.choices[0].message.content
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                # Recover the first JSON object if the model wrapped it in prose.
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1:
                    parsed = json.loads(text[start : end + 1])
                else:
                    raise
            return {
                "triplet_id": item["id"],
                "category": item.get("category"),
                "validator_model": model,
                "checks": parsed.get("checks", {}),
                "overall": parsed.get("overall", "needs_human_review"),
                "raw": text,
            }
        except Exception as e:
            log.warning("validate %s attempt=%d failed: %s", item.get("id"), attempt, e)
            await asyncio.sleep(1.5 ** attempt)
    return {
        "triplet_id": item["id"],
        "category": item.get("category"),
        "validator_model": model,
        "checks": {},
        "overall": "validator_error",
        "raw": "",
    }


async def validate_many(
    items: list[dict],
    *,
    model: str,
    concurrency: int,
) -> list[dict]:
    log = logging.getLogger("val")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        log.error("OPENAI_API_KEY not set")
        sys.exit(1)
    client = AsyncOpenAI(api_key=api_key)

    sem = asyncio.Semaphore(concurrency)

    async def one(item: dict) -> dict:
        async with sem:
            return await validate_one(client, item, model, log)

    t0 = time.time()
    tasks = [asyncio.create_task(one(it)) for it in items]
    results = await asyncio.gather(*tasks)
    log.info("Validated %d items in %.1fs", len(results), time.time() - t0)
    return results


def write_validation_log(results: list[dict], path: Path) -> None:
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
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_broken_examples() -> list[dict]:
    """Twelve deliberately-broken triplets used to confirm the validator catches errors."""
    return [
        {
            "id": "broken_lang_001",
            "category": "kinship_paternal_maternal",
            "language": "hindi",
            "anchor": "My uncle will come tomorrow",
            "near": "मेरे चाचा कल पहुंचेंगे",
            "far": "मेरे मामा कल आएंगे",
            "anchor_en": "My uncle will come tomorrow",
            "near_en": "My uncle will arrive tomorrow",
            "far_en": "My uncle will come tomorrow",
            "flattening_intent": "strong",
        },
        {
            "id": "broken_contrast_001",
            "category": "kinship_paternal_maternal",
            "language": "hindi",
            "anchor": "मेरे चाचा कल आएंगे",
            "near": "मेरे चाचा कल पहुंचेंगे",
            "far": "मेरे चाचा कल जाएंगे",
            "anchor_en": "My uncle will come tomorrow",
            "near_en": "My uncle will arrive tomorrow",
            "far_en": "My uncle will leave tomorrow",
            "flattening_intent": "strong",
        },
        {
            "id": "broken_translation_001",
            "category": "kinship_paternal_maternal",
            "language": "hindi",
            "anchor": "मेरे चाचा कल आएंगे",
            "near": "मेरे चाचा कल पहुंचेंगे",
            "far": "मेरे मामा कल आएंगे",
            "anchor_en": "My paternal uncle will come tomorrow",
            "near_en": "My paternal uncle will arrive tomorrow",
            "far_en": "My maternal uncle will come tomorrow",
            "flattening_intent": "strong",
        },
        {
            "id": "broken_minimal_001",
            "category": "kinship_relative_age",
            "language": "hindi",
            "anchor": "मेरा बड़ा भाई स्कूल गया",
            "near": "मेरा बड़ा भाई स्कूल गया",
            "far": "मेरा छोटा भाई स्कूल गया",
            "anchor_en": "My brother went to school",
            "near_en": "My brother went to school",
            "far_en": "My brother went to school",
            "flattening_intent": "partial",
        },
        {
            "id": "broken_minimal_002",
            "category": "formality_tv",
            "language": "hindi",
            "anchor": "तुम कहाँ जा रहे हो?",
            "near": "तुम कहाँ जा रहे हो?",
            "far": "आप कहाँ गए थे?",
            "anchor_en": "Where are you going?",
            "near_en": "Where are you going?",
            "far_en": "Where had you gone?",
            "flattening_intent": "strong",
        },
        {
            "id": "broken_contrast_002",
            "category": "ser_vs_estar",
            "language": "spanish",
            "anchor": "Carlos es nervioso",
            "near": "Carlos es muy nervioso",
            "far": "Carlos es triste",
            "anchor_en": "Carlos is nervous",
            "near_en": "Carlos is very nervous",
            "far_en": "Carlos is sad",
            "flattening_intent": "partial",
        },
        {
            "id": "broken_lex_001",
            "category": "gender_agreement_adjectives",
            "language": "spanish",
            "anchor": "El actor está cansado",
            "near": "El actor parece cansado",
            "far": "La actriz está cansada",
            "anchor_en": "The actor is tired",
            "near_en": "The actor seems tired",
            "far_en": "The actress is tired",
            "flattening_intent": "strong",
        },
        {
            "id": "broken_nonflat_001",
            "category": "ser_vs_estar",
            "language": "spanish",
            "anchor": "Carlos es aburrido",
            "near": "Carlos es muy aburrido",
            "far": "Carlos está aburrido",
            "anchor_en": "Carlos is boring",
            "near_en": "Carlos is very boring",
            "far_en": "Carlos is boring",
            "flattening_intent": "non_flattening",
        },
        {
            "id": "broken_consistency_001",
            "category": "formality_tv",
            "language": "hindi",
            "anchor": "तू कैसे हो?",
            "near": "तू कैसे हो?",
            "far": "आप कैसे हैं?",
            "anchor_en": "How are you?",
            "near_en": "How are you?",
            "far_en": "How are you?",
            "flattening_intent": "strong",
        },
        {
            "id": "broken_consistency_002",
            "category": "formality",
            "language": "spanish",
            "anchor": "Tú tienen frío",
            "near": "Tú tienen mucho frío",
            "far": "Usted tiene frío",
            "anchor_en": "You are cold",
            "near_en": "You are very cold",
            "far_en": "You are cold",
            "flattening_intent": "strong",
        },
        {
            "id": "broken_translation_002",
            "category": "kinship_relative_age",
            "language": "hindi",
            "anchor": "मेरा बड़ा भाई दिल्ली में रहता है",
            "near": "मेरा बड़ा भाई दिल्ली में निवास करता है",
            "far": "मेरा छोटा भाई दिल्ली में रहता है",
            "anchor_en": "My sister lives in Mumbai",
            "near_en": "My sister resides in Mumbai",
            "far_en": "My sister lives in Mumbai",
            "flattening_intent": "partial",
        },
        {
            "id": "broken_minimal_003",
            "category": "ser_vs_estar",
            "language": "spanish",
            "anchor": "Ana es alta",
            "near": "Ana es muy alta",
            "far": "El perro de Marta corre rápido por el parque grande hoy",
            "anchor_en": "Ana is tall",
            "near_en": "Ana is very tall",
            "far_en": "Marta's dog runs fast through the big park today",
            "flattening_intent": "partial",
        },
    ]


async def cmd_calibrate(args) -> int:
    log = logging.getLogger("val")
    phase1 = json.loads((ROOT / "triplets.json").read_text(encoding="utf-8"))["triplets"]
    if args.limit:
        phase1 = phase1[: args.limit]
    broken = make_broken_examples()

    log.info("Calibrating on %d Phase-1 items + %d broken examples", len(phase1), len(broken))
    p1_results = await validate_many(phase1, model=args.model, concurrency=args.concurrency)
    br_results = await validate_many(broken, model=args.model, concurrency=args.concurrency)

    # Compute pass rates.
    def pass_rate(results, checks=("language", "minimal_pair", "contrast_feature", "translation")):
        ok = 0
        for r in results:
            ck = r.get("checks") or {}
            if all((ck.get(c) or {}).get("verdict") == "pass" for c in checks):
                ok += 1
        return ok / max(1, len(results))

    p1_rate = pass_rate(p1_results)
    br_caught = 1 - pass_rate(br_results)
    log.info("Phase-1 pass rate (Checks 1-4): %.3f (gate: ≥0.95)", p1_rate)
    log.info("Broken catch rate (Checks 1-4 fail): %.3f (gate: ≥0.90)", br_caught)

    write_validation_log(p1_results + br_results, DATA / "validator_calibration.csv")
    summary = {
        "validator_model": args.model,
        "phase1_n": len(p1_results),
        "phase1_pass_rate": p1_rate,
        "broken_n": len(br_results),
        "broken_catch_rate": br_caught,
        "passes_gate": p1_rate >= 0.95 and br_caught >= 0.90,
    }
    (DATA / "validator_calibration_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


async def cmd_validate(args) -> int:
    log = logging.getLogger("val")
    src = Path(args.input)
    items = json.loads(src.read_text(encoding="utf-8"))["items"]
    if args.limit:
        items = items[: args.limit]
    log.info("Validating %d items with %s", len(items), args.model)
    results = await validate_many(items, model=args.model, concurrency=args.concurrency)
    out = Path(args.output)
    write_validation_log(results, out)

    # Stats.
    total = len(results)
    auto = sum(1 for r in results if r["overall"] == "auto_validated")
    flagged = sum(1 for r in results if r["overall"] == "flagged_for_review")
    review = sum(1 for r in results if r["overall"] == "needs_human_review")
    err = sum(1 for r in results if r["overall"] == "validator_error")
    summary = {
        "validator_model": args.model,
        "n_items": total,
        "auto_validated": auto,
        "flagged_for_review": flagged,
        "needs_human_review": review,
        "validator_error": err,
        "flag_rate": (flagged + review) / max(1, total),
    }
    (out.parent / f"{out.stem}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["calibrate", "validate"])
    parser.add_argument("--input", default="data/generated_triplets_filtered.json")
    parser.add_argument("--output", default="data/llm_validation_log.csv")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    if args.mode == "calibrate":
        return asyncio.run(cmd_calibrate(args))
    return asyncio.run(cmd_validate(args))


if __name__ == "__main__":
    sys.exit(main())
