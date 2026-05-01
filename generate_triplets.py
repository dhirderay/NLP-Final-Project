"""Async batched triplet generation against the Anthropic API.

For each category, splits the target item count into batches and fires
them concurrently (bounded by a semaphore). Each batch prompt includes
a random subset of Phase 1 exemplars. Raw per-batch JSON goes to
data/raw_generated/; the consolidated output goes to
data/generated_triplets.json.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path

from anthropic import AsyncAnthropic

from src.category_specs import CATEGORIES_TO_GENERATE, CATEGORY_SPECS, SPANISH_GENDER_NEUTRAL_NOUNS

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw_generated"
OUT_PATH = ROOT / "data" / "generated_triplets.json"

TARGET_PER_CATEGORY = 110
BATCH_SIZE = 22
CONCURRENCY = 12
MODEL_NAME = "claude-sonnet-4-6"
TEMPERATURE = 0.85
SEED = 20260501


def load_phase1_exemplars(category: str) -> list[dict]:
    raw = json.loads((ROOT / "triplets.json").read_text(encoding="utf-8"))
    return [t for t in raw["triplets"] if t["category"] == category]


def make_prompt(category: str, exemplars: list[dict], n_items: int, batch_id: int) -> str:
    spec = CATEGORY_SPECS[category]
    rng = random.Random(SEED + batch_id * 17 + hash(category) % 1000)
    sampled = rng.sample(exemplars, k=min(5, len(exemplars)))
    exemplar_block = "\n".join(json.dumps(e, ensure_ascii=False) for e in sampled)

    schema_note = """Each generated item must have exactly these fields:
{
  "id": "<auto-assigned later, leave as empty string>",
  "language": "<'hindi' or 'spanish'>",
  "category": "<the category key>",
  "anchor": "<source-language sentence>",
  "near": "<source-language paraphrase preserving contrast feature>",
  "far": "<source-language sentence with contrast feature flipped>",
  "anchor_en": "<English translation of anchor>",
  "near_en": "<English translation of near>",
  "far_en": "<English translation of far>",
  "notes": "<one short line: what contrast term changed in 'far'>",
  "source": "phase2b_generated",
  "flattening_intent": "<strong | partial | non_flattening>",
  "dialect": "<peninsular | general | n/a>",
  "validation_status": "pending"
}"""

    constraints_block = f"""CONTRAST FEATURE: {spec.contrast_feature_description}

NEAR rule: {spec.near_rule}
FAR rule: {spec.far_rule}

EXTRA CONSTRAINTS: {spec.extra_constraints}

TARGET FLATTENING INTENT for this category: '{spec.flattening_intent}'.
"""

    extra_lexical = ""
    if category == "gender_agreement_adjectives":
        extra_lexical = (
            "ALLOWED NOUN LEMMAS (use only these; both gendered surface forms are fine): "
            f"{sorted(SPANISH_GENDER_NEUTRAL_NOUNS)}\n"
        )

    instr = f"""Generate {n_items} new triplets for the category `{category}`.

These must follow the same probe design as the exemplars below. Each triplet contrasts a source-language distinction that English does not (or only partially) preserve.

{constraints_block}{extra_lexical}
DO NOT reuse any sentence (anchor/near/far) from the exemplars. Vary subjects, verbs, settings, and contrast lemmas.

Phase 1 exemplars (in JSON, for reference; do NOT copy):
{exemplar_block}

{schema_note}

Output: a single JSON object {{"items": [ ... {n_items} items ... ]}}.
No commentary, no markdown fence, just the JSON object. The id field can be the empty string; ids will be assigned post-hoc.
"""
    return instr


async def call_anthropic(client: AsyncAnthropic, prompt: str, batch_id: int) -> dict:
    """Issue one batched generation request, with exponential-backoff retries."""
    log = logging.getLogger("gen")
    for attempt in range(4):
        try:
            resp = await client.messages.create(
                model=MODEL_NAME,
                max_tokens=8192,
                temperature=TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text
            text = re.sub(r"^```(json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE)
            return {"text": text, "usage": resp.usage.model_dump() if resp.usage else None}
        except Exception as e:
            log.warning("batch=%d attempt=%d failed: %s", batch_id, attempt, e)
            await asyncio.sleep(2 ** attempt)
    raise RuntimeError(f"batch {batch_id} failed after retries")


async def generate_category(
    client: AsyncAnthropic,
    sem: asyncio.Semaphore,
    category: str,
    target: int,
    batch_size: int,
) -> list[dict]:
    log = logging.getLogger("gen")
    n_batches = (target + batch_size - 1) // batch_size
    exemplars = load_phase1_exemplars(category)

    async def one_batch(batch_id: int) -> list[dict]:
        async with sem:
            prompt = make_prompt(category, exemplars, batch_size, batch_id)
            t0 = time.time()
            r = await call_anthropic(client, prompt, batch_id)
            dt = time.time() - t0
            try:
                parsed = json.loads(r["text"])
                items = parsed.get("items", [])
            except json.JSONDecodeError as e:
                log.error("[%s b%d] JSON parse error: %s", category, batch_id, e)
                (RAW_DIR / f"{category}_batch{batch_id}_RAW_FAIL.txt").write_text(r["text"], encoding="utf-8")
                return []
            (RAW_DIR / f"{category}_batch{batch_id}.json").write_text(
                json.dumps({"items": items, "usage": r["usage"]}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            log.info("[%s b%d] %d items in %.1fs", category, batch_id, len(items), dt)
            return items

    tasks = [asyncio.create_task(one_batch(i)) for i in range(n_batches)]
    results = await asyncio.gather(*tasks)
    flat = [item for sublist in results for item in sublist]
    return flat


async def main_async() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("gen")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.error("ANTHROPIC_API_KEY not set")
        sys.exit(1)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    client = AsyncAnthropic(api_key=api_key)

    sem = asyncio.Semaphore(CONCURRENCY)

    t0 = time.time()
    coros = [
        generate_category(client, sem, cat, TARGET_PER_CATEGORY, BATCH_SIZE)
        for cat in CATEGORIES_TO_GENERATE
    ]
    by_cat = await asyncio.gather(*coros)
    elapsed = time.time() - t0

    all_items: list[dict] = []
    for cat, items in zip(CATEGORIES_TO_GENERATE, by_cat):
        prefix = {
            "kinship_paternal_maternal": "hi_kin2",
            "kinship_relative_age": "hi_age2",
            "formality_tv": "hi_for2",
            "ser_vs_estar": "es_ser2",
            "formality": "es_for2",
            "gender_agreement_adjectives": "es_gen2",
        }[cat]
        for i, it in enumerate(items, start=1):
            it["id"] = f"{prefix}_{i:03d}"
            it["category"] = cat
            it.setdefault("source", "phase2b_generated")
            it.setdefault("validation_status", "pending")
        all_items.extend(items)
        log.info("Category %s: collected %d items", cat, len(items))

    OUT_PATH.write_text(
        json.dumps({"items": all_items, "n": len(all_items), "elapsed_s": elapsed}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("Wrote %d items to %s in %.1fs", len(all_items), OUT_PATH, elapsed)


if __name__ == "__main__":
    asyncio.run(main_async())
