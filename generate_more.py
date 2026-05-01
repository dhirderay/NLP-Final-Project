"""Top-up generation against the OpenAI API.

Counts items already in data/raw_generated/ per category, then issues
enough additional generation calls (via gpt-4.1) to reach the target.
Using a second provider/family adds diversity and sidesteps the
Anthropic per-org output-tokens-per-minute cap. Outputs land in
data/raw_generated/{category}_gpt_batch{N}.json and are then merged
into data/generated_triplets.json.
"""

from __future__ import annotations

import asyncio
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI

from src.category_specs import CATEGORIES_TO_GENERATE, CATEGORY_SPECS, SPANISH_GENDER_NEUTRAL_NOUNS

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw_generated"
OUT_PATH = ROOT / "data" / "generated_triplets.json"

TARGET_PER_CATEGORY = 110
BATCH_SIZE = 22
CONCURRENCY = 12
MODEL_NAME = "gpt-4.1"
TEMPERATURE = 0.85
SEED = 20260501


def load_phase1_exemplars(category: str) -> list[dict]:
    raw = json.loads((ROOT / "triplets.json").read_text(encoding="utf-8"))
    return [t for t in raw["triplets"] if t["category"] == category]


def existing_count(category: str) -> int:
    """Count items already saved in raw_generated for this category."""
    n = 0
    for path in glob.glob(str(RAW_DIR / f"{category}_*.json")):
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            n += len(data.get("items", []))
        except Exception:
            continue
    return n


def make_prompt(category: str, exemplars: list[dict], n_items: int, batch_id: int) -> str:
    import random as _r
    spec = CATEGORY_SPECS[category]
    # Offset by 9999 so the OpenAI top-up samples a different exemplar
    # subset than the Anthropic primary run on the same category.
    rng = _r.Random(SEED + batch_id * 17 + hash(category) % 1000 + 9999)
    sampled = rng.sample(exemplars, k=min(5, len(exemplars)))
    exemplar_block = "\n".join(json.dumps(e, ensure_ascii=False) for e in sampled)

    schema_note = """Each generated item must have exactly these fields:
{
  "id": "",
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

    constraints = (
        f"CONTRAST FEATURE: {spec.contrast_feature_description}\n\n"
        f"NEAR rule: {spec.near_rule}\n"
        f"FAR rule: {spec.far_rule}\n\n"
        f"EXTRA CONSTRAINTS: {spec.extra_constraints}\n\n"
        f"TARGET FLATTENING INTENT for this category: '{spec.flattening_intent}'."
    )

    extra_lex = ""
    if category == "gender_agreement_adjectives":
        extra_lex = f"\nALLOWED NOUN LEMMAS: {sorted(SPANISH_GENDER_NEUTRAL_NOUNS)}"

    return (
        f"Generate {n_items} new triplets for the category `{category}`.\n\n"
        f"{constraints}{extra_lex}\n\n"
        "DO NOT reuse any sentence from the exemplars. Vary subjects, verbs, settings, and contrast lemmas.\n\n"
        f"Phase 1 exemplars (do NOT copy):\n{exemplar_block}\n\n"
        f"{schema_note}\n\n"
        f"Output: a single JSON object {{\"items\": [ ... {n_items} items ... ]}}.\n"
        "No commentary, no markdown fence - just the JSON object."
    )


async def call_openai(client: AsyncOpenAI, prompt: str, batch_id: int) -> dict:
    log = logging.getLogger("gen2")
    for attempt in range(4):
        try:
            r = await client.chat.completions.create(
                model=MODEL_NAME,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": "You are a careful linguistic data generator. Produce strictly valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=8000,
            )
            text = r.choices[0].message.content or ""
            return {"text": text}
        except Exception as e:
            log.warning("batch=%d attempt=%d failed: %s", batch_id, attempt, e)
            await asyncio.sleep(1.5 ** attempt)
    raise RuntimeError(f"batch {batch_id} failed after retries")


async def generate_for_category(client, sem, category, n_needed, batch_id_offset):
    log = logging.getLogger("gen2")
    n_batches = (n_needed + BATCH_SIZE - 1) // BATCH_SIZE
    exemplars = load_phase1_exemplars(category)

    async def one_batch(local_idx: int) -> list[dict]:
        bid = batch_id_offset + local_idx
        async with sem:
            t0 = time.time()
            prompt = make_prompt(category, exemplars, BATCH_SIZE, bid)
            r = await call_openai(client, prompt, bid)
            dt = time.time() - t0
            try:
                parsed = json.loads(r["text"])
                items = parsed.get("items", [])
            except json.JSONDecodeError as e:
                log.error("[%s b%d] JSON parse error: %s", category, bid, e)
                (RAW_DIR / f"{category}_gpt_batch{bid}_RAW_FAIL.txt").write_text(r["text"], encoding="utf-8")
                return []
            (RAW_DIR / f"{category}_gpt_batch{bid}.json").write_text(
                json.dumps({"items": items}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            log.info("[%s b%d] %d items in %.1fs", category, bid, len(items), dt)
            return items

    tasks = [asyncio.create_task(one_batch(i)) for i in range(n_batches)]
    results = await asyncio.gather(*tasks)
    return [it for sub in results for it in sub]


async def main_async():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("gen2")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        log.error("OPENAI_API_KEY not set")
        sys.exit(1)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI(api_key=api_key)
    sem = asyncio.Semaphore(CONCURRENCY)

    coros = []
    for cat in CATEGORIES_TO_GENERATE:
        have = existing_count(cat)
        need = max(0, TARGET_PER_CATEGORY - have)
        log.info("Category %s: have %d, need %d more", cat, have, need)
        if need > 0:
            coros.append(generate_for_category(client, sem, cat, need, batch_id_offset=100))

    t0 = time.time()
    await asyncio.gather(*coros)
    log.info("Top-up generation done in %.1fs", time.time() - t0)

    consolidate()


def consolidate():
    log = logging.getLogger("consolidate")
    all_items: list[dict] = []
    prefix = {
        "kinship_paternal_maternal": "hi_kin2",
        "kinship_relative_age": "hi_age2",
        "formality_tv": "hi_for2",
        "ser_vs_estar": "es_ser2",
        "formality": "es_for2",
        "gender_agreement_adjectives": "es_gen2",
    }
    counts = {}
    for cat in CATEGORIES_TO_GENERATE:
        items_for_cat = []
        for path in sorted(glob.glob(str(RAW_DIR / f"{cat}_*.json"))):
            try:
                data = json.loads(Path(path).read_text(encoding="utf-8"))
                for it in data.get("items", []):
                    items_for_cat.append(it)
            except Exception as e:
                log.warning("skip %s: %s", path, e)
        # Re-id and tag.
        for i, it in enumerate(items_for_cat, start=1):
            it["id"] = f"{prefix[cat]}_{i:03d}"
            it["category"] = cat
            it.setdefault("source", "phase2b_generated")
            it.setdefault("validation_status", "pending")
        counts[cat] = len(items_for_cat)
        all_items.extend(items_for_cat)
    OUT_PATH.write_text(
        json.dumps({"items": all_items, "n": len(all_items), "counts": counts}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("Consolidated %d items -> %s", len(all_items), OUT_PATH)
    log.info("Counts: %s", counts)


if __name__ == "__main__":
    asyncio.run(main_async())
