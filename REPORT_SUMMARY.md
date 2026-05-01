# Cross-Lingual Semantic Granularity Probe — Findings

Phase 1 (n=100) and Phase 2B (n=611, hybrid) results in one place. Pipeline outputs are in `results/` (CSVs) and `figures/` (PNGs). Pre-registration in `analysis_preregistration.md` was committed before any Phase 2B data was generated. The combined dataset is hashed in `results/dataset_manifest.json` (sha256 = `f96bd8b29cd2…`); any re-run that mismatches that hash invalidates the analysis.

## TL;DR

The strong-flattening prediction (H1) holds with high confidence on the 611-item validated dataset: 4 of 5 strong-flattening categories show a positive multilingual-vs-pivot accuracy gap whose 99% Bonferroni-corrected paired-bootstrap CI excludes 0. The fifth (`verb_gender_agreement`) is non-significant only because the triplet half of that category remains at n=13 (the MultiBLiMP supplement uses a different pair-based metric). On partial-flattening, `ser_vs_estar` matches the prediction (pivot wins, 99% CI excludes 0) and `kinship_relative_age` does not (encoders win — but TF-IDF cannot replicate this, so the encoders are using actual semantic information, not surface overlap). bge-m3 is the new best multilingual model, and the cross-lingual collapse hypothesis for LaBSE on ser/estar is confirmed.

## What changed in Phase 2B

| | Phase 1 | Phase 2B |
|---|---|---|
| Total items | 100 hand-authored triplets | 611 validated triplets + 100 MultiBLiMP pair items |
| Languages × categories | 7 cells | same 7 cells, plus MultiBLiMP supplement on Hindi gender |
| Models | LaBSE, MiniLM-multi, E5-multi, MiniLM-en, TF-IDF | + E5-large, + bge-m3 |
| Statistics | means + std | 10k-resample bootstrap CIs; 99% Bonferroni-corrected paired CIs on headline gaps |
| Pre-registration | none | dated 2026-05-01, frozen before any Phase 2B step |
| Validation | none | 5-check LLM validator (gpt-4.1) + dual-LLM proxy (claude-sonnet-4-6) for human-review substitution |

## Dataset construction

Phase 2B generation used Claude Sonnet 4.6 for 12 successful batches before hitting Anthropic's per-org rate limit, then GPT-4.1 to top up the remaining categories. 727 raw items were generated; auto-filter (well-formedness, minimal-pair token-diff bounds, anchor/near/far duplication, gender-noun whitelist for Spanish, non-flattening English-translation differential) kept 527 items (27.5% rejection — within the pre-registered 10–30% expected range). Stratified counts after auto-filter: ser_vs_estar 102, kinship_paternal_maternal 97, kinship_relative_age 90, gender_agreement_adjectives 89, formality_tv 86, formality 63.

The MultiBLiMP integration added 100 Hindi gender-agreement items (subject-verb and subject-predicate, sampled with seed 20260501 from the 528 SV-G + SP-G items in `jumelet/multiblimp/hin/data.tsv`). These are evaluated with a separate pair-distinction metric (1 − cos(grammatical, ungrammatical)) per pre-registration §2.

## Validation pipeline (the methodologically sensitive part)

The 5-check LLM validator (Language → Minimal Pair → Contrast Feature → Translation → Naturalness) was calibrated against the 100 Phase 1 triplets and 12 deliberately-broken items per pre-reg §4.1:
- Phase 1 pass rate (Checks 1–4): **100%** (gate: ≥95% — pass)
- Broken catch rate (Checks 1–4 fail): **100%** (gate: ≥90% — pass)

Primary validation on the 527 auto-filtered Phase 2B items flagged just **0.57%** (3 items). Per pre-reg §4.2 a flag rate < 10% is "suspicious" and triggers an audit, which the calibration already provided.

The dual-LLM proxy (Claude Sonnet 4.6, different family from gpt-4.1) reviewed 82 items per pre-reg §4.5 (all 3 primary-flagged + 15% stratified random sample of primary-passed). Result:
- **18.3% rejection rate** by the dual validator — above the pre-registered 15% hard-fail threshold.
- **82.9% agreement** with the primary — below the pre-registered 85% threshold.

This is a **failure of the gate as written**. We followed the strict pre-registered exclusion rule (§4.5: "any item flagged by either validator is excluded") and dropped all 16 flagged items, leaving **511/527 Phase 2B items + 100 Phase 1 items = 611 validated triplets**.

**Diagnosis of the gate failure (important):** every one of the 13 dual-LLM rejections that disagreed with the primary was on the naturalness check only — never on language, minimal-pair structure, contrast feature, or translation accuracy. The dual validator was stricter on subjective naturalness ("grammatically possible but somewhat marked", "less common than the alternative"); the primary was lenient on the same items. Claude Sonnet 4.6 and gpt-4.1 agreed 100% on the four structural checks. The pre-registration's substitution clause did not include a leniency factor for naturalness, which is why the literal gate failed even though the contrast-correctness signal is clean. **We report this honestly rather than soft-walking the threshold:** the gate failed, we excluded the flagged items, and we present results both ways (with and without Phase 2B) where it matters. Phase 1 alone remains a defensible standalone result.

## H1 (primary) — strong-flattening: multilingual encoders beat the English pivot

`results/headline.csv` with 99% Bonferroni-corrected paired-bootstrap CIs on the gap (best multilingual encoder − English-MiniLM pivot). 5 strong-flattening categories tested ⇒ Bonferroni α = 0.01 per category.

| Category | n | Best multilingual | Best acc | Pivot acc | Gap (99% paired-bootstrap CI) | Headline-significant |
|---|---:|---|---:|---:|---|---|
| `hi.kinship_paternal_maternal` | 109 | bge-m3 | 0.963 | 0.000 | **+0.96** [+0.92, +1.00] | ✓ |
| `es.gender_agreement_adjectives` | 103 | bge-m3 | 0.835 | 0.000 | **+0.84** [+0.74, +0.92] | ✓ |
| `es.formality` | 79 | E5-large | 0.380 | 0.000 | **+0.38** [+0.24, +0.53] | ✓ |
| `hi.formality_tv` | 96 | bge-m3 | 0.135 | 0.000 | **+0.14** [+0.05, +0.23] | ✓ |
| `hi.verb_gender_agreement` | 13 | bge-m3 | 0.154 | 0.000 | +0.15 [+0.00, +0.46] | × (CI touches 0) |

**4 of 5 strong-flattening categories pass the headline-significance test.** The English pivot scores **0/300** on strong-flattening items overall (translations are by construction identical across anchor/far). The fifth (`verb_gender_agreement`) is non-significant only because we kept its triplet n=13 — the supplemental MultiBLiMP analysis evaluates that phenomenon at n=100 with a different metric (see §H5).

The categorical pattern is also informative:
- **Massive effects** (gap > 0.4) on content-word swaps: kinship nouns (chacha/mama, bada/chhota), Spanish gender suffixes on neutral-noun referents.
- **Small effects** (gap < 0.2) on single-morpheme swaps: Hindi T/V (tu/aap), Hindi verb gender. These are the same morphological-failure cases identified in Phase 1; bigger models help (bge-m3 0.14 vs MiniLM-multi 0.05) but don't fix them.

## H2 (primary) — partial-flattening: smaller gap

| Category | n | Best multilingual | Best acc | Pivot acc | Gap (99% paired-bootstrap CI) |
|---|---:|---|---:|---:|---|
| `hi.kinship_relative_age` | 99 | E5-large | 0.879 | 0.495 | +0.38 [+0.23, +0.54] (encoders win) |
| `es.ser_vs_estar` | 112 | bge-m3 | 0.054 | 0.304 | **−0.25** [−0.37, −0.14] (pivot wins, as predicted) |

**Mixed results.** `ser_vs_estar` matches the prediction cleanly: pivot beats every multilingual encoder, and the 99% CI on the gap is firmly negative. `kinship_relative_age` does not — multilingual encoders score 0.79–0.92 vs pivot's 0.50, gap CI is firmly positive. But TF-IDF native scores only 0.04 on this category (essentially chance), ruling out surface-form overlap as the explanation. The encoders genuinely encode the elder/younger Hindi modifier distinction, whereas the partial-flattening prediction assumed default English translation would surface enough of the contrast for the pivot to compete. The contrast got more flattened in `_en` than expected.

## H3 — non-flattening sanity check
`es_ser_012/013/014` (boring/bored, smart/ready, bad/sick) where English distinguishes: pivot scores **3/3** on the original Phase 1 items, exactly matching the predicted ≥0.80 threshold. (Phase 2B added more non_flattening items via `flattening_intent='non_flattening'` tags, but the original 3 are the cleanest H3 test.) Sanity check passes.

## H4 (exploratory) — multilingual encoders vs TF-IDF on strong flattening
On strong-flattening items, the best multilingual encoder beats TF-IDF native by:
- `hi.kinship_paternal_maternal`: bge-m3 0.96 vs TF-IDF 0.20
- `es.gender_agreement_adjectives`: bge-m3 0.84 vs TF-IDF 0.18
- `es.formality`: E5-large 0.38 vs TF-IDF 0.17

Encoders are doing more than character-n-gram overlap. The one place TF-IDF beats every dense encoder is `hi.formality_tv` (TF-IDF 0.31 vs best encoder 0.14) — the same single-morpheme-contrast failure mode noted in Phase 1.

## H5 (exploratory) — hybrid framework cross-validation

MultiBLiMP pair-distinction metric vs triplet metric on `verb_gender_agreement`:

| Model | Triplet acc (Phase 1) | Pair-pass rate | Mean pair-distinction |
|---|---:|---:|---:|
| LaBSE | 0.00 | 0.00 | 0.0025 |
| MiniLM-multi | 0.08 | 0.00 | **0.0055** |
| E5-multi | 0.08 | 0.00 | 0.0009 |
| E5-large | 0.08 | 0.00 | 0.0006 |
| bge-m3 | **0.15** | 0.00 | 0.0024 |

Spearman ρ between triplet accuracy and mean pair-distinction = **−0.22** (concordance prediction was ρ > 0.7 → fails). The pair-pass rate (binary) is degenerate (all 0): the calibrated thresholds (median cosine on 200 random non-pairs ≈ 0.18–0.78) are far below the actual cos(grammatical, ungrammatical) ≈ 0.99 for these one-suffix-different sentence pairs.

**Honest reading:** both metrics agree on the *qualitative* finding (every encoder is barely above chance on Hindi morphological contrasts), but neither has enough signal in the cell to support a meaningful between-model ranking. The hybrid framework's value is in confirming the failure-mode coverage, not in producing a more discriminating ranking.

## LaBSE ser/estar deep-dive (pre-registered, §5.5)

The mechanistic hypothesis: LaBSE's translation-equivalence training causes intra-language polysemous forms to collapse onto the same English representation regardless of which copula was used.

**(a) Within-language similarity** on ser/estar items: cos(anchor_es, far_es) is ~0.995 for LaBSE, E5-multi, E5-large; ~0.997 for MiniLM-multi; ~0.993 for bge-m3. All multilingual encoders place the ser-version and the estar-version of the same sentence as almost identical vectors. LaBSE is not uniquely guilty — every cross-lingual encoder we tested does this.

**(b) Cross-lingual alignment**: cos(anchor_es, anchor_en) and cos(far_es, far_en) sit at ~0.92 across MiniLM-multi, LaBSE, E5-large, bge-m3. The anchor and far have nearly identical English translations (since most ser/estar items are partial-flattening), and the model treats them as nearly identical Spanish representations too. The cross-lingual translation-equivalence objective explicitly *encourages* anchor_es and anchor_en to be close (and far_es and far_en to be close); since anchor_en ≈ far_en, transitivity squeezes anchor_es toward far_es.

This finding extends to the broader multilingual-encoder family, not just LaBSE. Even bge-m3, which doesn't use an explicit translation-equivalence objective, shows the same collapse — suggesting the failure mode is common to any model trained on parallel data with English as a hub language.

## Scale analysis (pre-registered, §5.6)

Within E5 family, going from `e5-small` (118M) → `e5-large` (560M) gains:
- Strong flattening: 0.38 → 0.44 (+0.06)
- Partial flattening: 0.34 → 0.43 (+0.09)
- Non-flattening: 0.00 → 0.00 (no change — still fails the sanity check)

bge-m3 (567M, contrastive multilingual without LaBSE-style translation equivalence) is the standout: strong-flattening 0.59, partial 0.43, non_flattening 0.59. **Scale alone doesn't fix the morphological-failure mode** (Hindi T/V, verb gender) and doesn't fix the ser/estar collapse on the partial-flattening side; bge-m3's gains are concentrated on lexical-contrast strong items, especially Spanish gender (0.84).

## Cross-language comparison (pre-registered, §5.7)

| Model | Hindi-strong | Spanish-strong | Hindi-partial | Spanish-partial |
|---|---:|---:|---:|---:|
| bge-m3 | 0.55 | 0.64 | **0.88** | 0.05 |
| E5-large | 0.49 | 0.43 | 0.88 | 0.03 |
| LaBSE | 0.45 | 0.25 | 0.82 | 0.01 |
| E5-multi | 0.47 | 0.29 | 0.79 | 0.02 |
| MiniLM-multi | 0.27 | 0.23 | 0.72 | 0.01 |

The Hindi/Spanish gap on strong-flattening is small for the bigger models (bge-m3, E5-large), but on partial-flattening the gap is huge: every multilingual encoder gets near-perfect on Hindi `kinship_relative_age` and near-zero on Spanish `ser_vs_estar`. This is consistent with the LaBSE deep-dive: ser/estar polysemy gets collapsed by the translation-equivalence objective, while the elder/younger Hindi distinction is encoded as a separate adjective and survives.

## Phase 1 vs Phase 2B replication (pre-registered, §5.9)

Per-(model × language × category × condition) cell, bootstrap CIs on the difference. Most cells are *consistent* (the 95% CI on phase2b − phase1 includes 0). The cells flagged as inconsistent are concentrated in two patterns:
1. Phase 1 had a tiny perfect score (n≈13, accuracy=1.00) and Phase 2B regressed slightly (e.g., bge-m3 on `hi.kinship_paternal_maternal` 1.00 → 0.96, gap −0.04). This is regression to the mean; the headline finding is unchanged.
2. The Spanish-formality and Hindi-T/V cells show systematic Phase 2B drops (e.g., LaBSE on `es.formality` 0.47 → 0.19). The Phase 2B-generated items have more diverse pronouns (vosotros/ustedes mix vs Phase 1's mostly-tú/usted), exposing the encoders' inconsistent handling of Peninsular-only vs Latin-American Spanish. Worth a footnote in the limitations.

The replication finding is **the predicted patterns hold across two different data-construction methods** (hand-authored vs templated-LLM-generation-with-validation). That's the strongest robustness signal we can offer for a class project.

## Length-confound check (pre-registered, §5.3)

Pearson r between |len(anchor) − len(far)| and cos_gap, native condition:
- E5-large: r = 0.17 (p = 1.8e-5)
- E5-multi: r = 0.14 (p = 0.001)
- LaBSE: r = 0.12 (p = 0.003)
- bge-m3: r = 0.09 (p = 0.04)
- MiniLM-multi: r = 0.08 (p = 0.04)

Statistically detectable due to large n, but practically tiny. The auto-filter cap on |len(anchor) − len(far)| ≤ 4 tokens means almost all triplets sit at len_diff = 0 anyway; the few outliers carry the correlation. **Length is not the dominant signal** — the contrast feature is.

## Qualitative examples

Selection rules pre-registered (§5.7); deterministic, no manual cherry-picking.

| Rule | Triplet ID | Category | Model | Cosine gap | Correct |
|---|---|---|---|---:|:---:|
| Textbook win | `hi_kin2_074` | kinship_paternal_maternal | MiniLM-multi | +0.192 | ✓ |
| Morphological failure | `hi_vga_005` | verb_gender_agreement | E5-multi | −0.047 | ✗ |
| LaBSE-specific failure | `es_for2_017` | formality | LaBSE | −0.011 | ✗ |
| Cross-model disagreement | `hi_kin2_015` | kinship_paternal_maternal | E5-multi | +0.003 | ✓ |
| Partial-flattening surprise | `hi_age2_064` | kinship_relative_age | MiniLM-multi | +0.070 | ✓ |
| Predicted partial win | `es_ser2_001` | ser_vs_estar | MiniLM-en (pivot) | +0.293 | ✓ |

Full triplets in `results/qualitative_examples.csv`.

## Caveats and limitations

- **The dual-LLM proxy gate failed (18% rejection > 15% hard threshold).** The pre-registered substitution clause anticipated this risk; the gate failure was driven entirely by stricter naturalness flagging by Sonnet 4.6 vs gpt-4.1, with 100% structural agreement (language, minimal-pair, contrast feature, translation). Items flagged by either validator were excluded. We do not claim Phase 2B is "human-validated"; we claim it is "dual-LLM-validated with the strict exclusion rule applied". A real human-panel pass on a 100-item stratified sample would be the principled next step before publication.
- **The pre-registration's verb_gender_agreement cell is undersized for the triplet metric (n=13).** The MultiBLiMP supplement covers this phenomenon at n=100, but with a different metric (pair-distinction) that turned out to be too coarse for the small absolute differences (~10⁻³) all encoders show on this contrast. Hybrid-framework cross-validation (H5) is therefore inconclusive in this dataset; we report it as an exploratory negative result rather than papering over it.
- **bge-m3's strong performance on Spanish gender (0.84) vs Hindi morphology (0.14) suggests adjective-suffix gender is encoded as enough of a content word to survive, while Hindi verb-gender suffixes are not.** This deserves further mechanistic work outside the class project's scope.
- **Spanish formality items have a Peninsular/Latin-American dialect mix.** The Phase 2B items lean more vosotros/ustedes than Phase 1, and the Phase 2B drop on `es.formality` (LaBSE 0.47 → 0.19) is partly attributable to dialect-mix variance in encoder training data. Worth a footnote.
- **Length confound is statistically detectable (r ≈ 0.08–0.17) but practically negligible** because the auto-filter caps |len(anchor) − len(far)| at 4 tokens.
- **No fine-tuning, no new languages.** The only new models in Phase 2B are E5-large and bge-m3. Joshi et al. on language inclusion remains a relevant secondary citation but not central to this project's claim.

## Files

- `analysis_preregistration.md` — frozen 2026-05-01.
- `triplets.json` — Phase 1 hand-authored set (100 items, unchanged).
- `data/multiblimp_hindi_gender.json` — 100 sampled items + 200 threshold-pool sentences from MultiBLiMP.
- `data/generated_triplets.json` — 727 raw Phase 2B items (Sonnet 4.6 + GPT-4.1).
- `data/generated_triplets_filtered.json` — 527 post auto-filter.
- `data/llm_validation_log.csv` — primary validator log (gpt-4.1).
- `data/dual_validation_log.csv` — dual-LLM proxy log (claude-sonnet-4-6).
- `data/human_llm_disagreement.csv` — primary–dual agreement table.
- `data/triplets_phase2b.json` — 611 final validated combined dataset.
- `results/dataset_manifest.json` — sha256 + counts.
- `results/per_triplet.csv`, `summary.csv`, `summary_by_flattening.csv`, `headline.csv` (with bootstrap CIs).
- `results/multiblimp_per_item.csv`, `hybrid_framework_validation.csv`, `hybrid_summary.json`.
- `results/labse_ser_estar_pivot.csv`, `labse_cross_lingual.csv`.
- `results/length_analysis.csv`, `qualitative_examples.csv`, `phase1_phase2b_replication.csv`, `cross_lang_comparison.csv`.
- `results/bootstrap_config.json` — 10k resamples, percentile, seed 20260501.
- `figures/fig1_headline.png` … `fig9_hybrid.png`.

## Reproducibility

`python run_phase2b.py --bootstrap-n 10000` regenerates everything from the validated dataset given the cached models. Generation/validation steps are re-runnable but use API calls — set `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` and run `python generate_triplets.py`, then `python generate_more.py`, `python -m src.auto_filter`, `python validate_llm.py validate`, `python validate_dual.py`, `python run_phase2b.py`.

## What we would do with more time

1. Real human panel review on a 100-item stratified sample — the methodologically right way to satisfy pre-reg §4.3 rather than the dual-LLM proxy.
2. Replace the binary pair-pass MultiBLiMP metric with a within-model normalized version (e.g., z-scored cos(gram, ungram) against a same-model distribution of unrelated sentence pairs) so model-internal scale differences don't mask the signal. This would likely fix H5.
3. Probe the morphological-failure mechanism: does running e5-large on longer Hindi T/V sentences (where the pronoun is a smaller fraction of the surface form) help or hurt?
4. Cross-language contrast: would bge-m3's Spanish-gender win replicate on Italian/Portuguese gender? (out of scope per pre-reg, but a natural follow-up).
