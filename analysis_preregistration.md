# Phase 2B Analysis Pre-Registration

**Date frozen:** 2026-05-01
**Phase 1 dataset:** 100 hand-authored triplets (frozen 2026-04-30)
**Phase 2B target:** 700-item hybrid dataset (MultiBLiMP-derived + LLM-generated, both validated)

This document is committed *before* MultiBLiMP is downloaded, before any LLM generation runs, and before any embeddings are computed on Phase 2B data. Any analysis added after data inspection is explicitly labeled **post-hoc** in the final report.

---

## 1. Hypotheses

### H1 (primary, pre-registered)
On categories where the source-language distinction is *not* expressible in default English translation ("strong-flattening": `kinship_paternal_maternal`, `verb_gender_agreement`, `formality_tv`, `formality` (es), `gender_agreement_adjectives` (es)), at least one multilingual sentence encoder evaluated on source-language text achieves higher triplet accuracy than the English-translation pivot baseline (English MiniLM on `_en` fields).

**Formal statement.** For each strong-flattening category *c*, define
\(\Delta_c = \text{best\_multi\_acc}(c) - \text{pivot\_acc}(c)\).
H1 predicts \(\Delta_c > 0\), with the 95% paired-bootstrap CI excluding 0, for at least 4 of the 5 strong-flattening categories.

### H2 (primary, pre-registered)
On partial-flattening categories (`kinship_relative_age`, `ser_vs_estar`), the gap \(\Delta_c\) is closer to zero than in H1 — formally, the mean strong-flattening gap exceeds the mean partial-flattening gap with a 95% CI on the difference excluding 0.

### H3 (secondary, pre-registered)
On non-flattening sanity-check items (`es_ser_012`/`013`/`014`), the English-pivot baseline achieves ≥ 0.80 triplet accuracy. (Failure of H3 indicates a pipeline bug, not a finding.)

### H4 (exploratory, pre-registered as exploratory)
At least one multilingual encoder beats the TF-IDF surface-form baseline on strong-flattening categories. (If TF-IDF wins, the encoders are doing surface-form matching, not semantics.)

### H5 (exploratory, hybrid-framework specific)
On Hindi `verb_gender_agreement`, the model ranking induced by the **MultiBLiMP pair-distinction metric** (Phase 2B) and the **triplet metric** (Phase 1 + Phase 2B generated) is concordant (Spearman ρ > 0.7).

---

## 2. Metrics

All metrics are computed per (model × language × category × condition) cell.

| Metric | Definition | Use |
|---|---|---|
| **Triplet accuracy** | fraction of items with `cos(a,n) > cos(a,f)` | primary |
| **Mean cosine gap** | `mean(cos(a,n) - cos(a,f))` | effect size |
| **Pair-distinction (MultiBLiMP)** | `1 - cos(grammatical, ungrammatical)` per item; summarised as mean and as fraction below a calibrated threshold | hybrid only |

All cosine values use L2-normalised embeddings (dot product). All embedding models are run with their documented input convention (E5 family: `query: ` prefix; bge-m3: no prefix).

### 2.1 Pair-distinction threshold calibration (MultiBLiMP)

Threshold for the binary "pair-pass" version of the pair-distinction metric is calibrated as the **median cosine** between 200 randomly chosen *non-paired* sentences from the MultiBLiMP set, computed once per model and frozen. A higher pair-distinction (lower cos(gram, ungram)) implies the model is more sensitive to the morphological violation.

---

## 3. Comparisons and intervals

### 3.1 Bootstrap procedure
- 10,000 resamples
- Percentile method (2.5%, 97.5%)
- Fixed seed: `RANDOM_SEED = 20260501`
- For paired comparisons (e.g. multilingual vs pivot on same triplet), resample triplet IDs jointly, recompute both accuracies on the resample, take the difference. CI on the difference, not on the two accuracies separately.

### 3.2 Headline-claim CIs
Reported on:
- `Δ_c` for each strong-flattening category (per H1)
- mean \(\bar{\Delta}_{\text{strong}} - \bar{\Delta}_{\text{partial}}\) (per H2)
- pivot accuracy on non-flattening items (per H3)
- best-encoder vs TF-IDF gap on strong-flattening (per H4)
- Spearman ρ between MultiBLiMP-induced and triplet-induced model rankings (per H5)

### 3.3 Multiple-comparison correction
H1 spans 5 strong-flattening categories. We apply **Bonferroni** correction: a per-category gap is "headline-significant" only if the per-category 99% paired-bootstrap CI excludes 0 (corresponding to family-wise α = 0.05 across 5 tests). For exploratory analyses (per-model rankings, length-confound, scale, cross-language), we report uncorrected 95% CIs and **label them exploratory**.

### 3.4 Language- and model-level comparisons
Reported as 95% CIs without correction; explicitly tagged as exploratory in the report.

---

## 4. Validation gate decision rules (committed)

These thresholds are frozen here and may not be moved after seeing the rejection rate.

### 4.1 LLM validator calibration
- Run validator on all 100 Phase 1 triplets.
- Pass rate on Checks 1–4 must be ≥ 95% on Phase 1 data. Below 95% ⇒ validator prompt is broken; iterate prompts before proceeding.
- Run validator on 12 deliberately-broken items (constructed for this purpose).
- Catch rate must be ≥ 90%. Below 90% ⇒ validator is too lenient; iterate prompts.

### 4.2 LLM validation rate gate (on Phase 2B generated data)
- LLM-flag rate < 10% ⇒ suspicious (validator may be too lenient); audit by adding 5 deliberate breaks into the validation set, confirm catch rate.
- LLM-flag rate 10–30% ⇒ proceed to human spot-check.
- LLM-flag rate > 30% ⇒ generation prompt is broken; one regeneration cycle allowed; if second cycle still > 30%, revert.

### 4.3 Human spot-check gate (final, ground truth)
Sample composition: all LLM-flagged items + 15% random sample of LLM-passed items (stratified by category so each cell has ≥ 10 reviewed items).

- Human rejection rate ≤ 8% ⇒ Phase 2B passes; auto-validated items count as validated.
- 8–15% ⇒ soft fail; expand human review to 40% sample; if still > 10%, revert.
- > 15% ⇒ hard fail; revert to Phase 1 dataset, document in limitations.

### 4.4 Human–LLM agreement
- Agreement on items both reviewed must be ≥ 85%. Below ⇒ LLM validator unreliable; either revert or expand human sample to 40%.

### 4.5 Schedule constraint and substitution
This project's class deadline does not allow real human-panel review at the volume the protocol requires. The following substitution is committed in advance:
- A **dual-LLM cross-check** (different model than the primary validator) is run on the human-review sample composition described in 4.3, and acts as a proxy validator.
- **All gates (4.3, 4.4) are computed on the dual-LLM cross-check**, with the same thresholds.
- The substitution is disclosed prominently in the methods section and the limitations section. Any item flagged by either validator is excluded from the validated set and listed in the rejection log.
- Native-speaker spot-check on a smaller sample (~30 items) is performed *after* the LLM gates if a co-author is available, and reported alongside, but does not replace the gates above.

### 4.6 Validators-must-not-see-results constraint
LLM validators are run **before** any embedding model is run on the new data. Validator code does not import `src.embed` or `src.evaluate`. This is enforced by code-level isolation (the validation step writes a "validated_triplets.json" snapshot, and the embedding step reads only from that snapshot).

---

## 5. Pre-registered analyses to run on the validated dataset

(Each is described in `CLAUDE_CODE_PHASE2B_v2.md` §5 and must run on the *post-validation* dataset.)

1. Triplet-accuracy + cosine-gap per (model × language × category) with bootstrap CIs.
2. H1 paired-bootstrap test per strong-flattening category (Bonferroni-corrected).
3. H2 strong-vs-partial gap test (uncorrected, primary).
4. H3 non-flattening sanity check.
5. H4 best-encoder vs TF-IDF (exploratory, but pre-registered).
6. Length-confound analysis: Pearson + Spearman on `len_diff` vs `cos_diff` per model. **No "drop top-quintile" filter unless H4 fails** — that filter would be post-hoc.
7. Qualitative examples table: deterministic selection rules, frozen here:
   - "Textbook win": highest gap on `kinship_paternal_maternal`, multilingual model, Phase 1.
   - "Morphological failure": lowest gap on `verb_gender_agreement`, best multilingual model.
   - "LaBSE-specific failure": item where LaBSE fails but MiniLM-multi and E5-multi pass.
   - "Cross-model disagreement": item with maximum standard deviation of `correct` across the 3 multilingual models.
   - "Partial-flattening surprise": highest gap on `kinship_relative_age`, multilingual model.
   - "Predicted partial win": item where pivot beats all multilingual models on `ser_vs_estar`.
8. LaBSE ser/estar deep-dive: cosine values + cross-lingual collapse analysis.
9. Scale analysis: e5-small → e5-large within family (BAAI/bge-m3 added as a contrast point).
10. Cross-language comparison: average accuracy per model on Hindi-strong vs Spanish-strong items.
11. Hybrid framework cross-validation: Spearman ρ on per-model rankings between MultiBLiMP pair-metric and Phase 1 triplet metric on `verb_gender_agreement`.
12. Phase 1 vs Phase 2B replication check: per-category accuracy comparison with paired bootstrap.

---

## 6. Post-hoc analyses (must be labeled as such in the report)

Any of the following, if performed after seeing data, must be tagged "post-hoc" in the writeup:
- New metrics or thresholds.
- Removing categories or items based on observed difficulty.
- Adding models not in §5.
- Different bootstrap procedures (block bootstrap, etc.).
- Per-item drop based on cosine value distribution rather than a documented quality issue.

---

## 7. Reproducibility

- Random seed: `RANDOM_SEED = 20260501`.
- All API calls (generation + validation) run with `temperature = 0.0` (validation) or `temperature = 0.7` (generation, to allow diversity but seeded), and the seed is logged with each call.
- Generation outputs raw to `data/raw_generated/`, validation logs to `data/llm_validation_log.csv`.
- Snapshot dataset (`triplets_phase2b.json`) is hashed (sha256) and the hash is stored in `results/dataset_manifest.json`. Any re-run of analysis logs that hash; mismatch ⇒ analysis reverted.

---

## 8. What we will *not* claim

- No causal claims about embedding objectives. Cross-lingual alignment objective ↔ ser/estar collapse is presented as a *mechanistic hypothesis*, not a tested causal claim.
- No claims about languages other than Hindi and Spanish.
- No claims about morphological agreement beyond gender (number, person agreement are not in the validated dataset).
- No claims about sentence length effects unless H4-related length analysis shows them.

---

**End of pre-registration.** Subsequent files (code, data) reference this document. Any deviation from this plan is logged in `results/preregistration_deviations.md` with timestamp and justification.
