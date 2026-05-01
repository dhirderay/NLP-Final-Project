# Cross-Lingual Semantic Granularity Probe

Do multilingual sentence embeddings preserve semantic distinctions that the source language encodes but English does not? We test this on hand-authored and LLM-generated minimal-pair triplets in Hindi and Spanish, plus 100 gender-agreement items from MultiBLiMP, and compare native-language scoring against an English-translation pivot baseline.

The headline finding: on strong-flattening categories (kinship granularity, T-V formality, Spanish gender agreement on neutral nouns) multilingual encoders beat the English-pivot baseline by very large, statistically significant margins. On partial-flattening categories the picture is mixed; ser/estar in particular collapses across every multilingual encoder we tested. See [REPORT_SUMMARY.md](REPORT_SUMMARY.md) for the full writeup.

## Repository layout

```
.
├── triplets.json                       Phase 1 dataset (100 hand-authored triplets)
├── build_triplets.py                   reproducible builder for Phase 1 dataset
├── QA_NOTES.md                         per-item linguistic notes for Phase 1
├── analysis_preregistration.md         pre-registration, frozen 2026-05-01
├── REPORT_SUMMARY.md                   findings writeup
├── internal_proposal.md.pdf            original project proposal
│
├── run_all.py                          Phase 1 entry point
├── run_phase2b.py                      Phase 2B entry point (validated dataset + analysis)
├── build_figures.py                    rebuild all figures from saved CSVs
│
├── generate_triplets.py                async generation via Anthropic
├── generate_more.py                    async top-up via OpenAI
├── validate_llm.py                     primary validator (gpt-4.1)
├── validate_dual.py                    dual-LLM proxy (claude-sonnet-4-6)
├── validate_human.py                   CLI for the human review pass
│
├── src/
│   ├── load_data.py                    Triplet dataclass + loader
│   ├── embed.py                        sentence-transformer wrappers (E5 prefix etc.)
│   ├── evaluate.py                     triplet-accuracy + cosine-gap scoring
│   ├── baselines.py                    TF-IDF char-n-gram baseline
│   ├── flattening_labels.py            strong / partial / non_flattening labels
│   ├── category_specs.py               per-category generation/validation spec
│   ├── auto_filter.py                  rule-based quality filter
│   ├── multiblimp_integration.py       MultiBLiMP sampler + threshold pool
│   ├── hybrid_validation.py            MultiBLiMP pair metric + cross-validation
│   ├── bootstrap.py                    bootstrap CIs, paired and unpaired
│   ├── length_analysis.py              length-confound check
│   ├── qualitative.py                  deterministic qualitative-example selection
│   ├── replication_check.py            Phase 1 vs Phase 2B replication
│   ├── labse_analysis.py               cross-lingual collapse analysis
│   ├── plots.py                        Phase 1 figures (fig1-fig4)
│   └── plots_phase2b.py                Phase 2B figures (fig1-fig9)
│
├── data/
│   ├── multiblimp_hindi_gender.json    100 sampled MultiBLiMP items + threshold pool
│   ├── generated_triplets.json         727 raw Phase 2B items (consolidated)
│   ├── generated_triplets_filtered.json 527 items post auto-filter
│   ├── triplets_phase2b.json           611 final validated combined dataset
│   ├── auto_filter_log.csv             rejection reasons by item
│   ├── llm_validation_log.csv          gpt-4.1 verdicts (5 checks per item)
│   ├── dual_validation_log.csv         claude-sonnet-4-6 verdicts on the review sample
│   ├── human_llm_disagreement.csv      primary vs dual agreement per item
│   ├── validator_calibration.csv       calibration on Phase 1 + broken examples
│   └── raw_generated/                  per-batch JSON outputs (provenance)
│
├── results/
│   ├── per_triplet.csv                 every (triplet, model, condition) row
│   ├── summary.csv                     aggregated with bootstrap CIs
│   ├── summary_by_flattening.csv       same, grouped by flattening type
│   ├── headline.csv                    per-category multilingual-vs-pivot gap with 99% Bonferroni-corrected paired CIs
│   ├── multiblimp_per_item.csv         MultiBLiMP per-item pair distinctions
│   ├── hybrid_framework_validation.csv per-model triplet vs pair metrics
│   ├── hybrid_summary.json             Spearman rho between the two frameworks
│   ├── labse_ser_estar_pivot.csv       per-item ser/estar cosines across models
│   ├── labse_cross_lingual.csv         Spanish-English cross-lingual cosines
│   ├── length_analysis.csv             Pearson/Spearman per (model, condition)
│   ├── qualitative_examples.csv        deterministic example table
│   ├── phase1_phase2b_replication.csv  per-cell replication with paired CIs
│   ├── cross_lang_comparison.csv       Hindi vs Spanish accuracy summary
│   ├── dataset_manifest.json           sha256 + counts for the validated dataset
│   └── bootstrap_config.json           seed and method for reproducibility
│
└── figures/
    ├── fig1_headline.png               best multilingual vs English pivot, with 99% CIs
    ├── fig2_per_model.png              per-model accuracy by category
    ├── fig3_strong_vs_partial.png      strong vs partial flattening by model
    ├── fig4_cosine_gap.png             cosine-gap distributions
    ├── fig5_length_confound.png        cos_gap vs |anchor - far| length diff
    ├── fig6_labse_ser_estar.png        within-language collapse + cross-lingual alignment
    ├── fig7_scale.png                  accuracy vs approximate model size
    ├── fig8_cross_lang.png             Hindi vs Spanish per flattening
    └── fig9_hybrid.png                 MultiBLiMP pair metric vs triplet metric
```

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # see below for the package list
```

The package list is:

```
sentence-transformers
scikit-learn
matplotlib
pandas
numpy
scipy
anthropic
openai
datasets
huggingface_hub
httpx
tenacity
```

### Phase 1 (Phase 1 dataset only, no API keys required after model download)

```bash
python run_all.py --smoke      # quick 5-triplet sanity check
python run_all.py              # full eval, writes results/ and figures/
```

### Phase 2B (rebuild everything from validated snapshots)

```bash
python run_phase2b.py --bootstrap-n 10000
```

This step requires the validated dataset (`data/triplets_phase2b.json`) and reads the validation logs to construct it. It does not call any external API.

### End-to-end (regenerate the dataset from scratch)

These steps make API calls. Set `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` in your shell first.

```bash
python -m src.multiblimp_integration                # download Hindi MultiBLiMP gender items
python generate_triplets.py                        # primary generation, Anthropic
python generate_more.py                            # OpenAI top-up
python -m src.auto_filter                           # rule-based quality filter
python validate_llm.py calibrate                   # validator calibration on Phase 1 + broken examples
python validate_llm.py validate                    # primary LLM validation
python validate_dual.py                            # dual-LLM proxy review
python run_phase2b.py --bootstrap-n 10000          # final analysis
python build_figures.py                            # regenerate all figures
```

The entire chain finishes in roughly an hour on consumer hardware once the embedding models are cached. Generation/validation parallelism is bounded by per-org rate limits.

## Reproducibility

- Random seed: `RANDOM_SEED = 20260501` everywhere stochastic.
- Bootstrap config (`results/bootstrap_config.json`): 10,000 resamples, percentile method, fixed seed.
- Combined dataset hash (`results/dataset_manifest.json`): every analysis re-run logs this hash; mismatch invalidates the run.
- All API calls run with `temperature = 0` for validation and `temperature = 0.85` for generation, with seeded prompt sampling.

## Citation

If you use the MultiBLiMP integration directly, cite Jumelet et al. 2025 (the source dataset). The probe design and triplet construction are this project's contribution.
