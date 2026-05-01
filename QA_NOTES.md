# QA Notes — triplets.json

Things you should eyeball before running embeddings. Most items are fine, but these are the ones I'd want a native/fluent speaker to sanity-check.

## How to read these

For each flagged item I give the ID, what concerns me, and a suggested edit if you want one. If you change an item, edit `build_triplets.py` (not `triplets.json`) and re-run it — that way the build is reproducible.

## Hindi

### Naturalness / register

- **hi_kin_011** uses `पधारे` (padhaare) — the honorific is fine but pretty formal/Sanskritised. Reads more like a wedding invitation than everyday speech. If you want neutral register, swap to `आए` / `आये`.
- **hi_for_005, hi_for_006, hi_for_009** mix `तू` (intimate) and `आप` (formal) addressed at the same kind of interlocutor (a close friend / sibling). Pragmatically this is *meant* to be marked — that's the whole point of the formality contrast — but flag for your partner that the "far" sentence is socially awkward, not just formally different. That's fine for a probe; just don't let a reviewer think the data is broken.
- **hi_for_009** specifically: addressing a best friend with `आप` is unusually distant. Still grammatical; just very marked.
- The verb-gender-agreement set uses first-person past intransitives (`मैं ... गई/गया`). I picked these intentionally because the subject `main` is morphologically gender-neutral, so the only gender cue is the verb ending — clean for the probe. Double-check the verb forms (`गई/गया`, `पहुँची/पहुँचा`, etc.) are spelled the way you'd write them; I used the most common surface form but transliteration conventions vary.

### Script / spelling

- I used `आएंगे` rather than `आयेंगे` consistently. Both are acceptable; modern usage tends toward the former. If your partner's preferred convention is the latter, swap globally.
- Nukta usage: I avoided nukta on `ज` (so `गई` not `ग़ई` etc.) — standard for Hindi. Urdu-loan words like `इंतज़ार` would normally take a nukta but I didn't use any of those.

### Category labels

- `kinship_relative_age` (`hi_age_*`): English actually *can* mark elder/younger brother (`elder brother` / `younger brother`), so this is a **partial-flattening** category, not strong. The English translations all use `brother` / `sister` to force flattening, which is fair — that's how a default MT system would render `bhai` — but be honest about it in the writeup. This category is the cleanest test of "does the model use the actual Hindi lexical contrast vs. fall back to translation" because the translation route *can* succeed if the pivot system chooses to disambiguate.

## Spanish

### Dialect / regional variation

- **es_ser_002** "Mi madre está joven": grammatical but dialectally marked. Works in some Latin American dialects (esp. Caribbean), borderline in Peninsular. The contrast with "es joven" is real but the sentence itself sounds non-standard to many speakers. If you want a cleaner item, replace with something like "Carlos está nervioso" / "Carlos es nervioso".
- **es_ser_005** "Esta sopa es buena" / "Esta sopa está buena": this is a **partial-flattening** case. English collapses both to "is good" by default but actually *can* render the second as "tastes good" / "is delicious". I kept the flattening English on purpose so the probe sees the lexical contrast unambiguously — flag in the writeup.
- **es_ser_012/013/014** (aburrido/listo/malo): these are the famous "ser+adj vs. estar+adj different meaning" cases (boring/bored, smart/ready, bad/sick). Here English *does* distinguish and I rendered the distinction in the `_en` fields — these are intentional **non-flattening** items, included as in-category contrast. The probe should pass these even on the English-pivot baseline; that's the point.

### Vosotros / ustedes

- **es_for_004, es_for_011, es_for_017** use `vosotros` for plural informal vs. `ustedes` for plural formal. This is a **Peninsular-Spanish-only** distinction; in Latin American Spanish `ustedes` covers both registers. The probe is testing "does the embedding distinguish T-V" which is real in Peninsular usage, so the items are fine, but the model's behaviour on these depends on training-data dialect mix. Worth a footnote in the eval.

### Gender-agreement set

- I deliberately used English-gender-neutral nouns (`friend`, `doctor`, `teacher`, `child`, `colleague`, `journalist`, `artist`, `relative`, `baby`) so the Spanish gender doesn't leak into the English translation — that's the core trick that lets gender-agreement be a flattening category. Double-check this is preserved in every item; if any one of them translates to "actor / actress" instead of "actor", that item leaks and should be fixed.
- Adjective-only contrast (`alto/alta`, `cansado/cansada`, etc.) on the same noun: morphologically minimal, semantically minimal too (the only difference is referent gender). Good probe items.

## Per-item flattening labels (for the eval)

When you build the eval, you'll probably want to label each item as **strong-flattening** (English can't express the distinction without a gloss) or **partial-flattening** (English has the distinction available but a default translation discards it). Suggested labelling:

- **strong**: `kinship_paternal_maternal` (all), `verb_gender_agreement` (all), `formality_tv` (all — English `you` is single-form), `formality` Spanish (all — same reason), `gender_agreement_adjectives` (all if nouns are kept neutral).
- **partial**: `kinship_relative_age` (English has elder/younger brother), `ser_vs_estar` mostly strong but with three intentional non-flattening items (es_ser_012/013/014 — see above).

This labelling matters for the analysis: the prediction is that English-pivot beats the multilingual encoders on *partial* but loses on *strong*. If you're seeing the encoders win on partial too, something else is going on (probably the encoders are doing surface-form matching, not semantics).

## Counts

- Hindi: 50 (kinship_paternal_maternal 13 · kinship_relative_age 12 · verb_gender_agreement 13 · formality_tv 12)
- Spanish: 50 (ser_vs_estar 17 · formality 17 · gender_agreement_adjectives 16)
- Total: 100

## What I did not do

- I did not author any French or German triplets — the proposal is Hindi + Spanish only.
- I did not generate paraphrases automatically; every "near" sentence is a deliberate hand-written paraphrase that preserves the contrast feature, and every "far" sentence flips the contrast feature with minimal other lexical change.
- I did not try to balance for sentence length within triplets; the anchors and pairs are usually within ±2 tokens but I didn't enforce it. If you find length is a confound in your eval (cosine similarity is mildly length-sensitive), you may want to filter or normalise.
