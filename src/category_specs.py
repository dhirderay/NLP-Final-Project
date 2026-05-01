"""Per-category generation/validation specs.

Single source of truth for what each category tests, what counts as a
near/far contrast, the auto-filter constraints, and the category-specific
validator question. The generator and the LLM validator both read from
here so they can't drift out of sync.
"""

from __future__ import annotations

from dataclasses import dataclass

# Spanish nouns whose English translation is gender-neutral. Used to keep
# gender_agreement_adjectives items from leaking gender into _en fields
# (no actor/actress, etc.).
SPANISH_GENDER_NEUTRAL_NOUNS = {
    "amigo", "amiga", "amigos", "amigas",
    "doctor", "doctora", "doctores", "doctoras",
    "profesor", "profesora", "profesores", "profesoras",
    "maestro", "maestra", "maestros", "maestras",
    "niño", "niña", "niños", "niñas",
    "colega", "colegas",
    "periodista", "periodistas",
    "artista", "artistas",
    "pariente", "parientes",
    "bebé", "bebés", "bebe",
    "estudiante", "estudiantes",
    "abogado", "abogada", "abogados", "abogadas",
    "ingeniero", "ingeniera",
    "compañero", "compañera",
    "vecino", "vecina",
    "primo", "prima",
    "tío", "tía",
    "hermano", "hermana",
    "hijo", "hija",
}


@dataclass(frozen=True)
class CategorySpec:
    key: str
    language: str
    contrast_feature_description: str
    near_rule: str
    far_rule: str
    flattening_intent: str
    extra_constraints: str = ""
    validator_check_question: str = ""


CATEGORY_SPECS: dict[str, CategorySpec] = {
    "kinship_paternal_maternal": CategorySpec(
        key="kinship_paternal_maternal",
        language="hindi",
        contrast_feature_description=(
            "Hindi distinguishes paternal and maternal kinship terms (chacha = father's brother, "
            "mama = mother's brother; dada = paternal grandfather, nana = maternal grandfather; "
            "bua = father's sister, mausi = mother's sister; etc.). English collapses both to a "
            "single term ('uncle', 'grandfather', 'aunt'). The probe tests whether multilingual "
            "embeddings preserve the paternal/maternal distinction the source language encodes."
        ),
        near_rule=(
            "Same kinship term as anchor; different surface form (paraphrase: alternate verb, "
            "synonym, slight reordering). The paternal-or-maternal designation is preserved."
        ),
        far_rule=(
            "Swap the kinship term to its corresponding paternal/maternal counterpart "
            "(chacha ↔ mama, dada ↔ nana, etc.). Otherwise minimal change. The English "
            "translation should collapse to the same generic English kinship term."
        ),
        flattening_intent="strong",
        extra_constraints=(
            "Use only Hindi paternal/maternal pairs in {chacha/mama, chachi/mami, "
            "dada/nana, dadi/nani, bua/mausi, phupha/mausa, taauji/mamaji, taiji/mamiji, "
            "bhatija/bhanja, bhatiji/bhanji}. Do NOT mix unrelated kinship terms. "
            "anchor_en should equal far_en (the English translation collapses)."
        ),
        validator_check_question=(
            "Do anchor and near use the same paternal-or-maternal kinship term, and does far "
            "swap it to the corresponding paternal/maternal counterpart with otherwise minimal change?"
        ),
    ),
    "kinship_relative_age": CategorySpec(
        key="kinship_relative_age",
        language="hindi",
        contrast_feature_description=(
            "Hindi distinguishes relative-age within sibling kinship: bada bhai (elder brother) vs "
            "chhota bhai (younger brother), badi behen vs chhoti behen, etc. English can express "
            "the distinction (elder/younger brother) but a default translation collapses to "
            "'brother' / 'sister'."
        ),
        near_rule=(
            "Preserve the elder/younger designation; rephrase with synonym or reordering."
        ),
        far_rule=(
            "Flip the relative-age modifier (bada ↔ chhota / badi ↔ chhoti) on the same kinship "
            "term. Otherwise minimal change."
        ),
        flattening_intent="partial",
        extra_constraints=(
            "Use only sibling/cousin terms with bada/chhota or badi/chhoti modifiers. "
            "The English translation should default to 'brother'/'sister'/'cousin' "
            "(no 'elder/younger' qualifier in _en fields), since the prediction is that "
            "MT typically discards this distinction."
        ),
        validator_check_question=(
            "Do anchor and near use the same elder/younger modifier on the same kinship term, "
            "and does far flip the modifier?"
        ),
    ),
    "verb_gender_agreement": CategorySpec(
        key="verb_gender_agreement",
        language="hindi",
        contrast_feature_description=(
            "In Hindi, the verb agrees in gender with the subject (or with the agent in ergative "
            "constructions). The contrast is encoded in the verb suffix (e.g., gayi (FEM) vs "
            "gaya (MASC)). English loses this distinction (both translate as 'went'). For first-"
            "person past intransitives, the subject 'main' is morphologically gender-neutral, so "
            "the only gender cue is the verb."
        ),
        near_rule=(
            "Same gender on the verb; rephrase with a synonym or reordering, keeping the verb "
            "morphology consistent."
        ),
        far_rule=(
            "Flip the verb gender (gaya ↔ gayi, pahuncha ↔ pahunchi, kiya ↔ ki) on otherwise "
            "minimal change."
        ),
        flattening_intent="strong",
        extra_constraints=(
            "Use first-person past intransitives where the subject is morphologically gender-"
            "neutral ('main') so the only gender cue is the verb ending. Avoid third-person "
            "subjects whose pronoun would carry the gender. Use the most common surface forms "
            "(gayi/gaya, pahunchi/pahuncha, kahi/kaha)."
        ),
        validator_check_question=(
            "Do anchor and near use the same verb gender, and does far flip the verb gender on "
            "an otherwise minimal sentence with a gender-neutral subject (main)?"
        ),
    ),
    "formality_tv": CategorySpec(
        key="formality_tv",
        language="hindi",
        contrast_feature_description=(
            "Hindi has a three-way T-V distinction: tu (intimate), tum (familiar), aap (formal). "
            "English collapses all to 'you'. The probe tests whether multilingual embeddings "
            "preserve the formality level."
        ),
        near_rule=(
            "Preserve the same formality level (same pronoun: tu/tum/aap), with consistent verb "
            "agreement and any associated honorifics. Rephrase via synonym or reordering."
        ),
        far_rule=(
            "Switch to a different formality level (e.g. tu → aap, or tum → aap). Update the "
            "verb agreement and pronoun consistently. Otherwise minimal change."
        ),
        flattening_intent="strong",
        extra_constraints=(
            "Verb agreement and pronoun choice must be internally consistent within each "
            "sentence (no mixing tu-pronoun with aap-verb-form). The English translation should "
            "lose the distinction (use 'you' uniformly)."
        ),
        validator_check_question=(
            "Do anchor and near use the same Hindi formality level (tu/tum/aap) consistently in "
            "both pronoun and verb agreement, and does far switch to a different formality level "
            "with consistent updates throughout the sentence?"
        ),
    ),
    "ser_vs_estar": CategorySpec(
        key="ser_vs_estar",
        language="spanish",
        contrast_feature_description=(
            "Spanish distinguishes ser (essential/inherent property) from estar (state/condition). "
            "English usually collapses both to 'is/was'. For most adjectives the distinction is "
            "subtle and partial-flattening (the English translation could choose to disambiguate "
            "via lexical choice but typically does not). For the polysemous adjectives "
            "{aburrido, listo, malo}, ser+adj and estar+adj have *different* English equivalents "
            "(boring/bored, smart/ready, bad/sick) — these are non-flattening and the English "
            "translation should render the distinction."
        ),
        near_rule=(
            "Preserve the same copula choice (ser or estar); rephrase via synonym or reordering "
            "of the predicate."
        ),
        far_rule=(
            "Flip the copula (ser ↔ estar) on otherwise minimal change. Subject and adjective "
            "stay the same."
        ),
        flattening_intent="mixed",
        extra_constraints=(
            "For typical adjectives (alto, joven, contento, cansado, abierto, etc.) the English "
            "translation should default to 'is/was' for both anchor and far (partial-flattening). "
            "For aburrido/listo/malo specifically, render the contrast in English "
            "(boring/bored, smart/ready, bad/sick) and tag the item flattening_intent='non_flattening'. "
            "Avoid dialectally marked items (e.g., 'Mi madre está joven' is borderline)."
        ),
        validator_check_question=(
            "Do anchor and near use the same copula (ser or estar) and does far swap to the "
            "other copula with otherwise minimal change to the same subject/adjective pair?"
        ),
    ),
    "formality": CategorySpec(
        key="formality",
        language="spanish",
        contrast_feature_description=(
            "Spanish T-V distinction: tú (informal singular) vs usted (formal singular); "
            "vosotros (informal plural, Peninsular) vs ustedes (formal plural Peninsular, "
            "general 2nd plural in Latin American). English collapses to 'you'. The probe tests "
            "whether the embedding distinguishes formality."
        ),
        near_rule=(
            "Preserve formality (same pronoun and verb agreement); rephrase via synonym or "
            "reordering."
        ),
        far_rule=(
            "Switch to the contrasting formality form (tú ↔ usted, vosotros ↔ ustedes). Update "
            "verb agreement consistently. Otherwise minimal change."
        ),
        flattening_intent="strong",
        extra_constraints=(
            "Tag dialect: 'peninsular' for vosotros/ustedes contrasts; 'general' for tú/usted. "
            "Verb agreement must match the pronoun (no tú-pronoun with usted-verb-form). "
            "The English translation should use 'you' uniformly."
        ),
        validator_check_question=(
            "Do anchor and near use the same Spanish formality form (tú/usted/vosotros/ustedes) "
            "consistently in pronoun and verb agreement, and does far switch to a contrasting "
            "form with consistent agreement updates?"
        ),
    ),
    "gender_agreement_adjectives": CategorySpec(
        key="gender_agreement_adjectives",
        language="spanish",
        contrast_feature_description=(
            "Spanish adjectives agree in gender with their referent noun (alto/alta, "
            "cansado/cansada). When the noun is morphologically marked in Spanish but "
            "translates to a gender-neutral English noun (amigo/amiga → 'friend'), the gender "
            "is encoded only in the adjective ending and is lost in the English translation."
        ),
        near_rule=(
            "Preserve the referent gender (same adjective ending); rephrase via synonym or "
            "reordering."
        ),
        far_rule=(
            "Flip the adjective gender (alto ↔ alta) on the same noun. Otherwise minimal change."
        ),
        flattening_intent="strong",
        extra_constraints=(
            f"Use ONLY nouns from the gender-neutral whitelist: {sorted(SPANISH_GENDER_NEUTRAL_NOUNS)}. "
            "The English translation must use a gender-neutral noun (no 'actor/actress' "
            "leakage). The referent gender must be encoded only via the adjective ending in Spanish "
            "and be lost in the English translation."
        ),
        validator_check_question=(
            "Do anchor and near use the same gender on the adjective (referring to the same "
            "person), and does far flip the adjective gender on otherwise minimal change, while "
            "the English translation uses a gender-neutral noun?"
        ),
    ),
}


CATEGORIES_TO_GENERATE = [
    # MultiBLiMP covers verb_gender_agreement; we don't generate more for that.
    "kinship_paternal_maternal",
    "kinship_relative_age",
    "formality_tv",
    "ser_vs_estar",
    "formality",
    "gender_agreement_adjectives",
]
