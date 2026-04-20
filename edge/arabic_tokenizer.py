"""Arabic CST tokenizer — canonical library module.

Exports the :class:`ArabicCSTTokenizer` and all data tables used to
turn an Arabic sentence into a sequence of CST tokens aligned with
``src/tokenizer/cst-spec.ts`` v1.0.

This module is the **source of truth** for Arabic tokenization in the
project. Experiment-specific scripts (e.g. ``edge/training/tokenize_1m.py``
for the 1M-sentence Wikipedia run) import from here; nothing in this
file is scale- or dataset-specific.

Public surface
--------------
- :class:`ArabicCSTTokenizer`            — the tokenizer
- :data:`ARABIC_ROOT_TO_FIELD`           — root → semantic field map
- :data:`ARABIC_REL_MAP`                 — function word → REL token
- :data:`ARABIC_LIT_WORDS`               — words that stay as LIT
- :data:`ARABIC_STR_TRIGGERS`            — sentence-level STR markers
- :data:`ARABIC_NUMERALS`                — numeric words → ROOT:size
- :data:`ARABIC_PATTERN_TO_ROLE`         — وزن → CMP role map
- :data:`PRC0_TOKENS` / :data:`PRC1_TOKENS` / :data:`PRC2_TOKENS` /
  :data:`PRC3_TOKENS` / :func:`enc0_feat` — CAMeL clitic mappings

Typical usage
-------------
    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.morphology.analyzer import Analyzer
    from arabic_tokenizer import ArabicCSTTokenizer

    analyzer = Analyzer(MorphologyDB.builtin_db())
    tok = ArabicCSTTokenizer(analyzer)
    out = tok.tokenize("وسيكتبُ الأطفالُ رسالةً للمعلمة")
    print(out["tokens"])
"""

import json
import os
import re
import time
from collections import Counter
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# Arabic root → semantic field mapping (from arabic_experiment_v2.py)
# ═══════════════════════════════════════════════════════════════

ARABIC_ROOT_TO_FIELD: dict[str, str] = {}

def _add(field: str, *roots: str):
    for r in roots:
        ARABIC_ROOT_TO_FIELD[r] = field

_add("write", "ك.ت.ب", "خ.ط.ط", "س.ج.ل", "د.و.ن", "ر.ق.م", "ن.س.خ", "ط.ب.ع", "ن.ش.ر", "ص.د.ر", "و.ث.ق", "ص.ح.ف", "د.#.ن")
_add("know", "ع.ل.م", "ع.ر.ف", "د.ر.س", "ف.ه.م", "ث.ق.ف", "خ.ب.ر", "ف.ق.ه", "ب.ح.ث", "ر.ش.د", "ل.ق.ن", "و.ع.ي", "ح.ف.ظ", "ع.ل.#", "ع.#.م")
_add("speak", "ق.و.ل", "ك.ل.م", "ح.د.ث", "ن.ط.ق", "خ.ط.ب", "ص.ر.خ", "ن.د.ي", "ل.غ.و", "ح.ك.ي", "ع.ل.ن", "ذ.ك.ر", "ر.و.ي", "س.أ.ل", "ج.و.ب", "ف.س.ر", "و.ص.ف", "ب.ي.ن", "ش.ر.ح", "ق.#.ل", "ح.#.ث", "ب.#.ن")
_add("think", "ف.ك.ر", "ع.ق.ل", "ر.أ.ي", "ظ.ن.ن", "ح.س.ب", "ن.ظ.ر", "خ.م.ن", "ق.ر.ر", "ز.ع.م", "ر.#.ي")
_add("see", "ب.ص.ر", "ش.ه.د", "ل.ح.ظ", "ل.م.ح", "ر.ق.ب", "ت.ب.ع", "ر.ص.د")
_add("feel", "ح.ب.ب", "ش.ع.ر", "ح.ز.ن", "ف.ر.ح", "خ.و.ف", "غ.ض.ب", "ق.ل.ق", "ر.ض.ي", "أ.م.ل", "ن.د.م", "أ.ل.م", "س.ع.د", "ح.ن.ن", "ع.ش.ق", "ك.ر.ه", "ح.ي.ر", "ذ.ع.ر", "ف.ز.ع")
_add("move", "م.ش.ي", "ذ.ه.ب", "ر.ج.ع", "س.ي.ر", "ق.د.م", "ر.ح.ل", "ه.ج.ر", "ج.ر.ي", "ط.ي.ر", "ع.ب.ر", "ه.ب.ط", "ص.ع.د", "د.خ.ل", "خ.ر.ج", "ف.ر.ر", "س.ب.ح", "ق.ف.ز", "ز.ح.ف", "ر.ك.ب", "ذ.#.ب", "ج.#.ز", "ر.#.ح")
_add("give", "ع.ط.ي", "و.ه.ب", "ت.ب.ر", "م.ن.ح", "ق.د.م")
_add("take", "أ.خ.ذ", "ق.ب.ل", "س.ر.ق", "ن.ه.ب", "خ.ط.ف", "س.ل.ب")
_add("make", "ص.ن.ع", "ب.ن.ي", "ع.م.ل", "ش.ي.د", "خ.ل.ق", "أ.ن.ش")
_add("destroy", "ه.د.م", "ك.س.ر", "ح.ط.م", "ق.ت.ل", "م.ح.ق", "ف.ن.ي", "ح.ر.ق", "غ.ر.ق", "خ.ر.ب", "ع.د.م")
_add("change", "ب.د.ل", "غ.ي.ر", "ح.و.ل", "ط.و.ر", "ن.م.و", "ز.ي.د")
_add("exist", "ك.و.ن", "و.ج.د", "ح.ي.و", "ب.ق.ي", "ع.ي.ش")
_add("time", "و.ق.ت", "ز.م.ن", "ت.ر.خ", "ب.د.ء", "ن.ه.ي", "خ.ت.م", "م.ه.ل")
_add("place", "م.ك.ن", "م.و.ض", "ب.ل.د", "م.د.ن", "ق.ر.ي", "م.ن.ط", "ح.د.د", "ق.ط.ر", "و.ل.ي")
_add("possess", "م.ل.ك", "ح.و.ز", "ك.س.ب", "ف.ق.د", "ح.ر.م")
_add("trade", "ب.ي.ع", "ش.ر.ي", "ت.ج.ر", "ر.ب.ح", "خ.س.ر", "س.و.ق", "ث.م.ن")
_add("fight", "ح.ر.ب", "ق.ت.ل", "ج.ه.د", "ن.ض.ل", "د.ف.ع", "ه.ج.م", "ق.و.م", "غ.ز.و", "ف.ت.ح")
_add("enable", "ع.و.ن", "ن.ص.ر", "س.ع.ف", "غ.و.ث", "أ.ن.ق")
_add("govern", "ح.ك.م", "س.ي.س", "م.ل.ك", "أ.م.ر", "ق.و.د", "ر.ئ.س")
_add("create", "خ.ل.ق", "ب.د.ع", "أ.ن.ش", "و.ل.د", "ف.ط.ر", "ح.د.ث", "ك.و.ن")
_add("force", "ق.و.ي", "ض.غ.ط", "ج.ب.ر", "ق.ه.ر", "أ.ر.غ", "ش.د.د")
_add("body", "ج.س.م", "ر.أ.س", "ي.د.ي", "ق.ل.ب", "ع.ي.ن", "س.م.ع", "د.م.م", "ع.ظ.م", "ل.ح.م", "ج.ل.د")
_add("food", "أ.ك.ل", "ش.ر.ب", "ط.ع.م", "ط.ب.خ", "ج.و.ع", "ع.ط.ش", "ذ.و.ق", "ه.ض.م", "غ.ذ.ي", "#.ك.ل")
_add("nature", "ط.ب.ع", "أ.ر.ض", "ب.ح.ر", "ن.ه.ر", "ج.ب.ل", "ب.ر.ر", "ص.ح.ر", "غ.ا.ب", "و.ا.د", "س.ه.ل", "#.ر.ض", "ر.#.ض")
_add("weather", "م.ط.ر", "ر.ي.ح", "ث.ل.ج", "ح.ر.ر", "ب.ر.د", "ش.م.س", "غ.ي.م", "ع.ص.ف", "ف.ي.ض", "ج.ف.ف")
_add("animal", "ح.ي.و", "ط.ي.ر", "س.م.ك", "ح.ش.ر", "ذ.ئ.ب", "أ.س.د", "ف.ر.س", "ب.ق.ر", "غ.ن.م", "ج.م.ل", "ك.ل.ب")
_add("plant", "ز.ر.ع", "ن.ب.ت", "ش.ج.ر", "ث.م.ر", "ز.ه.ر", "ح.ص.د", "غ.ر.س", "ر.و.ض")
_add("color", "ل.و.ن", "ب.ي.ض", "س.و.د", "ح.م.ر", "خ.ض.ر", "ز.ر.ق", "ص.ف.ر")
_add("size", "ك.ب.ر", "ص.غ.ر", "ط.و.ل", "ق.ص.ر", "ع.ر.ض", "و.س.ع", "ض.ي.ق", "ع.م.ق", "ك.ث.ر", "ق.ل.ل")
_add("measure", "ق.ي.س", "و.ز.ن", "ع.د.د", "ح.س.ب", "م.س.ح", "ب.ع.د", "ق.ر.ب", "ن.ص.ف", "ع.#.د")
_add("connect", "و.ص.ل", "ر.ب.ط", "ج.م.ع", "ض.م.م", "ل.ح.م", "ش.ب.ك", "ع.ل.ق", "ز.و.ج")
_add("contain", "ض.م.ن", "ح.و.ي", "ش.م.ل", "م.ل.أ", "ف.ر.غ", "ض.#.ف")
_add("open", "ف.ت.ح", "غ.ل.ق", "ب.و.ب", "ق.ف.ل", "ك.ش.ف", "س.ت.ر")
_add("hold", "م.س.ك", "ق.ب.ض", "ع.ل.ق", "ح.م.ل", "ر.ف.ع")
_add("hide", "خ.ف.ي", "س.ت.ر", "ك.ت.م", "غ.ي.ب", "ح.ج.ب", "خ.ب.أ", "ب.ط.ن")
_add("gather", "ج.م.ع", "ح.ش.د", "ض.م.م", "ل.م.م", "ج.ن.ي", "ح.ص.ل", "ح.#.ل")
_add("send", "ر.س.ل", "ب.ع.ث", "و.ج.ه", "ن.ق.ل", "ب.ث.ث")
_add("social", "ش.ر.ك", "ج.و.ر", "أ.ه.ل", "ق.و.م", "ش.ع.ب", "أ.م.م", "ق.ب.ل", "ح.ز.ب")
_add("dwell", "س.ك.ن", "ع.م.ر", "ب.ن.ي", "ن.ز.ل", "أ.ق.م")
_add("need", "ح.و.ج", "ل.ز.م", "ض.ر.ر", "و.ج.ب")
_add("want", "ط.ل.ب", "ر.غ.ب", "ت.م.ن", "ش.ه.و", "ب.غ.ي")
_add("enable", "م.ك.ن", "أ.ذ.ن", "س.م.ح", "ق.د.ر", "ي.س.ر")
_add("decide", "ق.ر.ر", "ح.ك.م", "ف.ص.ل", "ع.ز.م")
_add("fix", "ص.ل.ح", "ر.م.م", "ع.د.ل", "ض.ب.ط")
_add("rest", "ر.ا.ح", "ن.و.م", "ه.د.أ", "و.ق.ف", "ت.و.ق")
_add("person", "ب.ش.ر", "إ.ن.س", "ر.ج.ل", "م.ر.أ", "ط.ف.ل", "ش.ي.خ", "ش.ب.ب", "ن.س.ب")
_add("name", "س.م.ي", "ل.ق.ب", "ع.ن.و", "و.س.م")
_add("art", "ف.ن.ن", "ج.م.ل", "ز.خ.ر", "ن.ق.ش", "ر.س.م", "ل.ح.ن", "غ.ن.ي", "ع.ز.ف", "ر.ق.ص", "م.ث.ل", "ص.و.ر")
_add("science", "ب.ح.ث", "ن.ظ.ر", "ح.ل.ل", "ق.ي.س", "ك.ش.ف", "ف.ح.ص")
_add("tech", "ت.ق.ن", "ب.ر.م", "ش.ب.ك", "ه.ن.د")
_add("material", "م.ع.د", "ح.ج.ر", "ح.د.د", "ذ.ه.ب", "ف.ض.ض", "ن.ح.س", "خ.ش.ب", "ز.ج.ج", "ق.م.ش", "ن.س.ج", "ذ.ل.ل")
_add("structure", "ش.ك.ل", "ه.ي.ك", "ن.ظ.م", "ص.ف.ف", "ر.ت.ب", "ط.ب.ق")
_add("quality", "ص.ف.ي", "ج.و.د", "ح.س.ن", "س.و.أ", "ن.ظ.ف", "ق.ب.ح", "ج.د.د", "ق.د.م", "ص.ع.ب", "س.ه.ل", "ك.م.م")
_add("sport", "ل.ع.ب", "ر.ي.ض", "س.ب.ق", "ف.و.ز", "ه.ز.م")
_add("work", "ف.ع.ل", "ن.ف.ذ")

# Additional high-frequency roots
_add("exist", "ك.#.ن", "م.#.#", "#.ج.د", "ح.#.#", "#.ل.#", "#.#.ض", "#.ف.#", "ج.#.#", "#.#.ن", "ه.#.#", "ش.#.#", "#.ن.ن", "#.ن.#", "#.#.م")
_add("speak", "#.#.ل", "ح.#.ث", "ل.غ.#", "#.ل", "م.ن", "ب.م", "س.ب.ب", "ب.ل.غ", "د.ع.#", "ل.#", "س.#.ل")
_add("move", "د.#.ل", "خ.#.ر", "م.ر.ر", "ط.ر.ق", "س.ب.ل", "#.ف.ر", "س.ر.ع")
_add("time", "#.خ.ر", "ب.د.#", "م.د.د", "د.#.ر", "ع.ن.د", "#.ر.خ", "س.ن.#", "خ.ل.ف", "ص.ب.ح", "ش.ه.ر")
_add("change", "ح.#.ل", "غ.#.ر", "#.ث.ر", "م.#.ز", "خ.ف.ض")
_add("measure", "#.ح.د", "ع.#.د", "ث.ن.#", "ث.ل.ث", "د.ر.ج", "ر.ب.ع", "د.ق.ق")
_add("quality", "ت.م.م", "ك.م.م", "ف.ض.ل", "ك.م.ل")
_add("body", "ر.#.س")
_add("place", "#.ر.ض", "#.س.ط", "#.ق.ع", "ع.ر.ب", "ج.ز.ر", "ج.ز.#")
_add("nature", "ر.#.ض", "س.ط.ح", "ح.#.ط", "ع.ذ.ب", "ج.ر.د", "م.ل.ح", "ن.ج.م", "ق.م.ر")
_add("govern", "ق.#.د", "ن.#.خ", "ج.م.ه.ر")
_add("force", "ق.#.ي", "غ.ل.ب", "ج.ب.ر", "ر.غ.م", "س.ل.ب")
_add("contain", "ض.#.ف", "غ.ل.ف", "خ.ز.ن")
_add("connect", "#.ل", "ع.ق.د")
_add("person", "ن.س.ب", "ف.ر.د", "ن.ف.س")
_add("weather", "ب.خ.ر")
_add("know", "ح.ق.ق", "ث.ب.ت", "ب.د.ه", "ك.#.ف", "د.ل.ل", "ج.ر.ب")
_add("tech", "ه.ن.#")
_add("enable", "#.ف.ق", "ج.#.ز", "ك.ف.#", "ط.#.ق")
_add("see", "ش.#.ر", "ظ.ه.ر", "ب.ر.ز", "#.ز.ر", "#.ض.ح")
_add("want", "#.ر.د", "ق.ص.د")
_add("write", "ر.م.ز", "#.ت.ر")
_add("structure", "ق.ط.ع", "ق.س.م", "ص.ن.ف", "ن.ق.ط", "ف.ر.ع", "ع.ن.ص.ر")
_add("trade", "#.ق.د", "س.ه.م")
_add("fight", "ج.#.ش", "ع.س.ك.ر")
_add("size", "#.ل.ف", "ك.ث.ف", "ح.ج.م", "ك.ل.ل")
_add("social", "ه.م.م", "س.ل.م", "ح.ل.ف", "ع.ض.#")
_add("make", "ج.#.ل", "ج.ع.ل", "ص.#.غ")
_add("destroy", "ع.د.م")
_add("think", "ف.ل.س.ف")
_add("enable", "د.ع.م")
_add("create", "ن.#.س")
_add("give", "#.ز.ع")
_add("contain", "ن.#.ع")
_add("name", "ر.#.م")
_add("science", "ر.ك.ز", "م.ر.س")
_add("place", "ج.غ.ر.ف")
_add("decide", "ع.م.د")
_add("work", "ف.ع.ل", "ن.ف.ذ")
_add("exist", "م.#", "ب", "ل", "ه", "ف.#")

# ═══════════════════════════════════════════════════════════════
# Arabic function words → CST tokens (aligned with cst-spec.ts v1.0)
#
# Old: FUNC:PREP, FUNC:CONJ, etc.
# New: REL:<relation>, STR:<marker>, LIT:<surface>, ROOT:size
# ═══════════════════════════════════════════════════════════════

# Word → REL token (prepositions, conjunctions, quantifiers, etc.)
ARABIC_REL_MAP = {
    # Prepositions → REL:<specific>
    "في": "REL:in", "من": "REL:from", "إلى": "REL:to", "على": "REL:on",
    "عن": "REL:about", "مع": "REL:with", "بين": "REL:between", "حول": "REL:around",
    "خلال": "REL:through", "منذ": "REL:since", "حتى": "REL:until", "نحو": "REL:to",
    "لدى": "REL:at", "عند": "REL:at", "فوق": "REL:above", "تحت": "REL:under",
    "أمام": "REL:infront", "خلف": "REL:behind", "بعد": "REL:after", "قبل": "REL:before",
    "دون": "REL:without", "ضد": "REL:against", "عبر": "REL:across", "ضمن": "REL:within",
    "لأجل": "REL:for", "بما": "REL:with",
    # Conjunctions → REL:<specific>
    "و": "REL:and", "أو": "REL:or", "ثم": "REL:then", "لكن": "REL:but",
    "بل": "REL:instead", "أم": "REL:or", "إذ": "REL:as",
    "كي": "REL:for", "حيث": "REL:where", "لأن": "REL:causes",
    "بينما": "REL:contrast", "كما": "REL:like", "مثل": "REL:like",
    "حين": "REL:when", "عندما": "REL:when", "لما": "REL:when",
    # إنّ وأخواتها (sisters of إنّ) → mapped to specific relations
    "لكنّ": "REL:but", "لكنه": "REL:but", "لكنها": "REL:but",
    "كأن": "REL:like", "كأنّ": "REL:like", "كأنه": "REL:like", "كأنها": "REL:like",
    "لعلّ": "REL:maybe", "لعله": "REL:maybe", "لعلها": "REL:maybe",
    # أدوات الاستثناء (exception) → REL:except
    "إلا": "REL:except", "سوى": "REL:except", "عدا": "REL:except", "خلا": "REL:except",
    # Demonstratives/Relatives → REL:<referential>
    "هذا": "REL:this", "هذه": "REL:this", "ذلك": "REL:those", "تلك": "REL:those",
    "الذي": "REL:which", "التي": "REL:which", "الذين": "REL:who", "اللذين": "REL:who",
    "اللاتي": "REL:who", "ما": "REL:what", "هؤلاء": "REL:these",
    # Determiners/Quantifiers → REL:<quantifier>
    "كل": "REL:all", "بعض": "REL:some", "أي": "REL:any", "غير": "REL:unlike",
    "كلا": "REL:both", "جميع": "REL:all", "سائر": "REL:all", "معظم": "REL:most",
    "أغلب": "REL:most", "عدة": "REL:several", "كثير": "REL:many", "قليل": "REL:few",
    "أكثر": "REL:more", "أحد": "REL:some", "أقل": "REL:less",
    # Restriction → REL:only
    "إنما": "REL:only", "إنّما": "REL:only",
    # Adverbs → REL:<specific>
    "أيضا": "REL:also", "أيضاً": "REL:also", "جدا": "REL:emphasis", "جداً": "REL:emphasis",
    "فقط": "REL:only", "تقريبا": "REL:almost", "تقريباً": "REL:almost",
    "حاليا": "REL:now", "حالياً": "REL:now",
}

# Words that emit as LIT:<word> (personal pronouns, auxiliaries, particles)
ARABIC_LIT_WORDS = {
    # Personal pronouns → LIT (like English I/he/she)
    "هو", "هي", "هم", "هن", "أنا", "نحن", "أنت", "أنتم",
    "أنتِ", "أنتن", "أنتنّ", "هما",
    # Possessive/reflexive
    "نفس", "ذات",
    # Auxiliaries (كان وأخواتها) → LIT
    "كان", "يكون", "أصبح", "ظل", "بات", "صار", "ليس",
    # Subordinating particles → LIT
    "إن", "أن", "أنّ", "لعل",
    # Vocative → LIT (يا has low semantic content)
    "يا",
}

# Words that trigger STR markers (sentence-level, detected separately)
#
# NOTE: Grammatically-critical particles are kept DISTINCT so the model can
# learn Arabic syntax (jussive vs. subjunctive mood, likely vs. counterfactual
# conditionals) and produce the correct surface form at generation time.
ARABIC_STR_TRIGGERS = {
    # Negation → split by grammatical scope (each governs a different mood/case)
    "لا": "STR:neg:general",   # general negation (+ indicative / imperative)
    "لم": "STR:neg:past",      # past negation (+ jussive)
    "لن": "STR:neg:future",    # future negation (+ subjunctive)
    "ليس": "STR:neg:nominal",  # nominal negation (+ accusative predicate)
    # Conditional → split by semantics (likely vs. hypothetical vs. counterfactual)
    "إذا": "STR:cond:likely",  # likely / realistic condition
    "لو": "STR:cond:hypo",     # hypothetical / unreal condition
    "لولا": "STR:cond:counter", # counterfactual (if-not-for)
    # Future → STR:future
    "سوف": "STR:future",
    # Question → STR:question
    "هل": "STR:question",
    # Emphasis/past → STR:emphasis / STR:past
    "قد": "STR:past", "لقد": "STR:emphasis",
    # إنّ as emphasis (when standalone, not لكنّ/كأنّ which are in REL)
    "إنّ": "STR:emphasis",
}

# Numerals → ROOT:size
ARABIC_NUMERALS = {
    "واحد", "اثنان", "ثلاثة", "أربعة", "خمسة", "ستة", "سبعة", "ثمانية",
    "تسعة", "عشرة", "عشر", "مئة", "مائة", "ألف", "مليون", "ثلث",
}

PROCLITICS = ["وال", "وب", "ول", "وك", "فال", "فب", "فل", "ال", "لل", "بال", "كال"]
# NOTE: `PROCLITICS` is no longer used for naive prefix stripping — CAMeL
# Tools decomposes clitics natively (prc0/prc1/prc2/prc3/enc0). Kept here
# only as documentation of the common fused-prefix shapes in MSA.


# ═══════════════════════════════════════════════════════════════
# Arabic pattern (وزن) → CMP role mapping
# The core of the Arabic algebra: root × pattern = concept
# Patterns normalized: vowel diacritics stripped, shadda (ّ) preserved
# ═══════════════════════════════════════════════════════════════

def _strip_vowels(text):
    """Strip vowel diacritics but keep shadda (ّ \u0651)."""
    if not text: return ""
    return re.sub(r'[\u064B-\u0650\u0652\u0670]', '', text)

ARABIC_PATTERN_TO_ROLE = {
    # Active participle (فَاعِل) → agent (the doer)
    "فاعل": "agent", "فاعلة": "agent", "فاعلون": "agent",
    "فاعلات": "agent", "فاعلين": "agent", "فواعل": "agent",
    # Passive participle (مَفْعُول) → patient (the receiver)
    "مفعول": "patient", "مفعولة": "patient",
    # Place noun (مَفْعَلَة / مَفْعَل)
    "مفعلة": "place", "مفاعل": "place",
    # Instrument (مِفْعَال / مِفْعَل)
    "مفعال": "place",
    # Verbal nouns → instance (the thing) / state (the act)
    "فعال": "instance",       # كِتَاب (book)
    "فعول": "instance",       # دُخُول (entry)
    "فعل": "instance",        # عِلْم (knowledge)
    "فعالة": "state",         # كِتَابَة (writing)
    "فعولة": "state",         # عُبُودَة
    "تفعيل": "instance",      # Form II VN: تعليم (teaching)
    "تفعلة": "instance",      # Form II VN variant
    "انفعال": "instance",     # Form VII VN
    "افتعال": "instance",     # Form VIII VN
    "استفعال": "instance",    # Form X VN
    # Mutual action (Form VI)
    "تفاعل": "mutual",        # تَعَاوُن (cooperation)
    # Process (Form III verbal noun)
    "مفاعلة": "process",      # مُكَاتَبَة (correspondence)
    # Intensifier (فَعَّال — has shadda, distinct from فَعَال)
    "فعّال": "intensifier", "فعّالة": "intensifier",
    # Form II active participle (مُفَعِّل → causer)
    "مفعّل": "causer", "مفعّلة": "causer",
    # Form X active participle (مُسْتَفْعِل → seeker)
    "مستفعل": "seeker", "مستفعلة": "seeker",
    # Quality / adjective patterns
    "فعيل": "quality", "فعيلة": "quality",
    "فعلان": "quality",
    "فعلى": "quality",        # feminine elative
}

# POS-based fallback (when pattern doesn't match or is absent)
POS_TO_ROLE = {
    "adj": "quality",
    "adj_comp": "quality",
    "adj_num": "quality",
}

# POS values that indicate named entities → emit LIT:<surface>
NER_POS = frozenset({"noun_prop"})


# ═══════════════════════════════════════════════════════════════
# CAMeL Tools clitic feature → CST token mapping
#
# Arabic orthography fuses conjunctions, prepositions, articles, and
# pronominal suffixes into one graphical word (e.g. وبكتابه = "and with
# his book"). CAMeL Tools exposes these as separate morphological
# features (prc0..prc3, enc0). We decompose each into its own CST token
# so the model sees the real grammatical structure instead of a single
# opaque LIT:وبكتابه.
# ═══════════════════════════════════════════════════════════════

# prc0 — article / negation proclitic
PRC0_TOKENS = {
    "Al_det":      ["FEAT:def"],
    "AlmA_neg":    ["FEAT:def", "STR:neg:general"],
    "mA_neg":      ["STR:neg:general"],
    "ma_neg":      ["STR:neg:general"],
    "lA_neg":      ["STR:neg:general"],
}

# prc1 — preposition / future / emphasis proclitic
PRC1_TOKENS = {
    "bi_prep":      ["REL:with"],
    "bi_part":      ["REL:with"],
    "bi_prog":      [],                 # progressive particle → silent
    "li_prep":      ["REL:for"],
    "li_jus":       ["REL:for"],
    "li_sub":       ["REL:for"],
    "libi_prep":    ["REL:for", "REL:with"],
    "la_prep":      ["REL:for"],
    "la_emph":      ["STR:emphasis"],
    "ka_prep":      ["REL:like"],
    "sa_fut":       ["STR:future"],
    "Ha_fut":       ["STR:future"],
    "laHa_emphfut": ["STR:emphasis", "STR:future"],
    "laHa_rcfut":   ["STR:future"],
    "min_prep":     ["REL:from"],
    "fiy_prep":     ["REL:in"],
    "Ea_prep":      ["REL:about"],
    "EalaY_prep":   ["REL:on"],
    "ta_prep":      ["REL:by"],         # oath particle (تالله)
    "wa_prep":      ["REL:with"],       # rare: oath/accompaniment wa
    "hA_dem":       ["REL:this"],
    # Vocative / interrogative variants → silent (low semantic content)
    "yA_voc":       [],
    "wA_voc":       [],
    "<i$_interrog": ["STR:question"],
}

# prc2 — conjunction proclitic
PRC2_TOKENS = {
    "wa_conj": ["REL:and"],
    "wa_part": ["REL:and"],
    "wa_sub":  ["REL:and"],
    "fa_conj": ["REL:then"],
    "fa_conn": ["REL:then"],
    "fa_rc":   ["REL:then"],
    "fa_sub":  ["REL:then"],
}

# prc3 — question proclitic
PRC3_TOKENS = {
    ">a_ques": ["STR:question"],
}

# enc0 — pronominal enclitic (3ms_poss, 1s_dobj, …) → FEAT:pron:<tag>
# The _poss/_dobj/_pron suffix is dropped because the model learns
# syntactic role from position; what matters is person-gender-number.
def enc0_feat(val):
    if not val or val in ("0", "na"):
        return None
    tag = val.split("_")[0]
    # sanity check: tags are 1s/1p/2d/2p/2ms/2fs/2mp/2fp/3d/3p/3ms/3fs/3mp/3fp
    if tag in _ENC_TAGS:
        return f"FEAT:pron:{tag}"
    return None

_ENC_TAGS = {
    "1s", "1p",
    "2d", "2p", "2ms", "2fs", "2mp", "2fp",
    "3d", "3p", "3ms", "3fs", "3mp", "3fp",
}

# Person-gender-number tags we emit as a single bundled FEAT token.
# Used for verb conjugation (who did the action) and non-default noun
# inflection (feminine / plural / dual).
_PGN_TAGS = _ENC_TAGS  # same shape; re-used for verb/content FEATs


def _collect_prefix_tokens(a):
    """Return the ordered list of CST tokens produced by the clitics of
    a CAMeL analysis, read outer→inner: prc2 (conj) → prc1 (prep/fut) →
    prc0 (article/neg) → prc3 (question)."""
    out = []
    for feat_name, table in (("prc2", PRC2_TOKENS),
                             ("prc1", PRC1_TOKENS),
                             ("prc0", PRC0_TOKENS),
                             ("prc3", PRC3_TOKENS)):
        v = a.get(feat_name)
        if v and v != "0" and v != "na" and v in table:
            out.extend(table[v])
    return out


def _pgn_tag(a):
    """Build a compact person-gender-number tag from an analysis.
    Returns one of _PGN_TAGS or None if all are default/unknown."""
    per = a.get("per", "na")
    gen = a.get("gen", "na")
    num = a.get("num", "na")
    # Normalize
    if num == "u": num = "s"
    if gen == "u": gen = "m"
    if per == "na" and gen == "na" and num == "na":
        return None
    # Build tag. Drop person for 3rd (most common) only when asked.
    if per in ("1",) and num in ("s", "p"):
        return f"{per}{num}"           # 1s, 1p (no gender distinction)
    if per in ("2", "3") and num == "d":
        return f"{per}d"               # 2d, 3d (no gender distinction)
    if per in ("2", "3") and num == "p" and gen == "na":
        return f"{per}p"               # 3p (ungendered plural, e.g. rational pl)
    if per in ("2", "3") and num in ("s", "p") and gen in ("m", "f"):
        return f"{per}{gen}{num}"      # 3ms, 3fs, 3mp, 3fp, 2ms, 2fs, 2mp, 2fp
    return None


# ═══════════════════════════════════════════════════════════════
# Tokenizer
# ═══════════════════════════════════════════════════════════════

def _build_wildcard_index():
    index = dict(ARABIC_ROOT_TO_FIELD)
    weak_letters = set("وياأإآءئؤ")
    for root, field in list(ARABIC_ROOT_TO_FIELD.items()):
        parts = root.split(".")
        if len(parts) != 3:
            continue
        for i in range(3):
            if parts[i] in weak_letters:
                v = list(parts); v[i] = "#"
                k = ".".join(v)
                if k not in index: index[k] = field
        for i in range(3):
            for j in range(i+1, 3):
                if parts[i] in weak_letters and parts[j] in weak_letters:
                    v = list(parts); v[i] = "#"; v[j] = "#"
                    k = ".".join(v)
                    if k not in index: index[k] = field
    return index


class ArabicCSTTokenizer:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.vocab: dict[str, int] = {}
        self.next_id = 0
        self.root_index = _build_wildcard_index()
        self.stats = Counter()

        # Special tokens (aligned with cst-spec.ts v1.0)
        for tok in ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]"]:
            self._get_id(tok)

        # Pre-register ROOT tokens (one per semantic field)
        for f in sorted(set(self.root_index.values())):
            self._get_id(f"ROOT:{f}")

        # Pre-register REL tokens
        for rel in sorted(set(ARABIC_REL_MAP.values())):
            self._get_id(rel)

        # Pre-register REL tokens that may come from clitic expansion
        for table in (PRC1_TOKENS, PRC2_TOKENS):
            for toks in table.values():
                for t in toks:
                    self._get_id(t)

        # Pre-register STR tokens
        for stk in sorted(set(ARABIC_STR_TRIGGERS.values())):
            self._get_id(stk)
        for t in ("STR:question", "STR:emphasis", "STR:future", "STR:past"):
            self._get_id(t)

        # Pre-register FEAT tokens (definiteness, inflection, enclitics, aspect)
        self._get_id("FEAT:def")
        for tag in sorted(_PGN_TAGS):
            self._get_id(f"FEAT:{tag}")              # bundled person-gen-num
            self._get_id(f"FEAT:pron:{tag}")         # enclitic pronoun
        for asp in ("p", "i", "c"):
            self._get_id(f"FEAT:asp:{asp}")
        # Non-default inflection markers for nouns / adjectives (ms is default).
        for t in ("FEAT:f", "FEAT:p", "FEAT:d"):
            self._get_id(t)

        # ROOT:size for numerals
        self._get_id("ROOT:size")

    def _get_id(self, token):
        if token in self.vocab: return self.vocab[token]
        tid = self.next_id; self.vocab[token] = tid; self.next_id += 1
        return tid

    def _strip(self, word):
        word = re.sub(r'[\u064B-\u065F\u0670]', '', word)
        return word.replace('\u0640', '')

    def _find_field(self, roots):
        for r in roots:
            if r in self.root_index: return self.root_index[r]
        return None

    # ── Core per-word analysis ─────────────────────────────────
    def _best_analysis(self, clean):
        """Return the first analysis with a usable root, or None.

        CAMeL analyzer returns candidates sorted by likelihood. We scan
        for the first one whose root is a real triliteral (skipping
        placeholders like NTWS / PUNC / DIGIT / FOREIGN) so downstream
        feature extraction sees real morphology.
        """
        analyses = self.analyzer.analyze(clean)
        for a in analyses:
            r = a.get("root", "")
            if r and r not in ("NTWS", "PUNC", "DIGIT", "FOREIGN"):
                return a
        return None

    def _extract_role(self, a):
        """Extract CMP role from camel-tools pattern or POS."""
        pattern = a.get("pattern") or ""
        norm = _strip_vowels(pattern)
        if norm and norm in ARABIC_PATTERN_TO_ROLE:
            return ARABIC_PATTERN_TO_ROLE[norm]
        pos = a.get("pos", "")
        if pos in POS_TO_ROLE:
            return POS_TO_ROLE[pos]
        return None

    def _word_tokens(self, clean):
        """Tokenize a single orthographic word into a list of CST tokens.

        Emission order for a fully-analyzed content word:
            [prc2 tokens]        conjunction  (و / ف …)
            [prc1 tokens]        preposition / future / emphasis
            [prc0 tokens]        article (ال) / attached negation
            [prc3 tokens]        question أ
            core token           CMP: / ROOT: / LIT:
            [gender/number FEAT] non-default inflection (f / p / d)
            [aspect FEAT]        verb only (asp:p / asp:i / asp:c)
            [pgn FEAT]           verb person-gender-number (e.g. 3mp)
            [enclitic pronoun]   FEAT:pron:<tag>
        """
        out = []

        # 1. Fast paths — function / LIT / numeral / STR trigger words bypass
        #    full morphological decomposition; they carry fixed semantics.
        if clean in ARABIC_STR_TRIGGERS:
            # Handled at sentence level — emit nothing at word position.
            return out
        if clean in ARABIC_REL_MAP:
            out.append(ARABIC_REL_MAP[clean]); self.stats["rel"] += 1
            return out
        if clean in ARABIC_LIT_WORDS:
            out.append(f"LIT:{clean}"); self.stats["lit"] += 1
            return out
        if clean in ARABIC_NUMERALS:
            out.append("ROOT:size"); self.stats["root"] += 1
            return out

        # 2. Morphological analysis
        a = self._best_analysis(clean)
        if a is None:
            out.append(f"LIT:{clean}"); self.stats["lit"] += 1
            return out

        pos = a.get("pos", "")

        # 3. Disambiguate ما using the analyzer POS
        if clean == "ما":
            if pos == "part_neg":
                out.append("STR:neg:general"); self.stats["str"] += 1
            elif pos in ("pron_interrog", "adv_interrog"):
                out.append("REL:what"); self.stats["rel"] += 1
            elif pos in ("pron_rel", "adv_rel"):
                out.append("REL:which"); self.stats["rel"] += 1
            else:
                out.append("REL:what"); self.stats["rel"] += 1
            return out

        # 4. Emit proclitic tokens (conjunction → prep → article → ques)
        for t in _collect_prefix_tokens(a):
            out.append(t)
            if t.startswith("REL:"): self.stats["rel"] += 1
            elif t.startswith("STR:"): self.stats["str"] += 1
            elif t.startswith("FEAT:"): self.stats["feat"] += 1

        # 5. Named entity (noun_prop) → LIT:<surface>
        if pos in NER_POS:
            out.append(f"LIT:{clean}"); self.stats["ner"] += 1
            # No further FEATs for proper nouns (they are opaque tokens).
            return out

        # 6. Core content token
        roots = [a.get("root", "")]
        field = self._find_field(roots)
        if not field:
            out.append(f"LIT:{clean}"); self.stats["lit"] += 1
            return out

        role = self._extract_role(a)
        if role:
            out.append(f"CMP:{field}:{role}"); self.stats["cmp"] += 1
        else:
            out.append(f"ROOT:{field}"); self.stats["root"] += 1

        # 7. Per-word feature tokens
        if pos == "verb":
            asp = a.get("asp", "na")
            if asp in ("p", "i", "c"):
                out.append(f"FEAT:asp:{asp}"); self.stats["feat"] += 1
            pgn = _pgn_tag(a)
            if pgn:
                out.append(f"FEAT:{pgn}"); self.stats["feat"] += 1
        else:
            # Nouns / adjectives / participles — emit only non-default
            # inflection to keep sequence length low.
            gen = a.get("gen", "na")
            num = a.get("num", "na")
            if gen == "f":
                out.append("FEAT:f"); self.stats["feat"] += 1
            if num == "p":
                out.append("FEAT:p"); self.stats["feat"] += 1
            elif num == "d":
                out.append("FEAT:d"); self.stats["feat"] += 1

        # 8. Enclitic pronoun (if any)
        enc = enc0_feat(a.get("enc0"))
        if enc:
            out.append(enc); self.stats["feat"] += 1

        return out

    # ── Sentence-level tokenization ────────────────────────────
    def tokenize(self, sentence):
        tokens = ["[BOS]"]
        ids = [self.vocab["[BOS]"]]

        words = re.findall(r'[\u0600-\u06FF\u0750-\u077F]+', sentence)

        # Sentence-level STR markers: standalone particles and punctuation.
        str_emitted = set()
        def _emit_str(marker):
            if marker in str_emitted: return
            tokens.append(marker); ids.append(self._get_id(marker))
            str_emitted.add(marker)
            self.stats["str"] += 1

        for w in words:
            c = self._strip(w)
            if c in ARABIC_STR_TRIGGERS:
                _emit_str(ARABIC_STR_TRIGGERS[c])

        tail = sentence.rstrip()
        if tail.endswith("؟") or tail.endswith("?"):
            _emit_str("STR:question")
        if tail.endswith("!"):
            _emit_str("STR:emphasis")

        prefix_count = len(tokens) - 1   # excl. [BOS]

        # Word-by-word tokenization using the per-word pipeline.
        word_forms = []                  # parallel to word_token_counts
        word_token_counts = []
        for word in words:
            clean = self._strip(word)
            w_tokens = self._word_tokens(clean)
            word_forms.append(clean)
            word_token_counts.append(len(w_tokens))
            for tok in w_tokens:
                tokens.append(tok); ids.append(self._get_id(tok))

        tokens.append("[EOS]"); ids.append(self.vocab["[EOS]"])
        return {
            "ids": ids,
            "tokens": tokens,
            "text": sentence,
            # Alignment metadata — used offline by build_lookups.py to map
            # Arabic surface words ↔ CST token sequences. Not fed to the
            # model.
            "words": word_forms,
            "word_token_counts": word_token_counts,
            "prefix_count": prefix_count,
        }

    def save_vocab(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

