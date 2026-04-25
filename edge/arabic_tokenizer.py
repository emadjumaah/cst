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
# Phase 1 — Merge in labeled roots imported from the Arabic Algebra
# Engine workspace via ``edge/import_aae_roots.py``. The JSON file is
# optional; if absent the tokenizer still runs with the core ~543 roots
# defined above.
# ═══════════════════════════════════════════════════════════════

def _load_aae_roots() -> int:
    """Merge ``edge/artifacts/aae_roots.json`` into ``ARABIC_ROOT_TO_FIELD``.

    Returns the number of new (not-previously-present) roots added. Existing
    entries are **not** overwritten — the hand-curated mappings at the top of
    this module always win on collision.
    """
    path = Path(__file__).resolve().parent / "artifacts" / "aae_roots.json"
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    added = 0
    for root, field in data.items():
        if root not in ARABIC_ROOT_TO_FIELD:
            ARABIC_ROOT_TO_FIELD[root] = field
            added += 1
    return added


_AAE_ROOTS_ADDED = _load_aae_roots()


# ═══════════════════════════════════════════════════════════════
# Phase 2 — Full Arabic root inventory (Zerrouki).
#
# ``edge/artifacts/arabic_roots_zerrouki.txt`` is Taha Zerrouki's
# arabic-roots corpus (GPL, ~9.5K triliteral + quadriliteral roots,
# Hamzas normalized to ء). Each root is emitted in dotted canonical
# form ``ك.ت.ب`` and pre-registered during tokenizer construction so
# every real Arabic root has a deterministic, always-present vocab
# slot — even ones that never appear in the training corpus.
#
# This is the v4 bulk root inventory: it doesn't add field labels
# (only 1,757 of the 9.5K are semantically mapped), but it guarantees
# that no legitimate Arabic root will ever be ``[UNK]`` downstream
# and that the transformer's embedding table reserves a contiguous
# region for Arabic-root tokens.
# ═══════════════════════════════════════════════════════════════

ARABIC_ROOTS_ZERROUKI: list[str] = []


def _load_zerrouki_roots() -> int:
    """Return dotted-canonical form of every root in the Zerrouki list.

    Populates module-level ``ARABIC_ROOTS_ZERROUKI`` with entries like
    ``ك.ت.ب`` (triliteral) or ``س.ر.م.د`` (quadriliteral). Skips lines
    that are blank, comments, or shorter than 3 characters.
    """
    path = Path(__file__).resolve().parent / "artifacts" / "arabic_roots_zerrouki.txt"
    if not path.exists():
        return 0
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return 0
    seen: set[str] = set()
    for line in text.splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        if len(raw) < 3 or len(raw) > 5:
            # Zerrouki list is triliteral+quadriliteral; skip anything else.
            continue
        dotted = ".".join(list(raw))
        if dotted not in seen:
            seen.add(dotted)
            ARABIC_ROOTS_ZERROUKI.append(dotted)
    return len(ARABIC_ROOTS_ZERROUKI)


_ZERROUKI_ROOTS_COUNT = _load_zerrouki_roots()


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

# Words that emit as LIT:<word> (personal pronouns, particles)
#
# NOTE: v3 removed كان وأخواتها (auxiliary verbs) from this set. They used
# to short-circuit to LIT, contributing ~80K LIT occurrences for a handful
# of forms that CAMeL can analyze correctly as verbs of the root ك.و.ن etc.
# Letting them flow through morphology yields proper CMP:exist:* tokens.
ARABIC_LIT_WORDS = {
    # Personal pronouns → LIT (like English I/he/she)
    "هو", "هي", "هم", "هن", "أنا", "نحن", "أنت", "أنتم",
    "أنتِ", "أنتن", "أنتنّ", "هما",
    # Possessive/reflexive
    "نفس", "ذات",
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

# ═══════════════════════════════════════════════════════════════
# v3 additions — calendar months, punctuation, digit normalization
# ═══════════════════════════════════════════════════════════════

# Gregorian month names (English + Levantine) → TIME:month:<en>. These
# were appearing ~5,000 times each as LIT tokens in v2 because CAMeL
# tags them as NTWS (non-triliteral stem / foreign).
ARABIC_MONTHS = {
    "يناير": "TIME:month:jan", "كانون الثاني": "TIME:month:jan",
    "فبراير": "TIME:month:feb", "شباط": "TIME:month:feb",
    "مارس": "TIME:month:mar", "آذار": "TIME:month:mar",
    "أبريل": "TIME:month:apr", "نيسان": "TIME:month:apr",
    "مايو": "TIME:month:may", "أيار": "TIME:month:may",
    "يونيو": "TIME:month:jun", "حزيران": "TIME:month:jun",
    "يوليو": "TIME:month:jul", "تموز": "TIME:month:jul",
    "أغسطس": "TIME:month:aug", "آب": "TIME:month:aug",
    "سبتمبر": "TIME:month:sep", "أيلول": "TIME:month:sep",
    "أكتوبر": "TIME:month:oct", "تشرين الأول": "TIME:month:oct",
    "نوفمبر": "TIME:month:nov", "تشرين الثاني": "TIME:month:nov",
    "ديسمبر": "TIME:month:dec", "كانون الأول": "TIME:month:dec",
}

# Arabic punctuation code points to strip before morphological analysis.
# They live inside the Arabic Unicode block so the simple letter regex
# pulls them in with words like ذلك،. Removing them prevents 100K+
# spurious LIT:، occurrences and fusion noise on word boundaries.
_ARABIC_PUNCT_RE = re.compile(
    r"[\u060C\u060D\u061B\u061E\u061F\u066A-\u066D\u06D4\u06DD-\u06DE]"
)

# Latin + Arabic-Indic digit sets (for NUM classification).
_DIGIT_CLASS = "0-9\u0660-\u0669\u06F0-\u06F9"
_DIGIT_RE = re.compile(f"^[{_DIGIT_CLASS}]+$")
_DECIMAL_RE = re.compile(f"^[{_DIGIT_CLASS}]+[.,\u066B][{_DIGIT_CLASS}]+$")
_PERCENT_RE = re.compile(f"^[{_DIGIT_CLASS}]+[.,\u066B]?[{_DIGIT_CLASS}]*[%\u066A]$")

# Translation table to fold Arabic-Indic digits to ASCII for magnitude checks.
_DIGIT_FOLD = str.maketrans(
    "\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669"
    "\u06F0\u06F1\u06F2\u06F3\u06F4\u06F5\u06F6\u06F7\u06F8\u06F9",
    "01234567890123456789",
)


def _num_token(raw: str) -> str | None:
    """Classify a numeric surface form into a typed NUM token.

    Returns one of ``NUM:year`` / ``NUM:percent`` / ``NUM:decimal`` /
    ``NUM:small`` / ``NUM:measure`` or ``None`` if ``raw`` is not numeric.
    """
    if not raw:
        return None
    if _PERCENT_RE.match(raw):
        return "NUM:percent"
    if _DECIMAL_RE.match(raw):
        return "NUM:decimal"
    if _DIGIT_RE.match(raw):
        folded = raw.translate(_DIGIT_FOLD)
        try:
            n = int(folded)
        except ValueError:
            return "NUM:small"
        if 1800 <= n <= 2100:
            return "NUM:year"
        if n < 1000:
            return "NUM:small"
        return "NUM:measure"
    return None


# Letters we will try to substitute for the CAMeL weak-root placeholder `#`.
# Order matters: و is the most common weak root letter in classical
# reconstructions, so we try it first for a deterministic canonical form.
_WEAK_SUBSTITUTES = ("و", "ي", "ا", "ء", "ى")

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
    # CAMeL Tools encodes patterns with ASCII digits ``1/2/3`` for the
    # root consonant slots and Arabic letters for affixes; diacritics
    # (fatha/damma/kasra/sukun) carry between slots. We strip the short
    # vowels (keeping shadda ``ّ``) and normalize hamzat-wasl ``ٱ`` → ``ا``
    # before lookup. See ``_extract_role``.
    #
    # Keys below are therefore CAMeL-shape patterns, *not* the Arabic
    # wazn names (فاعل / مفعول / ...). Comments give the wazn for clarity.

    # فاعل — active participle → agent (the doer)
    "1ا23":    "agent",   # كاتب / عالم / قاتل
    "1ا23ة":   "agent",   # كاتبة
    "1ا23ه":   "agent",   # كاتبة (alt tāʾ-hāʾ spelling)
    "1ا23ات":  "agent",   # كاتبات
    "1ا23ون":  "agent",   # كاتبون
    "1ا23ين":  "agent",   # كاتبين

    # مفعول — passive participle → patient (the receiver)
    "م12و3":   "patient", # مكتوب / معلوم
    "م12و3ة":  "patient", # مكتوبة
    "م12و3ه":  "patient",

    # مَفْعَلَة / مَفْعِلَة — place (feminine forms disambiguate from VN)
    "م123ة":   "place",   # مكتبة / مدرسة
    "م123ه":   "place",
    "م12ّ3ة":  "place",   # مدرسة (with shadda on R2)
    "م12ّ3ه":  "place",

    # مِفْعَال — instrument
    "م12ا3":   "place",   # مفتاح / مسمار / منشار (grouped with place)

    # فِعال / فُعول — Form I verbal noun → instance
    "12ا3":    "instance",  # كتاب / كلام
    "12و3":    "instance",  # دخول
    "12ّا3":   "instance",  # كتاب (geminate variant; also intensifier — see below)

    # فِعالة — Form I verbal noun of state → state
    "12ا3ة":   "state",     # كتابة / قراءة
    "12ا3ه":   "state",

    # تفعيل — Form II verbal noun → instance
    "ت12ي3":   "instance",

    # انفعال — Form VII verbal noun
    "ان12ا3":  "instance",

    # افتعال — Form VIII verbal noun
    "ا1ت2ا3":  "instance",

    # استفعال — Form X verbal noun
    "است12ا3": "instance",

    # تفاعل — Form VI verbal noun → mutual action
    "ت1ا23":   "mutual",

    # مفاعلة — Form III verbal noun → process
    "م1ا23ة":  "process",
    "م1ا23ه":  "process",

    # مفعِّل — Form II active participle → causer
    "م12ّ3":   "causer",   # معلّم / مدرّس
    "م12ّ3ة":  "causer",
    "م12ّ3ه":  "causer",

    # مستفعِل — Form X active participle → seeker
    "مست123":  "seeker",   # مستعمل / مستخدم / مستقبل

    # فعيل — adjective → quality
    "12ي3":    "quality",  # كبير / كريم / جميل
    "12ي3ة":   "quality",
    "12ي3ه":   "quality",

    # فعلان — adjective (hungry/thirsty-style) → quality
    "123ان":   "quality",

    # فعلى — feminine elative → quality
    "123ى":    "quality",
}

# POS-based fallback (when pattern doesn't match or is absent)
POS_TO_ROLE = {
    "adj": "quality",
    "adj_comp": "quality",
    "adj_num": "quality",
}

# POS values that indicate named entities → emit NE:<surface>
NER_POS = frozenset({"noun_prop"})

# CAMeL pseudo-roots that signal the analyzer gave up on triliteral
# structure. Any word whose analyses are all in this set should be
# treated as a foreign stem, not shoved into the generic LIT bucket.
NON_ROOT_MARKERS = frozenset({"NTWS", "PUNC", "DIGIT", "FOREIGN"})


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


def _is_critical_feat(token: str) -> bool:
    """Return True for FEAT tokens kept in critical-only mode.

    Critical FEAT inventory keeps generation-relevant agreement and time
    information while dropping high-inflation markers like definiteness
    and nominal f/p/d features.
    """
    if token.startswith("FEAT:asp:"):
        return True
    if token.startswith("FEAT:pron:"):
        return True
    if token.startswith("FEAT:") and token.count(":") == 1:
        # Bundled person-gender-number tag, e.g. FEAT:3mp.
        return token.split(":", 1)[1] in _PGN_TAGS
    return False


def _collect_prefix_tokens(a, *, critical_feat_only: bool = True):
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
            if critical_feat_only:
                out.extend(
                    tok for tok in table[v]
                    if not tok.startswith("FEAT:") or _is_critical_feat(tok)
                )
            else:
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

def _weak_expand(root: str):
    """Yield every substitution of ``#`` in ``root`` with a weak letter."""
    if "#" not in root:
        yield root
        return
    stack = [root]
    while stack:
        cur = stack.pop()
        i = cur.find("#")
        if i < 0:
            yield cur
            continue
        for letter in _WEAK_SUBSTITUTES:
            stack.append(cur[:i] + letter + cur[i + 1 :])


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
    def __init__(
        self,
        analyzer,
        vocab_path=None,
        cap=None,
        *,
        emit_root_pattern=False,
        emit_space_token=False,
        emit_atomic_composition=True,
        critical_feat_only=True,
    ):
        """Arabic CST tokenizer.

        Parameters
        ----------
        analyzer : CAMeL morphology analyzer (or mock with ``.analyze``).
        vocab_path : optional path to a frozen vocab JSON.
            When provided, the tokenizer runs in **frozen mode**: only
            tokens already in the vocab are emitted; anything else is
            routed to ``[UNK]``. Semantic tokens (ROOT:*, CMP:*, FEAT:*,
            REL:*, STR:*, NUM:*, TIME:*) must be present in the file —
            the builder in :func:`build_frozen_vocab` guarantees this.
        cap : optional integer. When set alongside ``vocab_path`` only
            entries with ``id < cap`` are kept from the file; anything
            past the cap becomes ``[UNK]``. Useful for comparing 8K /
            16K / 32K runs against the same master vocab.
        emit_root_pattern : bool, default False.
            When enabled, content words emit explicit morphological
            decomposition as ``ROOT:<canonical-root>`` + ``PAT:<pattern>``
            (when available) instead of ``CMP:<field>:<role>``.
        emit_space_token : bool, default False.
            When enabled, emits ``SPACE`` after each tokenized word to
            preserve explicit word boundaries in the token stream.
        emit_atomic_composition : bool, default True.
            When enabled, avoid multi-property ``CMP:<field>:<role>``
            emissions and split composition into atomic pieces
            ``ROOT:<field>`` + ``ROLE:<role>``.
        critical_feat_only : bool, default True.
            When enabled, emit only critical FEAT markers:
            ``FEAT:asp:*``, ``FEAT:<pgn>``, and ``FEAT:pron:<pgn>``.
            Non-critical FEAT markers (``FEAT:def``, ``FEAT:f``,
            ``FEAT:p``, ``FEAT:d``) are suppressed to reduce sequence
            length in Arabic runs.
        """
        self.analyzer = analyzer
        self.vocab: dict[str, int] = {}
        self.next_id = 0
        self.root_index = _build_wildcard_index()
        self.stats = Counter()
        self.frozen = False
        self.unk_id = 1   # filled in below
        self.emit_root_pattern = emit_root_pattern
        self.emit_space_token = emit_space_token
        self.emit_atomic_composition = emit_atomic_composition
        self.critical_feat_only = critical_feat_only

        if vocab_path is not None:
            self._load_frozen_vocab(vocab_path, cap=cap)
            return

        # Special tokens (aligned with cst-spec.ts v1.0)
        for tok in ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]"]:
            self._get_id(tok)
        self.unk_id = self.vocab["[UNK]"]

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

        # Pre-register FEAT tokens.
        if not self.critical_feat_only:
            self._get_id("FEAT:def")
        for tag in sorted(_PGN_TAGS):
            self._get_id(f"FEAT:{tag}")              # bundled person-gen-num
            self._get_id(f"FEAT:pron:{tag}")         # enclitic pronoun
        for asp in ("p", "i", "c"):
            self._get_id(f"FEAT:asp:{asp}")
        # Non-default noun/adj inflection markers are optional in compact mode.
        if not self.critical_feat_only:
            for t in ("FEAT:f", "FEAT:p", "FEAT:d"):
                self._get_id(t)

        # Optional v5 morphology decomposition tokens.
        if self.emit_root_pattern:
            for pattern in sorted(ARABIC_PATTERN_TO_ROLE):
                self._get_id(f"PAT:{pattern}")

        # Optional atomic composition inventory.
        if self.emit_atomic_composition:
            role_values = set(ARABIC_PATTERN_TO_ROLE.values()) | set(POS_TO_ROLE.values())
            for role in sorted(role_values):
                self._get_id(f"ROLE:{role}")

        # Optional explicit word-boundary token.
        if self.emit_space_token:
            self._get_id("SPACE")

        # ROOT:size for numerals
        self._get_id("ROOT:size")

        # v3 typed-literal vocabulary — category prefixes only. Actual
        # surface forms (NE:محمد, FOREIGN:كوفيد, ...) are added on demand.
        for t in (
            "NUM:year", "NUM:small", "NUM:decimal", "NUM:percent", "NUM:measure",
        ):
            self._get_id(t)
        for t in sorted(set(ARABIC_MONTHS.values())):
            self._get_id(t)

        # v4: pre-register all ~9.5K Arabic roots from the Zerrouki
        # list as ROOT:<dotted-canonical> tokens. Every real Arabic
        # root gets a deterministic vocab slot — including rare roots
        # absent from the training corpus — so frozen-mode tokenizers
        # emit a real token instead of [UNK] for out-of-distribution
        # but linguistically valid words.
        for dotted in ARABIC_ROOTS_ZERROUKI:
            self._get_id(f"ROOT:{dotted}")

    # ── Frozen vocab support ──────────────────────────────────
    def _load_frozen_vocab(self, path, cap=None):
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        if cap is not None:
            loaded = {t: i for t, i in loaded.items() if i < cap}
        if "[UNK]" not in loaded:
            raise ValueError(
                f"frozen vocab at {path} is missing [UNK]; "
                "every shipped vocab must reserve an UNK slot"
            )
        self.vocab = loaded
        self.next_id = max(loaded.values()) + 1 if loaded else 0
        self.unk_id = loaded["[UNK]"]
        self.frozen = True

    def _resolve(self, token):
        """Return ``(emit_token, id)`` honouring frozen-vocab mode.

        In frozen mode, a missing token becomes ``("[UNK]", unk_id)``
        and ``stats["unk"]`` is bumped. Callers should use this rather
        than appending token/id pairs directly so UNK routing is
        uniform across the word/sentence pipelines.
        """
        tid = self.vocab.get(token)
        if tid is not None:
            return token, tid
        if self.frozen:
            self.stats["unk"] += 1
            return "[UNK]", self.unk_id
        new_id = self.next_id
        self.vocab[token] = new_id
        self.next_id += 1
        return token, new_id

    def _get_id(self, token):
        if token in self.vocab: return self.vocab[token]
        if self.frozen:
            self.stats["unk"] += 1
            return self.unk_id
        tid = self.next_id; self.vocab[token] = tid; self.next_id += 1
        return tid

    def _strip(self, word):
        # Diacritics + tatweel.
        word = re.sub(r'[\u064B-\u065F\u0670]', '', word)
        word = word.replace('\u0640', '')
        # v3: strip Arabic punctuation that lives inside the Arabic
        # Unicode block and was leaking through the word-split regex.
        word = _ARABIC_PUNCT_RE.sub('', word)
        return word

    def _find_field(self, roots):
        """Look up the semantic field for any root in ``roots``.

        Tries exact match against the wildcard index first. If that
        fails and the root contains a CAMeL weak placeholder ``#``,
        substitute each placeholder with every candidate weak letter
        (و / ي / ا / ء / ى) and retry. This recovers roots like
        ``#.ح.د`` → و.ح.د (social), ``ك.#.ن`` → ك.و.ن (exist).
        """
        for r in roots:
            if not r:
                continue
            if r in self.root_index:
                return self.root_index[r]
            if "#" in r:
                for cand in _weak_expand(r):
                    if cand in self.root_index:
                        return self.root_index[cand]
        return None

    def _canonical_root(self, r: str) -> str:
        """Return a stable canonical form of a root for `ROOT:<raw>` emission.

        Substitutes every ``#`` with و so ``ك.#.ن`` and ``ك.و.ن`` share
        the same raw-root token even when the dictionary didn't resolve
        a semantic field.
        """
        return r.replace("#", "و") if r else r

    # ── Core per-word analysis ─────────────────────────────────
    def _best_analysis(self, clean):
        """Return (analysis, all_ntws) for the best CAMeL candidate.

        ``all_ntws`` is True when CAMeL returned at least one analysis
        but none carry a real triliteral root — the signal to route to
        ``FOREIGN:<surface>`` instead of ``LIT``.

        Selection priority (v4):

          1. **Distinct-lex NE** — a ``noun_prop`` analysis whose
             ``lex`` does not appear in any non-prop analysis. This is
             the key signal: CAMeL stores distinct lexemes for genuine
             names (``سُلَيْمان`` vs common ``سَلِيم``, ``القاهِرَة`` vs
             root-based ``قَهَرَ``). CAMeL's spurious "noun_prop over
             a common noun" false positives (noun_prop:``عَمَل``
             sharing lex with noun:``عَمَل``) fail this test.

          2. **Common reading** — the first ``pos != noun_prop``
             analysis with a real root. Handles ordinary nouns /
             verbs / adjectives and the ال-prefixed common cases
             (العمل، الحزب).

          3. **Fallback NE** — first ``noun_prop`` analysis (real
             or NTWS root) when no common reading is available.
             Handles ``محمد`` (only ever tagged noun_prop by CAMeL)
             and foreign place names like ``فرنسا``.

          4. Return ``None`` with ``all_ntws`` set so the caller
             picks ``FOREIGN:<surface>`` or ``LIT:<surface>``.

        The v3 heuristic of "prefer non-noun_prop always" under-fired
        NE by ~20x on Arabic Wikipedia because common proper nouns
        (محمد → ح.م.د → praise) were silently collapsed to ``ROOT:``.
        The v4 lex-distinctness rule recovers them while still
        demoting CAMeL's false-positive noun_prop analyses.
        """
        analyses = self.analyzer.analyze(clean)
        if not analyses:
            # v4: treat "CAMeL returned nothing" the same as "CAMeL
            # returned only NTWS" — the caller routes both to
            # FOREIGN:<surface>, collapsing the v3 LIT explosion of
            # transliterations (تاموكسفين، فولدمورت، قطالونيين, …)
            # into the FOREIGN bucket where they belong.
            return None, True

        prop_analyses: list[dict] = []
        non_prop_real: list[dict] = []
        non_prop_lexes: set[str] = set()
        has_real_root = False

        for a in analyses:
            r = a.get("root", "")
            pos = a.get("pos", "")
            real = bool(r) and r not in NON_ROOT_MARKERS
            if real:
                has_real_root = True
            if pos == "noun_prop":
                prop_analyses.append(a)
            elif real:
                non_prop_real.append(a)
                lex = a.get("lex", "")
                if lex:
                    non_prop_lexes.add(lex)

        # 1. Distinct-lex NE
        for a in prop_analyses:
            lex = a.get("lex", "")
            if lex and lex not in non_prop_lexes:
                return a, False

        # 2. Common reading
        if non_prop_real:
            return non_prop_real[0], False

        # 3. Fallback NE (no common reading available)
        if prop_analyses:
            return prop_analyses[0], False

        return None, not has_real_root

    def _extract_role(self, a):
        """Extract CMP role from camel-tools pattern or POS.

        CAMeL pattern strings use ASCII digits ``1/2/3`` for root
        consonant slots and carry diacritics between slots. We strip
        short vowels (keeping shadda) and normalize hamzat-wasl
        (``ٱ`` U+0671 → ``ا``) before looking up the role map.
        """
        pattern = a.get("pattern") or ""
        norm = _strip_vowels(pattern).replace("\u0671", "\u0627")
        if norm and norm in ARABIC_PATTERN_TO_ROLE:
            return ARABIC_PATTERN_TO_ROLE[norm]
        pos = a.get("pos", "")
        if pos in POS_TO_ROLE:
            return POS_TO_ROLE[pos]
        return None

    def _extract_pattern(self, a):
        """Extract normalized CAMeL pattern for ``PAT:<pattern>`` emission."""
        pattern = a.get("pattern") or ""
        norm = _strip_vowels(pattern).replace("\u0671", "\u0627")
        return norm or None

    def _word_tokens(self, clean):
        """Tokenize a single orthographic word into a list of CST tokens.

        Emission order for a fully-analyzed content word:
            [prc2 tokens]        conjunction  (و / ف …)
            [prc1 tokens]        preposition / future / emphasis
            [prc0 tokens]        article (ال) / attached negation
            [prc3 tokens]        question أ
            core token           CMP: / ROOT: / LIT:
            [gender/number FEAT] optional non-default inflection (f / p / d)
            [aspect FEAT]        verb only (asp:p / asp:i / asp:c)
            [pgn FEAT]           verb person-gender-number (e.g. 3mp)
            [enclitic pronoun]   FEAT:pron:<tag>

        In ``critical_feat_only`` mode (default), non-critical FEAT
        markers (article definiteness and nominal f/p/d) are suppressed.
        """
        out = []

        # 1. Fast paths — function / LIT / numeral / STR trigger words bypass
        #    full morphological decomposition; they carry fixed semantics.
        if clean in ARABIC_STR_TRIGGERS:
            # Handled at sentence level — emit nothing at word position.
            return out
        if clean in ARABIC_MONTHS:
            out.append(ARABIC_MONTHS[clean]); self.stats["time"] += 1
            return out

        # ``ما`` needs POS-aware disambiguation before REL fast-path.
        # Otherwise it is always captured as REL:what and negation is lost.
        if clean == "ما":
            a, _ = self._best_analysis(clean)
            pos = (a or {}).get("pos", "")
            if pos == "part_neg":
                out.append("STR:neg:general"); self.stats["str"] += 1
            elif pos in ("pron_interrog", "adv_interrog"):
                out.append("REL:what"); self.stats["rel"] += 1
            elif pos in ("pron_rel", "adv_rel"):
                out.append("REL:which"); self.stats["rel"] += 1
            else:
                out.append("REL:what"); self.stats["rel"] += 1
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
        a, all_ntws = self._best_analysis(clean)
        if a is None:
            # CAMeL gave up on triliteral structure. Treat as foreign /
            # non-Arabic stem rather than collapsing to plain LIT. This
            # is the main fix for the v2 LIT explosion: words like
            # أكتوبر / فرنسا / كوفيد that CAMeL tags NTWS.
            if all_ntws:
                out.append(f"FOREIGN:{clean}"); self.stats["foreign"] += 1
            else:
                out.append(f"LIT:{clean}"); self.stats["lit"] += 1
            return out

        pos = a.get("pos", "")

        # 3. Emit proclitic tokens (conjunction → prep → article → ques)
        for t in _collect_prefix_tokens(a, critical_feat_only=self.critical_feat_only):
            out.append(t)
            if t.startswith("REL:"): self.stats["rel"] += 1
            elif t.startswith("STR:"): self.stats["str"] += 1
            elif t.startswith("FEAT:"): self.stats["feat"] += 1

        # 4. Named entity (noun_prop) → NE:<surface>
        if pos in NER_POS:
            out.append(f"NE:{clean}"); self.stats["ne"] += 1
            # No further FEATs for proper nouns (they are opaque tokens).
            return out

        # 5. Core content token
        roots = [a.get("root", "")]
        raw_root = roots[0] if roots else ""

        role = self._extract_role(a)

        if self.emit_root_pattern:
            # v5 mode: explicit morphology decomposition.
            if raw_root:
                canon = self._canonical_root(raw_root)
                out.append(f"ROOT:{canon}")
                self.stats["root_raw"] += 1
            else:
                field = self._find_field(roots)
                if field:
                    out.append(f"ROOT:{field}")
                    self.stats["root"] += 1
                else:
                    out.append(f"LIT:{clean}")
                    self.stats["lit"] += 1
                    return out

            pat = self._extract_pattern(a)
            if pat:
                out.append(f"PAT:{pat}")
                self.stats["pat"] += 1
            if self.emit_atomic_composition and role:
                out.append(f"ROLE:{role}")
                self.stats["role"] += 1
        else:
            field = self._find_field(roots)
            if field:
                if role:
                    if self.emit_atomic_composition:
                        out.append(f"ROOT:{field}")
                        out.append(f"ROLE:{role}")
                        self.stats["root"] += 1
                        self.stats["role"] += 1
                    else:
                        out.append(f"CMP:{field}:{role}")
                        self.stats["cmp"] += 1
                else:
                    out.append(f"ROOT:{field}"); self.stats["root"] += 1
            else:
                # Phase 2: morphological collapsing. CAMeL returned a real
                # triliteral root but we have no semantic field for it. Emit
                # ``ROOT:<canonical-root>`` so every surface form of the
                # same root (including weak-variant spellings) shares one
                # vocab slot instead of each inflected form producing its
                # own ``LIT:<surface>`` entry.
                if raw_root:
                    canon = self._canonical_root(raw_root)
                    out.append(f"ROOT:{canon}"); self.stats["root_raw"] += 1
                else:
                    out.append(f"LIT:{clean}"); self.stats["lit"] += 1
                    return out

        # 6. Per-word feature tokens
        if pos == "verb":
            asp = a.get("asp", "na")
            if asp in ("p", "i", "c"):
                out.append(f"FEAT:asp:{asp}"); self.stats["feat"] += 1
            pgn = _pgn_tag(a)
            if pgn:
                out.append(f"FEAT:{pgn}"); self.stats["feat"] += 1
        else:
            # Optional nominal inflection (legacy mode only).
            if not self.critical_feat_only:
                gen = a.get("gen", "na")
                num = a.get("num", "na")
                if gen == "f":
                    out.append("FEAT:f"); self.stats["feat"] += 1
                if num == "p":
                    out.append("FEAT:p"); self.stats["feat"] += 1
                elif num == "d":
                    out.append("FEAT:d"); self.stats["feat"] += 1

        # 7. Enclitic pronoun (if any)
        enc = enc0_feat(a.get("enc0"))
        if enc:
            out.append(enc); self.stats["feat"] += 1

        return out

    # ── Sentence-level tokenization ────────────────────────────
    def tokenize(self, sentence):
        tokens = ["[BOS]"]
        ids = [self.vocab["[BOS]"]]

        # Split into surface "words": contiguous runs of Arabic letters,
        # Latin letters, or digits (with optional . , %). This is wider
        # than v2 so foreign words and numbers survive as their own units
        # instead of being silently dropped by the Arabic-only regex.
        raw_words = re.findall(
            r"[\u0600-\u06FF\u0750-\u077F]+"
            r"|[A-Za-z][A-Za-z\-']*"
            r"|[0-9\u0660-\u0669\u06F0-\u06F9][0-9\u0660-\u0669\u06F0-\u06F9.,\u066B]*[%\u066A]?",
            sentence,
        )

        # Sentence-level STR markers: standalone particles and punctuation.
        str_emitted = set()
        def _emit_str(marker):
            if marker in str_emitted: return
            tokens.append(marker); ids.append(self._get_id(marker))
            str_emitted.add(marker)
            self.stats["str"] += 1

        for w in raw_words:
            c = self._strip(w)
            if c in ARABIC_STR_TRIGGERS:
                _emit_str(ARABIC_STR_TRIGGERS[c])

        tail = sentence.rstrip()
        if tail.endswith("؟") or tail.endswith("?"):
            _emit_str("STR:question")
        if tail.endswith("!"):
            _emit_str("STR:emphasis")

        prefix_count = len(tokens) - 1   # excl. [BOS]

        def _emit_space() -> int:
            if not self.emit_space_token:
                return 0
            emit, eid = self._resolve("SPACE")
            tokens.append(emit)
            ids.append(eid)
            self.stats["space"] += 1
            return 1

        # Word-by-word tokenization using the per-word pipeline.
        word_forms = []                  # parallel to word_token_counts
        word_token_counts = []
        for raw in raw_words:
            # NUM pre-pass: typed NUM tokens for digit sequences.
            num_tok = _num_token(raw)
            if num_tok is not None:
                tokens.append(num_tok); ids.append(self._get_id(num_tok))
                self.stats["num"] += 1
                count = 1 + _emit_space()
                word_forms.append(raw); word_token_counts.append(count)
                continue

            # Foreign / Latin script word → FOREIGN:<surface> directly.
            if raw and raw[0].isascii() and raw[0].isalpha():
                lowered = raw.lower()
                emit, eid = self._resolve(f"FOREIGN:{lowered}")
                tokens.append(emit); ids.append(eid)
                self.stats["foreign"] += 1
                count = 1 + _emit_space()
                word_forms.append(raw); word_token_counts.append(count)
                continue

            clean = self._strip(raw)
            # Drop empty / single-char tokens — these were dominating the
            # v2 LIT tail (commas, isolated ه / م abbreviations, etc.)
            # and carry no semantic load at the word level.
            if len(clean) < 2:
                word_forms.append(raw); word_token_counts.append(0)
                continue

            w_tokens = self._word_tokens(clean)
            word_forms.append(clean)
            count = len(w_tokens)
            for tok in w_tokens:
                emit, eid = self._resolve(tok)
                tokens.append(emit); ids.append(eid)
            if count > 0:
                count += _emit_space()
            word_token_counts.append(count)

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
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════
# Frozen-vocab builder
# ═══════════════════════════════════════════════════════════════

# Token namespaces whose surfaces are *dynamic* (minted on demand from
# raw text) and therefore compete for the frequency-capped portion of
# the vocabulary. ``ROOT:`` is included because raw dotted roots (e.g.
# ``ROOT:ك.ت.ب``) are dataset-adaptive by design; field roots
# (e.g. ``ROOT:write``) remain core semantic via ``_is_core_semantic``.
_DYNAMIC_PREFIXES = ("NE:", "FOREIGN:", "LIT:", "ROOT:")


def _is_raw_root_token(token: str) -> bool:
    """Return True when ``token`` is a dotted lexical root token.

    Examples
    --------
    - raw lexical root: ``ROOT:ك.ت.ب`` -> True
    - semantic field root: ``ROOT:write`` -> False
    - numeric semantic root: ``ROOT:size`` -> False
    """
    if not token.startswith("ROOT:"):
        return False
    value = token.split(":", 1)[1]
    return "." in value


def _is_core_semantic(token: str) -> bool:
    """Return True for tokens that must always survive capping.

    v5 adaptive-vocab policy:

    - ``ROOT:<field>`` remains core-semantic.
    - raw dotted roots (``ROOT:ك.ت.ب``) are **not** core and are selected
      by corpus frequency when building a capped vocabulary.

    This keeps tokenizer knowledge complete (all roots analyzable) while
    keeping model vocabularies dataset-adaptive.
    """
    if token.startswith("["):
        return True   # [PAD], [UNK], [BOS], [EOS], [SEP]
    if token == "SPACE":
        return True
    if token.startswith("ROOT:"):
        return not _is_raw_root_token(token)
    if token.startswith(("CMP:", "PAT:", "ROLE:", "FEAT:", "REL:", "STR:",
                         "NUM:", "TIME:")):
        return True
    return False


def build_frozen_vocab(analyzer, sentences, cap, out_path=None, *,
                       progress_every=10000):
    """Build a deterministic capped vocab from a training corpus.

    Tokenizes ``sentences`` with an *unfrozen* tokenizer, counts how
    often each dynamic token (NE:/FOREIGN:/LIT:/raw-root) appears,
    then assembles a vocab of exactly ``cap`` entries:

      1. All core-semantic tokens (specials, ROOT:<field>, CMP:*, ...).
      2. Top-``K`` dynamic tokens by frequency, where ``K = cap -
         len(core)``. Ties broken by token string for determinism.

    Returns the new vocab dict. If ``out_path`` is given, the vocab
    is written to disk as JSON (pretty-printed, UTF-8).

    The returned vocab is designed to be passed straight to
    ``ArabicCSTTokenizer(analyzer, vocab_path=...)`` on the next run.
    """
    builder = ArabicCSTTokenizer(analyzer)
    dyn_counts: Counter = Counter()
    extra_core: set[str] = set()
    for i, s in enumerate(sentences, 1):
        out = builder.tokenize(s)
        for t in out["tokens"]:
            if _is_core_semantic(t):
                if t not in builder.vocab:
                    extra_core.add(t)
            elif t.startswith(_DYNAMIC_PREFIXES):
                dyn_counts[t] += 1
        if progress_every and i % progress_every == 0:
            print(f"  [build_frozen_vocab] scanned {i:,} sentences, "
                  f"{len(dyn_counts):,} dynamic surfaces so far", flush=True)

    core = {t: i for t, i in builder.vocab.items() if _is_core_semantic(t)}
    if "[UNK]" not in core:
        raise RuntimeError("builder vocab missing [UNK] — bug")
    core_total = len(core) + len(extra_core)
    budget = cap - core_total
    if budget < 0:
        raise ValueError(
            f"cap={cap} is smaller than core semantic token count "
            f"({core_total} = {len(core)} pre-registered + "
            f"{len(extra_core)} extra); raise the cap or trim the "
            f"semantic inventory"
        )

    # Deterministic order: frequency desc, then lexicographic.
    ranked = sorted(dyn_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    kept = ranked[:budget]

    # Re-number: pre-registered core tokens first (keep relative order),
    # then extra core-semantic tokens observed at tokenize time
    # (sorted for determinism), then dynamic tokens.
    final: dict[str, int] = {}
    for tok, _old in sorted(core.items(), key=lambda kv: kv[1]):
        final[tok] = len(final)
    for tok in sorted(extra_core):
        final[tok] = len(final)
    for tok, _freq in kept:
        final[tok] = len(final)

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(final, f, ensure_ascii=False, indent=2)

    # Report coverage — how much of the corpus actually hits the kept set.
    total_dyn = sum(dyn_counts.values())
    kept_dyn = sum(f for _t, f in kept)
    coverage = (kept_dyn / total_dyn) if total_dyn else 1.0
    print(f"  [build_frozen_vocab] cap={cap} core={len(core)} "
          f"kept_dynamic={len(kept)} unique_seen={len(dyn_counts):,} "
          f"dynamic_coverage={coverage:.4f} "
          f"predicted_unk_rate={1 - coverage:.4f}", flush=True)
    return final

