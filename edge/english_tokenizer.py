"""English CST tokenizer \u2014 canonical Python library module.

Python port of the TypeScript tokenizer in ``src/tokenizer/``. This module
is the **source of truth** for English tokenization in training / research
/ evaluation code; the TS implementation is kept only for the browser demo
in ``edge/demo/``.

Both implementations load the same data tables from ``data/tokenizer/*.json``
so the vocabularies can never drift. Small runtime divergence remains
because the underlying NLP libraries differ:

    TypeScript:  compromise  (lemma + NER)
    Python:      spaCy       (lemma + NER + POS, via en_core_web_sm)

Parity is verified by ``scripts/check_tokenizer_parity.py``.

Public surface
--------------
- :class:`EnglishCSTTokenizer`           \u2014 the tokenizer
- :func:`normalize`                      \u2014 stage 1
- :func:`detect_structure`               \u2014 stage 2
- :func:`detect_entities`                \u2014 stage 4
- :func:`decompose`                      \u2014 stage 5 (morphology)
- :func:`emit_token`                     \u2014 stage 7

The pipeline mirrors the TS version stage-for-stage:

    normalize \u2192 detect_structure \u2192 split_words \u2192 detect_entities
    \u2192 for each word: get_lemma \u2192 decompose \u2192 emit_token

Typical usage
-------------
    import spacy
    from edge.english_tokenizer import EnglishCSTTokenizer

    tok = EnglishCSTTokenizer(spacy.load("en_core_web_sm"))
    out = tok.tokenize("The writer sent a message to the teacher")
    print(out["tokens"])
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional

# \u2550\u2550 Data loading \u2550\u2550

_HERE = Path(__file__).resolve().parent
_DATA_DIR = _HERE.parent / "data" / "tokenizer"


def _load_json(name: str) -> Any:
    with open(_DATA_DIR / name) as f:
        return json.load(f)


SEMANTIC_FIELDS: dict[str, str] = _load_json("semantic_fields.json")
RELATION_MAP: dict[str, str] = _load_json("relation_map.json")
FUNCTION_WORDS: set[str] = set(_load_json("function_words.json"))
PREFIX_ROLES: dict[str, str] = _load_json("prefix_roles.json")
SUFFIX_ROLES: list[tuple[str, str]] = [tuple(p) for p in _load_json("suffix_roles.json")]
_STRUCTURE_PATTERNS_RAW: list[dict[str, str]] = _load_json("structure_patterns.json")
STRUCTURE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(p["pattern"], re.IGNORECASE if "i" in p.get("flags", "") else 0), p["token"])
    for p in _STRUCTURE_PATTERNS_RAW
]
CST_SPEC: dict[str, Any] = _load_json("cst_spec.json")
SPECIAL_TOKENS: dict[str, int] = CST_SPEC["special_tokens"]

# Lexeme-level role overrides used to align irregular derivations with
# intended semantic roles in atomic mode.
LEXICAL_ROLE_OVERRIDES: dict[str, str] = {
    "teacher": "causer",
    "student": "seeker",
    "hospital": "place",
}


# \u2550\u2550 Stage 1 \u2014 Normalize \u2550\u2550

_SMART_SINGLE = re.compile(r"[\u2018\u2019\u201A\u2039\u203A]")
_SMART_DOUBLE = re.compile(r"[\u201C\u201D\u201E\u00AB\u00BB]")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Stage 1 \u2014 lowercase + quote normalization + whitespace collapse.

    Byte-identical to ``src/tokenizer/normalizer.ts``.
    """
    t = text.lower()
    t = _SMART_SINGLE.sub("'", t)
    t = _SMART_DOUBLE.sub('"', t)
    t = _WS.sub(" ", t)
    return t.strip()


# \u2550\u2550 Stage 2 \u2014 Structure detection \u2550\u2550

def detect_structure(normalized_text: str) -> list[str]:
    """Stage 2 \u2014 whole-sentence STR markers (question/negation/past/...).

    Matches the 6 regex patterns in ``structure_patterns.json``.
    """
    tokens: list[str] = []
    for pattern, token in STRUCTURE_PATTERNS:
        if pattern.search(normalized_text) and token not in tokens:
            tokens.append(token)
    return tokens


# \u2550\u2550 Stage 3 \u2014 Word split \u2550\u2550

_SPLIT_PUNCT = re.compile(r"([.,!?;:'\"()\-])")
_PUNCT_ONLY = re.compile(r"^[.,!?;:'\"()\-\u2013\u2014\u2026/]+$")
_CONTRACTION_FRAGMENT = re.compile(
    r"^(s|t|re|ll|ve|d|m|don|won|didn|doesn|isn|wasn|aren|couldn|shouldn|wouldn|haven|hasn|weren|ain|e|u|o|x)$"
)


def split_words(text: str) -> list[str]:
    """Stage 3 \u2014 whitespace/punctuation split. Matches TS splitWords()."""
    spaced = _SPLIT_PUNCT.sub(r" \1 ", text)
    return [w for w in spaced.split() if w]


def _should_skip(word: str) -> bool:
    if not word:
        return True
    if _PUNCT_ONLY.match(word):
        return True
    if _CONTRACTION_FRAGMENT.match(word):
        return True
    return False


# \u2550\u2550 Stage 4 \u2014 NER \u2550\u2550

# spaCy ent_type_ \u2192 TS compromise categories. Only these become LIT entities.
# compromise only flags People / Places / Organizations; it does NOT flag
# adjectival forms like "Canadian" (NORP) as entities. We exclude NORP
# to avoid spurious LIT tokens for nationality adjectives.
_NER_LABELS = {"PERSON", "GPE", "LOC", "ORG", "FAC"}


def detect_entities(doc) -> set[str]:
    """Stage 4 \u2014 lowercase set of NE surface forms, one entry per token span.

    Takes a spaCy Doc (already parsed on the *normalized* text). Splits
    multi-word entity spans back into individual lowercase words so the
    per-word check in stage 7 sees e.g. ``{"new", "york"}`` for "New York".
    """
    ents: set[str] = set()
    for ent in getattr(doc, "ents", []):
        if ent.label_ not in _NER_LABELS:
            continue
        for tok in ent:
            w = tok.text.lower()
            if w:
                ents.add(w)
    return ents


# \u2550\u2550 Stage 5 \u2014 Morphology \u2550\u2550

def detect_prefix(word: str) -> Optional[dict[str, str]]:
    """TS detectPrefix \u2014 iterates PREFIX_ROLES in insertion order."""
    for prefix, role in PREFIX_ROLES.items():
        if word.startswith(prefix) and len(word) > len(prefix) + 2:
            return {"prefix": prefix, "role": role, "stem": word[len(prefix):]}
    return None


def detect_suffix(word: str, lemma: str) -> Optional[dict[str, str]]:
    """TS detectSuffix \u2014 walks SUFFIX_ROLES greedily (already longest-first)."""
    target = word  # TS does `word !== lemma ? word : word` which is just word
    for suffix, role in SUFFIX_ROLES:
        if target.endswith(suffix) and len(target) > len(suffix) + 2:
            if suffix == "s" and word == lemma:
                continue
            if suffix == "ed" and word == lemma:
                continue
            if suffix == "ly" and word == lemma:
                continue
            if suffix == "ing" and word == lemma:
                continue
            return {"suffix": suffix, "role": role}
    if word != lemma and word == lemma + "s":
        return {"suffix": "s", "role": "plural"}
    if word != lemma and word == lemma + "es":
        return {"suffix": "es", "role": "plural"}
    return None


def decompose(word: str, lemma: str) -> dict[str, Optional[str]]:
    """TS decompose \u2014 returns {root, role}."""
    p = detect_prefix(word)
    if p:
        return {"root": p["stem"], "role": p["role"]}
    s = detect_suffix(word, lemma)
    if s:
        stem = word[: len(word) - len(s["suffix"])]
        root = lemma if lemma != word else stem
        return {"root": root, "role": s["role"]}
    return {"root": lemma, "role": None}


# \u2550\u2550 Stage 6 \u2014 Semantic field lookup \u2550\u2550

_SUFFIX_STRIP_LIST = [
    "able", "ible", "tion", "sion", "ment", "ance", "ence", "ness",
    "ful", "less", "ery", "ory", "ary", "age", "ing", "ist", "ian",
    "ity", "er", "or", "ee", "ly", "al", "ed",
]


def resolve_field(root: Optional[str], lemma: str) -> Optional[str]:
    """TS resolveField \u2014 tries candidates + suffix-strip recursion.

    Handles silent-e dropping (``writ`` \u2192 ``write``) and nested affixes
    (``readable`` \u2192 strip ``able`` \u2192 ``read`` \u2192 field ``know``).
    """
    if not root and not lemma:
        return None

    candidates: list[str] = []
    if root:
        candidates.append(root)
        candidates.append(root + "e")
    candidates.append(lemma)
    candidates.append(lemma + "e")

    for c in candidates:
        f = SEMANTIC_FIELDS.get(c)
        if f:
            return f

    if root:
        for sfx in _SUFFIX_STRIP_LIST:
            if root.endswith(sfx) and len(root) > len(sfx) + 2:
                stem = root[: len(root) - len(sfx)]
                f = SEMANTIC_FIELDS.get(stem) or SEMANTIC_FIELDS.get(stem + "e")
                if f:
                    return f
    # Also strip from lemma itself \u2014 compensates for spaCy leaving gerunds
    # as-is (``banning`` lemma stays ``banning``) where compromise returns
    # the verb infinitive (``ban``).
    if lemma:
        for sfx in _SUFFIX_STRIP_LIST:
            if lemma.endswith(sfx) and len(lemma) > len(sfx) + 2:
                stem = lemma[: len(lemma) - len(sfx)]
                f = SEMANTIC_FIELDS.get(stem) or SEMANTIC_FIELDS.get(stem + "e")
                if f:
                    return f
    return None


# \u2550\u2550 Stage 7 \u2014 Emit \u2550\u2550

_NUMERIC = re.compile(r"^\d+$")


def emit_tokens(
    word: str,
    lemma: str,
    is_entity_word: bool,
    decomp: dict[str, Optional[str]],
    *,
    emit_atomic_composition: bool = True,
) -> list[dict[str, Any]]:
    """Stage-7 emission with optional atomic composition split.

    When ``emit_atomic_composition`` is enabled, a compositional token
    ``CMP:<field>:<role>`` is emitted as two atomic tokens:
    ``ROOT:<field>`` + ``ROLE:<role>``.
    """
    if is_entity_word:
        return [_make_token("LIT", f"LIT:{word}", word, 1.0)]

    if _NUMERIC.match(word):
        return [_make_token("ROOT", "ROOT:size", word, 0.9, field="size")]

    rel = RELATION_MAP.get(word)
    if rel:
        return [_make_token("REL", rel, word, 1.0)]

    if word in FUNCTION_WORDS:
        return [_make_token("LIT", f"LIT:{word}", word, 0.9)]

    root = decomp["root"]
    role = decomp["role"]
    field = resolve_field(root, lemma)
    role = LEXICAL_ROLE_OVERRIDES.get(lemma) or LEXICAL_ROLE_OVERRIDES.get(word) or role

    if field and role:
        if emit_atomic_composition:
            return [
                _make_token("ROOT", f"ROOT:{field}", word, 0.9, field=field),
                _make_token("ROLE", f"ROLE:{role}", word, 0.9, role=role),
            ]
        return [_make_token("CMP", f"CMP:{field}:{role}", word, 0.9, field=field, role=role)]
    if field:
        return [_make_token("ROOT", f"ROOT:{field}", word, 0.8, field=field)]
    return [_make_token("LIT", f"LIT:{word}", word, 0.5)]


def emit_token(
    word: str,
    lemma: str,
    is_entity_word: bool,
    decomp: dict[str, Optional[str]],
) -> dict[str, Any]:
    """Backward-compatible single-token stage-7 helper."""
    return emit_tokens(word, lemma, is_entity_word, decomp)[0]


def _make_token(
    type_: str,
    value: str,
    surface: str,
    confidence: float,
    field: Optional[str] = None,
    role: Optional[str] = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "type": type_,
        "value": value,
        "surface": surface,
        "id": 0,
        "confidence": confidence,
    }
    if field is not None:
        out["field"] = field
    if role is not None:
        out["role"] = role
    return out


# \u2550\u2550 spaCy adapter: lemma cache on a pre-parsed Doc \u2550\u2550

_VERB_POS = {"VERB", "AUX"}
_NOUN_POS = {"NOUN", "PROPN"}


def _spacy_lemma_for(word: str, doc, nlp_fallback=None) -> str:
    """Mimic the TS getLemma() heuristics under spaCy.

    TS runs ``nlp(word)`` on the single word, checks ``.verbs()`` then
    ``.nouns()`` and falls back to lowercase. To approximate:

    1. Look for the word in the sentence Doc. If present, use its
       ``.lemma_`` (lowercased). This is what spaCy produces in context.
    2. If absent, fall back to single-word parsing via ``nlp_fallback(word)``
       if provided, else lowercase.
    """
    target = word.lower()
    if doc is not None:
        for tok in doc:
            if tok.text.lower() == target:
                lem = (tok.lemma_ or target).lower()
                return lem or target
    if nlp_fallback is not None:
        d = nlp_fallback(word)
        if len(d) > 0:
            return (d[0].lemma_ or target).lower()
    return target


# \u2550\u2550 Main class \u2550\u2550


class EnglishCSTTokenizer:
    """English CST tokenizer.

    Parameters
    ----------
    nlp : spacy.Language
        A loaded spaCy pipeline with lemma + NER (at minimum
        ``en_core_web_sm``). Required on construction so the tokenizer
        itself stays stateless and picklable.
    """

    def __init__(self, nlp, *, emit_atomic_composition: bool = True):
        self._nlp = nlp
        self._lemma_cache: dict[str, str] = {}
        self._emit_atomic_composition = emit_atomic_composition

    # \u2500 Core API \u2500

    def tokenize(self, text: str) -> dict[str, Any]:
        normalized = normalize(text)
        structure_tokens = detect_structure(normalized)
        words = split_words(normalized)

        # One spaCy parse on the normalized text (used for lemmas + NER).
        doc = self._nlp(normalized)
        entity_words = detect_entities(doc)

        tokens: list[dict[str, Any]] = []

        for sval in structure_tokens:
            tokens.append(_make_token("STR", sval, "", 1.0))

        for word in words:
            if _should_skip(word):
                continue
            lemma = self._get_lemma(word, doc)
            is_ent = word.lower() in entity_words
            decomp = decompose(word, lemma)
            tokens.extend(
                emit_tokens(
                    word,
                    lemma,
                    is_ent,
                    decomp,
                    emit_atomic_composition=self._emit_atomic_composition,
                )
            )

        return {
            "tokens": tokens,
            "values": [t["value"] for t in tokens],
            "coverage": _coverage(tokens),
            "text": text,
        }

    def _get_lemma(self, word: str, doc) -> str:
        cached = self._lemma_cache.get(word)
        if cached is not None:
            return cached
        lem = _spacy_lemma_for(word, doc, nlp_fallback=self._nlp)
        self._lemma_cache[word] = lem
        return lem


# \u2550\u2550 Coverage \u2550\u2550

def _coverage(tokens: list[dict[str, Any]]) -> dict[str, Any]:
    stats = {
        "total": 0,
        "cmp": 0,
        "root": 0,
        "role": 0,
        "str": 0,
        "rel": 0,
        "lit": 0,
        "unk": 0,
    }
    for t in tokens:
        stats["total"] += 1
        ty = t["type"].lower()
        if ty == "special":
            stats["unk"] += 1
        elif ty in stats:
            stats[ty] += 1
    structured = (
        stats["cmp"] + stats["root"] + stats["role"] + stats["str"] + stats["rel"]
    )
    stats["structured"] = structured
    stats["ratio"] = structured / stats["total"] if stats["total"] else 0.0
    return stats
