"""English reasoning-level tokenizer ``T_R^en`` — spaCy-backed.

Uses spaCy's lemmatizer + POS tagger + dependency parser to produce
CST-shaped tokens per the English addendum
(``docs/cst-english-tokenizers.md``). The projection π from
:mod:`reasoning.tokenizer.projection` then drops surface-only tokens
(``DET:*``, ``AUX:do/have/be``, possessives, etc.) to yield the
reasoning-level sequence.

Backend
-------

Default model: ``en_core_web_sm``. Download once::

    python -m spacy download en_core_web_sm

If spaCy is not installed or the model is missing, the tokenizer
degrades to a small rule-based fallback (clearly marked in the
``_stub`` output field) so the pipeline still runs end-to-end.

Token emission order (default level)::

    [BOS] [STR markers] [DET / AUX / CMP / ROOT / LIT / REL] … [STR:clause_end] [EOS]

Role population
---------------

- ``ROOT:<lemma>`` — content words (NOUN, VERB, ADJ, ADV, PROPN, PRON)
- ``CMP:<lemma>:<role>`` — content words whose dependency label maps to
  an argument role (``nsubj → agent``, ``dobj → patient``, …)
- ``REL:*`` — conjunctions, prepositions, negation, quantifiers, some
  discourse markers
- ``STR:*`` — question, conditional, clause/sentence boundary
- ``LIT:*`` — numbers, named entities
- ``DET:*`` / ``AUX:*`` / ``POSS:*`` — surface-only, dropped by π
"""
from __future__ import annotations

import re
from typing import Any

from .projection import Projection


# ═══════════════════════════════════════════════════════════════
# Role maps
# ═══════════════════════════════════════════════════════════════

_DEP_TO_CMP: dict[str, str] = {
    "nsubj": "agent",
    "nsubjpass": "patient",
    "csubj": "agent",
    "dobj": "patient",
    "obj": "patient",
    "iobj": "recipient",
    "dative": "recipient",
    "pobj": "oblique",
    "attr": "attribute",
    "acomp": "attribute",
    "xcomp": "complement",
    "ccomp": "complement",
    "advmod": "manner",
    "npadvmod": "time",
}

_PREP_TO_REL: dict[str, str] = {
    "in": "REL:in", "on": "REL:on", "at": "REL:at",
    "with": "REL:with", "for": "REL:for", "by": "REL:by",
    "from": "REL:from", "to": "REL:to", "about": "REL:about",
    "of": "REL:of", "over": "REL:over", "under": "REL:under",
    "between": "REL:between", "through": "REL:through",
    "during": "REL:during", "before": "REL:before", "after": "REL:after",
}

_CONN_TO_REL: dict[str, str] = {
    "and": "REL:and", "or": "REL:or", "but": "REL:contrast",
    "however": "REL:contrast", "because": "REL:cause",
    "so": "REL:entail", "therefore": "REL:entail", "thus": "REL:entail",
    "while": "REL:while", "although": "REL:contrast", "though": "REL:contrast",
}

_NEG_LEMMAS: set[str] = {"not", "no", "never", "nothing", "nobody", "nowhere"}

_QUANT_TO_REL: dict[str, str] = {
    "all": "REL:quant:all", "every": "REL:quant:all", "each": "REL:quant:all",
    "some": "REL:quant:some", "any": "REL:quant:some",
    "many": "REL:quant:many", "few": "REL:quant:few",
    "most": "REL:quant:most", "none": "REL:quant:none",
}

_DETERMINERS: set[str] = {"a", "an", "the"}

_AUX_FUTURE: set[str] = {"will", "shall"}
_AUX_LEMMAS: set[str] = {"do", "have", "be"}

_CONTENT_POS: set[str] = {"NOUN", "VERB", "ADJ", "ADV", "PROPN", "PRON"}


# ═══════════════════════════════════════════════════════════════
# Tokenizer
# ═══════════════════════════════════════════════════════════════

class EnglishReasoningTokenizer:
    """spaCy-backed English tokenizer producing CST-shaped tokens."""

    def __init__(
        self,
        model: str = "en_core_web_sm",
        *,
        emit_atomic_composition: bool = True,
    ) -> None:
        self._projection = Projection("en")
        self._nlp = None
        self._stub = False
        self._emit_atomic_composition = emit_atomic_composition

        try:
            import spacy
            try:
                self._nlp = spacy.load(model)
            except OSError:
                # Model not downloaded yet.
                self._stub = True
        except Exception:
            # ImportError if spaCy not installed; AttributeError / other on
            # Python 3.13 where thinc/torch import chain uses removed stdlib APIs.
            self._stub = True

    # ── Public API ────────────────────────────────────────────
    def tokenize(self, sentence: str) -> dict[str, Any]:
        if self._stub:
            default_tokens = _fallback_tokens(sentence)
        else:
            default_tokens = self._default_tokens_spacy(sentence)
        reasoning_tokens = self._projection.project(default_tokens)
        return {
            "default_tokens": default_tokens,
            "reasoning_tokens": reasoning_tokens,
            "text": sentence,
            "_stub": self._stub,
        }

    def reasoning(self, sentence: str) -> list[str]:
        return self.tokenize(sentence)["reasoning_tokens"]

    def compression_ratio(self, sentence: str) -> float:
        r = self.tokenize(sentence)
        d = len(r["default_tokens"])
        return len(r["reasoning_tokens"]) / d if d else 0.0

    # ── spaCy path ────────────────────────────────────────────
    def _default_tokens_spacy(self, sentence: str) -> list[str]:
        assert self._nlp is not None
        doc = self._nlp(sentence)

        out: list[str] = ["[BOS]"]

        if doc.text.rstrip().endswith("?"):
            out.append("STR:question")
        if self._has_conditional(doc):
            out.append("STR:conditional")

        # Collect named-entity spans so we skip the individual tokens.
        ent_spans = {(e.start, e.end): e for e in doc.ents}
        token_in_ent: dict[int, tuple[int, int]] = {}
        for (s, e) in ent_spans:
            for i in range(s, e):
                token_in_ent[i] = (s, e)

        i = 0
        while i < len(doc):
            if i in token_in_ent:
                s, e = token_in_ent[i]
                ent = ent_spans[(s, e)]
                out.append(f"LIT:{ent.text.strip()}")
                i = e
                continue

            tok = doc[i]
            lemma = tok.lemma_.lower()
            pos = tok.pos_
            dep = tok.dep_
            lower = tok.text.lower()

            if not tok.text.strip():
                i += 1
                continue

            if pos == "PUNCT":
                if tok.text in {".", ";"}:
                    out.append("STR:clause_end")
                elif tok.text in {'"', "'", "“", "”", "‘", "’"}:
                    out.append("STR:quote")
                i += 1
                continue

            # Quantifiers first — spaCy tags "all/some/every" as DET, but
            # semantically they are quantificational relations.
            if lemma in _QUANT_TO_REL:
                out.append(_QUANT_TO_REL[lemma])
                i += 1
                continue

            if pos == "DET" or lower in _DETERMINERS:
                out.append(f"DET:{lemma}")
                i += 1
                continue

            if lemma in _AUX_FUTURE:
                out.append(f"AUX:{lemma}")
                i += 1
                continue

            if pos == "AUX" or lemma in _AUX_LEMMAS:
                out.append(f"AUX:{lemma}")
                i += 1
                continue

            if lemma in _NEG_LEMMAS or dep == "neg":
                out.append("REL:neg")
                i += 1
                continue

            if lemma in _CONN_TO_REL:
                out.append(_CONN_TO_REL[lemma])
                i += 1
                continue

            if pos == "ADP" or lemma in _PREP_TO_REL:
                out.append(_PREP_TO_REL.get(lemma, f"REL:{lemma}"))
                i += 1
                continue

            if tok.tag_ == "POS" or lower == "'s":
                out.append("POSS:'s")
                i += 1
                continue

            if pos == "NUM" or tok.like_num:
                out.append(f"LIT:{tok.text}")
                i += 1
                continue

            if pos in _CONTENT_POS:
                role = _DEP_TO_CMP.get(dep)
                if role:
                    if self._emit_atomic_composition:
                        out.append(f"ROOT:{lemma}")
                        out.append(f"ROLE:{role}")
                    else:
                        out.append(f"CMP:{lemma}:{role}")
                else:
                    out.append(f"ROOT:{lemma}")
                i += 1
                continue

            out.append(f"LIT:{tok.text}")
            i += 1

        out.append("[EOS]")
        return out

    @staticmethod
    def _has_conditional(doc) -> bool:
        for tok in doc:
            if tok.lemma_.lower() in {"if", "unless", "whenever"}:
                return True
        return False


# ═══════════════════════════════════════════════════════════════
# Fallback (no spaCy)
# ═══════════════════════════════════════════════════════════════

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*|\d+|[.?!;,]")


def _fallback_tokens(sentence: str) -> list[str]:
    out: list[str] = ["[BOS]"]
    for raw in _WORD_RE.findall(sentence):
        lo = raw.lower()
        if lo in _DETERMINERS:
            out.append(f"DET:{lo}")
        elif lo in _CONN_TO_REL:
            out.append(_CONN_TO_REL[lo])
        elif lo in _NEG_LEMMAS:
            out.append("REL:neg")
        elif lo in _QUANT_TO_REL:
            out.append(_QUANT_TO_REL[lo])
        elif lo in _PREP_TO_REL:
            out.append(_PREP_TO_REL[lo])
        elif lo in _AUX_FUTURE or lo in _AUX_LEMMAS:
            out.append(f"AUX:{lo}")
        elif lo == "?":
            out.append("STR:question")
        elif lo in {".", ";"}:
            out.append("STR:clause_end")
        elif raw.isdigit():
            out.append(f"LIT:{raw}")
        else:
            out.append(f"ROOT:{lo}")
    out.append("[EOS]")
    return out
