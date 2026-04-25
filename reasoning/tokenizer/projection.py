"""Projection π: default-level CST tokens → reasoning-level CST tokens.

Implements the language-agnostic parts of §4 of the spec
(``docs/two-level-tokenization.md``) and the Arabic-specific table from
``docs/cst-arabic-tokenizers.md``.

A projection is a pure function from a default token to either:
    - a reasoning token (possibly the same), or
    - ``None`` (drop).

The function has access to a small local context (``prev`` / ``next``
token) so rules like "collapse adjacent REL:and" can be expressed.

The English-language projection table will live here too once the
English default tokenizer exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator


# ═══════════════════════════════════════════════════════════════
# Shared drop rules (apply to both languages)
# ═══════════════════════════════════════════════════════════════

# Any token prefix listed here is dropped unconditionally at reasoning
# level. FEAT:* is the canonical surface-only marker from the Arabic
# default tokenizer.
_DROP_PREFIXES = (
    "FEAT:",        # inflection, enclitic pronouns, aspect, definiteness
)

# Reasoning level ignores individual sub-pieces (English BPE fallback);
# default tokens that look like ``SUB:<piece>`` reassemble to the lemma
# upstream, so at reasoning level they are redundant.
_DROP_EXACT: set[str] = set()


# ═══════════════════════════════════════════════════════════════
# Arabic-specific π table (see docs/cst-arabic-tokenizers.md §4)
# ═══════════════════════════════════════════════════════════════

# Proclitics emitted by edge/arabic_tokenizer.py are already in the
# REL:*/STR:* space, so most pass through untouched. The definite
# article comes through as FEAT:def and is dropped by _DROP_PREFIXES.

_AR_REMAP: dict[str, str] = {
    # Emphasis particles are not truth-conditional at reasoning level.
    "STR:emphasis": None,  # type: ignore[dict-item]
}


# ═══════════════════════════════════════════════════════════════
# English-specific π table (see docs/cst-english-tokenizers.md §4)
# ═══════════════════════════════════════════════════════════════

_EN_REMAP: dict[str, str | None] = {
    "DET:the": None,
    "DET:a": None,
    "DET:an": None,
    "AUX:do": None,
    "AUX:have": None,
    "AUX:be": None,
    "AUX:will": "STR:future",
    "AUX:shall": "STR:future",
    "POSS:'s": "REL:of",
}


# ═══════════════════════════════════════════════════════════════
# Projection
# ═══════════════════════════════════════════════════════════════

@dataclass
class Projection:
    """Language-parameterised projection function."""

    lang: str  # "ar" | "en"

    def __post_init__(self) -> None:
        if self.lang == "ar":
            self._remap = _AR_REMAP
        elif self.lang == "en":
            self._remap = _EN_REMAP
        else:
            raise ValueError(f"Unsupported language: {self.lang!r}")

    def project_token(self, tok: str) -> str | None:
        """Project a single default token to reasoning level (or drop)."""
        if tok in self._remap:
            return self._remap[tok]
        if tok in _DROP_EXACT:
            return None
        for prefix in _DROP_PREFIXES:
            if tok.startswith(prefix):
                return None
        return tok

    def project(self, tokens: Iterable[str]) -> list[str]:
        """Project a token sequence and apply collapse rules.

        Collapse rules:
          * adjacent identical ``REL:*`` collapse to one;
          * adjacent identical ``STR:clause_end`` collapse to one.
        """
        out: list[str] = []
        for tok in tokens:
            projected = self.project_token(tok)
            if projected is None:
                continue
            if out and out[-1] == projected and self._collapsible(projected):
                continue
            out.append(projected)
        return out

    @staticmethod
    def _collapsible(tok: str) -> bool:
        return tok.startswith("REL:") or tok == "STR:clause_end"


# ═══════════════════════════════════════════════════════════════
# Convenience
# ═══════════════════════════════════════════════════════════════

def project_arabic(tokens: Iterable[str]) -> list[str]:
    return Projection("ar").project(tokens)


def project_english(tokens: Iterable[str]) -> list[str]:
    return Projection("en").project(tokens)
