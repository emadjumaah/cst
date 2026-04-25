"""Arabic reasoning-level tokenizer ``T_R^ar``.

Wraps the default-level tokenizer from ``edge/arabic_tokenizer.py`` and
applies the projection π defined in :mod:`reasoning.tokenizer.projection`.

Usage
-----

    from reasoning.tokenizer.arabic import ArabicReasoningTokenizer

    tok = ArabicReasoningTokenizer.default()
    out = tok.tokenize("وسيكتبُ الأطفالُ رسالةً للمعلمة")
    print(out["reasoning_tokens"])   # coarse, logic-preserving
    print(out["default_tokens"])     # full detail (for reconstruction)

Design note
-----------

We do *not* reimplement morphology. The default tokenizer is the source
of truth; the reasoning tokenizer is purely a **post-processor** that
applies π. This guarantees §4 property 3 (monotonicity) by construction:
``T_R(s) == project(T_D(s))`` is literally how we compute it.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Make ``edge/`` importable whether we run from repo root or reasoning/.
_HERE = Path(__file__).resolve().parent
_EDGE = _HERE.parent.parent / "edge"
if str(_EDGE) not in sys.path:
    sys.path.insert(0, str(_EDGE))

from arabic_tokenizer import ArabicCSTTokenizer  # noqa: E402

from .projection import Projection  # noqa: E402


class ArabicReasoningTokenizer:
    """Composes the default tokenizer + Arabic projection."""

    def __init__(self, default_tokenizer: ArabicCSTTokenizer) -> None:
        self._default = default_tokenizer
        self._projection = Projection("ar")

    # ── Factory ───────────────────────────────────────────────
    @classmethod
    def default(
        cls,
        *,
        emit_root_pattern: bool = False,
        emit_space_token: bool = False,
        emit_atomic_composition: bool = True,
        critical_feat_only: bool = True,
    ) -> "ArabicReasoningTokenizer":
        """Build with CAMeL Tools' built-in MSA database.

        Parameters
        ----------
        emit_root_pattern:
            Forwarded to ``ArabicCSTTokenizer``. Enables ``ROOT+PAT``
            decomposition for content words.
        emit_space_token:
            Forwarded to ``ArabicCSTTokenizer``. Emits explicit ``SPACE``
            boundary token after each tokenized word.
        emit_atomic_composition:
            Forwarded to ``ArabicCSTTokenizer``. Splits compositional
            tokens into atomic ``ROOT`` + ``ROLE`` pieces.
        critical_feat_only:
            Forwarded to ``ArabicCSTTokenizer``. Keeps only critical
            FEAT markers (aspect, bundled person-gender-number, and
            enclitic pronoun tags).
        """
        from camel_tools.morphology.analyzer import Analyzer
        from camel_tools.morphology.database import MorphologyDB

        db = MorphologyDB.builtin_db()
        analyzer = Analyzer(db)
        return cls(
            ArabicCSTTokenizer(
                analyzer,
                emit_root_pattern=emit_root_pattern,
                emit_space_token=emit_space_token,
                emit_atomic_composition=emit_atomic_composition,
                critical_feat_only=critical_feat_only,
            )
        )

    # ── Core API ──────────────────────────────────────────────
    def tokenize(self, sentence: str) -> dict[str, Any]:
        """Return both levels and the alignment metadata.

        Returns
        -------
        dict with keys:
            ``default_tokens``    — list[str], the full-detail sequence
            ``reasoning_tokens``  — list[str], the projected sequence
            ``default_ids``       — list[int]
            ``text``              — original sentence
        """
        d = self._default.tokenize(sentence)
        reasoning_tokens = self._projection.project(d["tokens"])
        return {
            "default_tokens": d["tokens"],
            "default_ids": d["ids"],
            "reasoning_tokens": reasoning_tokens,
            "text": sentence,
        }

    def reasoning(self, sentence: str) -> list[str]:
        """Convenience: return only the reasoning-level tokens."""
        return self.tokenize(sentence)["reasoning_tokens"]

    # ── Stats ─────────────────────────────────────────────────
    def compression_ratio(self, sentence: str) -> float:
        """|T_R(s)| / |T_D(s)|  — lower is better compression."""
        r = self.tokenize(sentence)
        d_len = len(r["default_tokens"])
        return len(r["reasoning_tokens"]) / d_len if d_len else 0.0
