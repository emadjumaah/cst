"""Regression tests for English atomic composition emission.

These tests are pure and do not require spaCy models. They validate the
semantic mapping and suffix-role composition used by Stage 7 emission.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Repo-relative path: tests/ -> training/ -> edge/
EDGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EDGE))

from english_tokenizer import decompose, emit_tokens  # type: ignore


def _values(tokens: list[dict]) -> list[str]:
    return [t["value"] for t in tokens]


def test_library_maps_to_write_plus_place_atomic_by_default():
    parts = decompose("library", "library")
    vals = _values(emit_tokens("library", "library", False, parts))
    assert vals == ["ROOT:write", "ROLE:place"]


def test_writer_maps_to_write_plus_agent_atomic():
    parts = decompose("writer", "writer")
    vals = _values(emit_tokens("writer", "writer", False, parts))
    assert vals == ["ROOT:write", "ROLE:agent"]


def test_non_atomic_flag_keeps_legacy_cmp_shape():
    parts = decompose("library", "library")
    vals = _values(
        emit_tokens(
            "library",
            "library",
            False,
            parts,
            emit_atomic_composition=False,
        )
    )
    assert vals == ["CMP:write:place"]
