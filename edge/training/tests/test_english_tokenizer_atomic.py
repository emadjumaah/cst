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

from arabic_tokenizer import ArabicCSTTokenizer  # type: ignore
from english_tokenizer import decompose, emit_tokens  # type: ignore


class MockAnalyzer:
    """Returns scripted Arabic analyses keyed by surface form."""

    def __init__(self, table: dict[str, list[dict]]):
        self.table = table

    def analyze(self, word: str) -> list[dict]:
        return self.table.get(word, [])


def _values(tokens: list[dict]) -> list[str]:
    return [t["value"] for t in tokens]


def _ar_values(table: dict[str, list[dict]], text: str) -> list[str]:
    tok = ArabicCSTTokenizer(MockAnalyzer(table))
    # Arabic tokenizer output includes BOS/EOS sentinels; strip them for parity.
    return tok.tokenize(text)["tokens"][1:-1]


def test_library_maps_to_write_plus_place_atomic_by_default():
    parts = decompose("library", "library")
    vals = _values(emit_tokens("library", "library", False, parts))
    assert vals == ["ROOT:write", "ROLE:place"]


def test_writer_maps_to_write_plus_agent_atomic():
    parts = decompose("writer", "writer")
    vals = _values(emit_tokens("writer", "writer", False, parts))
    assert vals == ["ROOT:write", "ROLE:agent"]


def test_teacher_maps_to_know_plus_causer_atomic():
    parts = decompose("teacher", "teacher")
    vals = _values(emit_tokens("teacher", "teacher", False, parts))
    assert vals == ["ROOT:know", "ROLE:causer"]


def test_student_maps_to_know_plus_seeker_atomic():
    parts = decompose("student", "student")
    vals = _values(emit_tokens("student", "student", False, parts))
    assert vals == ["ROOT:know", "ROLE:seeker"]


def test_hospital_maps_to_health_plus_place_atomic():
    parts = decompose("hospital", "hospital")
    vals = _values(emit_tokens("hospital", "hospital", False, parts))
    assert vals == ["ROOT:health", "ROLE:place"]


def test_doctor_maps_to_health_plus_agent_atomic():
    parts = decompose("doctor", "doctor")
    vals = _values(emit_tokens("doctor", "doctor", False, parts))
    assert vals == ["ROOT:health", "ROLE:agent"]


def test_sender_maps_to_send_plus_agent_atomic():
    parts = decompose("sender", "sender")
    vals = _values(emit_tokens("sender", "sender", False, parts))
    assert vals == ["ROOT:send", "ROLE:agent"]


def test_message_maps_to_send_plus_instance_atomic():
    parts = decompose("message", "message")
    vals = _values(emit_tokens("message", "message", False, parts))
    assert vals == ["ROOT:send", "ROLE:instance"]


def test_cross_lingual_writer_converges_to_same_atomic_pair():
    english = _values(emit_tokens("writer", "writer", False, decompose("writer", "writer")))
    arabic = _ar_values(
        {
            "كاتب": [{
                "root": "ك.ت.ب",
                "pattern": "1َا2ِ3",
                "pos": "noun",
                "prc0": "0",
                "prc1": "0",
                "prc2": "0",
                "prc3": "0",
                "enc0": "0",
                "gen": "m",
                "num": "s",
            }],
        },
        "كاتب",
    )
    assert english == arabic == ["ROOT:write", "ROLE:agent"]


def test_cross_lingual_library_converges_to_same_atomic_pair():
    english = _values(emit_tokens("library", "library", False, decompose("library", "library")))
    arabic = _ar_values(
        {
            "مكتبة": [{
                "root": "ك.ت.ب",
                "pattern": "مَ1ْ2َ3َة",
                "pos": "noun",
                "prc0": "0",
                "prc1": "0",
                "prc2": "0",
                "prc3": "0",
                "enc0": "0",
                "gen": "f",
                "num": "s",
            }],
        },
        "مكتبة",
    )
    assert english == arabic == ["ROOT:write", "ROLE:place"]


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


def test_cross_lingual_teacher_converges_to_same_atomic_pair():
    english = _values(emit_tokens("teacher", "teacher", False, decompose("teacher", "teacher")))
    arabic = _ar_values(
        {
            "معلم": [{
                "root": "ع.ل.م",
                "pattern": "مُ1َ2ّ3",
                "pos": "noun",
                "prc0": "0",
                "prc1": "0",
                "prc2": "0",
                "prc3": "0",
                "enc0": "0",
                "gen": "m",
                "num": "s",
            }],
        },
        "معلم",
    )
    assert english == arabic == ["ROOT:know", "ROLE:causer"]


def test_cross_lingual_hospital_converges_to_same_atomic_pair():
    english = _values(emit_tokens("hospital", "hospital", False, decompose("hospital", "hospital")))
    arabic = _ar_values(
        {
            "مستشفى": [{
                "root": "ش.ف.ي",
                "pattern": "مست123",
                "pos": "noun",
                "prc0": "0",
                "prc1": "0",
                "prc2": "0",
                "prc3": "0",
                "enc0": "0",
                "gen": "m",
                "num": "s",
            }],
        },
        "مستشفى",
    )
    assert english == arabic == ["ROOT:health", "ROLE:place"]
