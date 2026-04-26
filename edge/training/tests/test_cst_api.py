"""Unit tests for edge/cst_api.py unified facade."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


EDGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EDGE))

from cst_api import CSTApi  # noqa: E402


class StubArabicTokenizer:
    def __init__(self) -> None:
        self.vocab = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[BOS]": 2,
            "[EOS]": 3,
            "ROOT:write": 10,
            "ROLE:agent": 11,
        }

    def tokenize(self, text: str) -> dict:
        _ = text
        return {
            "tokens": ["[BOS]", "ROOT:write", "ROLE:agent", "[EOS]"],
            "ids": [2, 10, 11, 3],
        }


class StubEnglishTokenizer:
    def tokenize(self, text: str) -> dict:
        _ = text
        return {
            "values": ["ROOT:write", "REL:to", "ROOT:person"],
            "coverage": {"ratio": 1.0},
        }


class StubLogicTokenizer:
    def __init__(self) -> None:
        self.vocab = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[BOS]": 2,
            "[EOS]": 3,
            "C:ACTION": 10,
            "RO:AGENT": 11,
            "L:IMPL": 12,
        }
        self._inv = {v: k for k, v in self.vocab.items()}

    def from_standard(self, tokens: list[str], *, add_bos_eos: bool = False) -> list[str]:
        out: list[str] = []
        if add_bos_eos:
            out.append("[BOS]")
        for t in tokens:
            if t == "ROOT:write":
                out.append("C:ACTION")
            elif t == "ROLE:agent":
                out.append("RO:AGENT")
            elif t in self.vocab:
                out.append(t)
            else:
                out.append("[UNK]")
        if add_bos_eos:
            out.append("[EOS]")
        return out

    def from_formal(self, text: str, *, add_bos_eos: bool = False) -> list[str]:
        _ = text
        out = ["L:IMPL"]
        if add_bos_eos:
            return ["[BOS]", *out, "[EOS]"]
        return out

    def to_ids(self, tokens: list[str]) -> list[int]:
        return [self.vocab.get(t, 1) for t in tokens]

    def to_tokens(self, ids: list[int]) -> list[str]:
        return [self._inv.get(i, "[UNK]") for i in ids]


@pytest.fixture
def api() -> CSTApi:
    return CSTApi(
        arabic_tokenizer=StubArabicTokenizer(),
        english_tokenizer=StubEnglishTokenizer(),
        logic_tokenizer=StubLogicTokenizer(),
    )


def test_standard_ar_strips_bos_eos_by_default(api: CSTApi) -> None:
    out = api.tokenize("test", lang="ar", level="standard")
    assert out["tokens"] == ["ROOT:write", "ROLE:agent"]
    assert out["ids"] == [10, 11]


def test_standard_ar_keeps_bos_eos_when_requested(api: CSTApi) -> None:
    out = api.tokenize("test", lang="ar", level="standard", add_bos_eos=True)
    assert out["tokens"] == ["[BOS]", "ROOT:write", "ROLE:agent", "[EOS]"]
    assert out["ids"] == [2, 10, 11, 3]


def test_standard_en_adds_bos_eos_when_requested(api: CSTApi) -> None:
    out = api.tokenize("test", lang="en", level="standard", add_bos_eos=True)
    assert out["tokens"] == ["[BOS]", "ROOT:write", "REL:to", "ROOT:person", "[EOS]"]
    assert out["ids"] is None


def test_logic_tokenize_from_text_projects_standard_tokens(api: CSTApi) -> None:
    out = api.tokenize("test", lang="ar", level="logic")
    assert out["tokens"] == ["C:ACTION", "RO:AGENT"]
    assert out["ids"] == [10, 11]


def test_project_rejects_unsupported_directions(api: CSTApi) -> None:
    with pytest.raises(ValueError):
        api.project(["ROOT:write"], source="logic", target="standard", lang="en")


def test_encode_logic_from_text_returns_logic_ids(api: CSTApi) -> None:
    ids = api.encode("test", lang="ar", level="logic")
    assert ids == [10, 11]


def test_encode_standard_english_requires_vocab_for_tokens(api: CSTApi) -> None:
    with pytest.raises(ValueError):
        api.encode(["ROOT:write"], lang="en", level="standard")


def test_encode_standard_english_with_vocab(api: CSTApi) -> None:
    ids = api.encode(
        ["ROOT:write", "ROOT:missing"],
        lang="en",
        level="standard",
        vocab={"[UNK]": 1, "ROOT:write": 50},
    )
    assert ids == [50, 1]


def test_decode_standard_english_accepts_both_vocab_shapes(api: CSTApi) -> None:
    # token -> id shape
    out_a = api.decode([50, 1], lang="en", level="standard", vocab={"ROOT:write": 50, "[UNK]": 1})
    assert out_a == ["ROOT:write", "[UNK]"]

    # id -> token shape
    out_b = api.decode([50, 1], lang="en", level="standard", vocab={50: "ROOT:write", 1: "[UNK]"})
    assert out_b == ["ROOT:write", "[UNK]"]


def test_untokenize_placeholder_raises(api: CSTApi) -> None:
    with pytest.raises(NotImplementedError):
        api.untokenize(["ROOT:write"], lang="en", level="standard")


def test_tokenize_formal_logic(api: CSTApi) -> None:
    out = api.tokenize_formal_logic("P -> Q", add_bos_eos=True)
    assert out["lang"] == "formal"
    assert out["level"] == "logic"
    assert out["tokens"] == ["[BOS]", "L:IMPL", "[EOS]"]
    assert out["ids"] == [2, 12, 3]
