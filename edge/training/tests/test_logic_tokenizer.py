"""Tests for edge/logic_tokenizer.py — closed-vocabulary logic tokenizer."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from logic_tokenizer import (  # type: ignore
    LOGIC_VOCAB,
    LOGIC_VOCAB_SIZE,
    LogicTokenizer,
)


# ───────────────────────── vocabulary ─────────────────────────

def test_vocab_size_is_closed():
    # 151 tokens exactly (6 specials + 15 operators + 8 quantifiers +
    # 18 relations + 10 time/modal + 10 roles + 30 concepts +
    # 20 arithmetic + 8 structure + 16 variables + 10 numbers)
    assert LOGIC_VOCAB_SIZE == 151


def test_specials_are_at_fixed_ids():
    assert LOGIC_VOCAB["[PAD]"] == 0
    assert LOGIC_VOCAB["[UNK]"] == 1
    assert LOGIC_VOCAB["[BOS]"] == 2
    assert LOGIC_VOCAB["[EOS]"] == 3


def test_no_duplicate_ids():
    ids = list(LOGIC_VOCAB.values())
    assert len(ids) == len(set(ids))
    assert set(ids) == set(range(LOGIC_VOCAB_SIZE))


def test_core_logical_tokens_present():
    for t in ("L:AND", "L:OR", "L:NOT", "L:IMPL", "L:IFF",
              "Q:ALL", "Q:SOME", "Q:EXISTS",
              "R:EQUALS", "R:IS", "R:HAS",
              "A:PLUS", "A:SIMPLIFY",
              "T:PAST", "T:FUTURE",
              "C:PERSON", "C:ACTION", "C:CONCEPT",
              "RO:AGENT", "RO:PATIENT",
              "N:ZERO", "N:ONE",
              "V:X", "V:Y"):
        assert t in LOGIC_VOCAB, t


# ───────────────────────── from_standard ──────────────────────

@pytest.fixture
def tk() -> LogicTokenizer:
    return LogicTokenizer()


def test_rel_if_then_maps_to_impl(tk):
    out = tk.from_standard(["REL:if", "ROOT:rain", "REL:then", "ROOT:earth"])
    assert out.count("L:IMPL") >= 1
    assert "[UNK]" not in out


def test_rel_all_maps_to_quantifier(tk):
    out = tk.from_standard(["REL:all", "CMP:person:agent", "REL:is", "ROOT:animal"])
    assert "Q:ALL" in out
    assert "C:PERSON" in out
    assert "RO:AGENT" in out
    assert "R:IS" in out
    assert "C:ANIMAL" in out


def test_str_negation_becomes_not(tk):
    for s in ("STR:neg:general", "STR:neg:past",
              "STR:neg:future", "STR:neg:nominal"):
        assert tk.from_standard([s]) == ["L:NOT"]


def test_str_conditional_becomes_impl(tk):
    for s in ("STR:cond:likely", "STR:cond:hypo", "STR:cond:counter"):
        assert tk.from_standard([s]) == ["L:IMPL"]


def test_legacy_str_labels_are_supported(tk):
    assert tk.from_standard(["STR:negation"]) == ["L:NOT"]
    assert tk.from_standard(["STR:condition"]) == ["L:IMPL"]
    assert tk.from_standard(["STR:past"]) == ["T:PAST"]


def test_feat_aspect_to_time(tk):
    assert tk.from_standard(["FEAT:asp:p"]) == ["T:PAST"]
    assert tk.from_standard(["FEAT:asp:i"]) == ["T:PRESENT"]
    assert tk.from_standard(["FEAT:asp:c"]) == ["T:PRESENT"]


def test_feat_other_dropped(tk):
    for f in ("FEAT:num:s", "FEAT:gen:m", "FEAT:per:3",
              "FEAT:case:n", "FEAT:mood:i", "FEAT:state:d"):
        assert tk.from_standard([f]) == []


def test_num_buckets(tk):
    assert tk.from_standard(["NUM:zero"]) == ["N:ZERO"]
    assert tk.from_standard(["NUM:one"]) == ["N:ONE"]
    assert tk.from_standard(["NUM:small"]) == ["N:SMALL"]
    assert tk.from_standard(["NUM:large"]) == ["N:LARGE"]
    assert tk.from_standard(["NUM:neg"]) == ["N:NEG"]
    # Unknown NUM subtype falls back to N:INT
    assert tk.from_standard(["NUM:unknown_subtype"]) == ["N:INT"]


def test_legacy_and_high_frequency_rel_tokens(tk):
    assert tk.from_standard(["REL:quant:all", "REL:quant:some", "REL:causes", "REL:more", "REL:less"]) == [
        "Q:ALL", "Q:SOME", "R:CAUSES", "R:GT", "R:LT",
    ]

    # Explicitly dropped relations should not become [UNK] in logic mode.
    assert tk.from_standard(["REL:to", "REL:for", "REL:with", "REL:on"]) == []


def test_time_prefix_collapses(tk):
    assert tk.from_standard(["TIME:month:1"]) == ["C:TIME_POINT"]
    assert tk.from_standard(["TIME:month:12"]) == ["C:TIME_POINT"]


def test_cmp_projects_concept_and_role(tk):
    # CMP:<field>:<role>
    assert tk.from_standard(["CMP:person:agent"]) == ["C:PERSON", "RO:AGENT"]
    assert tk.from_standard(["CMP:place:patient"]) == ["C:PLACE", "RO:PATIENT"]
    assert tk.from_standard(["CMP:move:agent"]) == ["C:ACTION", "RO:AGENT"]
    # Role with no projection drops to concept alone
    assert tk.from_standard(["CMP:person:instance"]) == ["C:PERSON"]
    # Unknown field falls to C:CONCEPT
    assert tk.from_standard(["CMP:nonexistent_field:agent"]) == [
        "C:CONCEPT", "RO:AGENT"
    ]


def test_root_projects_to_concept(tk):
    assert tk.from_standard(["ROOT:person"]) == ["C:PERSON"]
    assert tk.from_standard(["ROOT:animal"]) == ["C:ANIMAL"]
    assert tk.from_standard(["ROOT:unmapped_xyz"]) == ["C:CONCEPT"]


def test_raw_arabic_root_projects_via_field_map(tk):
    # كتب (k-t-b) resolves to field "write" which maps to action concept.
    assert tk.from_standard(["ROOT:ك.ت.ب"]) == ["C:ACTION"]


def test_role_and_pat_tokens_in_atomic_stream(tk):
    assert tk.from_standard(["ROLE:agent"]) == ["RO:AGENT"]
    assert tk.from_standard(["ROLE:quality"]) == []
    assert tk.from_standard(["PAT:1ا23"]) == []


def test_named_entity_and_foreign(tk):
    assert tk.from_standard(["NE:Cairo"]) == ["C:PERSON"]
    assert tk.from_standard(["FOREIGN:hello"]) == ["C:CONCEPT"]
    assert tk.from_standard(["LIT:@#$"]) == ["[UNK]"]


def test_specials_passthrough(tk):
    out = tk.from_standard(["[BOS]", "ROOT:person", "[EOS]"])
    assert out == ["[BOS]", "C:PERSON", "[EOS]"]


def test_adjacent_duplicates_collapse(tk):
    out = tk.from_standard(["REL:and", "REL:and", "ROOT:food"])
    assert out.count("L:AND") == 1

    out = tk.from_standard(["FEAT:asp:p", "FEAT:asp:p", "FEAT:asp:p"])
    assert out == ["T:PAST"]


def test_non_collapsible_tokens_preserved(tk):
    # C:PERSON is not in the collapsible set
    out = tk.from_standard(["ROOT:person", "ROOT:person"])
    assert out == ["C:PERSON", "C:PERSON"]


# ───────────────────────── from_formal ────────────────────────

def test_formal_unicode_quantifier(tk):
    out = tk.from_formal("\u2200x. P(x) \u2192 Q(x)")
    assert "Q:ALL" in out
    assert "L:IMPL" in out
    assert "V:X" in out
    assert out.count("L:LPAREN") == 2
    assert out.count("L:RPAREN") == 2


def test_formal_ascii_impl_and_ops(tk):
    out = tk.from_formal("(p && q) -> r")
    assert "L:AND" in out
    assert "L:IMPL" in out


def test_formal_arithmetic(tk):
    out = tk.from_formal("2*x + 3 = 7")
    assert "A:TIMES" in out
    assert "A:PLUS" in out
    assert "R:EQUALS" in out
    assert "V:X" in out


def test_formal_integer_buckets(tk):
    assert tk.from_formal("0") == ["N:ZERO"]
    assert tk.from_formal("1") == ["N:ONE"]
    assert tk.from_formal("42") == ["N:SMALL"]
    assert tk.from_formal("1000") == ["N:LARGE"]


def test_formal_keywords_english(tk):
    out = tk.from_formal("if p then q therefore r")
    assert "L:IMPL" in out
    assert "L:THEREFORE" in out


def test_formal_solve_keyword(tk):
    out = tk.from_formal("solve 2x + 3 = 7")
    assert "A:SOLVE" in out


def test_formal_iff(tk):
    assert "L:IFF" in tk.from_formal("p <-> q")
    assert "L:IFF" in tk.from_formal("p \u2194 q")


def test_formal_comparisons(tk):
    out = tk.from_formal("x >= 0 and y != 1")
    assert "R:GE" in out
    assert "R:NE" in out
    assert "L:AND" in out


# ───────────────────────── id conversion ──────────────────────

def test_to_ids_and_back(tk):
    toks = ["[BOS]", "Q:ALL", "V:X", "R:IS", "C:ANIMAL", "[EOS]"]
    ids = tk.to_ids(toks)
    assert all(isinstance(i, int) for i in ids)
    assert all(0 <= i < LOGIC_VOCAB_SIZE for i in ids)
    assert tk.to_tokens(ids) == toks


def test_unknown_token_to_unk(tk):
    assert tk.to_ids(["not-a-real-token"]) == [tk.unk_id]


def test_bos_eos_flag(tk):
    out = tk.from_standard(["ROOT:person"], add_bos_eos=True)
    assert out[0] == "[BOS]"
    assert out[-1] == "[EOS]"

    out = tk.from_formal("p and q", add_bos_eos=True)
    assert out[0] == "[BOS]"
    assert out[-1] == "[EOS]"


# ───────────────────────── determinism ────────────────────────

def test_vocab_is_stable_across_instances():
    a = LogicTokenizer()
    b = LogicTokenizer()
    assert a.vocab == b.vocab


def test_output_is_pure_function(tk):
    inputs = ["REL:if", "ROOT:rain", "REL:then", "ROOT:earth"]
    out1 = tk.from_standard(inputs)
    out2 = tk.from_standard(inputs)
    assert out1 == out2
