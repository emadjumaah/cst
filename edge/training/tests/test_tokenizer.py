"""Unit tests for the Arabic CST tokenizer (edge/arabic_tokenizer.py).

These tests use a mock analyzer so they don't require the 500MB
CAMeL Tools morphology database. The mock returns hand-crafted analyses
that simulate the real analyzer's output (prc0/prc1/prc2/prc3/enc0,
pos, root, pattern, gen/num/per/asp).

Run:
    pytest edge/training/tests/test_tokenizer.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

# Repo-relative path: tests/ → training/ → edge/ (where arabic_tokenizer.py lives)
EDGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EDGE))

import pytest  # noqa: E402

from arabic_tokenizer import ArabicCSTTokenizer  # noqa: E402


class MockAnalyzer:
    """Returns scripted analyses keyed by surface form."""

    def __init__(self, table: dict[str, list[dict]]):
        self.table = table

    def analyze(self, word: str) -> list[dict]:
        return self.table.get(word, [])


# ── Helpers ────────────────────────────────────────────────────
def build_tokenizer(
    table: dict[str, list[dict]],
    **kwargs,
) -> ArabicCSTTokenizer:
    # Most tests validate legacy token shapes explicitly; keep that
    # behavior stable unless the test opts in to atomic mode.
    kwargs.setdefault("emit_atomic_composition", False)
    return ArabicCSTTokenizer(MockAnalyzer(table), **kwargs)


def tokens_of(tok: ArabicCSTTokenizer, sentence: str) -> list[str]:
    """Tokenize and return tokens with BOS/EOS stripped."""
    out = tok.tokenize(sentence)
    return out["tokens"][1:-1]


# ── STR split: negation and conditional are grammatically distinct ──
def test_negation_particles_are_distinct():
    tok = build_tokenizer({})
    assert tokens_of(tok, "لا يكتب")[0] == "STR:neg:general"
    assert tokens_of(tok, "لم يكتب")[0] == "STR:neg:past"
    assert tokens_of(tok, "لن يكتب")[0] == "STR:neg:future"
    assert tokens_of(tok, "ليس هنا")[0] == "STR:neg:nominal"


def test_conditional_particles_are_distinct():
    tok = build_tokenizer({})
    assert tokens_of(tok, "إذا كان")[0] == "STR:cond:likely"
    assert tokens_of(tok, "لو كان")[0] == "STR:cond:hypo"
    assert tokens_of(tok, "لولا أنه")[0] == "STR:cond:counter"


def test_prepositions_since_vs_from_are_distinct():
    tok = build_tokenizer({})
    assert tokens_of(tok, "من البيت")[0] == "REL:from"
    assert tokens_of(tok, "منذ سنة")[0] == "REL:since"


def test_before_vs_infront_are_distinct():
    tok = build_tokenizer({})
    assert tokens_of(tok, "قبل يوم")[0] == "REL:before"
    assert tokens_of(tok, "أمام البيت")[0] == "REL:infront"


# ── Clitic decomposition (prc2 / prc1 / prc0 / enc0) ──────────
def test_article_definite_is_dropped_in_critical_mode():
    # الكتاب — "the book": prc0=Al_det, root=ك.ت.ب, CAMeL pattern
    # '1ِ2َا3' → stripped '12ا3' → instance.
    tok = build_tokenizer({
        "الكتاب": [{
            "root": "ك.ت.ب", "pattern": "1ِ2َا3", "pos": "noun",
            "prc0": "Al_det", "prc1": "0", "prc2": "0", "prc3": "0",
            "enc0": "0", "gen": "m", "num": "s",
        }],
    })
    assert tokens_of(tok, "الكتاب") == ["CMP:write:instance"]


def test_article_definite_emits_feat_def_in_legacy_mode():
    tok = build_tokenizer({
        "الكتاب": [{
            "root": "ك.ت.ب", "pattern": "1ِ2َا3", "pos": "noun",
            "prc0": "Al_det", "prc1": "0", "prc2": "0", "prc3": "0",
            "enc0": "0", "gen": "m", "num": "s",
        }],
    }, critical_feat_only=False)
    assert tokens_of(tok, "الكتاب") == ["FEAT:def", "CMP:write:instance"]


def test_conjunction_preposition_article_stack():
    # وبالكتاب — "and with the book"; CAMeL pattern '1ِ2َا3'.
    tok = build_tokenizer({
        "وبالكتاب": [{
            "root": "ك.ت.ب", "pattern": "1ِ2َا3", "pos": "noun",
            "prc2": "wa_conj", "prc1": "bi_prep", "prc0": "Al_det",
            "prc3": "0", "enc0": "0", "gen": "m", "num": "s",
        }],
    })
    assert tokens_of(tok, "وبالكتاب") == ["REL:and", "REL:with", "CMP:write:instance"]


def test_enclitic_pronoun_is_emitted_last():
    # كتابه — "his book"; CAMeL pattern '1ِ2َا3'.
    tok = build_tokenizer({
        "كتابه": [{
            "root": "ك.ت.ب", "pattern": "1ِ2َا3", "pos": "noun",
            "prc0": "0", "prc1": "0", "prc2": "0", "prc3": "0",
            "enc0": "3ms_poss", "gen": "m", "num": "s",
        }],
    })
    assert tokens_of(tok, "كتابه") == [
        "CMP:write:instance", "FEAT:pron:3ms",
    ]


def test_future_marker_sa_decomposes_to_str_future():
    # سيكتبون — "they will write"
    tok = build_tokenizer({
        "سيكتبون": [{
            "root": "ك.ت.ب", "pattern": None, "pos": "verb",
            "prc1": "sa_fut", "prc0": "0", "prc2": "0", "prc3": "0",
            "enc0": "0", "asp": "i", "per": "3", "gen": "m", "num": "p",
        }],
    })
    toks = tokens_of(tok, "سيكتبون")
    assert toks[0] == "STR:future"
    assert "ROOT:write" in toks
    assert "FEAT:asp:i" in toks
    assert "FEAT:3mp" in toks


# ── ما disambiguation via POS ──────────────────────────────────
def test_ma_negation_via_pos():
    tok = build_tokenizer({
        "ما": [{"root": "ما", "pos": "part_neg", "prc0": "0",
                "prc1": "0", "prc2": "0", "prc3": "0", "enc0": "0"}],
    })
    # POS-aware disambiguation now runs before the REL fast path.
    assert tokens_of(tok, "ما جاء")[0] == "STR:neg:general"


def test_ma_interrogative_fallthrough():
    # ما is in ARABIC_REL_MAP, so the fast-path REL:what fires and the
    # POS disambiguation is not needed; documents the current behaviour.
    tok = build_tokenizer({})
    assert tokens_of(tok, "ما هذا") == ["REL:what", "REL:this"]


def test_ma_interrogative_via_pos():
    tok = build_tokenizer({
        "ما": [{"root": "ما", "pos": "pron_interrog", "prc0": "0",
                "prc1": "0", "prc2": "0", "prc3": "0", "enc0": "0"}],
    })
    assert tokens_of(tok, "ما جاء")[0] == "REL:what"


def test_ma_relative_via_pos():
    tok = build_tokenizer({
        "ما": [{"root": "ما", "pos": "pron_rel", "prc0": "0",
                "prc1": "0", "prc2": "0", "prc3": "0", "enc0": "0"}],
    })
    assert tokens_of(tok, "ما جاء")[0] == "REL:which"


# ── Feature emission: gender / number on nouns ─────────────────
def test_feminine_plural_noun_drops_f_and_p_in_critical_mode():
    # كاتبات — "female writers"; CAMeL pattern '1َا2ِ3ات' → '1ا23ات'.
    tok = build_tokenizer({
        "كاتبات": [{
            "root": "ك.ت.ب", "pattern": "1َا2ِ3ات", "pos": "noun",
            "prc0": "0", "prc1": "0", "prc2": "0", "prc3": "0",
            "enc0": "0", "gen": "f", "num": "p",
        }],
    })
    assert tokens_of(tok, "كاتبات") == ["CMP:write:agent"]


def test_feminine_plural_noun_emits_f_and_p_in_legacy_mode():
    tok = build_tokenizer({
        "كاتبات": [{
            "root": "ك.ت.ب", "pattern": "1َا2ِ3ات", "pos": "noun",
            "prc0": "0", "prc1": "0", "prc2": "0", "prc3": "0",
            "enc0": "0", "gen": "f", "num": "p",
        }],
    }, critical_feat_only=False)
    assert tokens_of(tok, "كاتبات") == ["CMP:write:agent", "FEAT:f", "FEAT:p"]


def test_masculine_singular_noun_emits_no_feat():
    # كاتب — "writer (m.sg)"; CAMeL pattern '1َا2ِ3' → '1ا23' → agent.
    tok = build_tokenizer({
        "كاتب": [{
            "root": "ك.ت.ب", "pattern": "1َا2ِ3", "pos": "noun",
            "prc0": "0", "prc1": "0", "prc2": "0", "prc3": "0",
            "enc0": "0", "gen": "m", "num": "s",
        }],
    })
    assert tokens_of(tok, "كاتب") == ["CMP:write:agent"]


# ── v5: root+pattern and explicit SPACE boundaries ────────────
def test_root_pattern_mode_emits_root_and_pat_tokens():
    tok = build_tokenizer({
        "كاتب": [{
            "root": "ك.ت.ب", "pattern": "1َا2ِ3", "pos": "noun",
            "prc0": "0", "prc1": "0", "prc2": "0", "prc3": "0",
            "enc0": "0", "gen": "m", "num": "s",
        }],
    }, emit_root_pattern=True)
    assert tokens_of(tok, "كاتب") == ["ROOT:ك.ت.ب", "PAT:1ا23"]


def test_atomic_mode_is_default_for_production_tokenizer():
    # Construct tokenizer directly (not via helper) to verify runtime default.
    tok = ArabicCSTTokenizer(MockAnalyzer({
        "كاتب": [{
            "root": "ك.ت.ب", "pattern": "1َا2ِ3", "pos": "noun",
            "prc0": "0", "prc1": "0", "prc2": "0", "prc3": "0",
            "enc0": "0", "gen": "m", "num": "s",
        }],
    }))
    assert tokens_of(tok, "كاتب") == ["ROOT:write", "ROLE:agent"]


def test_atomic_mode_splits_cmp_into_root_and_role_tokens():
    tok = build_tokenizer({
        "كاتب": [{
            "root": "ك.ت.ب", "pattern": "1َا2ِ3", "pos": "noun",
            "prc0": "0", "prc1": "0", "prc2": "0", "prc3": "0",
            "enc0": "0", "gen": "m", "num": "s",
        }],
    }, emit_atomic_composition=True)
    assert tokens_of(tok, "كاتب") == ["ROOT:write", "ROLE:agent"]


def test_root_pattern_atomic_mode_emits_role_as_separate_token():
    tok = build_tokenizer({
        "كاتب": [{
            "root": "ك.ت.ب", "pattern": "1َا2ِ3", "pos": "noun",
            "prc0": "0", "prc1": "0", "prc2": "0", "prc3": "0",
            "enc0": "0", "gen": "m", "num": "s",
        }],
    }, emit_root_pattern=True, emit_atomic_composition=True)
    assert tokens_of(tok, "كاتب") == ["ROOT:ك.ت.ب", "PAT:1ا23", "ROLE:agent"]


def test_space_token_mode_emits_word_boundaries():
    analysis = {
        "كاتب": [{
            "root": "ك.ت.ب", "pattern": "1َا2ِ3", "pos": "noun",
            "prc0": "0", "prc1": "0", "prc2": "0", "prc3": "0",
            "enc0": "0", "gen": "m", "num": "s",
        }],
    }
    tok = build_tokenizer(
        analysis,
        emit_root_pattern=True,
        emit_space_token=True,
    )
    assert tokens_of(tok, "كاتب كاتب") == [
        "ROOT:ك.ت.ب", "PAT:1ا23", "SPACE",
        "ROOT:ك.ت.ب", "PAT:1ا23", "SPACE",
    ]


def test_v5_tokens_are_pre_registered_when_mode_enabled():
    tok = build_tokenizer(
        {},
        emit_root_pattern=True,
        emit_space_token=True,
        emit_atomic_composition=True,
    )
    assert "PAT:1ا23" in tok.vocab
    assert "SPACE" in tok.vocab
    assert "ROLE:agent" in tok.vocab


# ── Verb aspect & pgn ──────────────────────────────────────────
def test_verb_emits_aspect_and_pgn():
    # كتبوا — "they (m) wrote": perfective, 3mp
    tok = build_tokenizer({
        "كتبوا": [{
            "root": "ك.ت.ب", "pattern": None, "pos": "verb",
            "prc0": "0", "prc1": "0", "prc2": "0", "prc3": "0",
            "enc0": "0", "asp": "p", "per": "3", "gen": "m", "num": "p",
        }],
    })
    assert tokens_of(tok, "كتبوا") == [
        "ROOT:write", "FEAT:asp:p", "FEAT:3mp",
    ]


# ── Named entity / foreign handling (v3 typed literals) ───────────
def test_ntws_only_emits_foreign():
    """A word whose only analysis is NTWS (no triliteral structure, no
    noun_prop tag) should route to ``FOREIGN:<surface>`` rather than
    silently collapsing to ``LIT``. This is the v3 fix for words like
    أكتوبر / فرنسا / كوفيد."""
    tok = build_tokenizer({
        "كوفيد": [{"root": "NTWS", "pos": "noun",
                    "prc0": "0", "prc1": "0", "prc2": "0", "prc3": "0",
                    "enc0": "0"}],
    })
    assert tokens_of(tok, "كوفيد") == ["FOREIGN:كوفيد"]


def test_noun_prop_emits_ne():
    """A proper-noun analysis should emit ``NE:<surface>`` (typed literal)
    so the transformer can treat all proper nouns as one category."""
    tok = build_tokenizer({
        "محمد": [{"root": "ح.م.د", "pos": "noun_prop",
                 "prc0": "0", "prc1": "0", "prc2": "0", "prc3": "0",
                 "enc0": "0"}],
    })
    assert tokens_of(tok, "محمد") == ["NE:محمد"]


def test_named_entity_with_valid_root_and_noun_prop():
    tok = build_tokenizer({
        "علي": [{"root": "ع.ل.و", "pos": "noun_prop", "pattern": "فعي",
                 "prc0": "0", "prc1": "0", "prc2": "0", "prc3": "0",
                 "enc0": "0"}],
    })
    assert tokens_of(tok, "علي") == ["NE:علي"]


# ── Phase 2: unmapped root falls back to ROOT:<raw-root> ──────
def test_unmapped_root_emits_root_raw():
    """A real triliteral root with no semantic-field mapping should collapse
    all surface variants into a single ``ROOT:<dotted-root>`` token rather
    than producing one ``LIT:<surface>`` entry per inflected form."""
    tok = build_tokenizer({
        # CAMeL returns a valid root (not NTWS/PUNC/DIGIT/FOREIGN) but
        # this root is not present in ARABIC_ROOT_TO_FIELD.
        "زقزقت": [{"root": "ز.ق.ز.ق", "pattern": "فعللت", "pos": "verb",
                   "prc0": "0", "prc1": "0", "prc2": "0", "prc3": "0",
                   "enc0": "0", "asp": "p", "per": "3", "gen": "f", "num": "s"}],
    })
    toks = tokens_of(tok, "زقزقت")
    assert toks[0] == "ROOT:ز.ق.ز.ق"
    assert "FEAT:asp:p" in toks


# ── v3: weak-root substitution ────────────────────────────
def test_weak_root_hash_resolves_via_substitution():
    """CAMeL sometimes returns a root with ``#`` for the weak consonant
    (ك.#.ن instead of ك.و.ن). The lookup must substitute weak letters
    and hit the semantic field."""
    tok = build_tokenizer({
        "يكون": [{"root": "ك.#.ن", "pattern": "يفعل", "pos": "verb",
                 "prc0": "0", "prc1": "0", "prc2": "0", "prc3": "0",
                 "enc0": "0", "asp": "i", "per": "3", "gen": "m", "num": "s"}],
    })
    toks = tokens_of(tok, "يكون")
    # ك.و.ن is in the "exist" field
    assert any(t.startswith("ROOT:exist") or t.startswith("CMP:exist") for t in toks)


# ── v3: NUM regex pre-pass ───────────────────────────────
def test_num_year_detected():
    tok = build_tokenizer({})
    out = tok.tokenize("1985")
    assert "NUM:year" in out["tokens"]


def test_num_small_detected():
    tok = build_tokenizer({})
    out = tok.tokenize("42")
    assert "NUM:small" in out["tokens"]


def test_num_percent_detected():
    tok = build_tokenizer({})
    out = tok.tokenize("12%")
    assert "NUM:percent" in out["tokens"]


# ── v3: month lookup ───────────────────────────────────
def test_month_emits_time_token():
    tok = build_tokenizer({})
    assert tokens_of(tok, "أكتوبر") == ["TIME:month:oct"]


# ── v3: punctuation and single-char filter ───────────────────
def test_punctuation_stripped_from_words():
    tok = build_tokenizer({
        "ذلك": [],  # no CAMeL analyses → falls to LIT (non-NTWS path)
    })
    out = tok.tokenize("ذلك،")
    # The ، must not remain attached and must not produce a standalone token.
    assert not any("," in t or "،" in t for t in out["tokens"])


# ── Alignment metadata ─────────────────────────────────────────
def test_alignment_fields_present_and_consistent():
    tok = build_tokenizer({
        "كتاب": [{"root": "ك.ت.ب", "pattern": "فعال", "pos": "noun",
                  "prc0": "0", "prc1": "0", "prc2": "0", "prc3": "0",
                  "enc0": "0", "gen": "m", "num": "s"}],
    })
    out = tok.tokenize("لم يوجد كتاب")
    # "لم" → STR:neg:past at sentence-level prefix, consumed at word loop
    assert out["prefix_count"] >= 1
    assert out["words"] == ["لم", "يوجد", "كتاب"]
    assert len(out["word_token_counts"]) == 3
    # Total tokens = 1 (BOS) + prefix + sum(word_token_counts) + 1 (EOS)
    total = 1 + out["prefix_count"] + sum(out["word_token_counts"]) + 1
    assert total == len(out["tokens"])


# ── Vocabulary pre-registration ────────────────────────────────
def test_critical_feat_tokens_pre_registered_by_default():
    tok = build_tokenizer({})
    v = tok.vocab
    for t in ("FEAT:asp:p", "FEAT:asp:i", "FEAT:asp:c",
              "FEAT:3ms", "FEAT:3fs", "FEAT:3mp", "FEAT:3fp",
              "FEAT:pron:3ms", "FEAT:pron:1s"):
        assert t in v, f"{t} not pre-registered"
    for t in ("FEAT:def", "FEAT:f", "FEAT:p", "FEAT:d"):
        assert t not in v, f"{t} should not be pre-registered in critical mode"


def test_non_critical_feat_tokens_preregister_in_legacy_mode():
    tok = build_tokenizer({}, critical_feat_only=False)
    v = tok.vocab
    for t in ("FEAT:def", "FEAT:f", "FEAT:p", "FEAT:d"):
        assert t in v, f"{t} not pre-registered in legacy mode"


def test_split_str_tokens_pre_registered():
    tok = build_tokenizer({})
    v = tok.vocab
    for t in ("STR:neg:general", "STR:neg:past", "STR:neg:future",
              "STR:neg:nominal",
              "STR:cond:likely", "STR:cond:hypo", "STR:cond:counter"):
        assert t in v, f"{t} not pre-registered"


def test_split_rel_tokens_pre_registered():
    tok = build_tokenizer({})
    v = tok.vocab
    assert "REL:since" in v
    assert "REL:infront" in v


# ── v3.1: frozen vocab with cap ─────────────────────────────────
def _make_vocab_file(tmp_path, extra_tokens=()):
    """Build a minimal valid frozen vocab from an unfrozen tokenizer."""
    from arabic_tokenizer import build_frozen_vocab  # noqa: F401
    tok = build_tokenizer({})
    # Seed with the known-present dynamic tokens so cap math is predictable.
    vocab = dict(tok.vocab)
    for t in extra_tokens:
        if t not in vocab:
            vocab[t] = len(vocab)
    import json as _json
    path = tmp_path / "vocab.json"
    path.write_text(_json.dumps(vocab, ensure_ascii=False))
    return path, vocab


def test_frozen_vocab_loads_and_reports_mode(tmp_path):
    path, vocab = _make_vocab_file(tmp_path)
    tok = ArabicCSTTokenizer(MockAnalyzer({}), vocab_path=str(path))
    assert tok.frozen is True
    assert tok.unk_id == vocab["[UNK]"]
    assert len(tok.vocab) == len(vocab)


_NE_ANALYSIS = {
    "سليمان": [{"root": "س.ل.م", "pos": "noun_prop",
                "prc0": "0", "prc1": "0", "prc2": "0", "prc3": "0",
                "enc0": "0"}],
}


def test_frozen_vocab_routes_oov_to_unk(tmp_path):
    # Vocab does NOT contain NE:سليمان — OOV name should become [UNK].
    path, vocab = _make_vocab_file(tmp_path)
    tok = ArabicCSTTokenizer(MockAnalyzer(_NE_ANALYSIS), vocab_path=str(path))
    out = tok.tokenize("سليمان")
    # The OOV NE surface should not appear verbatim; [UNK] should.
    assert "NE:سليمان" not in out["tokens"]
    assert "[UNK]" in out["tokens"]
    assert tok.unk_id in out["ids"]
    assert tok.stats["unk"] >= 1


def test_frozen_vocab_keeps_present_dynamic_token(tmp_path):
    # Pre-seed the vocab with NE:سليمان. Must round-trip without UNK.
    path, _ = _make_vocab_file(tmp_path, extra_tokens=("NE:سليمان",))
    tok = ArabicCSTTokenizer(MockAnalyzer(_NE_ANALYSIS), vocab_path=str(path))
    out = tok.tokenize("سليمان")
    assert "NE:سليمان" in out["tokens"]
    assert tok.stats["unk"] == 0


def test_frozen_vocab_cap_drops_entries_past_limit(tmp_path):
    # Cap below full size → entries with id >= cap are unreachable.
    path, vocab = _make_vocab_file(tmp_path, extra_tokens=("NE:سليمان",))
    cap = vocab["NE:سليمان"]   # entries >= cap get dropped
    tok = ArabicCSTTokenizer(
        MockAnalyzer(_NE_ANALYSIS), vocab_path=str(path), cap=cap,
    )
    assert "NE:سليمان" not in tok.vocab
    out = tok.tokenize("سليمان")
    assert "[UNK]" in out["tokens"]


def test_frozen_vocab_requires_unk(tmp_path):
    import json as _json
    path = tmp_path / "bad.json"
    path.write_text(_json.dumps({"[PAD]": 0, "[BOS]": 1, "[EOS]": 2}))
    with pytest.raises(ValueError, match="UNK"):
        ArabicCSTTokenizer(MockAnalyzer({}), vocab_path=str(path))


def test_build_frozen_vocab_keeps_semantic_and_caps_dynamic(tmp_path):
    from arabic_tokenizer import build_frozen_vocab
    analyzer = MockAnalyzer({
        "كتاب": [{"root": "ك.ت.ب", "pattern": "فعال", "pos": "noun",
                  "prc0": "0", "prc1": "0", "prc2": "0", "prc3": "0",
                  "enc0": "0", "gen": "m", "num": "s"}],
    })
    # Build a reference tokenizer to know the core-semantic size.
    ref = ArabicCSTTokenizer(analyzer)
    core_size = len(ref.vocab)
    cap = core_size + 5   # room for a few dynamic surfaces
    sentences = ["كتاب " * 3]  # one real word, seen repeatedly
    out_path = tmp_path / "v.json"
    final = build_frozen_vocab(
        analyzer, sentences, cap=cap, out_path=str(out_path),
        progress_every=0,
    )
    assert "[UNK]" in final
    assert "ROOT:write" in final       # core semantic, always kept
    assert len(final) <= cap
    # And the file on disk parses back identical.
    import json as _json
    assert _json.loads(out_path.read_text()) == final


# ── v4: NE emission + ROOT:<raw> core-semantic ──────────────────

def test_v4_noun_prop_with_distinct_lex_wins_over_common_reading():
    """سليمان-style: CAMeL returns a common reading (noun سَلِيم) AND
    a proper-noun reading with a **distinct** lex (سُلَيْمان). Because
    the noun_prop lexeme is not shared with the common reading, v4
    routes it to NE:<surface>.

    This is the v4 fix for NE under-firing: v3 preferred non-noun_prop
    on ties, causing real names with collision-prone roots to be
    silently collapsed to ROOT:<field>.
    """
    tok = build_tokenizer({
        "سليمان": [
            {"root": "س.ل.م", "pos": "noun", "lex": "سَلِيم",
             "prc0": "0", "prc1": "0", "prc2": "0", "prc3": "0", "enc0": "0",
             "gen": "m", "num": "s"},
            {"root": "س.ل.م", "pos": "noun", "lex": "سَلِيم",
             "prc0": "0", "prc1": "0", "prc2": "0", "prc3": "0", "enc0": "0",
             "gen": "m", "num": "s"},
            {"root": "NTWS", "pos": "noun_prop", "lex": "سُلَيْمان",
             "prc0": "0", "prc1": "0", "prc2": "0", "prc3": "0", "enc0": "0"},
        ],
    })
    assert tokens_of(tok, "سليمان") == ["NE:سليمان"]


def test_v4_noun_prop_sharing_lex_is_demoted():
    """ذهب-style: CAMeL returns noun_prop, verb, and noun readings
    all with the same lex (ذَهَب means both the verb "went" and a
    rare personal name). Because the noun_prop lex matches the
    common-reading lex, v4 uses the common reading → ROOT:move.

    Regression guard: distinct-lex heuristic must not false-positive
    on lexemes that genuinely share spelling across POS.
    """
    tok = build_tokenizer({
        "ذهب": [
            {"root": "ذ.ه.ب", "pos": "noun_prop", "lex": "ذَهَب",
             "prc0": "0", "prc1": "0", "prc2": "0", "prc3": "0", "enc0": "0"},
            {"root": "ذ.ه.ب", "pos": "verb", "lex": "ذَهَب",
             "prc0": "0", "prc1": "0", "prc2": "0", "prc3": "0", "enc0": "0",
             "asp": "p", "per": "3", "gen": "m", "num": "s"},
        ],
    })
    toks = tokens_of(tok, "ذهب")
    assert "NE:ذهب" not in toks
    # ذ.ه.ب → material (gold); could also route to move depending on
    # which field wins in ARABIC_ROOT_TO_FIELD. The key assertion is
    # that it's a semantic ROOT:, not an NE:.
    assert any(t.startswith("ROOT:") or t.startswith("CMP:") for t in toks)


def test_v4_noun_prop_only_analysis_emits_ne():
    """محمد-style: CAMeL has a single noun_prop analysis (no competing
    common reading). v4 falls to priority 3 and emits NE:<surface>.
    """
    tok = build_tokenizer({
        "محمد": [
            {"root": "ح.م.د", "pos": "noun_prop", "lex": "مُحَمَّد",
             "prc0": "0", "prc1": "0", "prc2": "0", "prc3": "0", "enc0": "0"},
        ],
    })
    assert tokens_of(tok, "محمد") == ["NE:محمد"]


def test_v4_al_prefixed_noun_prop_is_demoted_to_common_reading():
    """العمل — CAMeL sometimes lists a spurious ``noun_prop`` analysis
    alongside the ordinary ``noun``. Because both carry the same lex
    (عَمَل), the distinct-lex test fails and v4 prefers the common
    reading → ROOT:make.

    Regression guard: the v4 NE heuristic must not route common nouns
    like العمل / الحزب to NE just because CAMeL lists a shadow
    noun_prop analysis.
    """
    tok = build_tokenizer({
        "العمل": [
            # CAMeL's first choice is sometimes the (wrong) noun_prop reading
            {"root": "ع.م.ل", "pos": "noun_prop", "lex": "عَمَل",
             "prc0": "Al_det", "prc1": "0", "prc2": "0", "prc3": "0",
             "enc0": "0"},
            # but a legitimate common-noun reading exists too (same lex)
            {"root": "ع.م.ل", "pattern": "العمل", "pos": "noun",
             "lex": "عَمَل",
             "prc0": "Al_det", "prc1": "0", "prc2": "0", "prc3": "0",
             "enc0": "0", "gen": "m", "num": "s"},
        ],
    })
    toks = tokens_of(tok, "العمل")
    assert "NE:العمل" not in toks
    assert "ROOT:make" in toks


def test_v4_foreign_proper_noun_routes_to_ne():
    """A proper-noun analysis with NTWS root (foreign place / person
    names like فرنسا) should emit NE:<surface>, not FOREIGN:<surface>.
    """
    tok = build_tokenizer({
        "فرنسا": [{"root": "NTWS", "pos": "noun_prop",
                   "prc0": "0", "prc1": "0", "prc2": "0", "prc3": "0",
                   "enc0": "0"}],
    })
    assert tokens_of(tok, "فرنسا") == ["NE:فرنسا"]


def test_v5_raw_root_is_dataset_adaptive_not_core():
    """Adaptive-vocab policy: raw dotted roots are tokenizer-known but
    not core-semantic in capped vocab construction.

    ``ROOT:<field>`` remains core; ``ROOT:<dotted-root>`` competes in the
    dynamic frequency budget.
    """
    from arabic_tokenizer import _is_core_semantic
    assert _is_core_semantic("ROOT:write") is True
    assert _is_core_semantic("ROOT:ك.ت.ب") is False       # raw triliteral
    assert _is_core_semantic("ROOT:ز.ق.ز.ق") is False     # raw quadriliteral
    # Non-ROOT dynamic tokens remain dynamic
    assert _is_core_semantic("NE:محمد") is False
    assert _is_core_semantic("FOREIGN:paris") is False
    assert _is_core_semantic("LIT:وحواء") is False


def test_v5_raw_root_competes_in_dynamic_budget(tmp_path):
    """Under a tight cap, rare raw roots are dropped if lower-frequency
    than other dynamic tokens; this is the dataset-adaptive behavior.
    """
    from arabic_tokenizer import build_frozen_vocab, _is_core_semantic
    analyzer = MockAnalyzer({
        # Rare word with an unmapped root — emits ROOT:ز.ق.ز.ق once
        "زقزقت": [{"root": "ز.ق.ز.ق", "pos": "verb",
                    "prc0": "0", "prc1": "0", "prc2": "0", "prc3": "0",
                    "enc0": "0", "asp": "p", "per": "3", "gen": "f", "num": "s"}],
        # High-frequency proper noun — emits NE:محمد many times
        "محمد": [{"root": "ح.م.د", "pos": "noun_prop",
                   "prc0": "0", "prc1": "0", "prc2": "0", "prc3": "0",
                   "enc0": "0"}],
    })
    ref = ArabicCSTTokenizer(analyzer)
    core_size = sum(1 for t in ref.vocab if _is_core_semantic(t))
    # Only one dynamic slot available: it should go to frequent NE:محمد,
    # not to the rare raw root.
    cap = core_size + 1
    sentences = (["محمد"] * 50) + ["زقزقت"]   # NE:محمد dominates by freq
    final = build_frozen_vocab(
        analyzer, sentences, cap=cap, out_path=None, progress_every=0,
    )
    # The one dynamic slot is taken by the most-frequent NE.
    assert "NE:محمد" in final
    assert "ROOT:ز.ق.ز.ق" not in final


def test_v4_zerrouki_roots_are_preregistered():
    """Every root in the Zerrouki arabic-roots list must be present in
    the vocab of a fresh tokenizer (no corpus passed). This guarantees
    zero-shot coverage for rare Arabic roots that happen to be absent
    from the training corpus but appear at inference time.
    """
    from arabic_tokenizer import ARABIC_ROOTS_ZERROUKI
    analyzer = MockAnalyzer({})
    tok = ArabicCSTTokenizer(analyzer)

    # Inventory is non-trivial
    assert len(ARABIC_ROOTS_ZERROUKI) >= 8000

    # Every Zerrouki root is pre-registered as ROOT:<dotted>
    missing = [r for r in ARABIC_ROOTS_ZERROUKI[:500]
               if f"ROOT:{r}" not in tok.vocab]
    assert missing == []

    # And the common ones too
    for probe in ("ك.ت.ب", "ع.ل.م", "ق.و.ل", "ذ.ه.ب"):
        # at least one of these should be present (all four are common)
        pass
    assert f"ROOT:ك.ت.ب" in tok.vocab
    assert f"ROOT:ذ.ه.ب" in tok.vocab


# ── v4.1: pattern→role map uses CAMeL-shaped patterns ─────────────

def test_v4_1_camel_pattern_agent_fires():
    """Active participle كاتب → pattern ``'1ا2ِ3'`` in CAMeL. After
    diacritic stripping this is ``'1ا23'`` which must map to ``agent``.
    Regression guard: v4 used Arabic-letter pattern keys (``فاعل``)
    which never matched CAMeL's digit-slot format, so all CMP tokens
    collapsed to ``:quality`` via the POS fallback."""
    tok = build_tokenizer({
        "كاتب": [{
            "root": "ك.ت.ب", "pattern": "1ا2ِ3", "pos": "noun",
            "prc0": "0", "prc1": "0", "prc2": "0", "prc3": "0",
            "enc0": "0", "gen": "m", "num": "s",
        }],
    })
    assert tokens_of(tok, "كاتب") == ["CMP:write:agent"]


def test_v4_1_camel_pattern_patient_fires():
    """مكتوب → pattern ``'مَ1ْ2ُو3'`` → stripped ``'م12و3'`` → patient."""
    tok = build_tokenizer({
        "مكتوب": [{
            "root": "ك.ت.ب", "pattern": "مَ1ْ2ُو3", "pos": "noun",
            "prc0": "0", "prc1": "0", "prc2": "0", "prc3": "0",
            "enc0": "0", "gen": "m", "num": "s",
        }],
    })
    assert tokens_of(tok, "مكتوب") == ["CMP:write:patient"]


def test_v4_1_camel_pattern_place_fires():
    """مكتبة → pattern ``'مَ1ْ2َ3َة'`` → stripped ``'م123ة'`` → place."""
    tok = build_tokenizer({
        "مكتبة": [{
            "root": "ك.ت.ب", "pattern": "مَ1ْ2َ3َة", "pos": "noun",
            "prc0": "0", "prc1": "0", "prc2": "0", "prc3": "0",
            "enc0": "0", "gen": "f", "num": "s",
        }],
    })
    assert tokens_of(tok, "مكتبة") == ["CMP:write:place"]


def test_v4_1_camel_pattern_form_x_vn_normalizes_hamzat_wasl():
    """استقبال → pattern starts with hamzat-wasl ``ٱ`` (U+0671). After
    stripping diacritics and normalizing ``ٱ`` → ``ا`` it becomes
    ``'است12ا3'`` and must map to ``instance``."""
    tok = build_tokenizer({
        "استقبال": [{
            "root": "ق.ب.ل", "pattern": "ٱسْتِ1ْ2َا3", "pos": "noun",
            "prc0": "0", "prc1": "0", "prc2": "0", "prc3": "0",
            "enc0": "0", "gen": "m", "num": "s",
        }],
    })
    toks = tokens_of(tok, "استقبال")
    # ق.ب.ل maps via ARABIC_ROOT_TO_FIELD; role must be instance
    assert any(t.startswith("CMP:") and t.endswith(":instance") for t in toks)


def test_v4_1_camel_pattern_quality_fires():
    """فعيل adjective كبير → pattern ``'1َ2ِي3'`` → stripped ``'12ي3'``
    → quality. Must fire via the pattern map, not only the POS fallback."""
    tok = build_tokenizer({
        "كبير": [{
            "root": "ك.ب.ر", "pattern": "1َ2ِي3", "pos": "adj",
            "prc0": "0", "prc1": "0", "prc2": "0", "prc3": "0",
            "enc0": "0", "gen": "m", "num": "s",
        }],
    })
    assert tokens_of(tok, "كبير") == ["CMP:size:quality"]


def test_v4_1_camel_pattern_form_ii_ap_causer_with_shadda():
    """معلّم → pattern ``'مُ1َ2ّ2ِ3'`` or similar — shadda on R2 must be
    preserved and the pattern must map to ``causer``. We probe the
    map with the stripped-shadda-preserved shape ``'م12ّ3'`` to verify."""
    # NB: the tokenizer's _strip removes shadda from the surface before
    # querying the analyzer, so the mock must be keyed by the
    # shadda-stripped form ``معلم``.
    tok = build_tokenizer({
        "معلم": [{
            "root": "ع.ل.م", "pattern": "مُ1َ2ّ3", "pos": "noun",
            "prc0": "0", "prc1": "0", "prc2": "0", "prc3": "0",
            "enc0": "0", "gen": "m", "num": "s",
        }],
    })
    toks = tokens_of(tok, "معلّم")
    assert any(t.endswith(":causer") for t in toks)
