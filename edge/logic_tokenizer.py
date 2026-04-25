"""CST Logic tokenizer — closed ~150-token inventory.

This is the **reasoning-level** tokenizer. It is language-independent:
given a semantically equivalent sentence in Arabic or English, the same
logic-token sequence should emerge.

Design contract
---------------

- **Closed vocabulary.** The inventory is fixed in ``edge/logic/vocab/``
  (specials, operators, quantifiers, relations, time/modal, roles,
  concepts, arithmetic, structural, variables, numbers). Anything that
  is not in the inventory is ``[UNK]``. No dynamic surfaces.
- **Two input modes.**
    1. *CST-standard stream.* Pass a list of CST-standard tokens
       (output of ``edge/arabic_tokenizer.py`` or the future English
       standard tokenizer). The logic tokenizer projects each one to
       a logic token or drops it.
    2. *Raw text.* Convenience path that takes plain text, runs the
       CST-standard tokenizer behind the scenes, and projects.
- **Deterministic.** No randomness, no side effects.
- **Separate model.** The trained logic model is a separate
  small transformer (~1-5M params) trained from scratch on
  logic-tokenized reasoning data. It is not a fine-tune of the
  CST-standard model.

See ``docs/spec/cst-logic-spec.md`` for the full spec.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

try:
    from arabic_tokenizer import ARABIC_ROOT_TO_FIELD
except Exception:
    # Logic tokenizer can still run for formal/English paths without Arabic tables.
    ARABIC_ROOT_TO_FIELD = {}

# ────────────────────────────────────────────────────────────────
# Load closed vocabulary
# ────────────────────────────────────────────────────────────────

_HERE = Path(__file__).resolve().parent
_VOCAB_DIR = _HERE / "logic" / "vocab"


def _load_vocab() -> dict[str, int]:
    """Load all vocab JSONs in a deterministic order."""
    vocab: dict[str, int] = {}

    # 1. Specials — list
    for t in json.loads((_VOCAB_DIR / "specials.json").read_text()):
        vocab[t] = len(vocab)

    # 2. Remaining buckets — dict (token → description); keys sorted so
    # the vocab ID assignment is reproducible regardless of JSON
    # pretty-printing choices.
    for name in ("operators", "quantifiers", "relations", "time_modal",
                 "roles", "concepts", "arithmetic", "structure",
                 "variables", "numbers"):
        data = json.loads((_VOCAB_DIR / f"{name}.json").read_text())
        for key in sorted(data):
            if key in vocab:
                raise RuntimeError(f"duplicate logic token: {key}")
            vocab[key] = len(vocab)
    return vocab


LOGIC_VOCAB: dict[str, int] = _load_vocab()
LOGIC_VOCAB_SIZE: int = len(LOGIC_VOCAB)

PAD_ID = LOGIC_VOCAB["[PAD]"]
UNK_ID = LOGIC_VOCAB["[UNK]"]
BOS_ID = LOGIC_VOCAB["[BOS]"]
EOS_ID = LOGIC_VOCAB["[EOS]"]


# ────────────────────────────────────────────────────────────────
# CST-standard → logic projection
# ────────────────────────────────────────────────────────────────

# Standard REL:* surface-form ⇒ logic token. Driven by the REL map in
# ``edge/arabic_tokenizer.py`` and the English default addendum.
_STD_REL_TO_LOGIC: dict[str, str | None] = {
    # Logical connectives
    "REL:and":       "L:AND",
    "REL:or":        "L:OR",
    "REL:but":       "L:AND",        # "but" is conjunction for truth-preservation
    "REL:not":       "L:NOT",
    "REL:neither":   "L:NOT",
    "REL:neg":       "L:NOT",        # legacy English path
    # Conditionals
    "REL:if":        "L:IMPL",
    "REL:then":      "L:IMPL",
    "REL:iff":       "L:IFF",
    # Conclusion / cause
    "REL:therefore": "L:THEREFORE",
    "REL:because":   "L:BECAUSE",
    "REL:so":        "L:THEREFORE",
    # Quantifier surfaces that sometimes arrive as REL
    "REL:all":       "Q:ALL",
    "REL:every":     "Q:ALL",
    "REL:each":      "Q:ALL",
    "REL:quant:all": "Q:ALL",        # legacy English path
    "REL:some":      "Q:SOME",
    "REL:any":       "Q:SOME",
    "REL:quant:some": "Q:SOME",      # legacy English path
    "REL:no":        "Q:NO",
    "REL:none":      "Q:NONE",
    "REL:most":      "Q:MOST",
    "REL:few":       "Q:FEW",
    "REL:many":      "Q:SOME",
    "REL:several":   "Q:SOME",
    "REL:both":      "Q:ALL",
    "REL:only":      "Q:UNIQUE",
    # Temporal / spatial that survive to logic
    "REL:before":    "R:BEFORE",
    "REL:after":     "R:AFTER",
    "REL:in":        "R:IN",
    "REL:now":       "T:PRESENT",
    "REL:maybe":     "M:MAY",
    # Possession / attribute
    "REL:has":       "R:HAS",
    "REL:of":        "R:PART_OF",
    "REL:is":        "R:IS",
    "REL:causes":    "R:CAUSES",
    "REL:more":      "R:GT",
    "REL:less":      "R:LT",

    # High-frequency relation surfaces with weak/no logic semantics.
    # Drop these explicitly to reduce avoidable [UNK] inflation.
    "REL:to":        None,
    "REL:for":       None,
    "REL:as":        None,
    "REL:with":      None,
    "REL:on":        None,
    "REL:from":      None,
    "REL:at":        None,
    "REL:about":     None,
    "REL:above":     None,
    "REL:across":    None,
    "REL:against":   None,
    "REL:almost":    None,
    "REL:also":      None,
    "REL:around":    None,
    "REL:behind":    None,
    "REL:between":   None,
    "REL:contrast":  None,
    "REL:emphasis":  None,
    "REL:except":    None,
    "REL:infront":   None,
    "REL:instead":   None,
    "REL:like":      None,
    "REL:since":     None,
    "REL:these":     None,
    "REL:this":      None,
    "REL:those":     None,
    "REL:through":   None,
    "REL:under":     None,
    "REL:unlike":    None,
    "REL:until":     None,
    "REL:what":      None,
    "REL:when":      None,
    "REL:where":     None,
    "REL:which":     None,
    "REL:who":       None,
    "REL:within":    None,
    "REL:without":   None,
}

# STR:* — sentence-structural markers.
_STD_STR_TO_LOGIC: dict[str, str | None] = {
    "STR:neg:general": "L:NOT",
    "STR:neg:past":    "L:NOT",
    "STR:neg:future":  "L:NOT",
    "STR:neg:nominal": "L:NOT",
    "STR:negation":    "L:NOT",      # legacy corpus label
    "STR:cond:likely":  "L:IMPL",
    "STR:cond:hypo":    "L:IMPL",
    "STR:cond:counter": "L:IMPL",
    "STR:condition":    "L:IMPL",    # legacy corpus label
    "STR:future":      "T:FUTURE",
    "STR:past":        "T:PAST",
    "STR:question":    "[Q]",
    "STR:emphasis":    None,  # drop — not truth-conditional
}

# FEAT:* — inflection markers. All dropped at logic level.
# Exception: aspectual time collapses to a time token.
_STD_FEAT_TO_LOGIC: dict[str, str | None] = {
    "FEAT:asp:p": "T:PAST",
    "FEAT:asp:i": "T:PRESENT",
    "FEAT:asp:c": "T:PRESENT",  # imperative collapses to present for logic
}

# NUM:* — numeric buckets.
_STD_NUM_TO_LOGIC: dict[str, str] = {
    "NUM:zero":    "N:ZERO",
    "NUM:one":     "N:ONE",
    "NUM:small":   "N:SMALL",
    "NUM:large":   "N:LARGE",
    "NUM:year":    "N:LARGE",
    "NUM:percent": "N:SMALL",
    "NUM:neg":     "N:NEG",
    "NUM:frac":    "N:FRAC",
    "NUM:real":    "N:REAL",
}

# TIME:month:* — all collapse to a single time-point marker.
# (Specific month identity is not truth-conditional in logic tasks.)
_STD_TIME_PREFIX = "TIME:"

# Semantic-field → concept bucket. Driven by the 55 CST fields in
# ``ARABIC_ROOT_TO_FIELD``; fields not listed here map to C:CONCEPT.
_FIELD_TO_CONCEPT: dict[str, str] = {
    # People / social
    "person":   "C:PERSON",
    "family":   "C:PERSON",
    "body":     "C:OBJECT",
    "social":   "C:GROUP",
    "social_g": "C:GROUP",
    # Places
    "place":    "C:PLACE",
    "nature":   "C:PLACE",
    # Actions / processes
    "move":     "C:ACTION",
    "make":     "C:ACTION",
    "create":   "C:ACTION",
    "destroy":  "C:ACTION",
    "fight":    "C:ACTION",
    "build":    "C:ACTION",
    "change":   "C:PROCESS",
    "enable":   "C:PROCESS",
    "give":     "C:ACTION",
    "take":     "C:ACTION",
    "gather":   "C:ACTION",
    "send":     "C:ACTION",
    "speak":    "C:ACTION",
    "write":    "C:ACTION",
    "fix":      "C:ACTION",
    "want":     "C:STATE",
    "feel":     "C:EMOTION",
    # Cognition
    "think":    "C:IDEA",
    "know":     "C:IDEA",
    "art":      "C:IDEA",
    "force":    "C:PROPERTY",
    "govern":   "C:RULE",
    "decide":   "C:ACTION",
    # Measures
    "size":     "C:SIZE",
    "color":    "C:COLOR",
    "time":     "C:TIME_POINT",
    "quality":  "C:PROPERTY",
    # Objects
    "food":     "C:OBJECT",
    "animal":   "C:ANIMAL",
    "material": "C:OBJECT",
    "trade":    "C:ACTION",
    "dwell":    "C:PLACE",
    "contain":  "C:WHOLE",
    "connect":  "C:RELATION",
    "exist":    "C:STATE",
}

# Role suffixes on CMP:<field>:<role> map to semantic roles.
_CMP_ROLE_TO_ROLE: dict[str, str] = {
    "agent":        "RO:AGENT",
    "patient":      "RO:PATIENT",
    "place":        "RO:LOCATION",
    "instance":     None,       # collapses to the concept itself
    "state":        None,
    "mutual":       None,
    "process":      None,
    "causer":       "RO:CAUSE",
    "seeker":       "RO:AGENT",
    "quality":      None,
    "intensifier":  None,
}


def _maybe_root_field(root_value: str) -> str | None:
    """Resolve a raw Arabic root token payload to a semantic field.

    Accepts dotted (``ك.ت.ب``) and undotted (``كتب``) forms. Returns
    ``None`` when no mapping is known.
    """
    field = ARABIC_ROOT_TO_FIELD.get(root_value)
    if field:
        return field

    # Support undotted raw roots by canonicalizing to dotted form.
    if "." not in root_value and len(root_value) in (3, 4):
        dotted = ".".join(root_value)
        return ARABIC_ROOT_TO_FIELD.get(dotted)
    return None


def _project_standard_token(tok: str) -> list[str]:
    """Project one CST-standard token into zero, one, or two logic tokens.

    Returns a list so callers can splice the result in place.
    """
    # Specials pass through verbatim
    if tok in ("[BOS]", "[EOS]", "[PAD]", "[UNK]", "[SEP]"):
        return [tok] if tok in LOGIC_VOCAB else []

    # REL: → logical / quantifier / relational
    if tok in _STD_REL_TO_LOGIC:
        mapped = _STD_REL_TO_LOGIC[tok]
        return [mapped] if mapped else []

    # STR:
    if tok in _STD_STR_TO_LOGIC:
        mapped = _STD_STR_TO_LOGIC[tok]
        return [mapped] if mapped else []

    # FEAT:
    if tok in _STD_FEAT_TO_LOGIC:
        mapped = _STD_FEAT_TO_LOGIC[tok]
        return [mapped] if mapped else []
    if tok.startswith("FEAT:"):
        return []  # all other FEAT:* dropped

    # NUM:
    if tok in _STD_NUM_TO_LOGIC:
        return [_STD_NUM_TO_LOGIC[tok]]
    if tok.startswith("NUM:"):
        return ["N:INT"]  # catch-all

    # PAT:* is morphology-internal in standard tokenizer; logic view keeps
    # a closed semantic vocabulary and drops pattern surfaces.
    if tok.startswith("PAT:"):
        return []

    # ROLE:* from atomic composition path.
    if tok.startswith("ROLE:"):
        role = tok.split(":", 1)[1]
        mapped = _CMP_ROLE_TO_ROLE.get(role)
        return [mapped] if mapped else []

    # TIME:month:* → time-point
    if tok.startswith(_STD_TIME_PREFIX):
        return ["C:TIME_POINT"]

    # CMP:<field>:<role>
    if tok.startswith("CMP:"):
        parts = tok.split(":")
        if len(parts) == 3:
            _, field, role = parts
            concept = _FIELD_TO_CONCEPT.get(field, "C:CONCEPT")
            role_tok = _CMP_ROLE_TO_ROLE.get(role)
            if role_tok:
                return [concept, role_tok]
            return [concept]
        return ["C:CONCEPT"]

    # ROOT:<field> — bare root
    if tok.startswith("ROOT:"):
        root_value = tok.split(":", 1)[1]
        # Prefer explicit field roots, then try lexical Arabic roots.
        field = root_value if root_value in _FIELD_TO_CONCEPT else _maybe_root_field(root_value)
        return [_FIELD_TO_CONCEPT.get(field, "C:CONCEPT") if field else "C:CONCEPT"]

    # NE:<surface> → typed person/place/thing. Without per-surface
    # gazetteer we default to C:PERSON; specific gazetteers can refine.
    if tok.startswith("NE:"):
        return ["C:PERSON"]

    # FOREIGN:<surface> → C:CONCEPT (unknown meaning)
    if tok.startswith("FOREIGN:"):
        return ["C:CONCEPT"]

    # LIT:<surface> → [UNK]
    if tok.startswith("LIT:"):
        return ["[UNK]"]

    # Unknown token — UNK
    return ["[UNK]"]


# ────────────────────────────────────────────────────────────────
# Raw-text helpers (regex pre-pass for formal-logic-style input)
# ────────────────────────────────────────────────────────────────

_FORMAL_TOKEN_RE = re.compile(
    r"""
    (?P<paren_l>\() |
    (?P<paren_r>\)) |
    (?P<iff>↔|<->|<=>) |
    (?P<impl>→|->|=>) |
    (?P<and>∧|&&|&) |
    (?P<or>∨|\|\||\|) |
    (?P<ne>≠|!=) |
    (?P<not>¬|~|!) |
    (?P<eq>==|=) |
    (?P<ge>≥|>=) |
    (?P<le>≤|<=) |
    (?P<gt>>) |
    (?P<lt><) |
    (?P<plus>\+) |
    (?P<minus>-) |
    (?P<times>×|\*) |
    (?P<div>÷|/) |
    (?P<pow>\^) |
    (?P<forall>∀) |
    (?P<exists>∃) |
    (?P<therefore>∴) |
    (?P<because>∵) |
    (?P<int>\d+) |
    (?P<var>[A-Za-z_][A-Za-z0-9_]*) |
    (?P<ws>\s+)
    """,
    re.VERBOSE,
)

_FORMAL_GROUP_TO_LOGIC: dict[str, str] = {
    "paren_l":   "L:LPAREN",
    "paren_r":   "L:RPAREN",
    "iff":       "L:IFF",
    "impl":      "L:IMPL",
    "and":       "L:AND",
    "or":        "L:OR",
    "not":       "L:NOT",
    "eq":        "R:EQUALS",
    "ne":        "R:NE",
    "ge":        "R:GE",
    "le":        "R:LE",
    "gt":        "R:GT",
    "lt":        "R:LT",
    "plus":      "A:PLUS",
    "minus":     "A:MINUS",
    "times":     "A:TIMES",
    "div":       "A:DIV",
    "pow":       "A:POW",
    "forall":    "Q:ALL",
    "exists":    "Q:SOME",
    "therefore": "L:THEREFORE",
    "because":   "L:BECAUSE",
}

_KEYWORD_TO_LOGIC_EN: dict[str, str] = {
    "and": "L:AND", "or": "L:OR", "not": "L:NOT", "but": "L:AND",
    "if": "L:IMPL", "then": "L:IMPL", "iff": "L:IFF",
    "therefore": "L:THEREFORE", "so": "L:THEREFORE",
    "because": "L:BECAUSE", "since": "L:BECAUSE",
    "all": "Q:ALL", "every": "Q:ALL", "each": "Q:ALL", "any": "Q:SOME",
    "forall": "Q:ALL", "for_all": "Q:ALL",
    "some": "Q:SOME", "no": "Q:NO", "none": "Q:NONE",
    "exists": "Q:EXISTS", "unique": "Q:UNIQUE",
    "most": "Q:MOST", "few": "Q:FEW",
    "is": "R:IS", "are": "R:IS", "was": "R:IS", "were": "R:IS",
    "isa": "R:ISA", "has": "R:HAS", "have": "R:HAS",
    "equals": "R:EQUALS",
    "true": "L:TRUE", "false": "L:FALSE",
    "simplify": "A:SIMPLIFY", "expand": "A:EXPAND",
    "factor": "A:FACTOR", "solve": "A:SOLVE", "evaluate": "A:EVAL",
    "before": "R:BEFORE", "after": "R:AFTER",
    "must": "M:MUST", "may": "M:MAY",
    "can": "M:CAN", "should": "M:SHOULD",
    "always": "T:ALWAYS", "never": "T:NEVER",
    "sometimes": "T:SOMETIMES",
    "past": "T:PAST", "future": "T:FUTURE", "present": "T:PRESENT",
    "premise": "S:PREMISE", "conclusion": "S:CONCLUSION",
    "step": "S:STEP", "assume": "S:ASSUME",
    "prove": "S:PROVE", "qed": "S:QED",
    "case": "S:CASE", "define": "S:DEFINE",
}

_VARIABLE_NAMES = frozenset(k for k in LOGIC_VOCAB if k.startswith("V:"))


def _integer_bucket(n: int) -> str:
    if n == 0: return "N:ZERO"
    if n == 1: return "N:ONE"
    if n < 0:  return "N:NEG"
    if n <= 100: return "N:SMALL"
    return "N:LARGE"


def _tokenize_formal(text: str) -> list[str]:
    """Tokenize a formal-logic / algebraic expression string."""
    out: list[str] = []
    for m in _FORMAL_TOKEN_RE.finditer(text):
        grp = m.lastgroup
        if grp in (None, "ws"): continue
        if grp in _FORMAL_GROUP_TO_LOGIC:
            out.append(_FORMAL_GROUP_TO_LOGIC[grp])
            continue
        if grp == "int":
            out.append(_integer_bucket(int(m.group())))
            continue
        if grp == "var":
            raw = m.group()
            low = raw.lower()
            if low in _KEYWORD_TO_LOGIC_EN:
                out.append(_KEYWORD_TO_LOGIC_EN[low])
                continue
            vslot = f"V:{raw[0].upper()}"
            out.append(vslot if vslot in _VARIABLE_NAMES else "C:CONCEPT")
    return out


# ────────────────────────────────────────────────────────────────
# Public tokenizer
# ────────────────────────────────────────────────────────────────

class LogicTokenizer:
    """Closed-vocabulary logic tokenizer.

    Usage::

        tk = LogicTokenizer()

        # From CST-standard tokens
        logic_toks = tk.from_standard(["[BOS]", "REL:if", "ROOT:rain",
                                       "REL:then", "ROOT:wet", "[EOS]"])

        # From raw formal text
        logic_toks = tk.from_formal("∀x. P(x) → Q(x)")

        ids = tk.to_ids(logic_toks)
    """

    def __init__(self) -> None:
        self.vocab = LOGIC_VOCAB
        self.pad_id = PAD_ID
        self.unk_id = UNK_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID

    # --- projection from CST-standard stream -------------------

    def from_standard(
        self, tokens: list[str], *, add_bos_eos: bool = False,
    ) -> list[str]:
        """Project a CST-standard token stream to logic tokens.

        Applies the projection table then a collapse pass that removes
        adjacent duplicates of ``L:AND``/``L:OR``/``L:NOT``/``T:*``.
        """
        out: list[str] = []
        if add_bos_eos:
            out.append("[BOS]")
        for tok in tokens:
            for projected in _project_standard_token(tok):
                if projected not in self.vocab:
                    projected = "[UNK]"
                if out and out[-1] == projected and _collapsible(projected):
                    continue
                out.append(projected)
        if add_bos_eos:
            out.append("[EOS]")
        return out

    # --- raw formal-logic / algebraic input --------------------

    def from_formal(
        self, text: str, *, add_bos_eos: bool = False,
    ) -> list[str]:
        toks = _tokenize_formal(text)
        out = ["[BOS]"] if add_bos_eos else []
        for t in toks:
            if t not in self.vocab:
                t = "[UNK]"
            if out and out[-1] == t and _collapsible(t):
                continue
            out.append(t)
        if add_bos_eos:
            out.append("[EOS]")
        return out

    # --- id conversion -----------------------------------------

    def to_ids(self, tokens: list[str]) -> list[int]:
        return [self.vocab.get(t, self.unk_id) for t in tokens]

    def to_tokens(self, ids: list[int]) -> list[str]:
        inv = {i: t for t, i in self.vocab.items()}
        return [inv.get(i, "[UNK]") for i in ids]


def _collapsible(tok: str) -> bool:
    """Tokens that collapse when repeated adjacently."""
    return (
        tok in ("L:AND", "L:OR", "L:NOT", "L:SEP",
                "T:PAST", "T:PRESENT", "T:FUTURE",
                "[Q]")
    )


# ────────────────────────────────────────────────────────────────
# CLI sanity check
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":  # pragma: no cover
    tk = LogicTokenizer()
    print(f"logic vocab size: {len(tk.vocab)}")
    samples_std = [
        ["[BOS]", "REL:if", "ROOT:rain", "REL:then", "ROOT:earth", "[EOS]"],
        ["[BOS]", "REL:all", "CMP:person:agent", "REL:is",
         "ROOT:animal", "[EOS]"],
    ]
    for s in samples_std:
        print(s, "→", tk.from_standard(s, add_bos_eos=False))
    samples_formal = [
        "∀x. P(x) → Q(x)",
        "(p ∧ q) → r",
        "2x + 3 = 7, solve for x",
    ]
    for s in samples_formal:
        print(repr(s), "→", tk.from_formal(s, add_bos_eos=False))
