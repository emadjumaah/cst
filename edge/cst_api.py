"""Internal unified CST API facade.

This module prepares a stable, single entry point for CST tokenization
workflows across levels:

- standard-level tokenization (Arabic / English)
- logic-level projection and closed-vocabulary encoding

It is intentionally internal for now. Exposure and packaging are handled
separately.
"""

from __future__ import annotations

from typing import Any
from typing import Mapping
from typing import Sequence


_LANGS = {"ar", "en"}
_LEVELS = {"standard", "logic"}


def _strip_bos_eos(tokens: list[str], ids: list[int] | None) -> tuple[list[str], list[int] | None]:
    """Strip exactly one BOS/EOS pair when present at the sequence edges."""
    if len(tokens) >= 2 and tokens[0] == "[BOS]" and tokens[-1] == "[EOS]":
        stripped_tokens = tokens[1:-1]
        if ids is None:
            return stripped_tokens, None
        if len(ids) == len(tokens):
            return stripped_tokens, ids[1:-1]
        return stripped_tokens, ids
    return tokens, ids


def _token_to_id_lookup(tokens: Sequence[str], vocab: Mapping[str, int], unk_id: int = 1) -> list[int]:
    return [vocab.get(t, unk_id) for t in tokens]


def _coerce_id_to_token_vocab(vocab: Mapping[Any, Any]) -> dict[int, str]:
    """Accept either id->token or token->id vocab mappings.

    Raises ValueError when the mapping cannot be interpreted.
    """
    if not vocab:
        raise ValueError("decode requires a non-empty vocabulary mapping")

    sample_key = next(iter(vocab.keys()))
    sample_val = vocab[sample_key]

    # id -> token mapping
    if isinstance(sample_key, int) and isinstance(sample_val, str):
        return {int(k): str(v) for k, v in vocab.items()}

    # token -> id mapping
    if isinstance(sample_key, str) and isinstance(sample_val, int):
        return {int(v): str(k) for k, v in vocab.items()}

    raise ValueError(
        "vocab must be either token->id (str->int) or id->token (int->str)",
    )


class CSTApi:
    """Unified facade for CST tokenization operations.

    Dependency injection is supported so tests can run without heavyweight
    runtime dependencies (CAMeL Tools / spaCy).
    """

    def __init__(
        self,
        *,
        arabic_tokenizer: Any | None = None,
        english_tokenizer: Any | None = None,
        logic_tokenizer: Any | None = None,
        ar_tokenizer_kwargs: dict[str, Any] | None = None,
        en_tokenizer_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._arabic_tokenizer = arabic_tokenizer
        self._english_tokenizer = english_tokenizer
        self._logic_tokenizer = logic_tokenizer
        self._ar_tokenizer_kwargs = dict(ar_tokenizer_kwargs or {})
        self._en_tokenizer_kwargs = dict(en_tokenizer_kwargs or {})

    # -----------------------------------------------------------------
    # Public API (pre-exposure contract)
    # -----------------------------------------------------------------

    def tokenize(
        self,
        text: str,
        *,
        lang: str,
        level: str = "standard",
        add_bos_eos: bool = False,
    ) -> dict[str, Any]:
        """Tokenize one string.

        Returns a normalized result shape:

        {
          "text": str,
          "lang": "ar"|"en"|"formal",
          "level": "standard"|"logic",
          "tokens": list[str],
          "ids": list[int] | None,
          "meta": dict,
        }
        """
        _validate_level(level)

        if level == "standard":
            _validate_lang(lang)
            if lang == "ar":
                return self._tokenize_standard_ar(text, add_bos_eos=add_bos_eos)
            return self._tokenize_standard_en(text, add_bos_eos=add_bos_eos)

        _validate_lang(lang)
        return self._tokenize_logic_from_text(text, lang=lang, add_bos_eos=add_bos_eos)

    def tokenize_batch(
        self,
        texts: Sequence[str],
        *,
        lang: str,
        level: str = "standard",
        add_bos_eos: bool = False,
    ) -> list[dict[str, Any]]:
        """Tokenize a batch of strings."""
        return [
            self.tokenize(t, lang=lang, level=level, add_bos_eos=add_bos_eos)
            for t in texts
        ]

    def tokenize_formal_logic(
        self,
        text: str,
        *,
        add_bos_eos: bool = False,
    ) -> dict[str, Any]:
        """Tokenize formal logic/algebraic expressions directly."""
        tk = self._get_logic_tokenizer()
        tokens = tk.from_formal(text, add_bos_eos=add_bos_eos)
        ids = tk.to_ids(tokens)
        return {
            "text": text,
            "lang": "formal",
            "level": "logic",
            "tokens": tokens,
            "ids": ids,
            "meta": {
                "source": "edge.logic_tokenizer.LogicTokenizer.from_formal",
                "formal_mode": True,
            },
        }

    def project(
        self,
        tokens: Sequence[str],
        *,
        source: str = "standard",
        target: str = "logic",
        lang: str | None = None,
        add_bos_eos: bool = False,
    ) -> list[str]:
        """Project tokens across levels.

        Currently supported:
        - standard -> logic (closed vocabulary projection)
        """
        if source != "standard" or target != "logic":
            raise ValueError(
                f"unsupported projection: {source}->{target}; only standard->logic is supported",
            )
        if lang is not None:
            _validate_lang(lang)

        # Keep unknown tokens, strip sequence wrappers, and let LogicTokenizer
        # own the closed-vocabulary mapping behavior.
        clean = [t for t in tokens if t not in ("[BOS]", "[EOS]")]
        return self._get_logic_tokenizer().from_standard(clean, add_bos_eos=add_bos_eos)

    def encode(
        self,
        tokens_or_text: Sequence[str] | str,
        *,
        lang: str,
        level: str = "standard",
        vocab: Mapping[str, int] | None = None,
        add_bos_eos: bool = False,
    ) -> list[int]:
        """Encode tokens (or text) to ids.

        Notes:
        - logic-level ids use the closed logic vocabulary.
        - Arabic standard-level ids use Arabic tokenizer vocabulary.
        - English standard-level ids require an explicit vocab mapping.
        """
        _validate_level(level)
        _validate_lang(lang)

        if isinstance(tokens_or_text, str):
            out = self.tokenize(
                tokens_or_text,
                lang=lang,
                level=level,
                add_bos_eos=add_bos_eos,
            )
            if out["ids"] is not None:
                return list(out["ids"])
            tokens = list(out["tokens"])
        else:
            tokens = list(tokens_or_text)

        if level == "logic":
            return self._get_logic_tokenizer().to_ids(tokens)

        if lang == "ar":
            ar_tok = self._get_arabic_tokenizer()
            unk_id = ar_tok.vocab.get("[UNK]", 1)
            return _token_to_id_lookup(tokens, ar_tok.vocab, unk_id=unk_id)

        if vocab is None:
            raise ValueError(
                "english standard encode requires vocab=token_to_id mapping",
            )
        unk_id = int(vocab.get("[UNK]", 1))
        return _token_to_id_lookup(tokens, vocab, unk_id=unk_id)

    def decode(
        self,
        ids: Sequence[int],
        *,
        lang: str,
        level: str = "standard",
        vocab: Mapping[Any, Any] | None = None,
    ) -> list[str]:
        """Decode ids to tokens.

        Notes:
        - logic-level ids use the closed logic vocabulary.
        - Arabic standard-level ids use Arabic tokenizer vocabulary.
        - English standard-level ids require an explicit vocab mapping.
        """
        _validate_level(level)
        _validate_lang(lang)

        if level == "logic":
            return self._get_logic_tokenizer().to_tokens(list(ids))

        if lang == "ar":
            ar_tok = self._get_arabic_tokenizer()
            inv = {v: k for k, v in ar_tok.vocab.items()}
            return [inv.get(int(i), "[UNK]") for i in ids]

        if vocab is None:
            raise ValueError(
                "english standard decode requires vocab mapping (token->id or id->token)",
            )
        id_to_token = _coerce_id_to_token_vocab(vocab)
        return [id_to_token.get(int(i), "[UNK]") for i in ids]

    def untokenize(
        self,
        tokens: Sequence[str],
        *,
        lang: str,
        level: str = "standard",
    ) -> str:
        """Untokenize tokens back to text.

        This is intentionally a contract placeholder for future exposure.
        Current tokenizer implementations do not provide a full untokenize
        implementation yet.
        """
        _validate_level(level)
        _validate_lang(lang)
        _ = tokens
        raise NotImplementedError(
            "untokenize is reserved for the public API milestone and is not implemented yet",
        )

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _tokenize_standard_ar(self, text: str, *, add_bos_eos: bool) -> dict[str, Any]:
        tok = self._get_arabic_tokenizer()
        raw = tok.tokenize(text)

        tokens = list(raw.get("tokens", []))
        ids = list(raw.get("ids", [])) if raw.get("ids") is not None else None
        if not add_bos_eos:
            tokens, ids = _strip_bos_eos(tokens, ids)

        return {
            "text": text,
            "lang": "ar",
            "level": "standard",
            "tokens": tokens,
            "ids": ids,
            "meta": {
                "source": "edge.arabic_tokenizer.ArabicCSTTokenizer",
                "native_has_bos_eos": True,
            },
        }

    def _tokenize_standard_en(self, text: str, *, add_bos_eos: bool) -> dict[str, Any]:
        tok = self._get_english_tokenizer()
        raw = tok.tokenize(text)

        tokens = list(raw.get("values", []))
        if add_bos_eos:
            tokens = ["[BOS]", *tokens, "[EOS]"]

        return {
            "text": text,
            "lang": "en",
            "level": "standard",
            "tokens": tokens,
            "ids": None,
            "meta": {
                "source": "edge.english_tokenizer.EnglishCSTTokenizer",
                "coverage": raw.get("coverage"),
            },
        }

    def _tokenize_logic_from_text(
        self,
        text: str,
        *,
        lang: str,
        add_bos_eos: bool,
    ) -> dict[str, Any]:
        std = self.tokenize(text, lang=lang, level="standard", add_bos_eos=False)
        logic_tokens = self.project(
            std["tokens"],
            source="standard",
            target="logic",
            lang=lang,
            add_bos_eos=add_bos_eos,
        )
        logic_ids = self._get_logic_tokenizer().to_ids(logic_tokens)

        return {
            "text": text,
            "lang": lang,
            "level": "logic",
            "tokens": logic_tokens,
            "ids": logic_ids,
            "meta": {
                "source": "edge.logic_tokenizer.LogicTokenizer.from_standard",
                "projection": {
                    "input_count": len(std["tokens"]),
                    "output_count": len(logic_tokens),
                    "delta": len(std["tokens"]) - len(logic_tokens),
                },
            },
        }

    def _get_arabic_tokenizer(self) -> Any:
        if self._arabic_tokenizer is not None:
            return self._arabic_tokenizer

        from camel_tools.morphology.analyzer import Analyzer
        from camel_tools.morphology.database import MorphologyDB

        try:
            from edge.arabic_tokenizer import ArabicCSTTokenizer
        except ImportError:
            from arabic_tokenizer import ArabicCSTTokenizer

        analyzer = Analyzer(MorphologyDB.builtin_db())
        self._arabic_tokenizer = ArabicCSTTokenizer(analyzer, **self._ar_tokenizer_kwargs)
        return self._arabic_tokenizer

    def _get_english_tokenizer(self) -> Any:
        if self._english_tokenizer is not None:
            return self._english_tokenizer

        import spacy

        try:
            from edge.english_tokenizer import EnglishCSTTokenizer
        except ImportError:
            from english_tokenizer import EnglishCSTTokenizer

        model = self._en_tokenizer_kwargs.get("model", "en_core_web_sm")
        tok_kwargs = {k: v for k, v in self._en_tokenizer_kwargs.items() if k != "model"}
        nlp = spacy.load(model)
        self._english_tokenizer = EnglishCSTTokenizer(nlp, **tok_kwargs)
        return self._english_tokenizer

    def _get_logic_tokenizer(self) -> Any:
        if self._logic_tokenizer is not None:
            return self._logic_tokenizer

        try:
            from edge.logic_tokenizer import LogicTokenizer
        except ImportError:
            from logic_tokenizer import LogicTokenizer

        self._logic_tokenizer = LogicTokenizer()
        return self._logic_tokenizer


def _validate_lang(lang: str) -> None:
    if lang not in _LANGS:
        raise ValueError(f"unsupported lang: {lang!r}; expected one of {_LANGS}")


def _validate_level(level: str) -> None:
    if level not in _LEVELS:
        raise ValueError(f"unsupported level: {level!r}; expected one of {_LEVELS}")
