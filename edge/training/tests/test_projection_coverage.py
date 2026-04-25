"""Guard tests for CST standard -> logic projection coverage.

These tests prevent silent drift where Arabic standard tokenizer outputs
(REL:* / STR:*) stop being handled by the logic projection layer,
causing avoidable [UNK] inflation in reasoning corpora.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Repo-relative path: tests/ -> training/ -> edge/
EDGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EDGE))

from arabic_tokenizer import ARABIC_REL_MAP, ARABIC_STR_TRIGGERS  # noqa: E402
from logic_tokenizer import _STD_REL_TO_LOGIC, _STD_STR_TO_LOGIC  # noqa: E402


def _missing_outputs(outputs: set[str], projection: dict[str, str | None]) -> list[str]:
    """Return sorted standard tokens not covered by a projection map.

    Presence in the map means "handled" even when mapped value is None
    (explicit drop policy).
    """
    return sorted(tok for tok in outputs if tok not in projection)


def test_all_arabic_rel_outputs_are_handled_by_logic_projection():
    rel_outputs = set(ARABIC_REL_MAP.values())
    missing = _missing_outputs(rel_outputs, _STD_REL_TO_LOGIC)
    assert not missing, (
        "Unmapped REL outputs from ARABIC_REL_MAP in _STD_REL_TO_LOGIC: "
        f"{missing}"
    )


def test_all_arabic_str_outputs_are_handled_by_logic_projection():
    str_outputs = set(ARABIC_STR_TRIGGERS.values())
    missing = _missing_outputs(str_outputs, _STD_STR_TO_LOGIC)
    assert not missing, (
        "Unmapped STR outputs from ARABIC_STR_TRIGGERS in _STD_STR_TO_LOGIC: "
        f"{missing}"
    )
