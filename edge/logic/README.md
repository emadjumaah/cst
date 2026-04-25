# CST Logic — source of truth

This folder is the **canonical home** of the CST-logic vocabulary.
Edit files here; then regenerate the USB bundle.

| Item                            | Location                                                                               |
| ------------------------------- | -------------------------------------------------------------------------------------- |
| Tokenizer                       | [../logic_tokenizer.py](../logic_tokenizer.py)                                         |
| Vocabulary (11 JSONs, 151 toks) | [vocab/](vocab)                                                                        |
| Spec                            | [../../plan/CST_LOGIC_SPEC.md](../../plan/CST_LOGIC_SPEC.md)                           |
| Tests                           | [../training/tests/test_logic_tokenizer.py](../training/tests/test_logic_tokenizer.py) |
| USB bundle (derived copy)       | `../../../usb-bundle/logic-v0.1/`                                                      |

## Refresh the USB bundle after changes

```bash
DST="/Volumes/Install mac/usb-bundle/logic-v0.1"
cp edge/logic_tokenizer.py      "$DST/"
cp edge/logic/vocab/*.json      "$DST/logic/vocab/"
python3 -m pytest edge/training/tests/test_logic_tokenizer.py -q
```

## Why a separate tokenizer?

- CST-standard v4.1: ~32k tokens, language-level, surface-preserving.
- CST-logic v0.1: 151 tokens, reasoning-level, language-independent.

Both trained as **separate models**. See the spec for projection rules.
