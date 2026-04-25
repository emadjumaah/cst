# CST Tiny Brain — Logic v0.1 showcase

Browser demo for the CST-logic tokenizer (151-token closed vocab) and
the small logic transformer.

## Status

- **Tokenizer:** shipped. `src/logic-tokenizer.ts` is a TS port of
  [edge/logic_tokenizer.py](../logic_tokenizer.py); both load the same
  11 JSON vocab files.
- **Model:** stubbed with a deterministic reasoner so the UI can be
  validated today. Swap in the real ONNX once Colab training finishes
  (see below).

## Run locally

```bash
cd edge/demo-logic
npm install
npm run dev
# open http://127.0.0.1:5178
```

Three tabs:

- **Prop logic** — type `p = T, q = F; (p AND q) -> r` and watch the
  tokenizer + answer update live.
- **Syllogism** — `All programmers are engineers. …` → `valid / invalid`.
- **Algebra** — `solve 2x + 3 = 7`, `simplify 2x + 3x - x`, `evaluate 3*4+2`.

Each run shows:

1. The CST-logic token sequence, color-coded by bucket (L/Q/R/C/RO/A/T/M/N/V/S)
2. The token IDs (what the model actually sees)
3. The answer
4. A CoT trace reconstructed from logic tokens

## Swap in the trained model

1. Export from Colab as ONNX with opset ≥ 17.
2. Copy to `edge/demo-logic/public/model_logic.onnx`.
3. In [src/model.ts](src/model.ts), replace the placeholder in
   `LogicModel.tryLoadOnnx()`:
   ```ts
   const ort = await import("onnxruntime-web");
   return await ort.InferenceSession.create("model_logic.onnx");
   ```
4. Rebuild: `npm run build`. The UI contract (answer + CoT + tokens) is
   unchanged — the stub and the ONNX speak the same interface.

## Build + preview

```bash
npm run build       # writes dist/
npm run preview     # serves dist/ locally
```

## Source-of-truth map

| Item                    | Location                                             |
| ----------------------- | ---------------------------------------------------- |
| Python tokenizer        | [../logic_tokenizer.py](../logic_tokenizer.py)       |
| Vocab JSONs (source)    | [../logic/vocab/](../logic/vocab/)                   |
| Vocab JSONs (for demo)  | `public/logic/vocab/` (copied from above)           |
| Spec                    | [../../plan/CST_LOGIC_SPEC.md](../../plan/CST_LOGIC_SPEC.md) |
| USB bundle              | `../../../usb-bundle/logic-v0.1/`                    |

## Refresh vocab after editing

```bash
cp edge/logic/vocab/*.json edge/demo-logic/public/logic/vocab/
```
