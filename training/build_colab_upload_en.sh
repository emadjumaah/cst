#!/usr/bin/env bash
# Build a Colab-ready upload folder with Python-tokenized CST + existing SPM baselines.
#
# Produces: training/colab-upload-en/
#   cst-8k/train-99963.jsonl              (Python-tokenized, capped 8K)
#   cst-8k/train-99963-vocab.json
#   cst-32k/train-99963.jsonl             (Python-tokenized, capped 32K)
#   cst-32k/train-99963-vocab.json
#   spm/train-99963-8k.jsonl              (copy — unchanged)
#   spm/train-99963-8k-vocab.json
#   spm/train-99963-32k.jsonl
#   spm/train-99963-32k-vocab.json
#   README.md
#
# After running this: tar/zip the folder and upload to Colab, then run
# `python training/experiments/colab_phase0.py --data-dir /content/colab-upload-en ...`

set -euo pipefail
cd "$(dirname "$0")/.."

PY_SRC="data/tokenized/cst-py/train-99963.jsonl"
OUT="training/colab-upload-en"

if [[ ! -f "$PY_SRC" ]]; then
  echo "ERROR: $PY_SRC not found. Run:" >&2
  echo "  python -m edge.tokenize --lang en --in data/tokenized/cst/train-100000.jsonl --text-field text --out $PY_SRC" >&2
  exit 1
fi

rm -rf "$OUT"
mkdir -p "$OUT/cst-8k" "$OUT/cst-32k" "$OUT/spm"

echo "[1/3] Cap CST vocab to 8K..."
python training/cap_cst_vocab.py 8000 --src "$PY_SRC" --out-dir "$OUT/cst-8k"

echo "[2/3] Cap CST vocab to 32K..."
python training/cap_cst_vocab.py 32000 --src "$PY_SRC" --out-dir "$OUT/cst-32k"

echo "[3/3] Copy SPM baselines..."
cp data/tokenized/spm/train-99963-8k.jsonl        "$OUT/spm/"
cp data/tokenized/spm/train-99963-8k-vocab.json   "$OUT/spm/"
cp data/tokenized/spm/train-99963-32k.jsonl       "$OUT/spm/"
cp data/tokenized/spm/train-99963-32k-vocab.json  "$OUT/spm/"

echo "[+] Bundle training scripts..."
mkdir -p "$OUT/scripts/experiments"
cp training/experiments/_core.py              "$OUT/scripts/experiments/"
cp training/experiments/run_multiseed.py      "$OUT/scripts/experiments/"
cp training/experiments/downstream_eval.py    "$OUT/scripts/experiments/"
cp training/experiments/prepare_downstream.py "$OUT/scripts/experiments/"
cp training/experiments/colab_phase0.py       "$OUT/scripts/experiments/"
touch "$OUT/scripts/__init__.py" "$OUT/scripts/experiments/__init__.py"

cat > "$OUT/README.md" <<'EOF'
# Colab upload — English Phase 0

Self-contained. No git clone needed.

## Contents

- `cst-8k/` `cst-32k/` — Python-tokenized CST corpora (via `edge/english_tokenizer.py`)
- `spm/` — SentencePiece BPE baselines
- `scripts/experiments/` — training code (_core, run_multiseed, colab_phase0, downstream_eval)

## Colab usage

Upload `colab-upload-en.tar.gz` to Colab, then in a GPU runtime:

```python
!pip install -q torch transformers

!tar -xzf colab-upload-en.tar.gz -C /content/
%cd /content/colab-upload-en

# 5 seeds x (CST-8K, SPM-8K, CST-32K, SPM-32K) = 20 runs
!python scripts/experiments/colab_phase0.py \
    --data-dir /content/colab-upload-en \
    --out-dir  /content/phase0_out \
    --langs en \
    --seeds 0 1 2 3 4 \
    --skip-downstream
```

Results land at `/content/phase0_out/results_en.json` and `summary.md`.
Checkpoints at `/content/phase0_out/checkpoints/<NAME>-seed<N>/`.

## Why `--skip-downstream`?

LAMBADA prepared files are not yet in this tarball. The multi-seed BPC
comparison is the core Phase 0 deliverable. Add LAMBADA later by:

1. Running `prepare_downstream.py` locally on raw LAMBADA
2. Dropping outputs into a `downstream/` subfolder
3. Re-running `colab_phase0.py` without `--skip-downstream`
EOF

echo
echo "Colab upload ready: $OUT"
du -sh "$OUT"/*
