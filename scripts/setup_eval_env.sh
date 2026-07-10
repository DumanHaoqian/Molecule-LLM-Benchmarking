#!/usr/bin/env bash
# Build the dedicated ChEBI-20 *evaluation* venv (separate from generation).
#
# Rationale: metric deps (FCD's ChemNet, Text2Mol's torch_geometric) can clash
# with the generation env. We isolate them here. CPU-only torch is used on
# purpose — evaluation runs over a few thousand molecules, which is fast enough
# on CPU and avoids the fragile cu128 build.
set -euo pipefail

VENV=/home/haoqian/Data/SAERAG/venvs/ChEBI-20-Eva

if [ ! -d "$VENV" ]; then
    echo "[setup-eval] creating venv at $VENV"
    python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --quiet --upgrade pip

echo "[setup-eval] installing CPU torch + metric deps"
pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
pip install --quiet \
    numpy \
    "scipy<1.14" \
    rdkit \
    nltk \
    "rouge_score>=0.1.2" \
    fcd_torch
# scipy<1.14: fcd_torch 1.0.7 calls linalg.sqrtm(..., disp=...) which newer
# scipy removed. 1.13.x still supports numpy 2.0.

python - <<'PY'
import nltk
for pkg in ["punkt", "punkt_tab", "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)
print("[setup-eval] nltk data ready")
PY

echo "[setup-eval] core deps done."
echo
echo "Text2Mol (optional) requires torch_geometric + the pretrained checkpoint:"
echo "  pip install torch_geometric transformers"
echo "  # download the Text2Mol checkpoint/embeddings, then set:"
echo "  export TEXT2MOL_DIR=/path/to/text2mol_resources"
echo "Without TEXT2MOL_DIR the Text2Mol column is reported as '—'."
