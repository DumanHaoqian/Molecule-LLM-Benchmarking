#!/usr/bin/env bash
# Install the extra deps the benchmark needs into the existing chemdfm venv.
# transformers / torch / datasets / rdkit / nltk are already present; this adds
# rouge_score and the NLTK corpora used by the caption metrics.
set -euo pipefail

VENV=/home/haoqian/Data/SAERAG/venvs/chemdfm
# shellcheck disable=SC1091
source "$VENV/bin/activate"

pip install --quiet "rouge_score>=0.1.2"

python - <<'PY'
import nltk
for pkg in ["punkt", "punkt_tab", "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)
print("[setup] nltk data ready")
PY

echo "[setup] done"
