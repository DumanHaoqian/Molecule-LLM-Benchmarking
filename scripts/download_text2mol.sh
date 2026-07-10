#!/usr/bin/env bash
# Download the Text2Mol metric resources into text2mol_resources/.
#   * test_outputfinal_weights.320.pt : Text2Mol MLP checkpoint (Box, ~424MB)
#   * m2v_model.pkl                    : mol2vec word vectors the checkpoint was
#                                        trained with (MolT5 repo, ~11MB)
# These are large binaries and are git-ignored. Point TEXT2MOL_DIR at the
# resulting directory when evaluating.
set -euo pipefail

DIR="${1:-$(cd "$(dirname "$0")/.." && pwd)/text2mol_resources}"
mkdir -p "$DIR"
cd "$DIR"

echo "[t2m] downloading Text2Mol checkpoint -> $DIR"
curl -L 'https://uofi.box.com/shared/static/es16alnhzfy1hpagf55fu48k49f8n29x' \
    -o test_outputfinal_weights.320.pt

echo "[t2m] downloading mol2vec model (MolT5 m2v_model.pkl)"
curl -L 'https://media.githubusercontent.com/media/blender-nlp/MolT5/main/evaluation/m2v_model.pkl' \
    -o m2v_model.pkl

echo "[t2m] done. Set:  export TEXT2MOL_DIR=$DIR"
ls -lh "$DIR"
