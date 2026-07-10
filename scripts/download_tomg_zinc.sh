#!/usr/bin/env bash
# Download the ZINC250k reference set used for TOMG-Bench MolCustom *novelty*
# (needed for the MolCustom WSR column). Git-ignored.
set -euo pipefail
DIR="${1:-$(cd "$(dirname "$0")/.." && pwd)/tomg_resources}"
mkdir -p "$DIR"
echo "[tomg] downloading ZINC250k reference -> $DIR/zinc250k.txt"
curl -L 'https://raw.githubusercontent.com/wengong-jin/icml18-jtnn/master/data/zinc/all.txt' \
    -o "$DIR/zinc250k.txt"
echo "[tomg] $(wc -l < "$DIR/zinc250k.txt") molecules. Set:"
echo "  export TOMG_ZINC_PATH=$DIR/zinc250k.txt"
