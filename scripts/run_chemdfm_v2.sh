#!/usr/bin/env bash
# Full run: ChemDFM-v2.0-14B on ChEBI-20 captioning (test) + TOMG-Bench mini.
# Two stages in their respective venvs; writes aggregate tables AND per-example
# scored jsonl (for bad-case analysis). Designed to be launched via nohup.
set -euo pipefail

REPO=/home/haoqian/Data/SAERAG/v3_Chem_SAE/Stage6_benchmarking
cd "$REPO"

GEN_PY=/home/haoqian/Data/SAERAG/venvs/chemdfm/bin/python
EVAL_PY=/home/haoqian/Data/SAERAG/venvs/ChEBI-20-Eva/bin/python
OUT=results/chemdfm_v2_full
BATCH=16
mkdir -p "$OUT"

export TEXT2MOL_DIR="$REPO/text2mol_resources"
export TOMG_ZINC_PATH="$REPO/tomg_resources/zinc250k.txt"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
echo "[driver $(ts)] START  out=$OUT batch=$BATCH"

# ---- Stage 1: generation (chemdfm venv) ----
echo "[driver $(ts)] GEN chebi20/captioning (3301)"
"$GEN_PY" -m molbench generate --benchmark chebi20 --model chemdfm-v2 \
    --task captioning --split test --batch-size "$BATCH" --out-dir "$OUT"

echo "[driver $(ts)] GEN tomg-mini (4500)"
"$GEN_PY" -m molbench generate --benchmark tomg-mini --model chemdfm-v2 \
    --batch-size "$BATCH" --out-dir "$OUT"

# ---- Stage 2: evaluation (ChEBI-20-Eva venv) — tables + per-example scores ----
echo "[driver $(ts)] EVAL chebi20/captioning"
"$EVAL_PY" -m molbench evaluate --benchmark chebi20 --models chemdfm-v2 \
    --task captioning --split test --out-dir "$OUT" --device cpu

echo "[driver $(ts)] EVAL tomg-mini"
"$EVAL_PY" -m molbench evaluate --benchmark tomg-mini --models chemdfm-v2 \
    --out-dir "$OUT" --device cpu

echo "[driver $(ts)] ALL DONE. Artifacts in $OUT/:"
echo "  *__scored.jsonl  = per-example structured I/O + scores (bad-case analysis)"
echo "  tables_*.md      = aggregate tables"
