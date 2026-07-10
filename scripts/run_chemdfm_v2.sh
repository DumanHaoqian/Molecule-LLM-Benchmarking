#!/usr/bin/env bash
# Resumable ChemDFM run: ChEBI-20 captioning + S2-TOMG-mini + evaluation.
set -euo pipefail

REPO=/home/haoqian/Data/SAERAG/v3_Chem_SAE/Stage6_benchmarking
cd "$REPO"

GEN_PY=/home/haoqian/Data/SAERAG/venvs/chemdfm/bin/python
EVAL_PY=/home/haoqian/Data/SAERAG/venvs/ChEBI-20-Eva/bin/python
MODEL=${MOLBENCH_MODEL:-chemdfm-v2}
OUT=${MOLBENCH_OUT:-results/chemdfm_v2_full}
NOFILE=${MOLBENCH_NOFILE:-65536}
STOP_AFTER=all
RESTART=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stop-after) STOP_AFTER="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --out-dir) OUT="$2"; shift 2 ;;
        --restart) RESTART=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

case "$STOP_AFTER" in
    chebi20/captioning|tomg-mini|all) ;;
    *) echo "invalid --stop-after: $STOP_AFTER" >&2; exit 2 ;;
esac
ulimit -n "$NOFILE"

mkdir -p "$OUT"
export TEXT2MOL_DIR="$REPO/text2mol_resources"
export TOMG_ZINC_PATH="$REPO/tomg_resources/zinc250k.txt"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

if [[ "$RESTART" -eq 1 && -f "$OUT/run.log" ]]; then
    archive="$OUT/history/$(date '+%Y%m%d-%H%M%S')-driver-restart-$$"
    mkdir -p "$archive"
    cp -p "$OUT/run.log" "$archive/run.log"
    sync "$archive/run.log"
    if ! cmp -s "$OUT/run.log" "$archive/run.log"; then
        echo "run.log changed while it was being archived; refusing restart" >&2
        exit 1
    fi
    archived_bytes=$(wc -c < "$archive/run.log")
    archived_sha256=$(sha256sum "$archive/run.log" | awk '{print $1}')
    rm "$OUT/run.log"
    printf '%s\n' \
        "{\"status\":\"archived\",\"reason\":\"driver --restart\",\"archived_at\":\"$(ts)\",\"run_log_bytes\":$archived_bytes,\"run_log_sha256\":\"$archived_sha256\"}" \
        > "$archive/failure.json"
fi

# The driver owns run.log so it can archive a previous run before reopening it.
# Launch with stdout/stderr redirected away from run.log itself.
exec > >(tee -a "$OUT/run.log") 2>&1

child=""
stop_requested=0
forward_stop() {
    stop_requested=1
    if [[ -n "$child" ]]; then
        kill -TERM "$child" 2>/dev/null || true
    fi
}
trap forward_stop INT TERM

run_stage() {
    echo "[driver $(ts)] $1"
    shift
    "$@" &
    child=$!
    set +e
    wait "$child"
    status=$?
    set -e
    child=""
    if [[ "$stop_requested" -eq 1 ]]; then
        exit 130
    fi
    return "$status"
}

echo "[driver $(ts)] START model=$MODEL out=$OUT restart=$RESTART stop_after=$STOP_AFTER"

restart_arg=()
if [[ "$RESTART" -eq 1 ]]; then restart_arg=(--restart); fi

run_stage "GEN chebi20/captioning (3301)" \
    "$GEN_PY" -m molbench generate --benchmark chebi20 --model "$MODEL" \
    --task captioning --split test --max-batch-size 16 --batching length-aware \
    --length-batch-policy '128:16,256:8,384:4,512:2,inf:1' \
    --token-budget 16384 --out-dir "$OUT" "${restart_arg[@]}"

if [[ "$STOP_AFTER" == "chebi20/captioning" ]]; then
    echo "[driver $(ts)] STOP_AFTER chebi20/captioning"
    exit 0
fi

# Only the first stage consumes --restart; later stages have independent stems.
run_stage "GEN tomg-mini (4500)" \
    "$GEN_PY" -m molbench generate --benchmark tomg-mini --model "$MODEL" \
    --max-batch-size 16 --batching length-aware --token-budget 16384 \
    --out-dir "$OUT"

if [[ "$STOP_AFTER" == "tomg-mini" ]]; then
    echo "[driver $(ts)] STOP_AFTER tomg-mini"
    exit 0
fi

run_stage "EVAL chebi20/captioning" \
    "$EVAL_PY" -m molbench evaluate --benchmark chebi20 --models "$MODEL" \
    --task captioning --split test --out-dir "$OUT" --device cpu

run_stage "EVAL tomg-mini" \
    "$EVAL_PY" -m molbench evaluate --benchmark tomg-mini --models "$MODEL" \
    --out-dir "$OUT" --device cpu

echo "[driver $(ts)] ALL DONE. Artifacts in $OUT/"
