#!/usr/bin/env bash
# Run ChemCoTBench V2/V1, ChEBI-20 captioning, and S2-TOMG-mini serially.
set -euo pipefail

REPO=${MOLBENCH_REPO:-/home/haoqian/Data/SAERAG/v3_Chem_SAE/Stage6_benchmarking}
OUT=${MOLBENCH_OUT:-results/chemdfm_r_full}
MODEL=${MOLBENCH_MODEL:-chemdfm-r}
NOFILE=${MOLBENCH_NOFILE:-65536}
RESTART=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --out-dir) OUT="$2"; shift 2 ;;
        --restart) RESTART=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

ulimit -n "$NOFILE"
cd "$REPO"
mkdir -p "$OUT/history"

if [[ "$RESTART" -eq 1 && -f "$OUT/run_chemdfm_r_all.log" ]]; then
    stamp=$(date '+%Y%m%d-%H%M%S')
    cp -p "$OUT/run_chemdfm_r_all.log" \
        "$OUT/history/${stamp}-run_chemdfm_r_all.log"
    rm "$OUT/run_chemdfm_r_all.log"
fi

exec > >(tee -a "$OUT/run_chemdfm_r_all.log") 2>&1
printf '%s\n' "$$" > "$OUT/chemdfm_r_all.pid"

child=""
stop_requested=0
cleanup() { rm -f "$OUT/chemdfm_r_all.pid"; }
forward_stop() {
    stop_requested=1
    [[ -z "$child" ]] || kill -TERM "$child" 2>/dev/null || true
}
trap cleanup EXIT
trap forward_stop INT TERM

run_stage() {
    echo "[master $(date '+%F %T')] $1"
    shift
    "$@" &
    child=$!
    set +e
    wait "$child"
    status=$?
    set -e
    child=""
    [[ "$stop_requested" -eq 0 ]] || exit 130
    return "$status"
}

restart_arg=()
[[ "$RESTART" -eq 0 ]] || restart_arg=(--restart)

echo "[master $(date '+%F %T')] START model=$MODEL out=$OUT nofile=$NOFILE"
run_stage "ChemCoTBench V2 + V1" \
    env MOLBENCH_NOFILE="$NOFILE" CHEMCOT_OUT="$OUT" \
    bash scripts/run_chemcotbench.sh --version both --model "$MODEL" \
    --out-dir "$OUT" "${restart_arg[@]}"

run_stage "ChEBI-20 captioning + S2-TOMG-mini" \
    env MOLBENCH_NOFILE="$NOFILE" MOLBENCH_MODEL="$MODEL" MOLBENCH_OUT="$OUT" \
    bash scripts/run_chemdfm_v2.sh --model "$MODEL" --out-dir "$OUT" \
    "${restart_arg[@]}"

echo "[master $(date '+%F %T')] ALL DONE"
