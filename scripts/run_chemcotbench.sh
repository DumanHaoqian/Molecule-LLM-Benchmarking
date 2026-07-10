#!/usr/bin/env bash
# Resumable ChemCoTBench V1/V2 generation and evaluation driver.
set -euo pipefail

REPO=${MOLBENCH_REPO:-/home/haoqian/Data/SAERAG/v3_Chem_SAE/Stage6_benchmarking}
GEN_PY=${CHEMCOT_GEN_PY:-/home/haoqian/Data/SAERAG/venvs/chemdfm/bin/python}
V1_EVAL_PY=${CHEMCOT_V1_EVAL_PY:-/home/haoqian/Data/SAERAG/venvs/chemcotbench-v1-eval/bin/python}
V2_EVAL_PY=${CHEMCOT_V2_EVAL_PY:-/home/haoqian/Data/SAERAG/venvs/chemcotbench-v2-eval/bin/python}
OUT=${CHEMCOT_OUT:-results/chemcotbench_full}
MODEL=chemdfm-v2
VERSION=v2
FAMILY=""
TASK=""
STOP_AFTER=""
RESTART=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --version) VERSION="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --family) FAMILY="$2"; shift 2 ;;
        --task) TASK="$2"; shift 2 ;;
        --stop-after) STOP_AFTER="$2"; shift 2 ;;
        --out-dir) OUT="$2"; shift 2 ;;
        --restart) RESTART=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

case "$VERSION" in v1|v2|both) ;; *) echo "invalid --version: $VERSION" >&2; exit 2 ;; esac
cd "$REPO"
mkdir -p "$OUT/history"

if [[ "$RESTART" -eq 1 && -f "$OUT/run_chemcotbench.log" ]]; then
    stamp=$(date '+%Y%m%d-%H%M%S')
    cp -p "$OUT/run_chemcotbench.log" "$OUT/history/${stamp}-run_chemcotbench.log"
    rm "$OUT/run_chemcotbench.log"
fi

exec > >(tee -a "$OUT/run_chemcotbench.log") 2>&1
printf '%s\n' "$$" > "$OUT/driver.pid"
child=""
stop_requested=0
cleanup() { rm -f "$OUT/driver.pid"; }
forward_stop() {
    stop_requested=1
    [[ -z "$child" ]] || kill -TERM "$child" 2>/dev/null || true
}
trap cleanup EXIT
trap forward_stop INT TERM

run_stage() {
    echo "[driver $(date '+%F %T')] $1"
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

run_version() {
    local version=$1 benchmark eval_py data_var data_dir
    if [[ "$version" == v1 ]]; then
        benchmark=chemcotbench
        eval_py=$V1_EVAL_PY
        data_var=CHEMCOTBENCH_V1_DATA_DIR
        data_dir="$REPO/resources/chemcotbench/v1"
    else
        benchmark=chemcotbench-v2
        eval_py=$V2_EVAL_PY
        data_var=CHEMCOTBENCH_V2_DATA_DIR
        data_dir="$REPO/resources/chemcotbench/v2"
    fi
    export "$data_var=$data_dir"

    local listing=()
    mapfile -t listing < <("$GEN_PY" -m molbench list-tasks --benchmark "$benchmark")
    local selected=()
    local line name family
    for line in "${listing[@]}"; do
        IFS=$'\t' read -r name family _ <<< "$line"
        [[ -z "$TASK" || "$name" == "$TASK" ]] || continue
        [[ -z "$FAMILY" || "$family" == "$FAMILY" ]] || continue
        selected+=("$name")
    done
    [[ ${#selected[@]} -gt 0 ]] || { echo "no tasks selected for $benchmark" >&2; exit 2; }

    for name in "${selected[@]}"; do
        args=(
            -m molbench generate --benchmark "$benchmark" --model "$MODEL"
            --task "$name" --max-batch-size 8 --batching length-aware
            --length-batch-policy '256:8,512:4,1024:2,inf:1'
            --token-budget 16384 --max-padding-ratio 1.20
            --long-prompt-threshold 1024 --out-dir "$OUT"
        )
        [[ "$RESTART" -eq 0 ]] || args+=(--restart)
        run_stage "GEN $benchmark/$name" "$GEN_PY" "${args[@]}"
        if [[ "$STOP_AFTER" == "$benchmark/$name" || "$STOP_AFTER" == "$name" ]]; then
            echo "[driver] STOP_AFTER $benchmark/$name"
            exit 0
        fi
    done

    eval_args=(-m molbench evaluate --benchmark "$benchmark" --models "$MODEL" --out-dir "$OUT")
    for name in "${selected[@]}"; do eval_args+=(--task "$name"); done
    run_stage "EVAL $benchmark (${#selected[@]} tasks)" "$eval_py" "${eval_args[@]}"
}

echo "[driver $(date '+%F %T')] START version=$VERSION model=$MODEL out=$OUT"
if [[ "$VERSION" == v2 || "$VERSION" == both ]]; then run_version v2; fi
if [[ "$VERSION" == v1 || "$VERSION" == both ]]; then run_version v1; fi
echo "[driver $(date '+%F %T')] ALL DONE"
