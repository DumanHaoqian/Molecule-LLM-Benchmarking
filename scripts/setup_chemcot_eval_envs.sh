#!/usr/bin/env bash
set -euo pipefail

ROOT=${VENV_ROOT:-/home/haoqian/Data/SAERAG/venvs}
VERSION=${1:-both}

create_env() {
    local name=$1
    local python=$2
    local target="$ROOT/$name"
    local current_version=""
    if [[ -x "$target/bin/python" ]]; then
        current_version=$("$target/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    fi
    if [[ "$current_version" != "3.10" ]]; then
        "$python" -m venv --clear "$target"
    fi
    "$target/bin/python" -m pip install --upgrade pip wheel 'setuptools<81'
    "$target/bin/python" -m pip install \
        'numpy<2' pandas requests tqdm 'scikit-learn==1.2.2' 'rdkit==2023.9.6' \
        'scipy<1.12' 'nltk>=3.9' 'huggingface_hub<1' fuzzywuzzy openpyxl networkx
    # PyTDC's full dependency set pins an obsolete RDKit and pulls unrelated
    # single-cell packages. ChemCoTBench only imports tdc.Oracle, so install the
    # pinned package without resolving that oversized, conflicting extras set.
    "$target/bin/python" -m pip install --no-deps pytdc==1.1.15
    echo "[setup] ready: $target"
}

case "$VERSION" in
    v1) create_env chemcotbench-v1-eval "${PYTHON_V1:-${PYTHON:-python3.10}}" ;;
    v2) create_env chemcotbench-v2-eval "${PYTHON_V2:-${PYTHON:-python3.10}}" ;;
    both)
        create_env chemcotbench-v1-eval "${PYTHON_V1:-${PYTHON:-python3.10}}"
        create_env chemcotbench-v2-eval "${PYTHON_V2:-${PYTHON:-python3.10}}"
        ;;
    *) echo "usage: $0 [v1|v2|both]" >&2; exit 2 ;;
esac
