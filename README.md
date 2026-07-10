# Molecule-LLM-Benchmarking

A **modular** harness for evaluating LLMs on molecule understanding / generation
benchmarks. Three independent axes — **models**, **benchmarks**, **metrics** —
so adding a new benchmark (e.g. ChemCoTBench) means adding one folder and
touching no existing code.

## Supported today

| Axis | Values |
|------|--------|
| Models     | `chemdfm-v2` (ChemDFM-v2.0-14B, direct), `chemdfm-r` (ChemDFM-R-14B, reasoning) |
| Benchmarks | `chebi20` ([`duongttr/chebi-20`](https://huggingface.co/datasets/duongttr/chebi-20)), `tomg` ([`phenixace/S2-TOMG-Bench`](https://huggingface.co/datasets/phenixace/S2-TOMG-Bench)) |

**chebi20** — tasks captioning (SMILES→text) + caption2smiles (text→SMILES).
Metrics: captioning = BLEU-2/4, ROUGE-1/2/L, METEOR, Text2Mol; caption2smiles =
BLEU, exact match, Levenshtein, MACCS/RDK/Morgan FTS, FCD, Text2Mol, Validity.
(MolT5 / ChEBI-20 protocol.)

**tomg** — text-based open molecule generation. 9 subtasks in 3 groups
(MolCustom / MolEdit / MolOpt). Metrics per group: **SR** (success rate) and
**WSR** (weighted success rate = SR × novelty for MolCustom, SR × similarity for
MolEdit/MolOpt), plus the average. MolCustom novelty needs a ZINC250k reference
(`scripts/download_tomg_zinc.sh`, `TOMG_ZINC_PATH`).

## Architecture

```
molbench/
  core/            # framework — no benchmark/model specifics
    model.py         Model ABC (generate → clean answers)
    task.py          Task ABC (build_prompt / postprocess / evaluate / columns)
    benchmark.py     Benchmark ABC (load + tasks)
    registry.py      name → model/benchmark registration
    io.py            prediction jsonl schema + paths
    runner.py        generic generate + evaluate + table rendering
  models/          # reusable across every benchmark
    chemdfm.py       ChemDFM v2/R, self-registered
  metrics/         # reusable metric library
    text.py  molecule.py  fcd.py  text2mol/
  benchmarks/      # one folder per benchmark = data + tasks + prompts + wiring
    chebi20/
  utils/chem.py    # canonicalize / extract_smiles / parse_answer / tokenize
  cli.py           # `generate` / `evaluate` subcommands
```

**Two-stage pipeline** (decoupled venvs so metric deps never clash with the
model runtime): `generate` (chemdfm venv) writes prediction jsonl →
`evaluate` (ChEBI-20-Eva venv) scores it and renders **one table per task**.
Because they are decoupled, metrics (incl. Text2Mol) can be recomputed without
re-running any model.

## Setup

Generation venv `/home/haoqian/Data/SAERAG/venvs/chemdfm` is already provisioned.
Evaluation venv:

```bash
bash scripts/setup_eval_env.sh        # builds ChEBI-20-Eva + metric deps (CPU)
bash scripts/download_text2mol.sh     # Text2Mol checkpoint + mol2vec model (~435MB)
```

## Run

```bash
# stage 1 — generate (chemdfm venv; resumable by default)
source /home/haoqian/Data/SAERAG/venvs/chemdfm/bin/activate
python -m molbench generate --benchmark chebi20 --model chemdfm-v2 --split test \
  --batching length-aware --max-batch-size 16 --out-dir results/full
python -m molbench generate --benchmark chebi20 --model chemdfm-r  --split test --out-dir results/full
#   --task {captioning|caption2smiles|all}  --limit N  --do-sample
#   --resume / --no-resume  --restart  --slice START:STOP
#   --length-batch-policy '128:16,256:8,384:4,512:2,inf:1'
#   --token-budget 16384

# stage 2 — evaluate (ChEBI-20-Eva venv)
source /home/haoqian/Data/SAERAG/venvs/ChEBI-20-Eva/bin/activate
export TEXT2MOL_DIR=$PWD/text2mol_resources          # enables the Text2Mol column
python -m molbench evaluate --benchmark chebi20 --models chemdfm-v2 chemdfm-r --split test --out-dir results/full
```

TOMG-Bench (`--limit` caps examples **per subtask**, so all 9 are covered):

```bash
# generate (chemdfm venv)
python -m molbench generate --benchmark tomg --model chemdfm-r --limit 10 --out-dir results/tomg
# evaluate (ChEBI-20-Eva venv)
export TOMG_ZINC_PATH=$PWD/tomg_resources/zinc250k.txt      # enables MolCustom novelty/WSR
python -m molbench evaluate --benchmark tomg --models chemdfm-r --out-dir results/tomg
```

Artifacts (git-ignored, under `--out-dir`):
`<benchmark>__<model>__<task>__<split>.jsonl` (+ `.meta.json` and `.run.json`),
`tables_<benchmark>_<split>.md`, `metrics_<benchmark>_<split>.json`.
Without `TEXT2MOL_DIR`, all other metrics still compute and Text2Mol shows `—`.

Generation and per-example evaluation are transactional. Every completed batch
is appended to a schema-v2 `.partial.jsonl`, flushed and `fsync`'d. Re-running
the same command resumes after validating the dataset/model/config/code
fingerprint. `--restart` archives active artifacts before starting over; it
never mixes incompatible runs. A forced process kill loses at most the current
batch. Final JSONL files are sorted by `example_index` and atomically replaced.

ChEBI captioning uses conservative SMILES-length batches by default: 16 up to
128 characters, 8 up to 256, 4 up to 384, 2 up to 512, and 1 above 512. The
model also enforces the token budget after applying its full chat template.
Progress logs report prompt/output lengths, peak reserved GPU memory, elapsed
time, ETA, and a heartbeat while a long batch is still computing.

The full v2 driver owns its log and supports stage boundaries:

```bash
nohup bash scripts/run_chemdfm_v2.sh --restart >/dev/null 2>&1 &
# Stop cleanly after ChEBI generation:
bash scripts/run_chemdfm_v2.sh --stop-after chebi20/captioning
```

Legacy smoke predictions must be explicitly migrated rather than guessed by
the reader:

```bash
python scripts/migrate_legacy_artifacts.py OLD.jsonl NEW.v2.jsonl --task captioning
```

## Adding a new benchmark (e.g. ChemCoTBench)

1. Create `molbench/benchmarks/chemcotbench/benchmark.py` with:
   - a `Benchmark` subclass implementing `load(split, limit)` and `tasks()`,
   - one `Task` subclass per direction implementing `build_prompt`,
   `postprocess`, `evaluate` (calling the shared `molbench.metrics`), and a
   `columns` list for its results table,
   - `register_benchmark("chemcotbench", ChemCoTBenchmark)` at import time.
2. Add `from . import chemcotbench` to `molbench/benchmarks/__init__.py`.
3. If it needs a new metric, add a function under `molbench/metrics/`.

No changes to `core/`, `models/`, or the CLI. Reuse existing metrics freely.

## Adding a new model

Add a `ModelSpec` in `molbench/models/` (path, params, system prompt,
`reasoning` flag) and `register_model(...)`. Generation is otherwise generic;
the model owns its chat template, system prompt, and answer parsing.

## Notes

- **ChemDFM-R** emits `<think>…</think><answer>…</answer>`; the model parses the
  `<answer>` block and gets extra `max_new_tokens` headroom for the trace.
- **Text2Mol** is the MLP association model (SciBERT + mol2vec); it needs
  `test_outputfinal_weights.320.pt` + MolT5's `m2v_model.pkl` (the model the
  checkpoint was trained with). Validated: matched ChEBI-20 gold pairs ~0.64,
  shuffled ~0.15. No torch_geometric required.
