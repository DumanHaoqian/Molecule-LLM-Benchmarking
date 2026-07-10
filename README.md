# Molecule-LLM-Benchmarking

A **modular** harness for evaluating LLMs on molecule understanding / generation
benchmarks. Three independent axes — **models**, **benchmarks**, **metrics** —
so adding a new benchmark (e.g. ChemCoTBench) means adding one folder and
touching no existing code.

## Supported today

| Axis | Values |
|------|--------|
| Models     | `chemdfm-v2` (ChemDFM-v2.0-14B, direct), `chemdfm-r` (ChemDFM-R-14B, reasoning) |
| Benchmarks | `chebi20`, `tomg`, `chemcotbench` (V1), `chemcotbench-v2` |

**chebi20** — tasks captioning (SMILES→text) + caption2smiles (text→SMILES).
Metrics: captioning = BLEU-2/4, ROUGE-1/2/L, METEOR, Text2Mol; caption2smiles =
BLEU, exact match, Levenshtein, MACCS/RDK/Morgan FTS, FCD, Text2Mol, Validity.
(MolT5 / ChEBI-20 protocol.)

**tomg** — text-based open molecule generation. 9 subtasks in 3 groups
(MolCustom / MolEdit / MolOpt). Metrics per group: **SR** (success rate) and
**WSR** (weighted success rate = SR × novelty for MolCustom, SR × similarity for
MolEdit/MolOpt), plus the average. MolCustom novelty needs a ZINC250k reference
(`scripts/download_tomg_zinc.sh`, `TOMG_ZINC_PATH`).

**chemcotbench** — the 1,495-sample gated V1 release, represented as 19
independently resumable source tasks. Metrics cover molecule understanding,
editing, optimization, reaction prediction, and mechanism selection.

**chemcotbench-v2** — 5,620 samples across 31 implementation subtasks. The
pinned official parsers and verifiers produce separate Layer 1 outcome, Layer 2
template-adherence, and Layer 3 process-validity results, then aggregate them
into the paper's 18 reporting tasks.

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

Generation and per-example evaluation are transactional. Every generated
record is appended separately to a schema-v3 `.partial.jsonl`, flushed and
`fsync`'d. Schema v2 artifacts have an explicit compatibility reader. Re-running
the same command resumes after validating the dataset/model/config/code
fingerprint. `--restart` archives active artifacts before starting over; it
never mixes incompatible runs. A forced process kill loses at most the current
batch. Final JSONL files are sorted by `example_index` and atomically replaced.

ChEBI captioning uses conservative SMILES-length batches by default: 16 up to
128 characters, 8 up to 256, 4 up to 384, 2 up to 512, and 1 above 512. The
model also enforces the token budget after applying its full chat template.
Progress logs report prompt/output lengths, peak reserved GPU memory, elapsed
time, ETA, and a heartbeat while a long batch is still computing.

Batch planning is based on both character buckets and the exact rendered chat
prompt token count. Inputs above `--long-prompt-threshold` are forced to batch
1, batches cannot exceed `--max-padding-ratio`, and CUDA OOM recursively splits
the physical batch. ChemCoTBench-V2 output budgets are task-specific values
derived from the committed ChemDFM tokenizer profile at
`resources/chemcotbench/v2_token_profile.json`.

## ChemCoTBench setup and run

Download and verify the pinned snapshots. V1 requires accepting its Hugging
Face access conditions and setting `HF_TOKEN`.

```bash
python scripts/fetch_chemcotbench.py --version v2
HF_TOKEN=... python scripts/fetch_chemcotbench.py --version v1
bash scripts/setup_chemcot_eval_envs.sh both
```

The evaluator-only environments use Python 3.10, scikit-learn 1.2.2, and
RDKit 2023.9.6 because the released JNK TDC oracle contains a legacy sklearn
pickle. They do not install PyTorch and remain isolated from the ChemDFM
generation environment. Oracle binaries are ignored by Git; their expected
sizes and SHA256 digests are tracked in
`resources/chemcotbench/oracle_sources.json`.

List the independent task names, run one family, or use the idempotent driver:

```bash
python -m molbench list-tasks --benchmark chemcotbench-v2
python -m molbench generate --benchmark chemcotbench-v2 --model chemdfm-v2 \
  --family mol_edit --max-batch-size 8 --max-padding-ratio 1.20 \
  --long-prompt-threshold 1024 --out-dir results/chemcotbench
bash scripts/run_chemcotbench.sh --version v2 --family mol_edit
```

V2 generation uses the exact released task-specific system/user prompts. The
model completion is retained as `raw_output`; model-level answer extraction is
stored separately as `answer_text`. A known defect in the pinned public release
omits both model-facing SMILES for `mol_und.smiles_equivalent.0097`; the loader
explicitly reconstructs them from the released canonical states and records an
`input_repair` object in that example and its run fingerprint.

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

## Adding a new benchmark

1. Create `molbench/benchmarks/<name>/benchmark.py` with:
   - a `Benchmark` subclass implementing `load_task(task_name, split, limit)`
   and `tasks()`,
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

- **ChemDFM-R** emits `<think>…</think><answer>…</answer>`; artifacts retain the
  complete completion and store the extracted `<answer>` separately. Tasks
  using their own formal-trace system prompt can disable the extra reasoning
  budget so the formal trace itself owns the full output allowance.
- **Text2Mol** is the MLP association model (SciBERT + mol2vec); it needs
  `test_outputfinal_weights.320.pt` + MolT5's `m2v_model.pkl` (the model the
  checkpoint was trained with). Validated: matched ChEBI-20 gold pairs ~0.64,
  shuffled ~0.15. No torch_geometric required.
