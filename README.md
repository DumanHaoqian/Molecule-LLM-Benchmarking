# Molecule-LLM-Benchmarking

Benchmarking harness for evaluating LLMs on **molecule understanding** and
**molecule generation** tasks.

## Currently supported

| Component | Value |
|-----------|-------|
| Dataset   | [`duongttr/chebi-20`](https://huggingface.co/datasets/duongttr/chebi-20) (ChEBI-20) |
| Models    | `ChemDFM-v2.0-14B` (direct), `ChemDFM-R-14B` (reasoning) |
| Tasks     | Molecule captioning (SMILES → text), Caption2SMILES (text → SMILES) |

### Tasks & metrics

- **Molecule captioning** — input SMILES, generate a description.
  Metrics: **BLEU-2, BLEU-4, ROUGE-1/2/L, METEOR, Text2Mol**.
- **Text-based molecule generation / caption2SMILES** — input a description,
  generate a SMILES. Metrics: **BLEU, exact match, Levenshtein, MACCS / RDK /
  Morgan FTS, FCD, Text2Mol, Validity**.

Follows the MolT5 / ChEBI-20 evaluation protocol.

## Two-stage design

Generation and evaluation run in **separate virtualenvs** so the heavy metric
dependencies never clash with the model runtime:

```
generate.py   (venv: chemdfm)       model → results/<model>__<task>__<split>.jsonl
     │
     ▼
evaluate.py   (venv: ChEBI-20-Eva)  predictions → metrics + two markdown tables
```

Because they are decoupled, you can (re)compute metrics — including Text2Mol —
without re-running any model.

## Setup

**Generation venv** (already provisioned): `/home/haoqian/Data/SAERAG/venvs/chemdfm`
(transformers, torch, datasets, rdkit).

**Evaluation venv** (`ChEBI-20-Eva`, CPU-only — fast enough, avoids the cu128 build):

```bash
bash scripts/setup_eval_env.sh          # builds venv + installs metric deps
bash scripts/download_text2mol.sh       # Text2Mol checkpoint + mol2vec model (~435MB)
```

## Run

### 1. Generate (chemdfm venv)

```bash
source /home/haoqian/Data/SAERAG/venvs/chemdfm/bin/activate

# smoke test: 10 examples, both tasks, both models
python generate.py --model chemdfm-v2 --task both --limit 10 --out-dir results/smoke
python generate.py --model chemdfm-r  --task both --limit 10 --out-dir results/smoke

# full ChEBI-20 test split
python generate.py --model chemdfm-v2 --task both --split test --out-dir results/full
python generate.py --model chemdfm-r  --task both --split test --out-dir results/full
```

### 2. Evaluate (ChEBI-20-Eva venv)

```bash
source /home/haoqian/Data/SAERAG/venvs/ChEBI-20-Eva/bin/activate
export TEXT2MOL_DIR=$PWD/text2mol_resources          # enables the Text2Mol column

python evaluate.py --results-dir results/full --models chemdfm-v2 chemdfm-r --split test
```

Outputs (git-ignored): `<model>__<task>__<split>.jsonl` (+ `.meta.json`) per run,
`tables_<split>.md` (the two tables), `metrics_<split>.json`.

If `TEXT2MOL_DIR` is unset, every other metric is still computed and Text2Mol
shows as `—`.

## Layout

```
benchmark/
  registry.py         # per-model config (path, params, system prompt, reasoning)
  model.py            # ChemDFM wrapper (batched, greedy, chat-template)
  generation.py       # stage 1: model → prediction jsonl
  data.py             # ChEBI-20 loader
  prompts.py          # per-task instruction templates
  smiles_utils.py     # canonicalization, answer/SMILES extraction, tokenization
  metrics_caption.py  # BLEU / ROUGE / METEOR
  metrics_molgen.py   # exact match / validity / FTS / Levenshtein / BLEU
  metrics_fcd.py      # FCD (fcd_torch)
  metrics_text2mol.py # Text2Mol (graceful when resources absent)
  text2mol_model.py   # Text2Mol MLP model + mol2vec featurizer (eval venv)
  evaluation.py       # stage 2: predictions → metrics + tables
  paths.py            # shared artifact path conventions
generate.py / evaluate.py   # CLIs for the two stages
scripts/setup_eval_env.sh   # build ChEBI-20-Eva venv
scripts/download_text2mol.sh# fetch Text2Mol checkpoint + mol2vec model
```

## Notes on the models

- **ChemDFM-v2.0-14B**: direct chat model; system prompt `"You are a helpful
  assistant."`; output is the answer as-is.
- **ChemDFM-R-14B**: reasoning model; emits `<think>…</think><answer>…</answer>`;
  the harness parses the `<answer>` block (larger `max_new_tokens` to fit the
  reasoning chain).

## Text2Mol metric

Text2Mol is the MLP association model from Edwards et al. (2021): SciBERT text
encoder + mol2vec molecule encoder, scored by cosine similarity. It needs two
downloaded artifacts (see `scripts/download_text2mol.sh`):

- `test_outputfinal_weights.320.pt` — the pretrained checkpoint
- `m2v_model.pkl` — **MolT5's** mol2vec word vectors (the checkpoint was trained
  with this specific model, not the larger DeepChem/samoturk one)

Validated: matched ChEBI-20 gold (SMILES, caption) pairs score ~0.64, shuffled
pairs ~0.15.

## Extending

- **New model**: add an entry to `benchmark/registry.py` (path, params, system
  prompt, `reasoning` flag). Generation is otherwise generic.
- **New dataset**: add a loader in `benchmark/data.py` and wire the columns in
  `benchmark/generation.py`.
