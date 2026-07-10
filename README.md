# Molecule-LLM-Benchmarking

Benchmarking harness for evaluating LLMs on **molecule understanding** and
**molecule generation** tasks.

## Currently supported

| Component | Value |
|-----------|-------|
| Dataset   | [`duongttr/chebi-20`](https://huggingface.co/datasets/duongttr/chebi-20) (ChEBI-20) |
| Model     | `ChemDFM-v2.0-14B` (local, Qwen2 arch) |
| Tasks     | Molecule captioning (SMILES → text), Caption2SMILES (text → SMILES) |

### Tasks & metrics

- **Molecule captioning** — input SMILES, generate a natural-language
  description. Metrics: **BLEU-2, BLEU-4, ROUGE-1/2/L, METEOR**.
- **Text-based molecule generation / caption2SMILES** — input a description,
  generate a SMILES. Metrics: **atom-level BLEU, exact match, Levenshtein,
  validity, MACCS / RDK / Morgan fingerprint Tanimoto (FTS)**.

These follow the standard MolT5 / ChEBI-20 evaluation protocol.

## Setup

Uses the existing `chemdfm` venv; this only adds `rouge_score` + NLTK corpora:

```bash
bash scripts/setup_env.sh
```

## Run

```bash
source /home/haoqian/Data/SAERAG/venvs/chemdfm/bin/activate

# quick smoke test (5 examples, both tasks)
python run_benchmark.py --task both --limit 5 --out-dir results/smoke

# full evaluation on the ChEBI-20 test split (3,301 examples)
python run_benchmark.py --task both --split test --batch-size 8 \
    --out-dir results/chemdfm
```

Outputs (under `--out-dir`, git-ignored):
- `captioning_<split>_preds.jsonl` — per-example SMILES / reference / prediction
- `caption2smiles_<split>_preds.jsonl` — description / gold / raw output / parsed SMILES
- `summary_<split>.json` — run config + aggregate metrics

### Useful flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--task` | `both` | `captioning`, `caption2smiles`, or `both` |
| `--split` | `test` | `train` / `validation` / `test` |
| `--limit` | none | cap #examples (debugging) |
| `--batch-size` | `8` | generation batch size |
| `--max-new-tokens` | `256` | max generated tokens |
| `--do-sample` | off | sample instead of greedy (greedy = reproducible default) |
| `--model-path` | ChemDFM-v2.0-14B | HF model dir |

## Layout

```
benchmark/
  model.py           # ChemDFM wrapper (batched, greedy, chat-template)
  data.py            # ChEBI-20 loader
  prompts.py         # per-task instruction templates
  smiles_utils.py    # RDKit canonicalization, SMILES extraction/tokenization
  metrics_caption.py # BLEU / ROUGE / METEOR
  metrics_molgen.py  # exact match / validity / FTS / Levenshtein / BLEU
  tasks.py           # orchestration + prediction dumping
run_benchmark.py     # CLI
scripts/setup_env.sh # extra deps
```

## Extending

- **New model**: add a wrapper exposing `generate(list[str]) -> list[str]`
  (see `benchmark/model.py`) and pass it to the functions in `benchmark/tasks.py`.
- **New dataset**: add a loader in `benchmark/data.py` and task functions in
  `benchmark/tasks.py`.
