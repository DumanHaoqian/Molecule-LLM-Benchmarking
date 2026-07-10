"""
baselines/cot_eval/utils.py
===========================
CoT 评估管线公共工具库：
  - API 客户端（claude-sonnet-4-6 / claude-opus-4-6）
  - CoT 解析器
  - RDKit 子结构验证器
  - 数据加载器（ring_count, murcko_scaffold）
"""

import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI
from rdkit import Chem

# ── API 配置 ────────────────────────────────────────────────────────────────
API_KEY  = os.environ.get("OPENAI_API_KEY", "")
API_BASE = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
SONNET   = "claude-sonnet-4-6"
OPUS     = "claude-opus-4-6"

# ── 路径配置 ────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(os.environ.get("CHEMCOTBENCH_ROOT", Path(__file__).resolve().parents[3]))
RING_COUNT_DATA = PROJECT_ROOT / "raw_data/difficulty/mol_und/ring_count.json"
RESULTS_ROOT    = PROJECT_ROOT / "results/cot_eval/mol_und/ring_count"
TEMPLATE_FILE   = RESULTS_ROOT / "gold_template.json"

MURCKO_DATA          = PROJECT_ROOT / "raw_data/difficulty/mol_und/murcko_scaffold.json"
MURCKO_RESULTS_ROOT  = PROJECT_ROOT / "results/cot_eval/mol_und/murcko_scaffold"
MURCKO_TEMPLATE_FILE = MURCKO_RESULTS_ROOT / "gold_template.json"

# ── OpenAI 客户端 ────────────────────────────────────────────────────────────
_client = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=API_KEY, base_url=API_BASE, timeout=120.0)
    return _client


def call_llm(
    prompt: str,
    model: str = SONNET,
    max_tokens: int = 1200,
    temperature: float = 0.3,
    system: str | None = None,
    retry: int = 3,
) -> tuple[str, dict]:
    """
    调用 LLM，返回 (response_text, usage_dict)。
    usage_dict = {"input_tokens": ..., "output_tokens": ..., "model": ...}
    不支持 system role，统一前置拼入 user 消息。
    """
    if system:
        content = f"[系统指令]\n{system}\n\n[用户]\n{prompt}"
    else:
        content = prompt

    for attempt in range(retry):
        try:
            resp = get_client().chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            usage = {
                "input_tokens":  resp.usage.prompt_tokens,
                "output_tokens": resp.usage.completion_tokens,
                "model":         model,
            }
            return resp.choices[0].message.content, usage
        except Exception as e:
            wait = 5 * (attempt + 1)
            print(f"  [LLM] 第{attempt+1}次调用失败({type(e).__name__}): {str(e)[:80]}")
            if attempt < retry - 1:
                print(f"  [LLM] 等待 {wait}s 后重试...")
                time.sleep(wait)
            else:
                raise


def estimate_cost(usage_list: list[dict]) -> float:
    """
    粗略估算 USD 成本（sonnet: $3/$15 per M tokens, opus: $15/$75）。
    """
    total = 0.0
    for u in usage_list:
        m = u.get("model", SONNET)
        if "opus" in m:
            total += u["input_tokens"] * 15e-6 + u["output_tokens"] * 75e-6
        else:
            total += u["input_tokens"] * 3e-6  + u["output_tokens"] * 15e-6
    return total


# ── 数据加载 ─────────────────────────────────────────────────────────────────

def load_ring_count(min_rounds: int = 1) -> list[dict]:
    """
    加载 ring_count 推理结果，展开每个有效 round 为独立记录。
    返回字段：smiles, ring_name, ring_smarts, gt_count, difficulty, pass_rate,
              round_idx, reasoning, pred_answer, is_correct, raw_response
    """
    with open(RING_COUNT_DATA) as f:
        raw = json.load(f)

    records = []
    for item in raw:
        rounds = item.get("rounds", [])
        if len(rounds) < min_rounds:
            continue
        for i, r in enumerate(rounds):
            parsed = r.get("parsed")
            if not isinstance(parsed, dict):
                continue
            reasoning = parsed.get("reasoning", "")
            if not reasoning:
                continue
            records.append({
                "smiles":       item["smiles"],
                "ring_name":    item["ring_name"],
                "ring_smarts":  item["ring"],
                "gt_count":     item["count"],
                "difficulty":   item["difficulty"],
                "pass_rate":    item["pass_rate"],
                "round_idx":    i,
                "reasoning":    reasoning,
                "pred_answer":  r.get("pred_answer"),
                "is_correct":   r.get("is_correct"),
                "raw_response": r.get("raw_response", ""),
                "_item_idx":    raw.index(item),  # 原始列表下标
            })
    return records


def sample_balanced(records: list[dict], n_correct: int, n_wrong: int,
                    seed: int = 42, ring_diversity: bool = True) -> list[dict]:
    """
    按对错均衡抽样，可选按 ring_name 多样性采样。
    """
    import random
    rng = random.Random(seed)

    correct = [r for r in records if r["is_correct"]]
    wrong   = [r for r in records if not r["is_correct"]]

    def diverse_sample(pool, n):
        if not ring_diversity:
            return rng.sample(pool, min(n, len(pool)))
        # 优先选不同 ring_name
        by_ring: dict[str, list] = {}
        for r in pool:
            by_ring.setdefault(r["ring_name"], []).append(r)
        selected, rings = [], list(by_ring.keys())
        rng.shuffle(rings)
        for ring in rings:
            if len(selected) >= n:
                break
            selected.append(rng.choice(by_ring[ring]))
        # 补足
        remaining = [r for r in pool if r not in selected]
        rng.shuffle(remaining)
        selected += remaining[:max(0, n - len(selected))]
        return selected[:n]

    return diverse_sample(correct, n_correct) + diverse_sample(wrong, n_wrong)


# ── RDKit 验证 ────────────────────────────────────────────────────────────────

def rdkit_substructure_count(smiles: str, smarts: str) -> int | None:
    """
    返回 SMILES 分子中 SMARTS 模式的匹配数，失败返回 None。
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    query = Chem.MolFromSmarts(smarts)
    if query is None:
        return None
    return len(mol.GetSubstructMatches(query))


def rdkit_verify_record(rec: dict) -> dict:
    """
    对单条记录运行 RDKit 验证，返回验证结果字段：
      rdkit_count      : RDKit 实际计数
      rdkit_agrees_gt  : rdkit_count == gt_count
      pred_agrees_rdkit: pred_answer == rdkit_count
      grounding_status : 'grounded'|'over_count'|'under_count'|'parse_fail'
    """
    rc = rdkit_substructure_count(rec["smiles"], rec["ring_smarts"])
    if rc is None:
        return {
            "rdkit_count": None,
            "rdkit_agrees_gt": None,
            "pred_agrees_rdkit": None,
            "grounding_status": "parse_fail",
        }
    agrees_gt  = (rc == rec["gt_count"])
    # pred_answer 可能是 int 或字符串，统一转 int
    raw_pred = rec["pred_answer"]
    try:
        pred_val = int(raw_pred) if raw_pred is not None else None
    except (ValueError, TypeError):
        pred_val = None

    pred_agrees = (pred_val == rc) if pred_val is not None else None
    if pred_val is None:
        status = "no_pred"
    elif pred_val == rc:
        status = "grounded"
    elif pred_val > rc:
        status = "over_count"
    else:
        status = "under_count"
    return {
        "rdkit_count": rc,
        "rdkit_agrees_gt": agrees_gt,
        "pred_agrees_rdkit": pred_agrees,
        "grounding_status": status,
    }


# ── 结果 I/O ──────────────────────────────────────────────────────────────────

def save_json(obj, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"  [saved] {path}")


def load_json(path: str | Path) -> dict | list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── murcko_scaffold 数据加载 ──────────────────────────────────────────────────

def _smiles_connectivity_match(smi_a: str, smi_b: str) -> bool:
    """
    用 InChIKey 前 14 位（连接层）判断两个 SMILES 的骨架是否一致，忽略立体化学。
    """
    try:
        from rdkit.Chem.inchi import MolToInchiKey
        ma = Chem.MolFromSmiles(smi_a)
        mb = Chem.MolFromSmiles(smi_b)
        if ma is None or mb is None:
            return False
        return MolToInchiKey(ma)[:14] == MolToInchiKey(mb)[:14]
    except Exception:
        return False


def load_murcko_scaffold(only_parsed: bool = True) -> list[dict]:
    """
    加载 murcko_scaffold 推理结果（murcko_100_sonnet.json）。
    每条 JSON 对象直接作为一条记录，无需展开 rounds。

    返回字段：
      smiles, largest_scaffold, mol_complexity, scaffold_rings,
      scaffold_ratio, difficulty, difficulty_score,
      reasoning, pred_scaffold, is_correct,
      raw_response, attempts

    is_correct 使用"宽松"标准（InChI 连接性比较，忽略立体化学），
    与修正后的 JSON 文件中的 is_correct 字段一致。
    """
    with open(MURCKO_DATA, encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    for item in raw:
        if only_parsed and item.get("parsed") is None:
            continue
        parsed = item.get("parsed") or {}
        reasoning = parsed.get("reasoning", "") or ""
        if only_parsed and not reasoning:
            continue
        records.append({
            "smiles":           item["smiles"],
            "largest_scaffold": item["largest_scaffold"],
            "mol_complexity":   item.get("mol_complexity"),
            "scaffold_rings":   item.get("scaffold_rings"),
            "scaffold_ratio":   item.get("scaffold_ratio"),
            "difficulty":       item.get("difficulty"),
            "difficulty_score": item.get("difficulty_score"),
            "reasoning":        reasoning,
            "pred_scaffold":    item.get("pred_scaffold"),
            "is_correct":       item.get("is_correct", False),
            "raw_response":     item.get("raw_response", ""),
            "attempts":         item.get("attempts", 1),
        })
    return records


def sample_murcko_balanced(records: list[dict], n_correct: int, n_wrong: int,
                            seed: int = 42) -> list[dict]:
    """
    按对错均衡抽样，同时尽量覆盖三个难度等级。
    """
    import random
    rng = random.Random(seed)

    correct = [r for r in records if r["is_correct"]]
    wrong   = [r for r in records if not r["is_correct"]]

    def diverse_sample(pool, n):
        by_diff: dict[str, list] = {}
        for r in pool:
            by_diff.setdefault(r.get("difficulty", "unknown"), []).append(r)
        selected = []
        diffs = list(by_diff.keys())
        rng.shuffle(diffs)
        for d in diffs:
            if len(selected) >= n:
                break
            selected.append(rng.choice(by_diff[d]))
        remaining = [r for r in pool if r not in selected]
        rng.shuffle(remaining)
        selected += remaining[:max(0, n - len(selected))]
        return selected[:n]

    return diverse_sample(correct, n_correct) + diverse_sample(wrong, n_wrong)
