"""Chemistry / text-answer utilities shared across benchmarks and metrics."""
from __future__ import annotations

import re
from typing import List, Optional

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# atom-wise SMILES tokenizer for atom-level BLEU (MolT5 style)
_SMI_REGEX = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])"
)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_THINK_RE = re.compile(r"</think>", re.DOTALL)


def tokenize_smiles(smiles: str) -> List[str]:
    return [t for t in _SMI_REGEX.findall(smiles) if t]


def canonicalize(smiles: str) -> Optional[str]:
    """RDKit canonical SMILES, or None if unparseable."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def parse_answer(text: str, reasoning: bool) -> str:
    """Extract the final answer from a (possibly reasoning) model output."""
    text = (text or "").strip()
    if not reasoning:
        return text
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()
    parts = _THINK_RE.split(text, maxsplit=1)
    tail = parts[1] if len(parts) > 1 else text
    return tail.replace("<answer>", "").replace("</answer>", "").strip()


def extract_smiles(text: str) -> str:
    """Pull the most likely SMILES out of a free-form answer."""
    if not text:
        return ""
    text = text.strip().replace("`", " ").strip()
    if canonicalize(text) is not None:
        return text
    candidates = re.split(r"\s+", text)
    valid = [(len(c), c) for c in candidates if canonicalize(c) is not None]
    if valid:
        valid.sort(reverse=True)
        return valid[0][1]
    return candidates[0] if candidates else ""
