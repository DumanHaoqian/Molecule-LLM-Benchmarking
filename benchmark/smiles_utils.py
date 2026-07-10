"""SMILES parsing / extraction helpers (RDKit-backed)."""
from __future__ import annotations

import re
from typing import Optional

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")  # silence RDKit parse warnings

# atom-wise SMILES tokenizer used for character/atom-level BLEU (MolT5 style)
_SMI_REGEX = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])"
)


def tokenize_smiles(smiles: str) -> list[str]:
    """Split a SMILES string into chemically meaningful tokens."""
    return [t for t in _SMI_REGEX.findall(smiles) if t]


def canonicalize(smiles: str) -> Optional[str]:
    """Return the RDKit canonical SMILES, or None if it does not parse."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_THINK_RE = re.compile(r"</think>", re.DOTALL)


def parse_answer(text: str, reasoning: bool) -> str:
    """Extract the final answer from a (possibly reasoning) model output.

    For reasoning models the answer sits inside ``<answer>...</answer>``; if the
    closing tag was truncated we fall back to whatever follows ``</think>``,
    then to the raw text. Non-reasoning outputs are returned stripped.
    """
    text = (text or "").strip()
    if not reasoning:
        return text
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()
    # truncated / malformed: take everything after the reasoning block
    parts = _THINK_RE.split(text, maxsplit=1)
    tail = parts[1] if len(parts) > 1 else text
    return tail.replace("<answer>", "").replace("</answer>", "").strip()


def extract_smiles(text: str) -> str:
    """Pull the most likely SMILES string out of a free-form model answer.

    Strategy: try the whole (stripped) string first; otherwise scan
    whitespace-separated candidates and return the first that RDKit can parse
    (preferring the longest such candidate). Falls back to the first token.
    """
    if not text:
        return ""
    text = text.strip()
    # strip common markdown / code fences
    text = text.replace("`", " ").strip()
    if canonicalize(text) is not None:
        return text

    candidates = re.split(r"\s+", text)
    valid = [(len(c), c) for c in candidates if canonicalize(c) is not None]
    if valid:
        valid.sort(reverse=True)  # longest valid candidate wins
        return valid[0][1]
    return candidates[0] if candidates else ""
