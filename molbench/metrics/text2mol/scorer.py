"""Text2Mol MLP association model (Edwards et al., 2021) — loader + scorer.

Reproduces the released ``test_outputfinal_weights.320.pt`` checkpoint:
  * text branch : SciBERT -> pooler_output -> Linear(768->300) -> LayerNorm(ln2)
  * mol branch  : mol2vec(300) -> MLP(300->600->600->300) -> LayerNorm(ln1)
  * score       : cosine similarity of the two 300-d embeddings

We reimplement ``mol2alt_sentence`` (pure RDKit) so we don't need the old
mol2vec package (gensim<4). The checkpoint was trained with MolT5's
``m2v_model.pkl`` (3003-word vocab) — NOT the larger DeepChem model.

Resources expected in the resource dir:
  * test_outputfinal_weights.320.pt
  * m2v_model.pkl
SciBERT config/tokenizer are fetched from HuggingFace on first use.
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem

CKPT_NAME = "test_outputfinal_weights.320.pt"
MOL2VEC_NAME = "m2v_model.pkl"
SCIBERT = "allenai/scibert_scivocab_uncased"
MAX_TEXT_LEN = 256


def mol2alt_sentence(mol, radius: int = 1) -> List[str]:
    radii = list(range(radius + 1))
    info: dict = {}
    AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)
    atoms = [a.GetIdx() for a in mol.GetAtoms()]
    dict_atoms = {x: {r: None for r in radii} for x in atoms}
    for element in info:
        for atom_idx, r in info[element]:
            dict_atoms[atom_idx][r] = element
    ids = []
    for atom in dict_atoms:
        for r in radii:
            ids.append(dict_atoms[atom][r])
    return [str(x) for x in ids if x is not None]


class MLPModel(nn.Module):
    def __init__(self, scibert_name: str = SCIBERT):
        super().__init__()
        from transformers import BertConfig, BertModel

        self.text_transformer_model = BertModel(BertConfig.from_pretrained(scibert_name))
        self.text_hidden1 = nn.Linear(768, 300)
        self.mol_hidden1 = nn.Linear(300, 600)
        self.mol_hidden2 = nn.Linear(600, 600)
        self.mol_hidden3 = nn.Linear(600, 300)
        self.temp = nn.Parameter(torch.tensor(0.07))
        self.ln1 = nn.LayerNorm(300)
        self.ln2 = nn.LayerNorm(300)
        self.relu = nn.ReLU()

    def forward(self, text_ids, text_mask, molecule):
        out = self.text_transformer_model(text_ids, attention_mask=text_mask)
        text_x = self.text_hidden1(out["pooler_output"])
        x = self.relu(self.mol_hidden1(molecule))
        x = self.relu(self.mol_hidden2(x))
        x = self.mol_hidden3(x)
        return self.ln2(text_x), self.ln1(x)


class Text2Mol:
    def __init__(self, resource_dir: str):
        from gensim.models import word2vec
        from transformers import BertTokenizerFast

        ckpt = os.path.join(resource_dir, CKPT_NAME)
        m2v = os.path.join(resource_dir, MOL2VEC_NAME)
        for p in (ckpt, m2v):
            if not os.path.exists(p):
                raise FileNotFoundError(p)

        self.kv = word2vec.Word2Vec.load(m2v)
        self.keys = self.kv.wv.key_to_index
        self.unk = self.kv.wv.get_vector("UNK")

        self.tokenizer = BertTokenizerFast.from_pretrained(SCIBERT)
        self.model = MLPModel()
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        state.pop("text_transformer_model.embeddings.position_ids", None)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def _featurize(self, smiles: str) -> Optional[np.ndarray]:
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            return None
        sent = mol2alt_sentence(mol, 1)
        if not sent:
            return None
        vecs = [self.kv.wv.get_vector(w) if w in self.keys else self.unk for w in sent]
        return np.sum(vecs, axis=0)

    @torch.no_grad()
    def pair_similarities(
        self, mols: List[str], texts: List[str], device: str = "cpu", batch_size: int = 32
    ) -> List[Optional[float]]:
        """Per-pair cosine similarity, aligned to input (None if mol invalid)."""
        self.model.to(device)
        results: List[Optional[float]] = [None] * len(mols)
        idxs, feats, keep_texts = [], [], []
        for i, (smi, txt) in enumerate(zip(mols, texts)):
            v = self._featurize(smi)
            if v is not None:
                idxs.append(i)
                feats.append(v)
                keep_texts.append(txt)
        for b in range(0, len(feats), batch_size):
            mol_vec = torch.tensor(
                np.stack(feats[b : b + batch_size]), dtype=torch.float32, device=device
            )
            enc = self.tokenizer(
                keep_texts[b : b + batch_size], padding=True, truncation=True,
                max_length=MAX_TEXT_LEN, return_tensors="pt",
            ).to(device)
            text_x, mol_x = self.model(enc["input_ids"], enc["attention_mask"], mol_vec)
            cos = torch.nn.functional.cosine_similarity(text_x, mol_x, dim=1).cpu().tolist()
            for j, c in zip(idxs[b : b + batch_size], cos):
                results[j] = float(c)
        return results

    def mean_similarity(
        self, mols: List[str], texts: List[str], device: str = "cpu", batch_size: int = 32
    ) -> float:
        sims = [s for s in self.pair_similarities(mols, texts, device, batch_size) if s is not None]
        return float(sum(sims) / len(sims)) if sims else 0.0


def load_text2mol(resource_dir: str) -> Text2Mol:
    return Text2Mol(resource_dir)
