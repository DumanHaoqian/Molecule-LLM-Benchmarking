"""Molecule-LLM benchmarking harness.

Evaluate LLMs on molecule understanding / generation benchmarks.
Currently supports the ChEBI-20 dataset (duongttr/chebi-20) with the
ChemDFM-v2.0-14B model on two tasks:

  * molecule captioning     : SMILES  -> natural-language description
  * caption2smiles          : description -> SMILES
"""
