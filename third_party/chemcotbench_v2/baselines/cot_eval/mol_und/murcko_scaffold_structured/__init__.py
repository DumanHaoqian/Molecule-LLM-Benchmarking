"""
murcko_scaffold Structured CoT Evaluation Pipeline
===================================================

Task description:
  Extract the Murcko scaffold from a given molecule. The Murcko scaffold is
  obtained by keeping all ring systems and the linker atoms connecting them,
  while removing all non-ring/non-linker substituents (side chains).

Structured output format (enforced via system prompt):
  [RING_SYSTEMS]:       <enumerated ring systems, e.g. "A: benzene, B: piperidine">
  [LINKERS]:            <linker chains connecting ring systems, or "none if fused">
  [SIDE_CHAINS_REMOVED]: <removed substituents, e.g. "-CH3, -OH, -COOH">
  [SCAFFOLD_SMILES]:    <predicted Murcko scaffold SMILES>
  Answer:               <SMILES>

Verification points:
  V1: [SCAFFOLD_SMILES] field present and non-empty
  V2: Value is RDKit-parseable (valid SMILES)
  V3: canonical([SCAFFOLD_SMILES]) == canonical(GT largest_scaffold)
  V4: canonical(Answer) == canonical(GT largest_scaffold)
  V5: canonical([SCAFFOLD_SMILES]) == canonical(Answer) (internal consistency)

State Score = (V1 + V2 + V3 + V4) / 4

What State Score measures:
  Evaluates whether the model can correctly identify the Murcko scaffold.
  V3 and V4 test correctness from two independent output locations (field vs answer
  line), providing redundancy. A full score requires a chemically correct scaffold
  expressed as a valid, canonical SMILES.
"""
