"""
ring_sys_scaffold Structured CoT Evaluation Pipeline
=====================================================

Task description:
  Yes/No classification. Given a molecule SMILES and a proposed scaffold SMILES,
  determine whether the proposed scaffold IS the ring system scaffold of that molecule.
  The ring system scaffold contains ONLY ring atoms from the molecule — all fused and
  spiro ring systems are preserved, but all non-ring substituents are removed.

Structured output format (enforced via system prompt):
  [MOL_RINGS]:               <ring systems found in the molecule>
  [SCAFFOLD_RINGS]:          <ring systems present in the proposed scaffold>
  [NON_RING_ATOMS_IN_SCAFFOLD]: <"Yes" if scaffold contains non-ring atoms; "No" otherwise>
  [TOPOLOGY_MATCH]:          <"Yes" if scaffold ring topology matches molecule; "No" otherwise>
  Answer:                    <Yes/No>

Verification points:
  V1: [MOL_RINGS], [NON_RING_ATOMS_IN_SCAFFOLD], and Answer all extracted
  V2: Both molecule SMILES and scaffold SMILES from the DATASET are parseable by RDKit
  V3: [NON_RING_ATOMS_IN_SCAFFOLD] matches RDKit computation on the dataset scaffold
      (whether any atom in ring_system_scaffold is NOT in a ring)
  V4: [TOPOLOGY_MATCH] field is logically consistent with Answer
      (TOPOLOGY_MATCH="Yes" ↔ Answer="Yes")
  V5: Answer == GT label ("Yes"/"No")

State Score = (V1 + V2 + V3 + V4) / 4

What State Score measures:
  Evaluates whether the model correctly identifies ring topology and detects non-ring
  atoms in the proposed scaffold. V3 compares the model's structural claim against
  RDKit ground truth on the dataset scaffold. V4 checks internal logical consistency.
"""
