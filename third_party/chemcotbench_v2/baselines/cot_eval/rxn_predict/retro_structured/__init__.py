"""
retro Structured CoT Evaluation Pipeline
=========================================

Task description:
  Solve a retrosynthesis problem — given a product SMILES (and optional reagents),
  predict the reactant SMILES. The model must identify the bond formed in the forward
  reaction, interpret reagent roles, and propose a set of reactants. GT is a
  dot-separated SMILES string of all reactants.

Structured output format (enforced via system prompt appended to query):
  [PRODUCT_FG]:      <key functional group and bond to disconnect>
  [REAGENT_ROLES]:   <each non-trivial reagent and its role>
  [DISCONNECTION]:   <retrosynthetic arrow notation>
  [REACTANT_SMILES]: <predicted reactant SMILES, dot-separated if multiple>
  Answer:            <SMILES>

Verification points:
  V1: [REACTANT_SMILES] field present and non-empty
  V2: All individual SMILES in [REACTANT_SMILES] (split by '.') parseable by RDKit
  V3: canonical set comparison — smiles_match_set(reactant_smiles_field, GT)
      where both sides are split on '.' and each fragment is canonicalized
  V4: [PRODUCT_FG] field extracted and non-empty (model identified functional group)
  V5: smiles_match_set(reactant_smiles_field, Answer) (internal consistency)

State Score = (V1 + V2 + V3 + V4) / 4

What State Score measures:
  Evaluates whether the model can correctly predict reactants via retrosynthetic
  reasoning. The set-based comparison in V3 handles cases where reactant order or
  notation differs. V4 checks that the model correctly identified the disconnection
  site, which is the core reasoning step.
"""
