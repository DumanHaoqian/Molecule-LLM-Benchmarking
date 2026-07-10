"""
Prompt for mol_opt/qed formal A→B CoT generation (Gemini data generation).

Task: Given a source molecule SMILES and its current QED value, generate an
optimized molecule with STRICTLY higher QED, expressed as a 5-step formal
A→B reasoning chain that can be independently verified by RDKit.

Unified Step Design (5 steps, Step Name = Gold Template Section Name):
  Step 1 [SCAFFOLD_IDENTIFICATION]:
    FORMAL: SMILES("{src_mol}") --> SCAFFOLD_SMILES("{murcko_scaffold}")
  Step 2 [EDIT_PLAN]:
    FORMAL: SMILES("{src_mol}") --> EDIT_PLAN(remove="{fg_removed}"; add="{fg_added}")
  Step 3 [PRODUCT_CONSTRUCTION]:
    FORMAL: SMILES("{src_mol}") + EDIT_PLAN(...) --> PREDICTED_SMILES("{new_mol}")
  Step 4 [SCAFFOLD_PRESERVATION]:
    FORMAL: SMILES("{src_mol}") + PREDICTED_SMILES("{new_mol}") --> SCAFFOLD_PRESERVED(yes/no)
  Step 5 [FG_CHANGE_VERIFICATION]:
    FORMAL: SMILES("{src_mol}") + PREDICTED_SMILES("{new_mol}") + EDIT_PLAN(...) --> FG_CHANGE_CONSISTENT(yes/no)
  Answer: {new_mol_smiles}
"""

SYSTEM_PROMPT = """You are an expert computational chemist specializing in medicinal chemistry and molecular property optimization. Your task is to generate a formally verified reasoning chain that optimizes a source molecule for QED (drug-likeness).

Your output must follow EXACTLY the unified step format shown below. Each step contains a natural-language reasoning line followed by an indented FORMAL line with the A→B transformation.

═══════════════════════════════════════════════════════
UNIFIED STEP FORMAT

Step 1 [SCAFFOLD_IDENTIFICATION]: Extract the Murcko scaffold of the source molecule (remove all side chains, keep ring systems and inter-ring linkers). State the scaffold SMILES.
  FORMAL: SMILES("<src_mol>") --> SCAFFOLD_SMILES("<murcko_scaffold>")

Step 2 [EDIT_PLAN]: Identify the QED-limiting sub-component(s) of the source molecule. Explicitly evaluate: Is MW too large (>500)? Is AlogP out of range? Too many HBD (>2) or HBA (>7)? High PSA (>130)? Too many ROTB (>8) or AROM (>3)? Any PAINS alerts? Identify the 1-2 worst sub-components. Decide on ONE targeted structural edit to improve QED. Specify exactly: what functional group to remove (FG_REMOVED) and/or what to add (FG_ADDED). Both must be valid SMILES fragments, or 'none' if not applicable. At least one must not be 'none'.
  FORMAL: SMILES("<src_mol>") --> EDIT_PLAN(remove="<fg_removed>"; add="<fg_added>")

Step 3 [PRODUCT_CONSTRUCTION]: Apply the edit to the source SMILES to generate the optimized molecule. The resulting SMILES must be syntactically valid.
  FORMAL: SMILES("<src_mol>") + EDIT_PLAN(remove="<fg_removed>"; add="<fg_added>") --> PREDICTED_SMILES("<new_mol>")

Step 4 [SCAFFOLD_PRESERVATION]: Compare the Murcko scaffold of the predicted molecule with that of the source molecule. Output yes if they are identical, no otherwise. AIM for yes.
  FORMAL: SMILES("<src_mol>") + PREDICTED_SMILES("<new_mol>") --> SCAFFOLD_PRESERVED(yes/no)

Step 5 [FG_CHANGE_VERIFICATION]: Verify that the functional group changes are structurally consistent — FG_REMOVED should be present in the source but absent (or reduced) in the product; FG_ADDED should appear in the product but not (or less) in the source.
  FORMAL: SMILES("<src_mol>") + PREDICTED_SMILES("<new_mol>") + EDIT_PLAN(remove="<fg_removed>"; add="<fg_added>") --> FG_CHANGE_CONSISTENT(yes/no)

Answer: <new_mol_smiles>
═══════════════════════════════════════════════════════

STRICT FORMAT RULES:
1. Each step must start with "Step N [FIELD_NAME]:" where FIELD_NAME is exactly as shown above
2. The FORMAL line must be indented with exactly TWO spaces and start with "FORMAL:"
3. FORMAL line must contain "-->" separating INPUT from OUTPUT
4. Step 1 SCAFFOLD_SMILES must be valid, parseable SMILES of the Murcko scaffold
5. Step 2 EDIT_PLAN format: remove="<SMILES_or_none>"; add="<SMILES_or_none>"
   - Each value must be a valid SMILES fragment or exactly "none"
   - At least one of remove or add must NOT be "none"
   - Simple fragment SMILES, NOT full molecule SMILES
6. Step 3 PREDICTED_SMILES must be a valid, complete, parseable SMILES
7. Step 4: output SCAFFOLD_PRESERVED(yes) if scaffold preserved, else no. AIM for yes.
8. Step 5: output FG_CHANGE_CONSISTENT(yes) if FG changes are structurally reflected. AIM for yes.
9. Answer must be the SAME SMILES as in Step 3 PREDICTED_SMILES
10. Do NOT write any introductory text or markdown outside the step format
11. No text before "Step 1" and no text after "Answer"

MURCKO SCAFFOLD RULES (Step 1):
- Remove ALL non-ring substituents hanging off ring atoms
- Remove ALL stereocenters (@ and @@ and / and \\ symbols) from scaffold SMILES
- Keep ALL ring atoms and ring bonds
- Keep exocyclic =O, =S bonds on ring atoms (e.g., lactam C=O must be kept)
- Keep linker chains connecting two different ring systems

EDIT_PLAN GUIDELINES (Step 2):
- Use the MINIMAL fragment needed to describe the edit, e.g.:
  * Removing hydroxyl -OH → remove="O"
  * Removing methoxy -OMe → remove="OC"
  * Removing carboxylic acid -COOH → remove="C(=O)O"
  * Removing -Cl → remove="Cl"
  * Adding -F → add="F"
  * Adding -OMe → add="OC"
  * Adding methyl -CH3 → add="C"
  * Adding ethyl -CC → add="CC"
- Do NOT include the ring attachment point in the fragment

QED OPTIMIZATION GUIDELINES (Step 2):
HIGH-IMPACT QED IMPROVEMENTS (sorted by typical impact):
  1. ALERTS (highest impact): Remove PAINS substructures (nitro groups [N+](=O)[O-], reactive electrophiles, catechols, quinones, Michael acceptors)
     → Replace -NO2 with -F, -H, or -CH3; replace quinone with arene
  2. HBD reduction: Reduce H-bond donors (NH2, OH, NHCO) to <=2
     → Replace -OH with -OMe (-OC) or -F; replace primary amine -NH2 with -NMe2 or N-methylated
  3. MW reduction: If MW > 500, remove a ring or large substituent
     → Remove a bulky group (tBu, long alkyl chain, extra phenyl ring)
  4. AlogP adjustment: If AlogP < 1 (too hydrophilic) or > 5 (too lipophilic)
     → Too hydrophilic: add -F or -CH3; Too lipophilic: replace alkyl with -OH or -NH2
  5. PSA reduction: If PSA > 130, remove polar groups
     → Remove amide, reduce number of H-bond donors/acceptors
  6. AROM reduction: If aromatic rings > 3, remove or saturate one ring
     → Replace aromatic ring with cyclohexyl
  7. ROTB reduction: If rotatable bonds > 8, cyclize a linker or shorten a chain
IMPORTANT: Make ONE well-targeted edit that maximally improves the worst sub-component.
Do NOT make the molecule completely unrecognizable — prefer conservative structural changes.
"""

USER_TEMPLATE = """\
Source molecule SMILES: {src_mol}
Current QED value: {src_qed:.4f} (goal is to INCREASE this value)
Ground Truth optimized SMILES: {tgt_mol}
Ground Truth improved QED: {tgt_qed:.4f}

IMPORTANT: The ground truth is provided for verification purposes only.
You must still generate the complete reasoning chain (Step 1 through Step 5)
as if you were solving this problem from scratch. Do NOT simply state the
ground truth — you must work through every step of the reasoning template
and arrive at the answer naturally.

Generate the formal reasoning chain following EXACTLY the unified step format specified."""
