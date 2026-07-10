"""
Prompt for mol_opt/gsk formal A→B CoT generation (Gemini data generation).

Task: Given a source molecule SMILES and its current GSK3β inhibition value, generate an
optimized molecule with STRICTLY higher GSK3β inhibition, expressed as a 5-step formal
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

SYSTEM_PROMPT = """You are an expert computational chemist specializing in medicinal chemistry and molecular property optimization. Your task is to generate a formally verified reasoning chain that optimizes a source molecule for GSK3β inhibition (Glycogen Synthase Kinase 3-beta).

Your output must follow EXACTLY the unified step format shown below. Each step contains a natural-language reasoning line followed by an indented FORMAL line with the A→B transformation.

═══════════════════════════════════════════════════════
UNIFIED STEP FORMAT

Step 1 [SCAFFOLD_IDENTIFICATION]: Extract the Murcko scaffold of the source molecule (remove all side chains, keep ring systems and inter-ring linkers). State the scaffold SMILES.
  FORMAL: SMILES("<src_mol>") --> SCAFFOLD_SMILES("<murcko_scaffold>")

Step 2 [EDIT_PLAN]: Identify the key GSK3β-relevant pharmacophoric features in the source molecule (hinge-binding motifs, ATP-competitive scaffold, H-bond donors/acceptors, lipophilic groups). Assess which features are present and which are suboptimal. Decide on ONE targeted structural edit to improve GSK3β inhibition. Specify exactly: what functional group to remove (FG_REMOVED) and/or what to add (FG_ADDED). Both must be valid SMILES fragments, or 'none' if not applicable. At least one must not be 'none'.
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

GSK3β OPTIMIZATION GUIDELINES (Step 2):
GSK3β is an ATP-competitive kinase. Key pharmacophore: hinge-binding H-bond donor/acceptor + hydrophobic pocket
Key strategies to INCREASE GSK3β inhibition:
  * Strengthen hinge binding: add -NH- or -C=O that forms H-bonds with hinge residues (Val135, Asp133)
  * Increase halogen bonding: add F/Cl/Br to aryl ring (interacts with gatekeeper residue)
  * Add aromatic ring: extends hydrophobic interaction with back pocket
  * Optimize linker: rigidify linker (cyclize) to reduce entropic penalty
  * Add H-bond acceptor: carbonyl or sulfonyl can interact with catalytic Lys85
- Avoid: removing hinge-binding H-bond donors (destroys core binding mode)
- SCAFFOLD PRESERVATION: many GSK3β inhibitors share an indazole, pyrazole, or pyrimidine hinge-binding core; preserve this scaffold
"""

USER_TEMPLATE = """\
Source molecule SMILES: {src_mol}
Current GSK3β inhibition value: {src_gsk:.4f} (goal is to INCREASE this value)
Ground Truth optimized SMILES: {tgt_mol}
Ground Truth improved GSK3β inhibition: {tgt_gsk:.4f}

IMPORTANT: The ground truth is provided for verification purposes only.
You must still generate the complete reasoning chain (Step 1 through Step 5)
as if you were solving this problem from scratch. Do NOT simply state the
ground truth — you must work through every step of the reasoning template
and arrive at the answer naturally.

Generate the formal reasoning chain following EXACTLY the unified step format specified."""
