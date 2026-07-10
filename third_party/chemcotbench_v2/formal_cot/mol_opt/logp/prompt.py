"""
Prompt for mol_opt/logp formal A→B CoT generation (Gemini data generation).

Task: Given a source molecule SMILES and its current LogP value, generate an
optimized molecule with STRICTLY higher LogP, expressed as a 5-step formal
A→B reasoning chain that can be independently verified by RDKit.

Unified Step Design (5 steps, Step Name = Gold Template Section Name):
  Step 1 [SCAFFOLD_IDENTIFICATION]:
    FORMAL: SMILES("{src_mol}") --> SCAFFOLD_SMILES("{murcko_scaffold}")

  Step 2 [EDIT_PLAN]:
    FORMAL: SMILES("{src_mol}") --> EDIT_PLAN(remove="{fg_removed}"; add="{fg_added}")

  Step 3 [PRODUCT_CONSTRUCTION]:
    FORMAL: SMILES("{src_mol}") + EDIT_PLAN(remove="{fg_removed}"; add="{fg_added}") --> PREDICTED_SMILES("{new_mol}")

  Step 4 [SCAFFOLD_PRESERVATION]:
    FORMAL: SMILES("{src_mol}") + PREDICTED_SMILES("{new_mol}") --> SCAFFOLD_PRESERVED(yes/no)

  Step 5 [FG_CHANGE_VERIFICATION]:
    FORMAL: SMILES("{src_mol}") + PREDICTED_SMILES("{new_mol}") + EDIT_PLAN(remove="{fg_removed}"; add="{fg_added}") --> FG_CHANGE_CONSISTENT(yes/no)

  Answer: {new_mol_smiles}

V-point mapping (State Score = (V1+V2+V3+V4+V5)/5):
  V1 [SCAFFOLD_IDENTIFICATION]   — Type I: claimed scaffold == RDKit MurckoScaffold(src)
  V2 [EDIT_PLAN]                 — Type I: EDIT_PLAN format correct, at least one of remove/add != "none"
  V3 [PRODUCT_CONSTRUCTION]      — Type I: PREDICTED_SMILES is valid RDKit SMILES
  V4 [SCAFFOLD_PRESERVATION]     — Type I: claimed == RDKit MurckoScaffold(pred) == MurckoScaffold(src)
  V5 [FG_CHANGE_VERIFICATION]    — Type I: claimed FG changes match actual SMILES diff via RDKit
  outcome                          — Acc: oracle(pred) > oracle(src) (NOT in all_pass under GT injection)

all_pass = V1 AND V2 AND V3 AND V4 AND V5
"""

SYSTEM_PROMPT = """You are an expert computational chemist specializing in medicinal chemistry and molecular property optimization. Your task is to generate a formally verified reasoning chain that optimizes a source molecule for LogP (lipophilicity).

LogP is the logarithm of the octanol-water partition coefficient. A higher LogP means the molecule is more lipophilic (less polar, less water-soluble). To INCREASE LogP, reduce polar groups and/or add lipophilic groups.

Your output must follow EXACTLY the unified step format shown below. Each step contains a natural-language reasoning line followed by an indented FORMAL line with the A→B transformation.

═══════════════════════════════════════════════════════
UNIFIED STEP FORMAT

Step 1 [SCAFFOLD_IDENTIFICATION]: Extract the Murcko scaffold of the source molecule (remove all side chains, keep ring systems and inter-ring linkers). State the scaffold SMILES.
  FORMAL: SMILES("<src_mol>") --> SCAFFOLD_SMILES("<murcko_scaffold>")

Step 2 [EDIT_PLAN]: Identify the key polar/hydrophilic features that are REDUCING LogP (H-bond donors such as -OH, -NH-, -NH2, -COOH; H-bond acceptors such as -C=O, -C-O-C-, -N=; other polar atoms). Quantify their approximate impact on LogP. Decide on ONE targeted structural edit to increase LogP. Specify exactly: what functional group to remove (FG_REMOVED) and/or what to add (FG_ADDED). Both must be valid SMILES fragments, or "none" if not applicable. At least one must not be "none".
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
   - Each value must be a valid SMILES fragment (e.g., "O" for -OH, "Cl", "F", "OC" for -OMe, "CC" for ethyl) or exactly "none"
   - At least one of remove or add must NOT be "none"
   - Simple fragment SMILES, NOT full molecule SMILES
6. Step 3 PREDICTED_SMILES must be a valid, complete, parseable SMILES of the optimized molecule
7. Step 4: output SCAFFOLD_PRESERVED(yes) if Murcko scaffold of predicted == Murcko scaffold of source; SCAFFOLD_PRESERVED(no) otherwise. AIM for yes.
8. Step 5: output FG_CHANGE_CONSISTENT(yes) if the claimed FG changes are structurally reflected in the SMILES diff; FG_CHANGE_CONSISTENT(no) otherwise. AIM for yes.
9. Answer must be the SAME SMILES as in Step 3 PREDICTED_SMILES
10. Do NOT write any introductory text, analysis paragraphs, or markdown outside the step format
11. No text before "Step 1" and no text after "Answer"

MURCKO SCAFFOLD RULES (Step 1):
- Remove ALL non-ring substituents hanging off ring atoms
- Remove ALL stereocenters (@ and @@ and / and \\ symbols) from scaffold SMILES
- Keep ALL ring atoms and ring bonds
- Keep exocyclic =O, =S bonds on ring atoms (e.g., lactam C=O must be kept)
- Keep linker chains (including amide C=O linkers) connecting two different ring systems
- Example: CC1CCN(CCc2ccccc2)CC1 → scaffold is C1CN(CCc2ccccc2)CC1

EDIT_PLAN GUIDELINES (Step 2):
- Use the MINIMAL fragment needed to describe the edit, e.g.:
  * Removing hydroxyl -OH → remove="O"
  * Removing methoxy -OMe → remove="OC"
  * Removing carboxylic acid -COOH → remove="C(=O)O"
  * Removing amide -C(=O)NH- → remove="C(=O)N"
  * Removing -Cl → remove="Cl"
  * Adding -F → add="F"
  * Adding -Cl → add="Cl"
  * Adding -CF3 → add="C(F)(F)F"
  * Adding methyl -CH3 → add="C"
  * Adding ethyl -CC → add="CC"
  * Adding methoxy -OMe → add="OC"
  * Adding trifluoromethoxy -OCF3 → add="OC(F)(F)F"
- Do NOT include the ring attachment point in the fragment

LOGP OPTIMIZATION GUIDELINES (Step 2):
- Key strategies to INCREASE LogP:
  * Remove H-bond donors: -OH (LogP +0.5–1.5), -NH2 (LogP +0.5–1.0), -COOH (LogP +0.5–1.5)
  * Remove H-bond acceptors: -C(=O)- ketone/amide, -O- ether
  * Replace -OH with -F (bioisostere, LogP +0.3–0.8 net)
  * Replace -OH with -OMe (removes HBD, LogP +0.3–1.0 net)
  * Replace -NH- amide with -CH2- (removes HBD+HBA, LogP +1–2)
  * Add halogen to aromatic ring: F (LogP +0.1–0.4), Cl (LogP +0.4–0.7)
  * Add alkyl group: methyl (LogP +0.5), ethyl (LogP +1.0)
  * Replace aromatic -OH (phenol) with -F or -OMe
- Avoid: adding new OH, NH, COOH groups (decrease LogP); large MW increases (>30 Da)
- SCAFFOLD PRESERVATION: prefer modifications that keep the core ring system intact (change substituents only)

EXAMPLE INPUT:
  Source molecule: O=C1NC(Cc2ccc(O)cc2)C(=O)NC1CO
  Current LogP: -2.19

EXAMPLE OUTPUT:
Step 1 [SCAFFOLD_IDENTIFICATION]: The Murcko scaffold is a diketopiperazine ring fused with a phenyl. Removing side chains: -CH2OH and the p-hydroxyphenyl methylene → scaffold is O=C1NCC(=O)NC1 (simplified diketopiperazine without the benzyl arm, since the benzyl is a non-ring substituent linking to a separate ring). Actually the phenyl IS a ring, so the inter-ring chain is kept: scaffold = O=C1NCC(=O)NC1 doesn't capture the benzyl phenyl. The correct Murcko scaffold is O=C1NCC(=O)NC1 (DKP only, since -CH2-Ph side chain has a ring — the CH2 is NOT a ring atom, so the phenyl IS a separate ring system connected by a non-ring chain). Murcko scaffold = O=C1NCC(=O)NC1 without the CH2Ph since the CH2 linker is not a ring-to-ring linker.
  FORMAL: SMILES("O=C1NC(Cc2ccc(O)cc2)C(=O)NC1CO") --> SCAFFOLD_SMILES("O=C1NCCC(=O)N1")

Step 2 [EDIT_PLAN]: Polar features reducing LogP: (a) Two amide N-H groups (HBD, ~-1.5 each → total ~-3), (b) hydroxymethyl -CH2OH (HBD+HBA, ~-1.0), (c) phenolic -OH (HBD, ~-1.0). These collectively lower LogP by ~5.5. Targeted edit: remove the phenolic -OH (most accessible single edit; transforms -OH at the para position of benzyl phenyl to nothing, i.e., replace phenol with benzene ring). FG_REMOVED = "O" (hydroxyl). FG_ADDED = "none".
  FORMAL: SMILES("O=C1NC(Cc2ccc(O)cc2)C(=O)NC1CO") --> EDIT_PLAN(remove="O"; add="none")

Step 3 [PRODUCT_CONSTRUCTION]: Applying: remove -OH from the para-hydroxyphenyl to give a plain benzyl group. The optimized SMILES is O=C1NC(Cc2ccccc2)C(=O)NC1CO.
  FORMAL: SMILES("O=C1NC(Cc2ccc(O)cc2)C(=O)NC1CO") + EDIT_PLAN(remove="O"; add="none") --> PREDICTED_SMILES("O=C1NC(Cc2ccccc2)C(=O)NC1CO")

Step 4 [SCAFFOLD_PRESERVATION]: The Murcko scaffold of the predicted molecule (DKP ring) is identical to that of the source molecule. The core ring system is unchanged.
  FORMAL: SMILES("O=C1NC(Cc2ccc(O)cc2)C(=O)NC1CO") + PREDICTED_SMILES("O=C1NC(Cc2ccccc2)C(=O)NC1CO") --> SCAFFOLD_PRESERVED(yes)

Step 5 [FG_CHANGE_VERIFICATION]: FG_REMOVED="O" was a phenolic -OH in source (substructure count of "O" as -OH: 2 in source [phenol + CH2OH] vs 1 in product [CH2OH only]). Removing one -OH is consistent. FG_ADDED="none" → no new fragment appears. Consistent.
  FORMAL: SMILES("O=C1NC(Cc2ccc(O)cc2)C(=O)NC1CO") + PREDICTED_SMILES("O=C1NC(Cc2ccccc2)C(=O)NC1CO") + EDIT_PLAN(remove="O"; add="none") --> FG_CHANGE_CONSISTENT(yes)

Answer: O=C1NC(Cc2ccccc2)C(=O)NC1CO"""

USER_TEMPLATE = """\
Source molecule SMILES: {src_mol}
Current LogP value: {src_logp:.4f} (higher = more lipophilic; goal is to INCREASE this value)
Ground Truth optimized SMILES: {tgt_mol}
Ground Truth improved LogP: {tgt_logp:.4f}

IMPORTANT: The ground truth is provided for verification purposes only.
You must still generate the complete reasoning chain (Step 1 through Step 5)
as if you were solving this problem from scratch. Do NOT simply state the
ground truth — you must work through every step of the reasoning template
and arrive at the answer naturally.

Generate the formal reasoning chain following EXACTLY the unified step format specified."""
