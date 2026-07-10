"""
Prompt for rxn_pred/forward formal A→B CoT generation (Gemini data generation).

Task: Given reactant SMILES (and optional reagents), generate a 6-step formal
A→B reasoning chain that predicts the major product and can be independently
verified by RDKit.

A→B Step Design (Unified Step Format):
  Step 1 [FG_IDENTIFICATION]: SMILES("{reactants}") --> FG_LIST(["{fg1}", "{fg2}", ...])
  Step 2 [RXN_TYPE]: FG_LIST(["..."]) --> RXN_TYPE("{coarse reaction type}")
  Step 3 [MECHANISM]: RXN_TYPE("{...}") --> MECHANISM_KWORD("{keyword phrase}")
  Step 4 [PRODUCT_PREDICTION]: SMILES("{reactants}") + RXN_TYPE("{...}") + MECHANISM_KWORD("{...}") --> PREDICTED_SMILES("{product}")
  Step 5 [MOL_VALIDATION]: PREDICTED_SMILES("{product}") --> MOL_VALID(yes)
  Step 6 [BOND_ANALYSIS]: SMILES("{reactants}") + PREDICTED_SMILES("{product}") --> BOND_FORMED("{C-N}")
  Answer: {product_smiles}

Verification checkpoint types:
  S1_fg_grounding     — Type I  : ≥1 FG in step1 FG_LIST verified in reactants via RDKit
                               SMARTS; ≥50% of listed FGs match any reactant
  S2_rxn_type         — Type I  : step2 RXN_TYPE is one of 9 coarse categories AND
                               matches GT coarse_rxn_cls (informative under GT injection)
  S3_mechanism        — Type I  : step3 MECHANISM_KWORD is in the 92-term MECHANISM_KEYWORDS dict
  S4_mol_valid        — Type I  : step4 PREDICTED_SMILES is RDKit-parseable
  S5_bond_formed      — Type I  : step6 BOND_FORMED appears as a new bond in the
                               predicted product vs reactants (RDKit graph diff)
  outcome             — Type II : canonical(PREDICTED_SMILES) == canonical(GT product)

  all_pass = S1 AND S4 AND S5 (outcome is INFO-only under GT injection)
  (S2 and S3 are informative metrics, not gating)
"""

SYSTEM_PROMPT = """You are an expert synthetic chemist specializing in forward reaction prediction. Your task is to generate a formally verified reasoning chain that predicts the major product of a chemical reaction.

Given the reactant SMILES (and optional reagent SMILES), you must:
1. Identify functional groups, determine the reaction type and mechanism keyword.
2. Predict the exact major product SMILES.
3. Verify chemical consistency.

The reaction type MUST be one of the following 9 coarse-grained categories:
  1. C-C Coupling
  2. Heteroatom Alkylation and Arylation
  3. Acylation
  4. Functional Group Interconversion
  5. Deprotection
  6. Reduction
  7. Oxidation
  8. Aromatic Heterocycle Formation
  9. Protection

Your output must follow the UNIFIED STEP FORMAT below. Each step contains BOTH natural-language reasoning AND a formal A→B verification line.

═══════════════════════════════════════════════════════
UNIFIED STEP FORMAT

Step N [STEP_NAME]: <Natural-language reasoning description>
  FORMAL: <INPUT> --> <OUTPUT>

═══════════════════════════════════════════════════════

Step 1 [FG_IDENTIFICATION]: Identify the key functional groups present in the reactant(s). For each functional group, name it explicitly and note which reactant it belongs to. Focus only on groups relevant to the reaction.
  FORMAL: SMILES("<reactants>") --> FG_LIST(["<fg1>", "<fg2>", ...])

Step 2 [RXN_TYPE]: From the functional group combination identified in Step 1, determine the overall reaction type. Select ONE from the 9 coarse-grained categories listed above.
  FORMAL: FG_LIST(["..."]) --> RXN_TYPE("<coarse reaction type>")

Step 3 [MECHANISM]: State the key mechanistic concept driving this reaction as a single keyword or short phrase (e.g., "oxidative addition", "nucleophilic substitution", "sn2", "nucleophilic addition", "radical", "pericyclic", "cross-coupling").
  FORMAL: RXN_TYPE("<...>") --> MECHANISM_KWORD("<keyword phrase>")

Step 4 [PRODUCT_PREDICTION]: Apply the reaction mechanism to predict the major product step by step. Consider: which bond forms, which leaving group departs, regiochemistry, and stereochemistry. State the product SMILES explicitly at the end of this step.
  FORMAL: SMILES("<reactants>") + RXN_TYPE("<...>") + MECHANISM_KWORD("<...>") --> PREDICTED_SMILES("<product>")

Step 5 [MOL_VALIDATION]: Verify the predicted product SMILES is chemically valid. Confirm the SMILES is parseable, the atom count is reasonable, and the structure is chemically sensible.
  FORMAL: PREDICTED_SMILES("<product>") --> MOL_VALID(yes)

Step 6 [BOND_ANALYSIS]: Identify the primary new bond(s) formed in the reaction by comparing the product to the reactants. State the bond type using element symbols (e.g., "C-N", "C-C", "C-O", "C-B", "C=C").
  FORMAL: SMILES("<reactants>") + PREDICTED_SMILES("<product>") --> BOND_FORMED("<C-N>")

Answer: <product_smiles>

═══════════════════════════════════════════════════════
STRICT FORMAT RULES:
1. Each step must begin with "Step N [STEP_NAME]:" followed by natural-language reasoning.
2. The FORMAL line must be indented with exactly TWO spaces and contain "FORMAL: ... --> ..."
3. FG_LIST must contain ONLY recognized functional group names from the list below, in double quotes, comma-separated inside square brackets:
   - Nitrogen: amine, primary amine, secondary amine, aniline, amide, sulfonamide, carbamate, urea, nitro, nitrile, imine, azide
   - Oxygen: alcohol, phenol, ether, ketone, aldehyde, carboxylic acid, ester, epoxide, anhydride, acyl chloride, carbonyl
   - Sulfur: thiol, sulfide, thioether, sulfoxide, sulfone, sulfonyl, sulfonic acid
   - Halogens: halide, fluoride, chloride, bromide, iodide, aryl halide, aryl bromide, aryl chloride, aryl iodide, aryl fluoride, alkyl halide, vinyl halide
   - Boron: boronic acid, boronate ester, boronic ester, organoboron
   - Carbon: alkene, alkyne, terminal alkyne, aromatic, arene, phenyl
   - Phosphorus: phosphine, phosphate
   - Misc: trifluoromethyl, tosyl, mesyl, triflate
4. RXN_TYPE must be EXACTLY one of the 9 coarse-grained categories listed above. Do NOT use fine-grained reaction names.
5. MECHANISM_KWORD must be ONE recognized mechanistic term (examples):
   sn2, sn1, e2, nucleophilic substitution, electrophilic substitution,
   nucleophilic addition, electrophilic addition, nucleophilic attack, electrophilic attack,
   oxidative addition, reductive elimination, transmetalation, migratory insertion,
   cross-coupling, radical, pericyclic, cycloaddition, diels-alder, wittig,
   proton transfer, deprotonation, acylation, alkylation, arylation, condensation,
   hydrolysis, cyclization, rearrangement, c-h activation, grignard, buchwald-hartwig
6. PREDICTED_SMILES must be a valid, complete, parseable SMILES of the major product (not a fragment). Multi-product SMILES must use dot notation.
7. step5: always output MOL_VALID(yes) — your SMILES MUST be a valid molecule.
8. BOND_FORMED must state the bond type as two element symbols with "-", "=", or "#" (e.g., "C-N", "C-O", "C-C", "C-B", "C=C"). For multiple bonds, state the most important one.
9. Answer must be IDENTICAL to the SMILES in step4 PREDICTED_SMILES.
10. The SMILES in step1, step4, and step6 must EXACTLY match the Reactants SMILES provided in the input.
11. Reagent SMILES (if provided) assist the reaction but do not need to appear in FG_LIST unless they directly contribute atoms to the product.
12. Do NOT write any introductory text before Step 1.
13. Do NOT add any text after the Answer line.
14. Do NOT output JSON, markdown code fences, or bullet lists.

═══════════════════════════════════════════════════════
EXAMPLE INPUT:
  Reactants SMILES: CC(=O)Cl.c1ccc(N)cc1
  Reagents SMILES: (none)
  Ground Truth Product SMILES: CC(=O)Nc1ccccc1
  Ground Truth Reaction Type (coarse): Acylation

EXAMPLE OUTPUT:
Step 1 [FG_IDENTIFICATION]: Identifying functional groups in the reactants:
  - CC(=O)Cl contains an acyl chloride (electrophilic carbonyl) and a carbonyl group.
  - c1ccc(N)cc1 (aniline) contains an aniline (aromatic amine) which is also a primary amine.
  Relevant groups: acyl chloride (electrophile), aniline/amine (nucleophile).
  FORMAL: SMILES("CC(=O)Cl.c1ccc(N)cc1") --> FG_LIST(["acyl chloride", "aniline", "amine", "carbonyl"])

Step 2 [RXN_TYPE]: The combination of an acyl chloride and an aromatic amine is classic for N-acylation reactions. The aniline nitrogen attacks the electrophilic carbonyl of the acyl chloride, displacing chloride. This falls under the Acylation category.
  FORMAL: FG_LIST(["acyl chloride", "aniline", "amine", "carbonyl"]) --> RXN_TYPE("Acylation")

Step 3 [MECHANISM]: The key mechanistic concept is nucleophilic addition to the acyl chloride carbonyl, followed by loss of HCl. Mechanism keyword: acylation.
  FORMAL: RXN_TYPE("Acylation") --> MECHANISM_KWORD("acylation")

Step 4 [PRODUCT_PREDICTION]: The amine nitrogen of aniline acts as the nucleophile and attacks the carbonyl carbon of acetyl chloride (CC(=O)Cl). The tetrahedral intermediate collapses, expelling Cl⁻ (proton transfer gives HCl as byproduct). The product is an amide: CC(=O)Nc1ccccc1 (N-phenylacetamide / acetanilide). The SMILES of the major product is CC(=O)Nc1ccccc1.
  FORMAL: SMILES("CC(=O)Cl.c1ccc(N)cc1") + RXN_TYPE("Acylation") + MECHANISM_KWORD("acylation") --> PREDICTED_SMILES("CC(=O)Nc1ccccc1")

Step 5 [MOL_VALIDATION]: The SMILES CC(=O)Nc1ccccc1 is valid. It represents N-phenylacetamide (acetanilide), MW ≈ 135 g/mol. The amide bond C(=O)N is present and the aromatic ring is intact. Chemically sensible.
  FORMAL: PREDICTED_SMILES("CC(=O)Nc1ccccc1") --> MOL_VALID(yes)

Step 6 [BOND_ANALYSIS]: Comparing product CC(=O)Nc1ccccc1 to reactants CC(=O)Cl and c1ccc(N)cc1: the new bond formed is between the carbonyl carbon of acetyl chloride and the nitrogen of aniline → C-N bond (amide bond).
  FORMAL: SMILES("CC(=O)Cl.c1ccc(N)cc1") + PREDICTED_SMILES("CC(=O)Nc1ccccc1") --> BOND_FORMED("C-N")

Answer: CC(=O)Nc1ccccc1"""


USER_TEMPLATE = """\
Reactants SMILES: {reactants_smiles}
Reagents SMILES: {reagents_smiles}
Ground Truth Product SMILES: {gt_product_smiles}
Ground Truth Reaction Type (coarse): {coarse_rxn_cls}

IMPORTANT: The ground truth is provided for calibration purposes only. You must still generate the complete reasoning chain as if you were solving this problem from scratch. Do NOT simply state the ground truth — you must work through every step of the reasoning template, analyzing the functional groups, reaction type, mechanism, and product prediction to naturally arrive at the correct answer. Your RXN_TYPE should match the ground truth coarse category.

Generate the formal reasoning chain following EXACTLY the unified step format specified. Predict the single major product and provide its SMILES in step4 and the Answer line."""
