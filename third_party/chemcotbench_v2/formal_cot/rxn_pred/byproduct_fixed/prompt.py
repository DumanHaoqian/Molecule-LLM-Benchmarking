"""
Prompt for rxn_pred/byproduct_fixed formal A→B CoT generation (Gemini data generation).

Task: Given reactant SMILES, optional reagent SMILES, and the known main product SMILES,
generate a 6-step formal A→B reasoning chain that predicts the byproduct and can be
independently verified by RDKit.

A→B Step Design (Unified Step Format):
  Step 1 [FG_IDENTIFICATION]: SMILES("{reactants}") --> FG_LIST(["{fg1}", "{fg2}", ...])
  Step 2 [RXN_TYPE]: FG_LIST(["..."]) --> RXN_TYPE("{coarse category}")
  Step 3 [ATOMIC_DELTA]: SMILES("{reactants}") + SMILES("{main_product}") --> ATOMIC_DELTA(["{elem1}", ...])
  Step 4 [LEAVING_FRAGMENT]: ATOMIC_DELTA(["..."]) --> LEAVING_FRAGMENT_SMILES("{lf_smiles}")
  Step 5 [FRAGMENT_SOURCE]: LEAVING_FRAGMENT_SMILES("{lf_smiles}") + SMILES("{reactants}") --> FRAGMENT_IN_REACTANT(yes)
  Step 6 [BYPRODUCT_FORMATION]: LEAVING_FRAGMENT_SMILES("{lf_smiles}") --> BYPRODUCT_SMILES("{bp_smiles}")
  Answer: {byproduct_smiles}

Verification checkpoint types:
  S1_fg_grounding          — Type I (info): ≥1 FG in step1 FG_LIST verified in reactants via RDKit SMARTS
  S2_rxn_type              — Type I (GATES): step2 RXN_TYPE is one of 9 coarse categories
  S3_atomic_delta_grounded — Type I (GATES): ATOMIC_DELTA element symbols ⊆ heavy_atom_elements(reactants)
  S4_fragment_parseable    — Type I (GATES): LEAVING_FRAGMENT_SMILES is RDKit-parseable
  S5_fragment_in_reactant  — Type I (GATES): heavy_atom_elements(LEAVING_FRAGMENT) ⊆ heavy_atom_elements(reactants)
  S6_element_coherence     — Type I (GATES): heavy_atom_elements(LEAVING_FRAGMENT) ⊆ heavy_atom_elements(BYPRODUCT)
  outcome                  — Type II(info): canonical(BYPRODUCT_SMILES) == canonical(GT byproduct)

  all_pass = S2 AND S3 AND S4 AND S5 AND S6
  (S1 and outcome are informative; outcome is info-only under GT injection)
"""

SYSTEM_PROMPT = """You are an expert synthetic chemist specializing in byproduct prediction. Your task is to generate a formally verified reasoning chain that predicts the byproduct of a chemical reaction.

Given the reactant SMILES, optional reagent SMILES, and the known main product SMILES, you must:
1. Identify functional groups and determine the coarse reaction type.
2. Compute what atoms are "lost" from reactants compared to the main product.
3. Identify the exact leaving fragment and predict the byproduct SMILES.
4. Verify chemical consistency.

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

Step 1 [FG_IDENTIFICATION]: Identify the key functional groups present in the reactant(s). For each functional group, name it explicitly and note which reactant it belongs to. Focus on groups relevant to the reaction mechanism.
  FORMAL: SMILES("<reactants>") --> FG_LIST(["<fg1>", "<fg2>", ...])

Step 2 [RXN_TYPE]: From the functional group combination identified in Step 1, determine the overall coarse reaction type. Select ONE from the 9 coarse-grained categories listed above.
  FORMAL: FG_LIST(["..."]) --> RXN_TYPE("<coarse reaction type>")

Step 3 [ATOMIC_DELTA]: Compare the heavy atoms in the reactants versus the main product. Identify which element(s) appear in the reactants but are absent (or in reduced quantity) in the main product — these are the elements of the leaving fragment. List these element symbols.
  FORMAL: SMILES("<reactants>") + SMILES("<main_product>") --> ATOMIC_DELTA(["<elem1>", "<elem2>", ...])

Step 4 [LEAVING_FRAGMENT]: From the atomic delta identified in Step 3, determine the exact molecular fragment that physically departs from the reactant skeleton as the leaving group. Write its SMILES. This fragment must (a) contain the elements identified in Step 3, and (b) exist in one of the reactants.
  FORMAL: ATOMIC_DELTA(["..."]) --> LEAVING_FRAGMENT_SMILES("<lf_smiles>")

Step 5 [FRAGMENT_SOURCE]: Verify that the leaving fragment identified in Step 4 is chemically sourced from the reactants. Check that all heavy-atom elements of the leaving fragment are present in the reactant pool. State whether this check passes: yes or no.
  FORMAL: LEAVING_FRAGMENT_SMILES("<lf_smiles>") + SMILES("<reactants>") --> FRAGMENT_IN_REACTANT(yes)

Step 6 [BYPRODUCT_FORMATION]: Determine the final byproduct SMILES. The byproduct may be the leaving fragment itself (e.g., [Cl-], O) or a species it combines with to form (e.g., Ph3P + O → Ph3P=O). Write the byproduct SMILES. Its heavy-atom elements must include all heavy-atom elements in the leaving fragment.
  FORMAL: LEAVING_FRAGMENT_SMILES("<lf_smiles>") --> BYPRODUCT_SMILES("<bp_smiles>")

Answer: <byproduct_smiles>

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
   - Metals: organomagnesium, organolithium, organozinc, grignard reagent
   - Misc: trifluoromethyl, tosyl, mesyl, triflate, silyl ether, protecting group
4. RXN_TYPE must be EXACTLY one of the 9 coarse-grained categories listed above. Do NOT use fine-grained reaction names.
5. ATOMIC_DELTA must be a list of ELEMENT SYMBOLS (e.g., ["Cl"], ["Br", "O"], ["C", "O"]) for the heavy atoms that appear in reactants but are absent or reduced in the main product. Use standard periodic table symbols (capitalized first letter).
6. LEAVING_FRAGMENT_SMILES must be a valid, parseable SMILES of the fragment that departs from the reactant skeleton (e.g., "[Cl-]", "[Br-]", "CC(C)(C)OC(=O)O", "O").
7. step5: always output FRAGMENT_IN_REACTANT(yes) — your leaving fragment MUST be sourced from the reactants.
8. BYPRODUCT_SMILES must be a valid, complete, parseable SMILES of the byproduct. Multi-component byproducts must use dot notation.
9. Answer must be IDENTICAL to the SMILES in step6 BYPRODUCT_SMILES.
10. The SMILES in step1 and step3 must EXACTLY match the Reactants SMILES provided in the input.
11. The SMILES in step3 (main product) must EXACTLY match the Main Product SMILES provided in the input.
12. Do NOT write any introductory text before Step 1.
13. Do NOT add any text after the Answer line.
14. Do NOT output JSON, markdown code fences, or bullet lists.

═══════════════════════════════════════════════════════
EXAMPLE INPUT:
  Reactants SMILES: CC(=O)Cl.c1ccc(N)cc1
  Reagents SMILES: (none)
  Main Product SMILES: CC(=O)Nc1ccccc1
  Ground Truth Byproduct SMILES: [Cl-]
  Ground Truth Coarse Reaction Type: Acylation

EXAMPLE OUTPUT:
Step 1 [FG_IDENTIFICATION]: Identifying functional groups in the reactants:
  - CC(=O)Cl contains an acyl chloride (electrophilic carbonyl) and a carbonyl group.
  - c1ccc(N)cc1 (aniline) contains an aniline (aromatic amine) which is also a primary amine.
  Relevant groups: acyl chloride (electrophile), aniline/primary amine (nucleophile).
  FORMAL: SMILES("CC(=O)Cl.c1ccc(N)cc1") --> FG_LIST(["acyl chloride", "aniline", "primary amine", "carbonyl"])

Step 2 [RXN_TYPE]: An acyl chloride reacting with an aromatic amine is classic N-acylation. The amine nitrogen attacks the electrophilic carbonyl carbon of the acyl chloride, displacing chloride. Under the 9 coarse categories, this is Acylation.
  FORMAL: FG_LIST(["acyl chloride", "aniline", "primary amine", "carbonyl"]) --> RXN_TYPE("Acylation")

Step 3 [ATOMIC_DELTA]: Reactants contain C, N, O, Cl (heavy atoms). Main product CC(=O)Nc1ccccc1 contains C, N, O only. Cl is present in reactants but absent in main product → ATOMIC_DELTA = ["Cl"].
  FORMAL: SMILES("CC(=O)Cl.c1ccc(N)cc1") + SMILES("CC(=O)Nc1ccccc1") --> ATOMIC_DELTA(["Cl"])

Step 4 [LEAVING_FRAGMENT]: The Cl in ATOMIC_DELTA comes from the C-Cl bond of CC(=O)Cl. When the amine attacks the carbonyl, the C-Cl bond breaks heterolytically, releasing Cl- as the leaving fragment.
  FORMAL: ATOMIC_DELTA(["Cl"]) --> LEAVING_FRAGMENT_SMILES("[Cl-]")

Step 5 [FRAGMENT_SOURCE]: [Cl-] contains only Cl. The reactant CC(=O)Cl contains Cl → the fragment is sourced from reactants → FRAGMENT_IN_REACTANT = yes.
  FORMAL: LEAVING_FRAGMENT_SMILES("[Cl-]") + SMILES("CC(=O)Cl.c1ccc(N)cc1") --> FRAGMENT_IN_REACTANT(yes)

Step 6 [BYPRODUCT_FORMATION]: The chloride [Cl-] is the primary byproduct under the reaction conditions (recorded as the ionic form in synthetic databases).
  FORMAL: LEAVING_FRAGMENT_SMILES("[Cl-]") --> BYPRODUCT_SMILES("[Cl-]")

Answer: [Cl-]"""


USER_TEMPLATE = """\
Reactants SMILES: {reactants_smiles}
Reagents SMILES: {reagents_smiles}
Main Product SMILES: {main_product_smiles}
Ground Truth Byproduct SMILES: {gt_byproduct_smiles}
Ground Truth Coarse Reaction Type: {coarse_rxn_cls}

IMPORTANT: The ground truth byproduct and reaction type are provided for calibration purposes only. You must still generate the complete reasoning chain as if you were solving this problem from scratch. Do NOT simply state the ground truth — you must work through every step of the reasoning template, analyzing functional groups, reaction type, atomic delta, leaving fragment, and byproduct formation to naturally arrive at the correct answer. Your BYPRODUCT_SMILES should match the ground truth byproduct and your RXN_TYPE should match the ground truth coarse category.

Generate the formal reasoning chain following EXACTLY the unified step format specified. Predict the byproduct and provide its SMILES in step6 and the Answer line."""
