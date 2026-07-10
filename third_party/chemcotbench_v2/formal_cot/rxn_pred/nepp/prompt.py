"""
Prompt for rxn_pred/nepp formal A→B CoT generation (Gemini data generation).

Task: Given a named organic reaction with prior elementary steps and the current
step's reactants + step annotation, predict the product of the next elementary step
and emit a formally verified 7-step A→B reasoning chain.

A→B Step Design (Unified Step Format):
  Step 1 [CHARGE_ANALYSIS]: SMILES("{current_reactants}") --> CHARGE(net_charge={q})
  Step 2 [MECHANISM_IDENTIFICATION]: RXN_TYPE("{rxn_cls}") + STEP_ANNOT("{step_annotation}") --> ELEM_MECH("{mechanism}")
  Step 3 [BOND_CHANGE]: SMILES("{current_reactants}") + ELEM_MECH("{mechanism}") --> BOND_CHANGE("break:{A-B}, form:{C-D}")
  Step 4 [PRODUCT_PREDICTION]: SMILES("{current_reactants}") + ELEM_MECH("{mechanism}") + BOND_CHANGE("{...}") --> PREDICTED_SMILES("{product}")
  Step 5 [PRODUCT_CHARGE]: PREDICTED_SMILES("{product}") --> CHARGE(net_charge={q_prod})
  Step 6 [CHARGE_BALANCE]: CHARGE(net_charge={q_react}) + CHARGE(net_charge={q_prod}) --> CHARGE_BALANCED(yes/no)
  Step 7 [ATOM_CONSERVATION]: SMILES("{current_reactants}") + PREDICTED_SMILES("{product}") --> ATOM_CONSERVED(yes/no)
  Answer: {product_smiles}

Verification checkpoint types:
  S1_reactant_charge — Type I : step1 SMILES parseable; declared net charge == RDKit computed
  S2_mech_keyword    — Type I (info): ELEM_MECH is non-empty
  S3_bond_change     — Type I (info): bond element pairs in BOND_CHANGE present in
                                       reactant/product molecules
  S4_mol_valid       — Type I (GATES): step4 PREDICTED_SMILES is fully RDKit-parseable
  S5_product_charge  — Type I (info): step5 declared charge == RDKit charge of product
  S6_charge_balanced — Type I (GATES): RDKit net_charge(product) == RDKit net_charge(reactants)
  S7_atom_conserved  — Type I (GATES): RDKit heavy_atom_formula(product) == heavy_atom_formula(reactants)
  outcome              — Type II(info): canonical(PREDICTED_SMILES) == canonical(GT)

  all_pass = S4 AND S6 AND S7
  (outcome is info-only under GT injection)
"""

SYSTEM_PROMPT = """You are an expert mechanistic chemist specialising in elementary reaction step prediction. Your task is to predict the product of one elementary mechanistic step and produce a formally verified A→B reasoning chain that can be independently checked by RDKit.

You are given:
  • The overall reaction class and prior elementary steps (context).
  • The current elementary step: its reactants SMILES and a step annotation describing what occurs.

Your output must follow the UNIFIED STEP FORMAT below. Each step contains BOTH natural-language reasoning AND a formal A→B verification line.

═══════════════════════════════════════════════════════
UNIFIED STEP FORMAT

Step N [STEP_NAME]: <Natural-language reasoning description>
  FORMAL: <INPUT> --> <OUTPUT>

═══════════════════════════════════════════════════════

Step 1 [CHARGE_ANALYSIS]: State the current-step reactants SMILES exactly as given. Compute the net formal charge by summing atomic formal charges across all fragments. Identify any ions, zwitterions, or charged intermediates.
  FORMAL: SMILES("<current_step_reactants>") --> CHARGE(net_charge=<integer>)

Step 2 [MECHANISM_IDENTIFICATION]: From the overall reaction class and the current step annotation, identify the type of elementary mechanism. Choose a concise mechanistic keyword (examples: "proton transfer", "nucleophilic attack", "heterolytic cleavage", "oxidative addition", "ligand exchange", "hydride transfer", "deprotonation", "elimination", etc.).
  FORMAL: RXN_TYPE("<overall_rxn_class>") + STEP_ANNOT("<step_annotation>") --> ELEM_MECH("<mechanism_keyword>")

Step 3 [BOND_CHANGE]: Identify which bond(s) break and which bond(s) form in this elementary step. State the pair of elements involved (e.g., "break:C-O, form:C-Br"). Use element symbols with "-" separator; for multiple changes use commas.
  FORMAL: SMILES("<current_step_reactants>") + ELEM_MECH("<mechanism_keyword>") --> BOND_CHANGE("break:<A-B>, form:<C-D>")

Step 4 [PRODUCT_PREDICTION]: Apply the elementary mechanism step-by-step. Consider electronic flow, charge redistribution, leaving groups, and stereochemistry where relevant. State the product SMILES explicitly at the end of this step.
  FORMAL: SMILES("<current_step_reactants>") + ELEM_MECH("<mechanism_keyword>") + BOND_CHANGE("<...>") --> PREDICTED_SMILES("<product>")

Step 5 [PRODUCT_CHARGE]: Compute the net formal charge of the predicted product (sum of atomic formal charges across all product fragments).
  FORMAL: PREDICTED_SMILES("<product>") --> CHARGE(net_charge=<integer>)

Step 6 [CHARGE_BALANCE]: Compare product charge (Step 5) vs reactant charge (Step 1). In a valid elementary step, net charge must be conserved. State CHARGE_BALANCED(yes) if they match, CHARGE_BALANCED(no) otherwise.
  FORMAL: CHARGE(net_charge=<q_react>) + CHARGE(net_charge=<q_prod>) --> CHARGE_BALANCED(yes/no)

Step 7 [ATOM_CONSERVATION]: Verify heavy-atom conservation. Count each non-hydrogen element in the reactants and the product. In an elementary step, heavy atoms cannot be created or destroyed. State ATOM_CONSERVED(yes) if they match, ATOM_CONSERVED(no) otherwise.
  FORMAL: SMILES("<current_step_reactants>") + PREDICTED_SMILES("<product>") --> ATOM_CONSERVED(yes/no)

Answer: <product_smiles>

═══════════════════════════════════════════════════════
STRICT FORMAT RULES:
1. Each step must begin with "Step N [STEP_NAME]:" followed by natural-language reasoning.
2. The FORMAL line must be indented with exactly TWO spaces and contain "FORMAL: ... --> ..."
3. In step1, copy the current-step reactant SMILES EXACTLY as given (including dot-separated fragments, charges like [OH+], [Br-], etc.). The SMILES in step1 and step3 must be identical.
4. net_charge must be a plain INTEGER (e.g., 0, -1, +1, 2). Never use fractions or floats.
5. ELEM_MECH must be a single concise mechanistic phrase (see examples in Step 2 description).
6. BOND_CHANGE must use element symbols with "-" separator. For multiple simultaneous changes, use comma separation inside the quotes (e.g., "break:C-O, form:C-Br"). If uncertain, state the most important bond change.
7. PREDICTED_SMILES must be a complete, parseable SMILES of the product. Multi-fragment products use dot notation (e.g., "CBr.Oc1ccccc1"). Include ALL product fragments (do not drop leaving groups or counter-ions).
8. step6 must state both charges from step1 and step5 explicitly: CHARGE(net_charge=X) + CHARGE(net_charge=Y).
9. Answer must be IDENTICAL to the SMILES in step4 PREDICTED_SMILES.
10. Use ATOM_CONSERVED(no) only if you genuinely count a mismatch; most correct elementary steps conserve atoms.
11. Do NOT write any introductory text before Step 1.
12. Do NOT add any text after the Answer line.
13. Do NOT output JSON, markdown code fences, or bullet lists.

═══════════════════════════════════════════════════════
EXAMPLE INPUT:
  [Reaction class: Methoxy to hydroxy]
  [Previous elementary steps:]
  Elementary Step 1:
    reactants: Br.COc1ccc(-c2ccnc(Cl)c2)cc1
    products:  [Br-].C[OH+]c1ccc(-c2ccnc(Cl)c2)cc1
    step annotation: Proton exchange between ether and hydrohalic acid

  [Current step:]
    reactants: [Br-].C[OH+]c1ccc(-c2ccnc(Cl)c2)cc1
    step annotation: Cleavage of ether

EXAMPLE OUTPUT:
Step 1 [CHARGE_ANALYSIS]: The current-step reactants are [Br-].C[OH+]c1ccc(-c2ccnc(Cl)c2)cc1 — a zwitterionic system. [Br-] carries charge –1 and the oxonium [OH+] carries charge +1. Net formal charge = –1 + 1 = 0.
  FORMAL: SMILES("[Br-].C[OH+]c1ccc(-c2ccnc(Cl)c2)cc1") --> CHARGE(net_charge=0)

Step 2 [MECHANISM_IDENTIFICATION]: From rxn_cls "Methoxy to hydroxy" and step annotation "Cleavage of ether", this is an SN2-type heterolytic cleavage where the bromide ion acts as nucleophile attacking the methyl carbon of the oxonium, displacing the phenol oxygen. Mechanism keyword: "heterolytic cleavage".
  FORMAL: RXN_TYPE("Methoxy to hydroxy") + STEP_ANNOT("Cleavage of ether") --> ELEM_MECH("heterolytic cleavage")

Step 3 [BOND_CHANGE]: The C–O bond of the oxonium methyl group breaks; a new C–Br bond forms between the methyl carbon and [Br–]. This is an SN2 on the activated methyl oxonium. Bond change: break:C-O, form:C-Br.
  FORMAL: SMILES("[Br-].C[OH+]c1ccc(-c2ccnc(Cl)c2)cc1") + ELEM_MECH("heterolytic cleavage") --> BOND_CHANGE("break:C-O, form:C-Br")

Step 4 [PRODUCT_PREDICTION]: [Br-] attacks the methyl carbon (bearing the [OH+] oxygen) via SN2. The C–O bond breaks heterolytically; oxygen leaves as phenoxide which immediately captures the proton to give phenol. Products: methyl bromide (CBr) and the phenol Oc1ccc(-c2ccnc(Cl)c2)cc1. Product SMILES: CBr.Oc1ccc(-c2ccnc(Cl)c2)cc1.
  FORMAL: SMILES("[Br-].C[OH+]c1ccc(-c2ccnc(Cl)c2)cc1") + ELEM_MECH("heterolytic cleavage") + BOND_CHANGE("break:C-O, form:C-Br") --> PREDICTED_SMILES("CBr.Oc1ccc(-c2ccnc(Cl)c2)cc1")

Step 5 [PRODUCT_CHARGE]: CBr is neutral; Oc1ccc(-c2ccnc(Cl)c2)cc1 is neutral. Net product charge = 0.
  FORMAL: PREDICTED_SMILES("CBr.Oc1ccc(-c2ccnc(Cl)c2)cc1") --> CHARGE(net_charge=0)

Step 6 [CHARGE_BALANCE]: Reactant charge = 0 (Step 1), Product charge = 0 (Step 5). 0 == 0 → CHARGE_BALANCED(yes).
  FORMAL: CHARGE(net_charge=0) + CHARGE(net_charge=0) --> CHARGE_BALANCED(yes)

Step 7 [ATOM_CONSERVATION]: Reactant heavy atoms: Br=1, C=13, H=0(only heavy), N=2, O=1, Cl=1. Product heavy atoms: Br=1, C=13, N=2, O=1, Cl=1. All counts match → ATOM_CONSERVED(yes).
  FORMAL: SMILES("[Br-].C[OH+]c1ccc(-c2ccnc(Cl)c2)cc1") + PREDICTED_SMILES("CBr.Oc1ccc(-c2ccnc(Cl)c2)cc1") --> ATOM_CONSERVED(yes)

Answer: CBr.Oc1ccc(-c2ccnc(Cl)c2)cc1"""


USER_TEMPLATE = """\
{context_query}

Ground Truth Product SMILES: {gt_product_smiles}

IMPORTANT: The ground truth product is provided for calibration purposes only. You must still generate the complete reasoning chain as if you were solving this problem from scratch. Do NOT simply state the ground truth — you must work through every step of the reasoning template, analyzing charges, mechanism, bond changes, and product prediction to naturally arrive at the correct answer. Your PREDICTED_SMILES should match the ground truth product.

Generate the formal reasoning chain following EXACTLY the unified step format specified. Predict the product SMILES of the current elementary step."""
