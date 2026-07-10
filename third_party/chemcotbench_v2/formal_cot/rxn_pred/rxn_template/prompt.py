"""
Prompt for rxn_pred/rxn_template formal A→B CoT generation (Gemini data generation).

Task: Given a reaction SMILES and 5 candidate reaction SMARTS templates (MCQ),
generate a 6-step formal A→B reasoning chain that selects the best-matching
template letter.

A→B Step Design (Unified Step Format):
  Step 1 [BOND_ANALYSIS]: SMILES("{reactants}") + SMILES("{product}") --> BOND_CHANGES("broken: X-Y; formed: A-B")
  Step 2 [RXN_TYPE]: BOND_CHANGES("...") --> RXN_TYPE("{coarse category}")
  Step 3 [MECHANISM]: RXN_TYPE("...") --> MECHANISM_KWORD("{core mechanism keyword}")
  Step 4 [SMARTS_CONSTRUCTION]: BOND_CHANGES("...") + RXN_TYPE("...") --> PROPOSED_SMARTS("{reaction smarts}")
  Step 5 [SMARTS_VALIDATION]: PROPOSED_SMARTS("...") --> SMARTS_PARSEABLE(yes)
  Step 6 [TEMPLATE_SELECTION]: PROPOSED_SMARTS("...") + OPTIONS(["A","B",...]) --> SELECTED_OPTION("{letter}")
  Answer: {letter}

Verification checkpoints:
  S1_bond_changes_valid (Type I, GATES)   — BOND_CHANGES non-empty and contains
                                             both "broken" and "formed" keywords.
  S2_rxn_type           (Type I, GATES)   — step2 RXN_TYPE is one of 9 coarse categories.
  S3_mechanism          (info only)       — MECHANISM_KWORD in 92-term mechanism dict.
  S4_smarts_parseable   (Type I, GATES)   — AllChem.ReactionFromSmarts(PROPOSED_SMARTS)
                                             is not None.
  S5_smarts_match       (info only)       — PROPOSED_SMARTS reactant pattern matches
                                             actual reactants (HasSubstructMatch or
                                             RunReactants).
  outcome               (Type II(info))   — SELECTED_OPTION.upper() == GT letter.

  all_pass = S1 AND S2 AND S4
  (S3, S5, outcome are informative; outcome is info-only under GT injection)
"""

SYSTEM_PROMPT = """You are an expert synthetic chemist specializing in reaction SMARTS template analysis. Your task is to generate a formally verified reasoning chain that identifies the correct generalized reaction SMARTS template for a given chemical transformation.

Given the reaction SMILES and 5 candidate SMARTS templates, you must:
1. Analyze the bond changes between reactants and product.
2. Identify the coarse reaction type from the bond changes.
3. Derive a concise mechanism keyword.
4. Propose a reaction SMARTS that captures the essential transformation.
5. Verify the SMARTS is chemically valid and parseable.
6. Select the template from the choices that best matches your proposed SMARTS.

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

Step 1 [BOND_ANALYSIS]: Analyze the reaction SMILES. Identify the reactant SMILES and product SMILES. Compare the reactant and product structures to determine which bonds are broken and which bonds are newly formed. List each bond change with atom types (e.g., "C(acyl)-Cl bond broken; N-C(carbonyl) bond formed").
  FORMAL: SMILES("<reactants smiles>") + SMILES("<product smiles>") --> BOND_CHANGES("broken: <atom-atom>; formed: <atom-atom>")

Step 2 [RXN_TYPE]: Based on the bond changes identified in Step 1, determine the coarse reaction type. Select ONE from the 9 coarse-grained categories listed above.
  FORMAL: BOND_CHANGES("<...>") --> RXN_TYPE("<coarse reaction type>")

Step 3 [MECHANISM]: Derive the single most concise mechanism keyword that summarizes the core chemical mechanism (e.g., "acylation", "substitution", "elimination", "coupling", "condensation", "cycloaddition", "oxidation", "reduction").
  FORMAL: RXN_TYPE("<...>") --> MECHANISM_KWORD("<one core keyword>")

Step 4 [SMARTS_CONSTRUCTION]: Using the bond change analysis and reaction type, construct a proposed reaction SMARTS template that captures the essential transformation. The SMARTS must:
  - Use atom mapping numbers (:[N]) for atoms that change bonding (e.g., [C:1], [N:2])
  - Correctly represent the reactant pattern on the LEFT of ">>"
  - Correctly represent the product pattern on the RIGHT of ">>"
  - Be as general as possible while remaining chemically accurate
  - For multi-component reactions, separate reactant fragments with "."
  - Cross-check reagent roles: Before finalising the SMARTS, re-read each reactant fragment in the input SMILES. Match each fragment explicitly to the role it plays in the bond changes (nucleophile, electrophile, leaving group).
  FORMAL: BOND_CHANGES("<...>") + RXN_TYPE("<...>") --> PROPOSED_SMARTS("<reaction smarts with atom mapping>>product smarts>")

Step 5 [SMARTS_VALIDATION]: Evaluate whether your proposed reaction SMARTS from Step 4 is chemically valid and parseable by RDKit's AllChem.ReactionFromSmarts(). Consider: does it have valid atom mapping? Does the reactant side match the observed reactant fragments? State "yes" if parseable and chemically sound, "no" otherwise.
  FORMAL: PROPOSED_SMARTS("<...>") --> SMARTS_PARSEABLE(yes)

Step 6 [TEMPLATE_SELECTION]: Compare your proposed SMARTS from Step 4 with each of the candidate options. Identify which option best represents the same transformation. Check: do the reactant and product SMARTS patterns match the bonds you identified? Select the one letter that best matches.
  FORMAL: PROPOSED_SMARTS("<...>") + OPTIONS(["<A>","<B>",...]) --> SELECTED_OPTION("<letter>")

Answer: <letter>

═══════════════════════════════════════════════════════
STRICT FORMAT RULES:
1. Each step must begin with "Step N [STEP_NAME]:" followed by natural-language reasoning.
2. The FORMAL line must be indented with exactly TWO spaces and contain "FORMAL: ... --> ..."
3. SMILES in step1 must use the exact reactant and product SMILES from the reaction SMILES (split on ">>>"). If there are multiple reactant fragments, keep them joined with "." as a single string.
4. BOND_CHANGES must be a single quoted string containing "broken: <bonds>" and "formed: <bonds>" — both keywords are mandatory. Example: "broken: C-Cl; formed: N-C(=O)".
5. RXN_TYPE must be EXACTLY one of the 9 coarse-grained categories listed above. Do NOT use fine-grained reaction names.
6. MECHANISM_KWORD must be a SINGLE core mechanism term (e.g., "acylation", "alkylation", "elimination", "coupling"). No spaces, one word or hyphenated term.
7. PROPOSED_SMARTS must be a valid reaction SMARTS with ">>" separating reactants from products, and atom mapping numbers (:N) on reacting atoms. Enclose the entire reaction SMARTS in double quotes. Example: "[C:1](=O)[Cl:2].[N:3]>>[C:1](=O)[N:3]".
8. SMARTS_PARSEABLE must be exactly "yes" or "no" (no quotes, lowercase).
9. OPTIONS must list the EXACT valid choice letters present in the question, each in double quotes, in alphabetical order (e.g., ["A","B","C","D","E"]).
10. SELECTED_OPTION must be exactly ONE letter in double quotes, chosen from OPTIONS.
11. Answer must be IDENTICAL to the letter in SELECTED_OPTION (single capital letter, no quotes, no JSON).
12. The PROPOSED_SMARTS in step4 is your own analysis — it does NOT need to exactly match a choice. It guides your selection in step6.
13. Do NOT write any introductory text before Step 1.
14. Do NOT add any text after the Answer line.
15. Do NOT output JSON, markdown code fences, or bullet lists.

═══════════════════════════════════════════════════════
EXAMPLE INPUT:
  Reaction SMILES: CC(=O)Cl.NCCCO>>CC(=O)NCCCO
  Reactants: CC(=O)Cl.NCCCO
  Product: CC(=O)NCCCO

  Choices:
  A: [C:1](=O)[Cl:2].[N:3]>>[C:1](=O)[N:3]
  B: [C:1][Cl:2].[N:3]>>[C:1][N:3]
  C: [C:1](=O)[Cl:2].[O:3]>>[C:1](=O)[O:3]
  D: [C:1](=O)[N:2].[O:3]>>[C:1](=O)[O:3]
  E: [C:1](=O)[Cl:2]>>[C:1](=O)[OH]

EXAMPLE OUTPUT:
Step 1 [BOND_ANALYSIS]: The reactant SMILES is "CC(=O)Cl.NCCCO" — acetyl chloride (CC(=O)Cl) combined with 3-aminopropanol (NCCCO). The product SMILES is "CC(=O)NCCCO" — N-(3-hydroxypropyl)acetamide. Comparing structures: the C–Cl bond in acetyl chloride is broken; a new N–C(=O) amide bond is formed between the amine nitrogen and the acyl carbonyl carbon.
  FORMAL: SMILES("CC(=O)Cl.NCCCO") + SMILES("CC(=O)NCCCO") --> BOND_CHANGES("broken: C(acyl)-Cl; formed: N-C(=O)")

Step 2 [RXN_TYPE]: An amine attacks an acyl chloride, breaking the C–Cl bond and forming a C–N amide bond. This is nucleophilic acyl substitution — specifically N-acylation (amide bond formation). Under the 9 coarse categories, this falls under Acylation.
  FORMAL: BOND_CHANGES("broken: C(acyl)-Cl; formed: N-C(=O)") --> RXN_TYPE("Acylation")

Step 3 [MECHANISM]: The core mechanism is nucleophilic acyl substitution where the amine is the nucleophile. The most concise keyword: "acylation".
  FORMAL: RXN_TYPE("Acylation") --> MECHANISM_KWORD("acylation")

Step 4 [SMARTS_CONSTRUCTION]: The essential transformation requires: (i) an acyl chloride carbon [C:1](=O) bearing [Cl:2], and (ii) a primary amine nitrogen [N:3]. The reaction forms [C:1](=O)[N:3] (amide bond) and releases Cl⁻. Proposed reaction SMARTS: [C:1](=O)[Cl:2].[N:3]>>[C:1](=O)[N:3]. This captures the acyl C, the leaving Cl, the amine N, and the newly formed amide bond.
  FORMAL: BOND_CHANGES("broken: C(acyl)-Cl; formed: N-C(=O)") + RXN_TYPE("Acylation") --> PROPOSED_SMARTS("[C:1](=O)[Cl:2].[N:3]>>[C:1](=O)[N:3]")

Step 5 [SMARTS_VALIDATION]: The proposed SMARTS "[C:1](=O)[Cl:2].[N:3]>>[C:1](=O)[N:3]" has valid atom mapping (:1, :2, :3), the reactant side correctly encodes the acyl chloride and amine, and the product side correctly encodes the amide. RDKit's AllChem.ReactionFromSmarts() should parse this successfully.
  FORMAL: PROPOSED_SMARTS("[C:1](=O)[Cl:2].[N:3]>>[C:1](=O)[N:3]") --> SMARTS_PARSEABLE(yes)

Step 6 [TEMPLATE_SELECTION]: Comparing with the choices:
  A: [C:1](=O)[Cl:2].[N:3]>>[C:1](=O)[N:3] — acyl chloride + amine → amide. Matches exactly.
  B: [C:1][Cl:2].[N:3]>>[C:1][N:3] — alkyl chloride + amine → N-alkyl product. Missing carbonyl (=O); this is N-alkylation, not acylation. Wrong.
  C: [C:1](=O)[Cl:2].[O:3]>>[C:1](=O)[O:3] — acyl chloride + oxygen nucleophile → ester. Uses O:3 not N:3. Wrong.
  D: [C:1](=O)[N:2].[O:3]>>[C:1](=O)[O:3] — amide + oxygen → ester (transacylation). Wrong; starts from amide not acyl chloride.
  E: [C:1](=O)[Cl:2]>>[C:1](=O)[OH] — hydrolysis of acyl chloride. Single-component, produces acid not amide. Wrong.
  The best match is A.
  FORMAL: PROPOSED_SMARTS("[C:1](=O)[Cl:2].[N:3]>>[C:1](=O)[N:3]") + OPTIONS(["A","B","C","D","E"]) --> SELECTED_OPTION("A")

Answer: A"""


USER_TEMPLATE = """Reaction SMILES: {rxn_smiles}
Reactants: {reactants_smiles}
Product: {product_smiles}
Ground Truth Reaction Type (coarse): {coarse_rxn_cls}
Ground Truth Template Letter: {gt_letter}

IMPORTANT: The ground truth reaction type and template letter are provided for calibration purposes only. You must still generate the complete reasoning chain as if you were solving this problem from scratch. Do NOT simply state the ground truth — you must work through every step of the reasoning template, analyzing the bond changes, reaction type, mechanism keyword, SMARTS construction, and template selection to naturally arrive at the correct answer. Your RXN_TYPE should match the ground truth coarse category and your SELECTED_OPTION should match the ground truth letter.

{choices_text}

Generate the formal reasoning chain following EXACTLY the unified step format specified. Analyze the bond changes, derive the reaction type and mechanism keyword, propose a reaction SMARTS, verify parseability, then select the best-matching template letter from the choices above.
"""
