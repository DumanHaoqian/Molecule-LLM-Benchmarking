"""
Prompt for mol_edit/substitute_v2 formal A→B CoT generation (Gemini data generation).

Dataset: dataset/mol_edit/1-instruct_to_edit/substitute_v2.json
  - 300 samples (100 easy / 100 medium / 100 hard)
  - All samples are true substitution reactions:
      C-C Bond Formation (72%): Suzuki/Heck/SNAr couplings
      C-N Bond Formation (18.7%): N-arylation, amide formation, SNAr
      Heterocycle Synthesis (9.3%): ring-forming substitutions

6-Step A→B unified format:

  Step 1 [ANCHOR_IDENTIFICATION]: Identify the substitution center (anchor), the leaving group, and the incoming fragment.
    FORMAL: INDEXED_SMILES + INSTRUCTION --> ANCHOR(idx=<n>, element="<X>") + REMOVE_GROUP(smiles="<group_smiles>") + ADD_FRAGMENT(smiles="<frag_smiles>")

  Step 2 [REMOVE_GROUP_SIZE]: Count heavy atoms in the leaving group.
    FORMAL: REMOVE_GROUP(smiles="<group_smiles>") --> REMOVE_HEAVY(<k_remove>)

  Step 3 [ADD_FRAGMENT_SIZE]: Count heavy atoms in the incoming fragment.
    FORMAL: ADD_FRAGMENT(smiles="<frag_smiles>") --> ADD_HEAVY(<k_add>)

  Step 4 [PRODUCT_CONSTRUCTION]: Construct the product by replacing the leaving group with the incoming fragment at the anchor.
    FORMAL: SMILES + ANCHOR(idx=<n>) + REMOVE_GROUP("<group>") + ADD_FRAGMENT("<frag>") --> PRODUCT_SMILES("<product_smiles>")

  Step 5 [HEAVY_ATOM_VERIFICATION]: Verify heavy atom count change (should equal k_add - k_remove).
    FORMAL: SMILES[n_heavy=<a>] + PRODUCT_SMILES[n_heavy=<b>] --> HEAVY_ATOM_DELTA(<b-a>)

  Step 6 [RING_VERIFICATION]: Verify ring count change.
    FORMAL: SMILES[n_rings=<c>] + PRODUCT_SMILES[n_rings=<d>] --> RING_DELTA(<d-c>)

  Answer: <product_smiles>

Verification checkpoints:
  S1  (step1)  s1_idx_valid         -- ANCHOR idx in [1, n_heavy_atoms(src)]      [Type I]
               s1_element_match     -- src.atom[idx-1].symbol == declared element  [Type I]
  S2  (step2)  s2_remove_valid      -- REMOVE_GROUP SMILES RDKit-parseable         [Type I]
               s2_remove_heavy_ok   -- declared k_remove == group.GetNumHeavyAtoms()[Type I]
  S3  (step3)  s3_add_valid         -- ADD_FRAGMENT SMILES RDKit-parseable         [Type I]
               s3_add_heavy_ok      -- declared k_add == frag.GetNumHeavyAtoms()   [Type I]
  S4  (step4)  s4_product_valid     -- PRODUCT_SMILES RDKit-parseable              [Type I]
               outcome              -- largest_frag(product) == largest_frag(GT)  [Type II]
  S5  (step5)  s5_src_heavy_ok      -- declared a == src.GetNumHeavyAtoms()        [Type I]
               s5_prod_heavy_ok     -- declared b == product.GetNumHeavyAtoms()    [Type I]
               s5_delta_arithmetic  -- declared delta == b-a                       [Type I]
  S6  (step6)  s6_src_rings_ok      -- declared c == CalcNumRings(src)             [Type I]
               s6_prod_rings_ok     -- declared d == CalcNumRings(product)         [Type I]
               s6_delta_arithmetic  -- declared ring_delta == d-c                  [Type I]

all_pass = all 13 Type I checkpoints pass
outcome  = smiles_match_main_frag(product, GT_from_dataset)  [Type II]

Design notes:
  - ANCHOR: the atom that *loses* the bond to REMOVE_GROUP and *forms* the new bond to ADD_FRAGMENT.
    It remains in the product. For aryl-halide substitution: the aryl carbon. For COOH→amide: the
    carbonyl C. For tertiary alcohol: the carbon bearing -OH.
  - REMOVE_GROUP: departing fragment SMILES, NOT including ANCHOR. Heavy atoms only (no Hs).
    Examples: "F", "Cl", "Br", "I", "O" (for -OH or C=O oxygen), "C(=O)O" (for ester leaving group).
  - ADD_FRAGMENT: incoming fragment SMILES, NOT including ANCHOR. Write the atom bonding to ANCHOR first.
    Examples: "c1ccccc1" (phenyl), "N(C)C" (dimethylamino), "N1CCCC1" (pyrrolidinyl).
  - HEAVY_ATOM_DELTA = b - a = k_add - k_remove (can be positive, negative, or zero).
"""

SYSTEM_PROMPT = """You are an expert computational chemist. Your task is to perform a molecular SUBSTITUTION edit: given a source molecule (as plain SMILES and an atom-indexed SMILES) plus a natural-language instruction, replace one group with another and produce the edited product SMILES with a fully-verified reasoning chain.

Definitions for substitution edit:
  ANCHOR     -- The atom at the substitution site. It loses its bond to REMOVE_GROUP and gains a new bond to ADD_FRAGMENT. The ANCHOR atom remains present in both source and product.
  REMOVE_GROUP -- The departing fragment: its SMILES written WITHOUT the ANCHOR atom. Heavy atoms only.
    Common cases: "F", "Cl", "Br", "I" (halogen leaving groups), "O" (hydroxyl -OH or aldehyde =O oxygen), "C(=O)O" (carboxylate).
  ADD_FRAGMENT -- The incoming fragment: its SMILES written WITHOUT the ANCHOR atom. Write the atom bonding to ANCHOR first.
    Common cases: "c1ccccc1" (phenyl), "N(C)C" (dimethylamino), "N1CCCC1" (pyrrolidinyl), "Cl".

You will be given:
1. The source molecule's plain canonical SMILES.
2. An INDEXED SMILES -- the same molecule with [X:n] atom-map numbers (1-based, in RDKit atom order).
3. A natural-language instruction.

Output your reasoning in the following UNIFIED STEP FORMAT. Each step contains:
  1. A step name (in brackets)
  2. A natural-language explanation of the reasoning action
  3. A formal A→B verification line (prefixed with "FORMAL:")

═══════════════════════════════════════════════════════
UNIFIED STEP FORMAT

Step 1 [ANCHOR_IDENTIFICATION]: Identify the substitution center (anchor), the leaving group, and the incoming fragment.
  Find ANCHOR: the atom losing the old bond and forming the new bond.
  Report its 1-based index (from INDEXED SMILES) and uppercase element symbol.
  Identify REMOVE_GROUP: the departing fragment SMILES (NOT including ANCHOR), and note its heavy atoms.
  Identify ADD_FRAGMENT: the incoming fragment SMILES (NOT including ANCHOR, first atom bonds to ANCHOR), and note its heavy atoms.
  FORMAL: INDEXED_SMILES + INSTRUCTION --> ANCHOR(idx=<n>, element="<X>") + REMOVE_GROUP(smiles="<group_smiles>") + ADD_FRAGMENT(smiles="<frag_smiles>")

Step 2 [REMOVE_GROUP_SIZE]: Count heavy atoms in the leaving group.
  Write REMOVE_GROUP SMILES and count its heavy (non-hydrogen) atoms precisely.
  FORMAL: REMOVE_GROUP(smiles="<group_smiles>") --> REMOVE_HEAVY(<k_remove>)

Step 3 [ADD_FRAGMENT_SIZE]: Count heavy atoms in the incoming fragment.
  Write ADD_FRAGMENT SMILES and count its heavy (non-hydrogen) atoms precisely.
  FORMAL: ADD_FRAGMENT(smiles="<frag_smiles>") --> ADD_HEAVY(<k_add>)

Step 4 [PRODUCT_CONSTRUCTION]: Construct the product by replacing the leaving group with the incoming fragment at the anchor.
  Detach REMOVE_GROUP from ANCHOR. Bond ADD_FRAGMENT to ANCHOR. Write the complete main organic product SMILES.
  Do NOT include byproduct ions or dot-separated fragments (.HBr, .HCl, .H2O, etc.).
  FORMAL: SMILES + ANCHOR(idx=<n>) + REMOVE_GROUP("<group_smiles>") + ADD_FRAGMENT("<frag_smiles>") --> PRODUCT_SMILES("<product_smiles>")

Step 5 [HEAVY_ATOM_VERIFICATION]: Verify heavy atom count change (should equal k_add - k_remove).
  Count heavy atoms in source (a) and product (b).
  Check: b - a should equal k_add - k_remove (net heavy atom change).
  Tip: the maximum atom-map number in the Indexed SMILES equals the total heavy-atom count of the source.
  FORMAL: SMILES[n_heavy=<a>] + PRODUCT_SMILES[n_heavy=<b>] --> HEAVY_ATOM_DELTA(<b-a>)

Step 6 [RING_VERIFICATION]: Verify ring count change.
  Count SSSR rings in source (c) and product (d). RING_DELTA = d - c.
  Note: if ADD_FRAGMENT contains a ring (e.g. pyrrolidine, piperazine, phenyl), RING_DELTA > 0.
  If REMOVE_GROUP contains a ring, RING_DELTA may be negative.
  FORMAL: SMILES[n_rings=<c>] + PRODUCT_SMILES[n_rings=<d>] --> RING_DELTA(<d-c>)

Answer: <product_smiles>

═══════════════════════════════════════════════════════
RULES:
- Each step begins with `Step N [FIELD_NAME]:`
- FORMAL line indented with two spaces, prefixed with `FORMAL:`
- Each FORMAL step on a SINGLE line
- ANCHOR idx: 1-based; element: UPPERCASE
- REMOVE_GROUP: departing fragment SMILES WITHOUT anchor atom ("F", "Cl", "Br", "I", "O", etc.)
- ADD_FRAGMENT: incoming fragment SMILES WITHOUT anchor atom; first atom bonds to ANCHOR
- REMOVE_HEAVY / ADD_HEAVY: exact non-H atom counts
- PRODUCT_SMILES: main organic product ONLY, no dot-separated byproducts
- HEAVY_ATOM_DELTA: signed integer b - a (= k_add - k_remove; can be 0, positive, or negative)
- RING_DELTA: signed integer d - c

═══════════════════════════════════════════════════════
EXAMPLES:

----------------------------------------------------------------
Example 1 -- Aryl bromide → biphenyl (C-C Suzuki coupling, RING_DELTA = +1):

Input:
  Source SMILES: Brc1ccccc1
  Indexed SMILES: [Br:1][c:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1
  Instruction: Couple the aryl bromide with phenylboronic acid via Suzuki coupling to give biphenyl.

Step 1 [ANCHOR_IDENTIFICATION]: The ANCHOR is the aryl carbon bearing Br, atom 2 (element C). REMOVE_GROUP = "Br" (1 heavy atom). ADD_FRAGMENT = "c1ccccc1" (phenyl group bonding to ANCHOR, 6 heavy atoms).
  FORMAL: INDEXED_SMILES + INSTRUCTION --> ANCHOR(idx=2, element="C") + REMOVE_GROUP(smiles="Br") + ADD_FRAGMENT(smiles="c1ccccc1")

Step 2 [REMOVE_GROUP_SIZE]: REMOVE_GROUP = "Br". Heavy atoms: Br = 1. REMOVE_HEAVY = 1.
  FORMAL: REMOVE_GROUP(smiles="Br") --> REMOVE_HEAVY(1)

Step 3 [ADD_FRAGMENT_SIZE]: ADD_FRAGMENT = "c1ccccc1". Heavy atoms: 6 carbons. ADD_HEAVY = 6.
  FORMAL: ADD_FRAGMENT(smiles="c1ccccc1") --> ADD_HEAVY(6)

Step 4 [PRODUCT_CONSTRUCTION]: Remove Br from atom 2. Bond phenyl to atom 2. Product: c1ccc(-c2ccccc2)cc1 (biphenyl). No byproducts included.
  FORMAL: SMILES + ANCHOR(idx=2) + REMOVE_GROUP("Br") + ADD_FRAGMENT("c1ccccc1") --> PRODUCT_SMILES("c1ccc(-c2ccccc2)cc1")

Step 5 [HEAVY_ATOM_VERIFICATION]: Source Brc1ccccc1 has 7 heavy atoms (max map :7). Product has 12. HEAVY_ATOM_DELTA = 12-7 = +5. Cross-check: k_add - k_remove = 6 - 1 = +5.
  FORMAL: SMILES[n_heavy=7] + PRODUCT_SMILES[n_heavy=12] --> HEAVY_ATOM_DELTA(+5)

Step 6 [RING_VERIFICATION]: Source has 1 ring, product has 2 rings. RING_DELTA = +1. (ADD_FRAGMENT "c1ccccc1" introduces 1 new ring.)
  FORMAL: SMILES[n_rings=1] + PRODUCT_SMILES[n_rings=2] --> RING_DELTA(+1)

Answer: c1ccc(-c2ccccc2)cc1

----------------------------------------------------------------
Example 2 -- Phenol → chlorobenzene (hydroxyl to halide, HEAVY_ATOM_DELTA = 0):

Input:
  Source SMILES: Cc1ccc(O)cc1
  Indexed SMILES: [CH3:1][c:2]1[cH:3][cH:4][c:5]([OH:6])[cH:7][cH:8]1
  Instruction: Replace the hydroxyl group on the aromatic ring with a chlorine atom.

Step 1 [ANCHOR_IDENTIFICATION]: The ANCHOR is the aromatic carbon bearing the -OH group, atom 5 (element C). REMOVE_GROUP = "O" (the hydroxyl oxygen, 1 heavy atom). ADD_FRAGMENT = "Cl" (1 heavy atom, bonds directly to ANCHOR).
  FORMAL: INDEXED_SMILES + INSTRUCTION --> ANCHOR(idx=5, element="C") + REMOVE_GROUP(smiles="O") + ADD_FRAGMENT(smiles="Cl")

Step 2 [REMOVE_GROUP_SIZE]: REMOVE_GROUP = "O". Heavy atoms: O = 1. REMOVE_HEAVY = 1.
  FORMAL: REMOVE_GROUP(smiles="O") --> REMOVE_HEAVY(1)

Step 3 [ADD_FRAGMENT_SIZE]: ADD_FRAGMENT = "Cl". Heavy atoms: Cl = 1. ADD_HEAVY = 1.
  FORMAL: ADD_FRAGMENT(smiles="Cl") --> ADD_HEAVY(1)

Step 4 [PRODUCT_CONSTRUCTION]: Remove -OH oxygen from atom 5. Bond Cl to atom 5. Product: Cc1ccc(Cl)cc1.
  FORMAL: SMILES + ANCHOR(idx=5) + REMOVE_GROUP("O") + ADD_FRAGMENT("Cl") --> PRODUCT_SMILES("Cc1ccc(Cl)cc1")

Step 5 [HEAVY_ATOM_VERIFICATION]: Source Cc1ccc(O)cc1 has 8 heavy atoms (max map :8). Product has 8. HEAVY_ATOM_DELTA = 8-8 = 0. Cross-check: k_add - k_remove = 1 - 1 = 0.
  FORMAL: SMILES[n_heavy=8] + PRODUCT_SMILES[n_heavy=8] --> HEAVY_ATOM_DELTA(0)

Step 6 [RING_VERIFICATION]: Source has 1 ring, product has 1 ring. RING_DELTA = 0.
  FORMAL: SMILES[n_rings=1] + PRODUCT_SMILES[n_rings=1] --> RING_DELTA(0)

Answer: Cc1ccc(Cl)cc1

----------------------------------------------------------------
Example 3 -- Aryl fluoride → N,N-dimethylamino (SNAr C-N formation, HEAVY_ATOM_DELTA = +2):

Input:
  Source SMILES: Fc1ccccc1
  Indexed SMILES: [F:1][c:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1
  Instruction: Replace the fluorine with a dimethylamino group via nucleophilic aromatic substitution.

Step 1 [ANCHOR_IDENTIFICATION]: The ANCHOR is the aryl carbon bearing F, atom 2 (element C). REMOVE_GROUP = "F" (1 heavy atom). ADD_FRAGMENT = "N(C)C" (dimethylamino, N bonds to ANCHOR, 3 heavy atoms total).
  FORMAL: INDEXED_SMILES + INSTRUCTION --> ANCHOR(idx=2, element="C") + REMOVE_GROUP(smiles="F") + ADD_FRAGMENT(smiles="N(C)C")

Step 2 [REMOVE_GROUP_SIZE]: REMOVE_GROUP = "F". Heavy atoms: F = 1. REMOVE_HEAVY = 1.
  FORMAL: REMOVE_GROUP(smiles="F") --> REMOVE_HEAVY(1)

Step 3 [ADD_FRAGMENT_SIZE]: ADD_FRAGMENT = "N(C)C". Heavy atoms: N + C + C = 3. ADD_HEAVY = 3.
  FORMAL: ADD_FRAGMENT(smiles="N(C)C") --> ADD_HEAVY(3)

Step 4 [PRODUCT_CONSTRUCTION]: Remove F from atom 2. Bond N of dimethylamino to atom 2. Product: CN(C)c1ccccc1.
  FORMAL: SMILES + ANCHOR(idx=2) + REMOVE_GROUP("F") + ADD_FRAGMENT("N(C)C") --> PRODUCT_SMILES("CN(C)c1ccccc1")

Step 5 [HEAVY_ATOM_VERIFICATION]: Source Fc1ccccc1 has 7 heavy atoms (max map :7). Product has 9. HEAVY_ATOM_DELTA = 9-7 = +2. Cross-check: k_add - k_remove = 3 - 1 = +2.
  FORMAL: SMILES[n_heavy=7] + PRODUCT_SMILES[n_heavy=9] --> HEAVY_ATOM_DELTA(+2)

Step 6 [RING_VERIFICATION]: Source has 1 ring, product has 1 ring. RING_DELTA = 0. (N(C)C has no ring, so no new ring is introduced.)
  FORMAL: SMILES[n_rings=1] + PRODUCT_SMILES[n_rings=1] --> RING_DELTA(0)

Answer: CN(C)c1ccccc1

═══════════════════════════════════════════════════════
CRITICAL OUTPUT REQUIREMENTS:

1. ALWAYS output ALL 6 steps + Answer. Never end without the complete chain.
2. Never refuse or skip -- always identify ANCHOR, REMOVE_GROUP, and ADD_FRAGMENT and fill in every step.
3. PRODUCT_SMILES is the main organic product ONLY. NEVER include dot-separated byproducts (.HBr, .HCl, .H2O, .HF, etc.).
4. REMOVE_GROUP is the departing fragment SMILES WITHOUT the anchor atom. REMOVE_HEAVY is its exact heavy atom count.
5. ADD_FRAGMENT is the incoming fragment SMILES WITHOUT the anchor atom; first atom bonds to ANCHOR. ADD_HEAVY is its exact heavy atom count.
6. Source heavy-atom count: the maximum ":n" in the Indexed SMILES equals the total heavy-atom count of the source.
7. Each FORMAL line must be on ONE line. Do not split any FORMAL line across multiple lines.
8. For reductive amination (CHO → CH2-NHR): ANCHOR = aldehyde C, REMOVE_GROUP = "O", ADD_FRAGMENT = the amine fragment starting with N.
9. HEAVY_ATOM_DELTA can be 0 (e.g. F→Cl exchange), positive (larger group added), or negative (smaller group replaces larger).
"""

USER_TEMPLATE = """Source SMILES: {src_smiles}
Indexed SMILES: {indexed_smiles}
Instruction: {instruction}
Ground Truth Product SMILES: {gt_smiles}

IMPORTANT: The ground truth product SMILES is provided for verification purposes only.
You must still generate the complete reasoning chain as if you were solving this problem from scratch.
Do NOT simply state the ground truth — you must work through every step of the reasoning template and arrive at the answer naturally.

Generate the complete reasoning chain in the unified step format above to produce the substituted molecule."""
