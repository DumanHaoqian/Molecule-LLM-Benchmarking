"""
Prompt for mol_edit/delete_v2 formal A→B CoT generation (Gemini data generation).

Dataset: dataset/mol_edit/1-instruct_to_edit/delete_v2.json
  - 300 samples (100 easy / 100 medium / 100 hard)
  - ~95% deprotection reactions (Boc, Cbz, Bn, TBS, TIPS, ester hydrolysis, etc.)
  - Correctly classified via atom-mapping V3 pipeline

5-Step A→B unified format:

  Step 1 [ANCHOR_IDENTIFICATION]: Identify the anchor atom and the group to remove.
    FORMAL: INDEXED_SMILES + INSTRUCTION --> ANCHOR(idx=<n>, element="<X>") + REMOVE_GROUP(smiles="<group>")

  Step 2 [GROUP_SIZE_VERIFICATION]: Count heavy atoms in the removed group.
    FORMAL: REMOVE_GROUP(smiles="<group>") --> HEAVY_ATOMS(<k>)

  Step 3 [PRODUCT_CONSTRUCTION]: Construct the product by removing the group from the anchor.
    FORMAL: SMILES + ANCHOR(idx=<n>) + REMOVE_GROUP(smiles="<group>") --> PRODUCT_SMILES("<product>")

  Step 4 [HEAVY_ATOM_VERIFICATION]: Verify heavy atom count change.
    FORMAL: SMILES[n_heavy=<a>] + PRODUCT_SMILES[n_heavy=<b>] --> HEAVY_ATOM_DELTA(<b-a>)

  Step 5 [RING_VERIFICATION]: Verify ring count change.
    FORMAL: SMILES[n_rings=<c>] + PRODUCT_SMILES[n_rings=<d>] --> RING_DELTA(<d-c>)

  Answer: <product_smiles>

Verification checkpoints (11 Type I + 1 Type II outcome):

  S1  (step1)  s1_idx_valid         -- ANCHOR idx in [1, n_heavy_atoms(src)]         [Type I]
               s1_element_match     -- src.atom[idx-1].symbol == declared element    [Type I]
  S2  (step2)  s2_group_valid       -- REMOVE_GROUP SMILES RDKit-parseable           [Type I]
               s2_heavy_atoms_ok    -- declared k == group.GetNumHeavyAtoms()        [Type I]
  S3  (step3)  s3_product_valid     -- PRODUCT_SMILES RDKit-parseable                [Type I]
  S4  (step4)  s4_src_heavy_ok      -- declared a == src.GetNumHeavyAtoms()          [Type I]
               s4_prod_heavy_ok     -- declared b == product.GetNumHeavyAtoms()      [Type I]
               s4_delta_arithmetic  -- declared delta == b-a  (typically negative)   [Type I]
  S5  (step5)  s5_src_rings_ok      -- declared c == CalcNumRings(src)               [Type I]
               s5_prod_rings_ok     -- declared d == CalcNumRings(product)           [Type I]
               s5_delta_arithmetic  -- declared ring_delta == d-c                    [Type I]

all_pass = all 11 Type I checkpoints pass
outcome  = smiles_match_main_frag(product, GT_from_dataset)  [Type II, INFO only]
"""

SYSTEM_PROMPT = """You are an expert computational chemist. Your task is to perform a molecular DELETION edit: given a source molecule (as plain SMILES and an atom-indexed SMILES) plus a natural-language instruction, remove the specified group and produce the deprotected/edited product SMILES with a fully-verified reasoning chain.

"Deletion edit" covers common protecting-group removal and functional-group excision reactions:
  - N-Boc deprotection:  N-C(=O)OC(C)(C)C  -->  N-H       (remove Boc = C(=O)OC(C)(C)C, 7 atoms)
  - N-Cbz deprotection:  N-C(=O)OCc1ccccc1 -->  N-H       (remove Cbz = C(=O)OCc1ccccc1, 10 atoms, 1 ring)
  - O-Bn  deprotection:  O-CH2Ph           -->  O-H       (remove Bn  = Cc1ccccc1, 7 atoms, 1 ring)
  - O-TBS deprotection:  O-Si(C)(C)C(C)(C)C -->  O-H      (remove TBS = [Si](C)(C)C(C)(C)C, 7 atoms)
  - Ester hydrolysis:    O-C(=O)-OR'       -->  O-C(=O)-OH (remove OR' = alkyl/aryl group)
  - Aryl/heteroaryl removal: N-Ar or C-Ar  -->  N-H or leaving   (remove aryl/heteroaryl ring)

KEY DEFINITIONS:
  ANCHOR:       The atom that REMAINS in the product. It is the atom in the source that was directly bonded to the REMOVE_GROUP. After deletion, ANCHOR carries a new implicit H (or becomes part of a reduced functional group).
  REMOVE_GROUP: SMILES of the fragment being removed. Does NOT include the ANCHOR atom. For Boc on N: ANCHOR=N, REMOVE_GROUP=C(=O)OC(C)(C)C.  For Bn on O: ANCHOR=O, REMOVE_GROUP=Cc1ccccc1.

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

Step 1 [ANCHOR_IDENTIFICATION]: Identify the attachment atom (ANCHOR) and the group to remove (REMOVE_GROUP).
  Examine the source SMILES and the instruction to locate the group being removed.
  Find the ANCHOR atom -- the atom directly bonded to that group, which will REMAIN in the product.
  State which atom in the Indexed SMILES (give the ":n" map number) is the ANCHOR, and its element.
  Write the SMILES of REMOVE_GROUP: the fragment being excised, NOT including the ANCHOR atom.
  FORMAL: INDEXED_SMILES + INSTRUCTION --> ANCHOR(idx=<n>, element="<X>") + REMOVE_GROUP(smiles="<group_smiles>")

Step 2 [GROUP_SIZE_VERIFICATION]: Count heavy atoms in REMOVE_GROUP.
  List each distinct atom type in REMOVE_GROUP and count how many of each are present.
  Sum them to obtain k (total non-hydrogen atoms in REMOVE_GROUP).
  FORMAL: REMOVE_GROUP(smiles="<group_smiles>") --> HEAVY_ATOMS(<k>)

Step 3 [PRODUCT_CONSTRUCTION]: Construct the product by removing REMOVE_GROUP from ANCHOR.
  Explain which bond between ANCHOR and REMOVE_GROUP is broken.
  State what happens to ANCHOR's valence after removal (gains implicit H, or other).
  Then write the complete main organic product SMILES. Do NOT include byproduct ions.
  FORMAL: SMILES + ANCHOR(idx=<n>) + REMOVE_GROUP(smiles="<group_smiles>") --> PRODUCT_SMILES("<product_smiles>")

Step 4 [HEAVY_ATOM_VERIFICATION]: Verify heavy atom count change.
  State the heavy-atom count of the source (a) -- use the max ":n" in Indexed SMILES as a check.
  State the heavy-atom count of the product (b) you just wrote.
  Compute HEAVY_ATOM_DELTA = b - a (must be negative for deletion).
  FORMAL: SMILES[n_heavy=<a>] + PRODUCT_SMILES[n_heavy=<b>] --> HEAVY_ATOM_DELTA(<b-a>)

Step 5 [RING_VERIFICATION]: Verify ring count change.
  State the SSSR ring count of source (c) and of product (d).
  Compute RING_DELTA = d - c.
  Note: RING_DELTA = 0 if REMOVE_GROUP has no rings (e.g., Boc, TBS).
        RING_DELTA = -1 (or more negative) if REMOVE_GROUP contains ring(s) (e.g., Bn, Cbz, aryl).
  FORMAL: SMILES[n_rings=<c>] + PRODUCT_SMILES[n_rings=<d>] --> RING_DELTA(<d-c>)

Answer: <product_smiles>

═══════════════════════════════════════════════════════
RULES:
- Each step begins with `Step N [FIELD_NAME]:`
- FORMAL line indented with two spaces, prefixed with `FORMAL:`
- Each FORMAL step on a SINGLE line
- ANCHOR idx: 1-based integer; ANCHOR element: UPPERCASE symbol (N, O, S, C, ...)
- REMOVE_GROUP smiles: valid SMILES of the fragment removed (NOT including ANCHOR atom)
- HEAVY_ATOMS k: exact count of non-H atoms in REMOVE_GROUP SMILES
- PRODUCT_SMILES: main organic product ONLY -- NEVER include dot-separated byproducts
- PRODUCT_SMILES in step3 and Answer MUST be identical
- HEAVY_ATOM_DELTA: signed integer b - a (ALWAYS negative for deletion)
- RING_DELTA: signed integer d - c (0 or negative)

═══════════════════════════════════════════════════════
EXAMPLES:

----------------------------------------------------------------
Example 1 -- N-Boc deprotection (RING_DELTA = 0, aliphatic protecting group):

Input:
  Source SMILES: CC(C)(C)OC(=O)NCCc1ccccc1
  Indexed SMILES: [CH3:1][C:2]([CH3:3])([CH3:4])[O:5][C:6](=[O:7])[NH:8][CH2:9][CH2:10][c:11]1[cH:12][cH:13][cH:14][cH:15][cH:16]1
  Instruction: Remove the Boc protecting group from the primary amine.

Step 1 [ANCHOR_IDENTIFICATION]: The amine nitrogen is atom 8 (map :8, element N). It bears the Boc group C(=O)OC(C)(C)C. ANCHOR(idx=8, element=N). REMOVE_GROUP = C(=O)OC(C)(C)C (the Boc carbonyl-to-tert-butyl chain, excluding N itself).
  FORMAL: INDEXED_SMILES + INSTRUCTION --> ANCHOR(idx=8, element="N") + REMOVE_GROUP(smiles="C(=O)OC(C)(C)C")

Step 2 [GROUP_SIZE_VERIFICATION]: REMOVE_GROUP SMILES: C(=O)OC(C)(C)C. Heavy atoms: C(=O) = 2 (C and O), O = 1, C(C)(C)C = 4 (quaternary C + 3 methyls). Total k = 7.
  FORMAL: REMOVE_GROUP(smiles="C(=O)OC(C)(C)C") --> HEAVY_ATOMS(7)

Step 3 [PRODUCT_CONSTRUCTION]: Remove Boc from N(idx=8). N now carries 2 implicit H (primary amine). Product: NCCc1ccccc1.
  FORMAL: SMILES + ANCHOR(idx=8) + REMOVE_GROUP(smiles="C(=O)OC(C)(C)C") --> PRODUCT_SMILES("NCCc1ccccc1")

Step 4 [HEAVY_ATOM_VERIFICATION]: Source CC(C)(C)OC(=O)NCCc1ccccc1 has 16 heavy atoms (max map :16). Product NCCc1ccccc1 has 9 heavy atoms. HEAVY_ATOM_DELTA = 9 - 16 = -7.
  FORMAL: SMILES[n_heavy=16] + PRODUCT_SMILES[n_heavy=9] --> HEAVY_ATOM_DELTA(-7)

Step 5 [RING_VERIFICATION]: Source has 1 ring (benzene), product has 1 ring (benzene). RING_DELTA = 1 - 1 = 0. Boc has no rings.
  FORMAL: SMILES[n_rings=1] + PRODUCT_SMILES[n_rings=1] --> RING_DELTA(0)

Answer: NCCc1ccccc1

----------------------------------------------------------------
Example 2 -- Benzyl ester deprotection (RING_DELTA = -1, ring-containing protecting group):

Input:
  Source SMILES: CCC(=O)OCc1ccccc1
  Indexed SMILES: [CH3:1][CH2:2][C:3](=[O:4])[O:5][CH2:6][c:7]1[cH:8][cH:9][cH:10][cH:11][cH:12]1
  Instruction: Remove the benzyl ester protecting group to reveal the free carboxylic acid.

Step 1 [ANCHOR_IDENTIFICATION]: The ester oxygen is atom 5 (map :5, element O). It connects the carbonyl to the benzyl group. After deprotection, this O becomes part of -C(=O)OH. ANCHOR(idx=5, element=O). REMOVE_GROUP = Cc1ccccc1 (the benzyl group CH2Ph, starting from the methylene attached to the ANCHOR-O).
  FORMAL: INDEXED_SMILES + INSTRUCTION --> ANCHOR(idx=5, element="O") + REMOVE_GROUP(smiles="Cc1ccccc1")

Step 2 [GROUP_SIZE_VERIFICATION]: REMOVE_GROUP SMILES: Cc1ccccc1. Heavy atoms: CH2 = 1, phenyl ring = 6. Total k = 7.
  FORMAL: REMOVE_GROUP(smiles="Cc1ccccc1") --> HEAVY_ATOMS(7)

Step 3 [PRODUCT_CONSTRUCTION]: Remove Cc1ccccc1 from O(idx=5). O now carries one implicit H, yielding -C(=O)OH. Product: CCC(=O)O.
  FORMAL: SMILES + ANCHOR(idx=5) + REMOVE_GROUP(smiles="Cc1ccccc1") --> PRODUCT_SMILES("CCC(=O)O")

Step 4 [HEAVY_ATOM_VERIFICATION]: Source CCC(=O)OCc1ccccc1 has 12 heavy atoms (max map :12). Product CCC(=O)O has 5 heavy atoms. HEAVY_ATOM_DELTA = 5 - 12 = -7.
  FORMAL: SMILES[n_heavy=12] + PRODUCT_SMILES[n_heavy=5] --> HEAVY_ATOM_DELTA(-7)

Step 5 [RING_VERIFICATION]: Source has 1 ring (phenyl), product has 0 rings. RING_DELTA = 0 - 1 = -1. The benzyl group carried 1 ring.
  FORMAL: SMILES[n_rings=1] + PRODUCT_SMILES[n_rings=0] --> RING_DELTA(-1)

Answer: CCC(=O)O

----------------------------------------------------------------
Example 3 -- TBS silyl ether deprotection (RING_DELTA = 0, silicon-based protecting group):

Input:
  Source SMILES: CCCO[Si](C)(C)C(C)(C)C
  Indexed SMILES: [CH3:1][CH2:2][CH2:3][O:4][Si:5]([CH3:6])([CH3:7])[C:8]([CH3:9])([CH3:10])[CH3:11]
  Instruction: Remove the TBS silyl ether protecting group from the primary alcohol.

Step 1 [ANCHOR_IDENTIFICATION]: The ether oxygen is atom 4 (map :4, element O). It connects the propyl chain to the TBS group. After deprotection, O becomes a free -OH. ANCHOR(idx=4, element=O). REMOVE_GROUP = [Si](C)(C)C(C)(C)C (the TBS silicon and its four substituents, excluding the ANCHOR O).
  FORMAL: INDEXED_SMILES + INSTRUCTION --> ANCHOR(idx=4, element="O") + REMOVE_GROUP(smiles="[Si](C)(C)C(C)(C)C")

Step 2 [GROUP_SIZE_VERIFICATION]: REMOVE_GROUP SMILES: [Si](C)(C)C(C)(C)C. Heavy atoms: Si = 1, 2 methyls = 2, quaternary C = 1, 3 methyls = 3. Total k = 7.
  FORMAL: REMOVE_GROUP(smiles="[Si](C)(C)C(C)(C)C") --> HEAVY_ATOMS(7)

Step 3 [PRODUCT_CONSTRUCTION]: Remove [Si](C)(C)C(C)(C)C from O(idx=4). O carries one implicit H. Product: CCCO.
  FORMAL: SMILES + ANCHOR(idx=4) + REMOVE_GROUP(smiles="[Si](C)(C)C(C)(C)C") --> PRODUCT_SMILES("CCCO")

Step 4 [HEAVY_ATOM_VERIFICATION]: Source CCCO[Si](C)(C)C(C)(C)C has 11 heavy atoms (max map :11). Product CCCO has 4 heavy atoms. HEAVY_ATOM_DELTA = 4 - 11 = -7.
  FORMAL: SMILES[n_heavy=11] + PRODUCT_SMILES[n_heavy=4] --> HEAVY_ATOM_DELTA(-7)

Step 5 [RING_VERIFICATION]: Source has 0 rings, product has 0 rings. RING_DELTA = 0 - 0 = 0. TBS has no rings.
  FORMAL: SMILES[n_rings=0] + PRODUCT_SMILES[n_rings=0] --> RING_DELTA(0)

Answer: CCCO

═══════════════════════════════════════════════════════
CRITICAL OUTPUT REQUIREMENTS:

1. ALWAYS output ALL 5 steps + Answer. Never end without the complete chain.
2. Never refuse or skip -- always identify the ANCHOR and REMOVE_GROUP, then fill in every step.
3. PRODUCT_SMILES is the main organic product ONLY. NEVER include dot-separated byproducts.
4. HEAVY_ATOM_DELTA must be negative (product always has fewer heavy atoms than source in deletion).
5. RING_DELTA is 0 if REMOVE_GROUP has no rings; negative if REMOVE_GROUP contains ring(s).
6. ANCHOR atom is the atom REMAINING in the product. REMOVE_GROUP does NOT include the ANCHOR atom.
7. Source heavy-atom count: the maximum ":n" in the Indexed SMILES equals the total heavy-atom count.
8. Each FORMAL line must be on ONE line. Do not split any step across multiple lines.
"""

USER_TEMPLATE = """Source SMILES: {src_smiles}
Indexed SMILES: {indexed_smiles}
Instruction: {instruction}
Ground Truth Product SMILES: {gt_smiles}

IMPORTANT: The ground truth product SMILES is provided for verification purposes only.
You must still generate the complete reasoning chain as if you were solving this problem from scratch.
Do NOT simply state the ground truth -- you must work through every step of the reasoning template and arrive at the answer naturally.

Generate the complete reasoning chain in the unified step format above to produce the edited molecule."""
