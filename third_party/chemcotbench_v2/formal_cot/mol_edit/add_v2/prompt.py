"""
Prompt for mol_edit/add_v2 formal A→B CoT generation (Gemini data generation).

Dataset: dataset/mol_edit/1-instruct_to_edit/add_v2.json
  - 300 samples (100 easy / 100 medium / 100 hard)
  - All samples are true addition reactions:
      C-C Bond Formation (52.7%): Suzuki, Heck, Stille couplings
      C-N Bond Formation (47.3%): N-acylation, N-alkylation, Buchwald-Hartwig couplings

5-Step A→B unified format:

  Step 1 [ANCHOR_IDENTIFICATION]: Identify the anchor atom and leaving group.
    FORMAL: INDEXED_SMILES + INSTRUCTION --> ANCHOR(idx=<n>, element="<X>") + LEAVING(smiles="<smiles_or_none>")

  Step 2 [FRAGMENT_IDENTIFICATION]: Extract incoming fragment and count heavy atoms.
    FORMAL: INSTRUCTION --> ADD_FRAGMENT(smiles="<frag>", heavy_atoms=<k>)

  Step 3 [PRODUCT_CONSTRUCTION]: Construct the product.
    FORMAL: SMILES + ANCHOR(idx=<n>) + LEAVING("<leaving>") + ADD_FRAGMENT(smiles="<frag>") --> PRODUCT_SMILES("<product>")

  Step 4 [HEAVY_ATOM_VERIFICATION]: Verify heavy atom count change.
    FORMAL: SMILES[n_heavy=<a>] + PRODUCT_SMILES[n_heavy=<b>] --> HEAVY_ATOM_DELTA(<b-a>)

  Step 5 [RING_VERIFICATION]: Verify ring count change.
    FORMAL: SMILES[n_rings=<c>] + PRODUCT_SMILES[n_rings=<d>] --> RING_DELTA(<d-c>)

  Answer: <product_smiles>

Verification checkpoints per step:
  S1  (step1)  s1_idx_valid         -- ANCHOR idx in [1, n_heavy_atoms(src)]      [Type I]
               s1_element_match     -- src.atom[idx-1].symbol == declared element  [Type I]
  S2  (step2)  s2_frag_valid        -- ADD_FRAGMENT SMILES RDKit-parseable         [Type I]
               s2_heavy_atoms_ok    -- declared k == frag.GetNumHeavyAtoms()       [Type I]
  S3  (step3)  s3_product_valid     -- PRODUCT_SMILES RDKit-parseable              [Type I]
  S4  (step4)  s4_src_heavy_ok      -- declared a == src.GetNumHeavyAtoms()        [Type I]
               s4_prod_heavy_ok     -- declared b == product.GetNumHeavyAtoms()    [Type I]
               s4_delta_arithmetic  -- declared delta == b-a                       [Type I]
  S5  (step5)  s5_src_rings_ok      -- declared c == CalcNumRings(src)             [Type I]
               s5_prod_rings_ok     -- declared d == CalcNumRings(product)         [Type I]
               s5_delta_arithmetic  -- declared ring_delta == d-c                  [Type I]

all_pass = all 11 Type I checkpoints pass
outcome  = smiles_match_main_frag(product, GT_from_dataset)  [Type II, INFO only]
"""

SYSTEM_PROMPT = """You are an expert computational chemist. Your task is to perform a molecular ADDITION edit: given a source molecule (as plain SMILES and an atom-indexed SMILES) plus a natural-language instruction, produce the edited product SMILES with a fully-verified reasoning chain.

"Addition edit" covers two sub-types:
  TYPE A -- Pure addition: a new fragment bonds to an atom in the source. No heavy atom is removed.
    Examples: N-acylation (NH2 + acyl --> NHC(=O)R, only H departs), O-acylation (OH + acyl --> OC(=O)R, only H departs), N-alkylation (NH + alkyl --> NR, only H departs).

  TYPE B -- Replacement addition: an incoming fragment displaces a heavy-atom leaving group at the ANCHOR.
    Examples: Suzuki coupling (Ar-Br + ArB(OH)2 --> Ar-Ar, Br departs), Buchwald-Hartwig (Ar-Br + amine --> Ar-NR2, Br departs), halide displacement (C-Cl + NHR2 --> C-NR2, Cl departs).

LEAVING group rules:
  - If only an H atom departs (N-acylation, O-acylation, N-alkylation on sp3 N/O) --> LEAVING = "none"
  - If the instruction involves Suzuki/Heck/Stille/Buchwald/C-N coupling or halide displacement --> LEAVING = the halogen SMILES ("Br", "Cl", "I", or "F")
  - For other leaving groups (e.g., -OH in an exchange): write the SMILES of the departing heavy-atom group (e.g., "O")

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

Step 1 [ANCHOR_IDENTIFICATION]: Identify the attachment atom (ANCHOR) and the leaving group (LEAVING).
  Find ANCHOR: the atom in the source where the new bond to the incoming fragment forms.
  Report its 1-based index (from INDEXED SMILES) and uppercase element symbol.
  Identify LEAVING: "none" for TYPE A (only H departs) or the heavy-atom leaving group SMILES for TYPE B.
  FORMAL: INDEXED_SMILES + INSTRUCTION --> ANCHOR(idx=<n>, element="<X>") + LEAVING(smiles="<smiles_or_none>")

Step 2 [FRAGMENT_IDENTIFICATION]: Extract the incoming fragment (ADD_FRAGMENT) and count its heavy atoms.
  Write the standalone SMILES of the new group bonding to ANCHOR, and count its heavy (non-hydrogen) atoms precisely.
  Do NOT include the ANCHOR atom itself -- only the new incoming atoms.
  FORMAL: INSTRUCTION --> ADD_FRAGMENT(smiles="<frag_smiles>", heavy_atoms=<k>)

Step 3 [PRODUCT_CONSTRUCTION]: Construct the product by bonding ADD_FRAGMENT to ANCHOR.
  TYPE A: bond ANCHOR to ADD_FRAGMENT (no heavy atoms removed).
  TYPE B: remove LEAVING from ANCHOR, then bond ADD_FRAGMENT to ANCHOR.
  Write the complete main organic product SMILES. Do NOT include byproduct ions or dot-separated fragments.
  FORMAL: SMILES + ANCHOR(idx=<n>) + LEAVING("<leaving>") + ADD_FRAGMENT(smiles="<frag_smiles>") --> PRODUCT_SMILES("<product_smiles>")

Step 4 [HEAVY_ATOM_VERIFICATION]: Verify heavy atom count change.
  Count heavy atoms in source (a) and product (b). HEAVY_ATOM_DELTA = b - a.
  Tip: the maximum atom-map number in the Indexed SMILES equals the total heavy-atom count of the source.
  FORMAL: SMILES[n_heavy=<a>] + PRODUCT_SMILES[n_heavy=<b>] --> HEAVY_ATOM_DELTA(<b-a>)

Step 5 [RING_VERIFICATION]: Verify ring count change.
  Count SSSR rings in source (c) and product (d). RING_DELTA = d - c.
  FORMAL: SMILES[n_rings=<c>] + PRODUCT_SMILES[n_rings=<d>] --> RING_DELTA(<d-c>)

Answer: <product_smiles>

═══════════════════════════════════════════════════════
RULES:
- Each step begins with `Step N [FIELD_NAME]:`
- FORMAL line indented with two spaces, prefixed with `FORMAL:`
- Each FORMAL step on a SINGLE line
- ANCHOR idx: 1-based integer; ANCHOR element: UPPERCASE symbol (C, N, O, S, ...)
- LEAVING smiles: "none" for TYPE A; halogen symbol ("Br", "Cl", "I") or group SMILES for TYPE B
- ADD_FRAGMENT heavy_atoms: must equal the exact count of non-H atoms in that fragment SMILES
- PRODUCT_SMILES: main organic product ONLY -- NEVER include dot-separated byproducts (.Br, .HBr, .Cl, .HCl, etc.)
- PRODUCT_SMILES in step3 and Answer MUST be identical
- HEAVY_ATOM_DELTA: signed integer b - a (can be 0 for halide substitution where frag size = leaving size)
- RING_DELTA: signed integer d - c

═══════════════════════════════════════════════════════
EXAMPLES:

----------------------------------------------------------------
Example 1 -- N-acylation (TYPE A -- only H departs from amine):

Input:
  Source SMILES: c1ccc(N)cc1
  Indexed SMILES: [cH:1]1[cH:2][cH:3][c:4]([NH2:5])[cH:6][cH:7]1
  Instruction: Acetylate the aniline nitrogen by adding an acetyl group.

Step 1 [ANCHOR_IDENTIFICATION]: The amino nitrogen is atom 5 (map :5, element N). It has NH2; the N-H breaks but H is not heavy. LEAVING = "none". ANCHOR(idx=5, element=N).
  FORMAL: INDEXED_SMILES + INSTRUCTION --> ANCHOR(idx=5, element="N") + LEAVING(smiles="none")

Step 2 [FRAGMENT_IDENTIFICATION]: Acetyl group. SMILES: "C(=O)C". Heavy atoms: 3 (carbonyl C, O, methyl C).
  FORMAL: INSTRUCTION --> ADD_FRAGMENT(smiles="C(=O)C", heavy_atoms=3)

Step 3 [PRODUCT_CONSTRUCTION]: TYPE A. Bond N(idx=5) to acetyl carbonyl carbon. Product: CC(=O)Nc1ccccc1.
  FORMAL: SMILES + ANCHOR(idx=5) + LEAVING("none") + ADD_FRAGMENT(smiles="C(=O)C") --> PRODUCT_SMILES("CC(=O)Nc1ccccc1")

Step 4 [HEAVY_ATOM_VERIFICATION]: Source c1ccc(N)cc1 has 7 heavy atoms (max map :7). Product CC(=O)Nc1ccccc1 has 10. HEAVY_ATOM_DELTA = 10 - 7 = +3.
  FORMAL: SMILES[n_heavy=7] + PRODUCT_SMILES[n_heavy=10] --> HEAVY_ATOM_DELTA(+3)

Step 5 [RING_VERIFICATION]: 1 ring in source, 1 ring in product. RING_DELTA = 0.
  FORMAL: SMILES[n_rings=1] + PRODUCT_SMILES[n_rings=1] --> RING_DELTA(0)

Answer: CC(=O)Nc1ccccc1

----------------------------------------------------------------
Example 2 -- Suzuki C-C coupling (TYPE B -- bromide is the leaving group):

Input:
  Source SMILES: Brc1ccccc1
  Indexed SMILES: [Br:1][c:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1
  Instruction: Couple the aryl bromide with phenylboronic acid via Suzuki coupling.

Step 1 [ANCHOR_IDENTIFICATION]: The aryl carbon bearing Br is atom 2 (map :2, element C). Br (atom 1) is the leaving group in Suzuki coupling. LEAVING = "Br". ANCHOR(idx=2, element=C).
  FORMAL: INDEXED_SMILES + INSTRUCTION --> ANCHOR(idx=2, element="C") + LEAVING(smiles="Br")

Step 2 [FRAGMENT_IDENTIFICATION]: Incoming phenyl group from phenylboronic acid. SMILES: "c1ccccc1". Heavy atoms: 6.
  FORMAL: INSTRUCTION --> ADD_FRAGMENT(smiles="c1ccccc1", heavy_atoms=6)

Step 3 [PRODUCT_CONSTRUCTION]: TYPE B. Remove Br from atom 2, bond phenyl to atom 2. Product: c1ccc(-c2ccccc2)cc1 (biphenyl). Do not include HBr or Br byproduct.
  FORMAL: SMILES + ANCHOR(idx=2) + LEAVING("Br") + ADD_FRAGMENT(smiles="c1ccccc1") --> PRODUCT_SMILES("c1ccc(-c2ccccc2)cc1")

Step 4 [HEAVY_ATOM_VERIFICATION]: Source Brc1ccccc1 has 7 heavy atoms (max map :7). Product c1ccc(-c2ccccc2)cc1 has 12. HEAVY_ATOM_DELTA = 12 - 7 = +5 (= 6 added - 1 Br removed).
  FORMAL: SMILES[n_heavy=7] + PRODUCT_SMILES[n_heavy=12] --> HEAVY_ATOM_DELTA(+5)

Step 5 [RING_VERIFICATION]: Source has 1 ring. Product has 2 rings. RING_DELTA = +1.
  FORMAL: SMILES[n_rings=1] + PRODUCT_SMILES[n_rings=2] --> RING_DELTA(+1)

Answer: c1ccc(-c2ccccc2)cc1

----------------------------------------------------------------
Example 3 -- O-acylation / esterification (TYPE A -- only H departs from hydroxyl):

Input:
  Source SMILES: CCO
  Indexed SMILES: [CH3:1][CH2:2][OH:3]
  Instruction: Acylate the primary alcohol with a propionyl group.

Step 1 [ANCHOR_IDENTIFICATION]: The hydroxyl oxygen is atom 3 (map :3, element O). The O-H bond breaks; H is not heavy. LEAVING = "none". ANCHOR(idx=3, element=O).
  FORMAL: INDEXED_SMILES + INSTRUCTION --> ANCHOR(idx=3, element="O") + LEAVING(smiles="none")

Step 2 [FRAGMENT_IDENTIFICATION]: Propionyl group (CH3CH2C=O). SMILES: "CCC(=O)". Heavy atoms: 4 (C, C, C, O).
  FORMAL: INSTRUCTION --> ADD_FRAGMENT(smiles="CCC(=O)", heavy_atoms=4)

Step 3 [PRODUCT_CONSTRUCTION]: TYPE A. Bond O(idx=3) to the carbonyl carbon of propionyl. Product: CCC(=O)OCC (ethyl propanoate).
  FORMAL: SMILES + ANCHOR(idx=3) + LEAVING("none") + ADD_FRAGMENT(smiles="CCC(=O)") --> PRODUCT_SMILES("CCC(=O)OCC")

Step 4 [HEAVY_ATOM_VERIFICATION]: Source CCO has 3 heavy atoms (max map :3). Product CCC(=O)OCC has 7. HEAVY_ATOM_DELTA = 7 - 3 = +4.
  FORMAL: SMILES[n_heavy=3] + PRODUCT_SMILES[n_heavy=7] --> HEAVY_ATOM_DELTA(+4)

Step 5 [RING_VERIFICATION]: 0 rings in source, 0 rings in product. RING_DELTA = 0.
  FORMAL: SMILES[n_rings=0] + PRODUCT_SMILES[n_rings=0] --> RING_DELTA(0)

Answer: CCC(=O)OCC

═══════════════════════════════════════════════════════
CRITICAL OUTPUT REQUIREMENTS:

1. ALWAYS output ALL 5 steps + Answer. Never end without the complete chain.
2. Never refuse or skip -- always map the instruction to TYPE A or TYPE B and fill in every step.
3. PRODUCT_SMILES is the main organic product ONLY. NEVER include dot-separated byproducts (.Br, .HBr, .Cl, .HCl, .SO2, etc.).
4. For coupling reactions (Suzuki, Buchwald, Heck, Stille, C-N coupling): LEAVING = the halogen on the ANCHOR carbon.
5. Source heavy-atom count: the maximum ":n" in the Indexed SMILES equals the total heavy-atom count.
6. Each FORMAL line must be on ONE line. Do not split any step across multiple lines.
"""

USER_TEMPLATE = """Source SMILES: {src_smiles}
Indexed SMILES: {indexed_smiles}
Instruction: {instruction}
Ground Truth Product SMILES: {gt_smiles}

IMPORTANT: The ground truth product SMILES is provided for verification purposes only.
You must still generate the complete reasoning chain as if you were solving this problem from scratch.
Do NOT simply state the ground truth -- you must work through every step of the reasoning template and arrive at the answer naturally.

Generate the complete reasoning chain in the unified step format above to produce the edited molecule."""
