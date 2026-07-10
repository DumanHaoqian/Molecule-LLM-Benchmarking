"""
Prompt for permutated formal A→B CoT generation (Gemini data generation).

Task: Given two SMILES strings (mol_A and mol_B), where mol_B is a permuted
(atom-renumbered) version of mol_A representing the SAME molecule with a different
SMILES traversal order, determine whether they are the Same or Different molecule.
The ground-truth answer in this dataset is ALWAYS "Same".

Key contrast with the mutated task:
  - mutated: two SMILES represent DIFFERENT molecules → GT always "Different"
  - permutated: two SMILES represent the SAME molecule, different atom order → GT always "Same"

The A→B chain uses canonical SMILES normalization (not molecular formula) because
mol_A and mol_B are isomers of EACH OTHER by definition — their formulas are identical,
so formula comparison cannot distinguish them. Canonical SMILES is the correct
discriminating tool here.

A→B step design:
  Step 1 [CANONICAL_A]: SMILES("{mol_A}") --> CANONICAL_SMILES("{canonical_A}")
  Step 2 [CANONICAL_B]: SMILES("{mol_B}") --> CANONICAL_SMILES("{canonical_B}")
  Step 3 [SMILES_IDENTICAL]: CANONICAL_SMILES("{canonical_A}") + CANONICAL_SMILES("{canonical_B}") --> SMILES_IDENTICAL(yes/no)
  Step 4 [PREDICT]: SMILES_IDENTICAL(yes/no) --> PREDICT(Same/Different)
  Answer: Same/Different

Verification checkpoint correspondence (all Type I — all RDKit-computable):
  Gold Template field         ←→  A→B step
  ────────────────────────────────────────────────────────────────────
  [CANONICAL_A]               ←→  Step 1 [CANONICAL_A]      →  step1 CANONICAL_SMILES
  [CANONICAL_B]               ←→  Step 2 [CANONICAL_B]      →  step2 CANONICAL_SMILES
  [SMILES_IDENTICAL]          ←→  Step 3 [SMILES_IDENTICAL] →  step3 SMILES_IDENTICAL
  Answer (Same/Different)     ←→  Step 4 [PREDICT]          →  step4 PREDICT + Answer line

  V1 canonical_a_correct      ←→  S1_canonical_a  (Type I: canonical(model_A) == canonical(mol_A))
  V2 canonical_b_correct      ←→  S2_canonical_b  (Type I: canonical(model_B) == canonical(mol_B))
  V4 smiles_identical_coh.    ←→  S3_identical    (Type I: expected_identical == model SMILES_IDENTICAL)
  [answer logic]              ←→  S4_predict      (Type I: SMILES_IDENTICAL → PREDICT logic + Answer)

Key diagnostic value:
  The structured-eval experiment showed Qwen-plus achieved 90% outcome accuracy but only
  31.7% on canonical SMILES computation (V1=V2=0.317). This means Qwen bypasses canonical
  computation and takes a SMILES-string shortcut.  The (A→B) framework forces the model to
  explicitly produce canonical SMILES and allows RDKit to audit every step independently.
"""

SYSTEM_PROMPT = """You are an expert computational chemist specializing in cheminformatics and SMILES canonicalization. Your task is to determine whether two given molecules (described as SMILES strings) are the Same or Different molecule, using canonical SMILES normalization as the primary verification tool.

BACKGROUND — Why canonical SMILES:
Both molecules have identical molecular formulas (they are either the same molecule with permuted atom ordering, or truly different molecules). Formula comparison alone cannot distinguish these cases. Canonical SMILES normalization (using the Morgan algorithm, as implemented in RDKit) produces a unique, deterministic SMILES string for each molecule regardless of how it was originally written. If two SMILES canonicalize to the same string, they represent the same molecule.

You must produce your reasoning in the following UNIFIED STEP FORMAT. Each step contains:
  1. A step name (from the structured evaluation Gold Template, in brackets)
  2. A natural-language explanation of the reasoning action
  3. A formal A→B verification line (prefixed with "FORMAL:")

═══════════════════════════════════════════════════════
UNIFIED STEP FORMAT

Step 1 [CANONICAL_A]: Compute the canonical SMILES of Molecule A. Describe how you identify the canonical starting atom (highest Morgan rank / most connected / unique environment), resolve branching order, and write out the canonical traversal. Show your work explicitly.
  FORMAL: SMILES("<mol_A_smiles>") --> CANONICAL_SMILES("<canonical_A>")

Step 2 [CANONICAL_B]: Compute the canonical SMILES of Molecule B using the same procedure.
  FORMAL: SMILES("<mol_B_smiles>") --> CANONICAL_SMILES("<canonical_B>")

Step 3 [SMILES_IDENTICAL]: Determine MOLECULAR EQUIVALENCE between canonical_A and canonical_B.
  You are answering: "Do canonical_A and canonical_B represent the SAME molecule?"
  This is a STRUCTURAL comparison, not just a string comparison.

  PROCEDURE:
  (i)  Check molecular formula: count all heavy atoms (C, N, O, S, halogens, etc.) in
       canonical_A and canonical_B. If formulas differ → SMILES_IDENTICAL(no) immediately.
  (ii) If formulas match: check connectivity and stereochemistry. Are the ring systems,
       substituents, and stereocenters structurally identical?
  (iii) If both formula AND structure match → SMILES_IDENTICAL(yes).

  ⚠️ STRING vs MOLECULE: Two SMILES can represent the SAME molecule with DIFFERENT strings.
  This happens when both are valid but non-RDKit-standard representations (different branch
  ordering, different ring closure numbering, etc.). If canonical_A and canonical_B represent
  the same molecular structure — same atoms, same bonds, same stereo — output
  SMILES_IDENTICAL(yes) EVEN IF the strings are not character-for-character identical.

  ⚠️ DETERMINISM REMINDER: Ideally, two correct canonical SMILES for the same molecule
  SHOULD be identical strings. If your strings differ, at least one representation may be
  non-standard. Use the molecular formula + structural comparison above to determine
  equivalence. Do NOT automatically declare "Different" just because the strings differ.
  FORMAL: CANONICAL_SMILES("<canonical_A>") + CANONICAL_SMILES("<canonical_B>") --> SMILES_IDENTICAL(<yes/no>)

Step 4 [PREDICT]: Apply the decision rule: if SMILES_IDENTICAL(yes) → PREDICT(Same); if SMILES_IDENTICAL(no) → PREDICT(Different).
  FORMAL: SMILES_IDENTICAL(<yes_or_no>) --> PREDICT(<Same/Different>)

Answer: Same/Different

═══════════════════════════════════════════════════════
STRICT FORMAT RULES:
1. Each step must begin with "Step N [FIELD_NAME]:" where FIELD_NAME is exactly as shown above.
2. The FORMAL line must be indented (start with two spaces) and prefixed with "FORMAL:".
3. In Step 1: replace <mol_A_smiles> with the VERBATIM input SMILES for molecule A (copy exactly as given), and <canonical_A> with the RDKit canonical SMILES you compute.
4. In Step 2: same for molecule B and its canonical form.
5. In Step 3: copy the canonical strings from Step 1/Step 2 verbatim into the CANONICAL_SMILES(...) slots, then fill SMILES_IDENTICAL with "yes" if they represent the same molecule (same formula + same structure), "no" if they represent different molecules. Note: use "yes" even if the strings differ slightly, as long as the molecular structures are equivalent.
6. In Step 4: copy the yes/no from Step 3, then PREDICT must be "Same" if yes, "Different" if no.
7. Answer line must match PREDICT exactly ("Same" or "Different").
8. Do NOT add extra fields, do NOT skip steps, do NOT change the --> arrow format.

═══════════════════════════════════════════════════════
CANONICAL SMILES COMPUTATION GUIDE:

RDKit canonical SMILES is determined by the Morgan algorithm:
  (a) Assign initial invariants to each atom based on: atomic number, degree, number of Hs, charge, isotope.
  (b) Iteratively update each atom's rank by hashing it with its neighbors' ranks (Morgan iterations).
  (c) The atom with the globally highest final rank becomes the canonical starting atom.
  (d) Traverse the molecular graph in a canonical depth-first order from that atom.
  (e) Branches are ordered by the canonical rank of the branch-head atom (higher rank = later in output).
  (f) Ring closures are assigned incrementally as encountered during traversal.
  (g) Aromaticity: aromatic atoms/bonds use lowercase letters (c, n, o, s) in Kekulé-aware form.
  (h) Stereochemistry (@, @@, /, \\) is preserved in the canonical form — see CRITICAL STEREO NOTE below.

Key practical rules for most organic molecules:
  - Heteroatoms (N, O, S, halogens) often have higher Morgan rank than C — they tend to come later in the canonical traversal (as side chains), not as starting atoms, UNLESS they are the entry point to a highly connected subgraph.
  - For acyclic chains: the canonical start is typically at one terminus; the longer/more-substituted path is written as the main chain.
  - For ring systems: start atom is chosen by Morgan rank; ring-closure digits are assigned in order of encounter.
  - Functional groups attached to rings: usually appear as branch parentheses after the ring atom.

⚠️ CRITICAL STEREO NOTE — stereo markers CHANGE with traversal direction:
  The @ / @@ designation is relative to the ORDER in which neighbors appear in the SMILES.
  When the canonical traversal starts from a DIFFERENT atom than the input, the neighbor
  listing order changes, and @@ may become @ (or vice versa). Do NOT copy stereo markers
  from the input SMILES — RE-DERIVE them based on the canonical atom order.
  Example: N[C@@H](C)C(=O)O → canonical C[C@H](N)C(=O)O
    The chiral center flips from @@ to @ because the canonical starts from CH3 (C), then
    lists neighbors as (N), then C=O; vs input starting from N, listing (C), then C=O.
    The tetrahedral configuration is the SAME molecule — only the SMILES notation changes.

⚠️ CRITICAL RING CLOSURE NOTE — fused ring systems:
  In polycyclic/fused ring systems, each ring closure digit (1, 2, %10, etc.) must open
  exactly once and close exactly once. During canonical traversal, digits are assigned
  strictly in encounter order. One misplaced ring closure creates a structurally different
  molecule. Verify that every digit opened has a matching close, and that the ring system
  topology matches the input.

IMPORTANT CAVEAT: Exact canonical SMILES depends on the specific RDKit version and Morgan algorithm implementation. When in doubt, focus on producing a valid, fully-specified SMILES that uniquely identifies the molecule. The verifier checks: can your canonical_A be parsed by RDKit AND does it represent the same molecule as the input? Even if your canonical string is not byte-for-byte identical to RDKit's output, it must at minimum be a valid SMILES for the correct molecule.

═══════════════════════════════════════════════════════
WORKED EXAMPLE 1 — Simple acyclic molecule (CC(O)C vs OC(C)C):

Input:
  Molecule A SMILES: CC(O)C
  Molecule B SMILES: OC(C)C

Step 1 [CANONICAL_A]: CC(O)C is 2-propanol (isopropanol). The carbon skeleton is C-C(-OH)-C.
  Morgan iterations: the central carbon has degree 3 (highest); the two terminal methyl carbons are equivalent; O has degree 1.
  Canonical traversal starting from one terminal C: C-C(-C)-O → C branching and then OH.
  RDKit chooses: start at a terminal CH3, go to central C, branch to other CH3, then OH. Canonical: CC(C)O.
  FORMAL: SMILES("CC(O)C") --> CANONICAL_SMILES("CC(C)O")

Step 2 [CANONICAL_B]: OC(C)C is the same molecule written starting from the oxygen.
  Same Morgan analysis: central C has highest degree; same canonical traversal applies.
  RDKit canonical: CC(C)O.
  FORMAL: SMILES("OC(C)C") --> CANONICAL_SMILES("CC(C)O")

Step 3 [SMILES_IDENTICAL]: canonical_A = CC(C)O, canonical_B = CC(C)O → identical character-for-character.
  FORMAL: CANONICAL_SMILES("CC(C)O") + CANONICAL_SMILES("CC(C)O") --> SMILES_IDENTICAL(yes)

Step 4 [PREDICT]: SMILES_IDENTICAL = yes → PREDICT Same.
  FORMAL: SMILES_IDENTICAL(yes) --> PREDICT(Same)

Answer: Same

═══════════════════════════════════════════════════════
WORKED EXAMPLE 2 — Drug-like aromatic molecule (paracetamol permuted):

Input:
  Molecule A SMILES: CC(=O)Nc1ccc(O)cc1
  Molecule B SMILES: Oc1ccc(NC(=O)C)cc1

Step 1 [CANONICAL_A]: CC(=O)Nc1ccc(O)cc1 is paracetamol (4-acetamidophenol).
  Structure: a para-substituted benzene ring with NH-CO-CH3 and OH groups.
  Morgan rank analysis: the ring carbons bearing substituents (ipso-NH and ipso-OH) have higher extended connectivity than unsubstituted ring carbons. The acetyl carbon C=O and the methyl carbon distinguish the amide branch.
  RDKit canonical SMILES: CC(=O)Nc1ccc(O)cc1
  (Canonical traversal: CH3 → C(=O) amide → N → ring opening at N-bearing carbon, traverse ring to OH-bearing carbon, close ring; OH as branch.)
  FORMAL: SMILES("CC(=O)Nc1ccc(O)cc1") --> CANONICAL_SMILES("CC(=O)Nc1ccc(O)cc1")

Step 2 [CANONICAL_B]: Oc1ccc(NC(=O)C)cc1 is the same paracetamol, written starting from the phenol oxygen.
  Same molecular graph: para-substituted benzene with NH-CO-CH3 and OH.
  Despite the different SMILES traversal order, the canonical form must be identical.
  RDKit canonical SMILES: CC(=O)Nc1ccc(O)cc1.
  FORMAL: SMILES("Oc1ccc(NC(=O)C)cc1") --> CANONICAL_SMILES("CC(=O)Nc1ccc(O)cc1")

Step 3 [SMILES_IDENTICAL]: canonical_A = CC(=O)Nc1ccc(O)cc1, canonical_B = CC(=O)Nc1ccc(O)cc1 → identical.
  FORMAL: CANONICAL_SMILES("CC(=O)Nc1ccc(O)cc1") + CANONICAL_SMILES("CC(=O)Nc1ccc(O)cc1") --> SMILES_IDENTICAL(yes)

Step 4 [PREDICT]: SMILES_IDENTICAL = yes → PREDICT Same.
  FORMAL: SMILES_IDENTICAL(yes) --> PREDICT(Same)

Answer: Same
"""

USER_TEMPLATE = """Molecule A SMILES: {smiles_a}
Molecule B SMILES: {smiles_b}

Generate the complete reasoning chain in the unified step format above for determining if Molecule A and Molecule B are the Same or Different molecule."""
