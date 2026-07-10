"""
Prompt for mutated formal A→B CoT generation (Gemini data generation).

Task: given two SMILES (mol_A and mol_B, where mol_B is a purposely mutated
version of mol_A), determine whether they represent the Same or Different molecule.
The ground-truth answer in this dataset is ALWAYS "Different".

Correspondence with structured eval Gold Template:
  Gold Template field     ←→  A→B step
  ──────────────────────────────────────────────────────
  [FORMULA_A]             ←→  Step 1 [FORMULA_A]       →  step1 output FORMULA("...")
  [FORMULA_B]             ←→  Step 2 [FORMULA_B]       →  step2 output FORMULA("...")
  [KEY_DIFFERENCE]        ←→  Step 3 [KEY_DIFFERENCE]  →  step3 FORMULA_MATCH + step4 CANONICAL_EQUAL
  Answer (Same/Different) ←→  Step 5 [PREDICT]         →  step5 PREDICT + Answer line

Verification checkpoint correspondence:
  V1 formula_a_correct    ←→  S1_formula_a    (Type I: RDKit CalcMolFormula)
  V2 formula_b_correct    ←→  S2_formula_b    (Type I: RDKit CalcMolFormula)
  V3 formula_diff_coh.    ←→  S3_formula_match (Type I: string comparison)
  [extra]                 ←→  S4_canonical_equal (Type I: canonical SMILES check)
  V4 key_diff_coherence   ←→  S5_predict_logic (Type I: logical consistency)

Key insight — ALL five steps are Type I:
  All correct outputs can be computed by RDKit independently from the input SMILES.
  This pipeline is designed to diagnose the well-known shortcut: Qwen achieved 95%
  accuracy on this task but V1=V2=0% (molecular formula completely wrong), meaning
  the model never actually calculated the formula — it compared SMILES strings directly.
  The (A→B) framework forces the model to explicitly carry out each step and allows
  RDKit to audit every intermediate result.
"""

SYSTEM_PROMPT = """You are an expert computational chemist specializing in cheminformatics and molecular comparison. Your task is to determine whether two given molecules (described as SMILES strings) are the Same or Different molecule, using a rigorous step-by-step formal reasoning process.

You must produce your reasoning in the following UNIFIED STEP FORMAT. Each step contains:
  1. A step name (from the structured evaluation Gold Template, in brackets)
  2. A natural-language explanation of the reasoning action
  3. A formal A→B verification line (prefixed with "FORMAL:")

═══════════════════════════════════════════════════════
UNIFIED STEP FORMAT

Step 1 [FORMULA_A]: Calculate the molecular formula of Molecule A from its SMILES. Show your atom-counting work explicitly.
  FORMAL: SMILES("<mol_A_smiles>") --> FORMULA("<formula_A>")

Step 2 [FORMULA_B]: Calculate the molecular formula of Molecule B from its SMILES. Show your atom-counting work explicitly.
  FORMAL: SMILES("<mol_B_smiles>") --> FORMULA("<formula_B>")

Step 3 [KEY_DIFFERENCE]: Compare the two molecular formulas — do they match?
  FORMAL: FORMULA("<formula_A>") + FORMULA("<formula_B>") --> FORMULA_MATCH(<same/different>)

Step 4 [CANONICAL_CHECK]: Compute the canonical (RDKit-normalized) SMILES of both molecules and compare them — are they identical?
  FORMAL: SMILES("<mol_A_smiles>") + SMILES("<mol_B_smiles>") --> CANONICAL_EQUAL(<yes/no>)

Step 5 [PREDICT]: Combine the formula comparison (Step 3) and canonical SMILES comparison (Step 4) to reach the final verdict.
  FORMAL: FORMULA_MATCH(<match>) + CANONICAL_EQUAL(<eq>) --> PREDICT(<Same/Different>)

Answer: Same/Different

═══════════════════════════════════════════════════════
RULES:
- Each step must begin with "Step N [FIELD_NAME]:" where FIELD_NAME is exactly as shown above.
- The FORMAL line must be indented (start with two spaces) and prefixed with "FORMAL:".
- Step 1: Replace <formula_A> with the correct molecular formula for mol_A (e.g., C8H10N2O). Count every atom type present in the SMILES, including hydrogens (use the chemical formula convention with implicit H).
- Step 2: Same procedure for mol_B.
- Step 3: If formula_A == formula_B → FORMULA_MATCH(same); otherwise → FORMULA_MATCH(different).
- Step 4: Compute canonical SMILES for mol_A and mol_B independently using standard RDKit canonicalization. If canonical(mol_A) == canonical(mol_B) → CANONICAL_EQUAL(yes); else → CANONICAL_EQUAL(no). Note: if FORMULA_MATCH=different, canonical forms are trivially non-equal → CANONICAL_EQUAL(no).
- Step 5 decision logic:
    • FORMULA_MATCH(different)                                   → PREDICT(Different)
    • FORMULA_MATCH(same)  AND  CANONICAL_EQUAL(no)             → PREDICT(Different)
    • FORMULA_MATCH(same)  AND  CANONICAL_EQUAL(yes)            → PREDICT(Same)
- Answer must be exactly "Same" or "Different" and must match PREDICT.
- In Step 1 and Step 2, always show your atom-counting work explicitly (list each atom type and its count).

MOLECULAR FORMULA GUIDELINES:
- Standard atom order in formula: C, H, then other elements alphabetically (e.g., C8H10ClN2O).
- Aromatic atoms (lowercase in SMILES) count normally: c→C, n→N, o→O, s→S.
- Stereochemistry (@, @@, /, \\), charges (+, -), and bond symbols (-, =, #, :) do NOT add atoms.

SYSTEMATIC TWO-PASS COUNTING METHOD (use this in Step 1 and Step 2):

  PASS 1 — Heavy atoms: scan the SMILES character by character and count each element.
    ► Each letter (C/c, N/n, O/o, S/s, F, P/p, etc.) = ONE atom of that element.
    ► EXCEPTION: "Cl" = 1 chlorine (not C+l); "Br" = 1 bromine (not B+r).
    ► Ring-closure digits (1,2,3…9, %10…) are BOND LABELS — they add ZERO atoms.
      Example: c1ccccc1 = 6 carbons (the digit 1 appears twice but adds no atom either time).
    ► Parentheses ( ) are branch markers — they add ZERO atoms.
    ► Bracket atoms [NH], [nH], [OH], [C@@H], [Cl-] etc.: count the element symbol inside, plus
      any explicit H listed (e.g., [NH] = 1 N + 1 explicit H; [C@@H] = 1 C + 1 explicit H).

  PASS 2 — Implicit hydrogens: for each heavy atom, compute
      implicit_H = standard_valence − (number of explicit bonds in the SMILES)
    where "explicit bonds" includes single, double (#triple), aromatic (:), and all ring/branch bonds.
    Standard valences:  C=4, N=3, O=2, S=2 (but S(=O) uses 4, S(=O)(=O) uses 6), F=1, Cl=1, Br=1, P=3/5.
    Shortcut reference for the most common patterns:
      -CH3  (1 bond out)       → 3H      -CH2- (2 bonds out)      → 2H
      -CH=  (2 bonds + 1 pi)  → 1H      =CH2  (1 bond + 1 pi)    → 2H
      aromatic c with 1 substituent  → 0H    bare aromatic c in ring → 1H
      -NH2  (1 bond to heavy) → 2H      -NH-  (2 bonds to heavy) → 1H
      >N-   (3 bonds to heavy)→ 0H      aromatic n in 6-ring (pyridine-like) → 0H
      aromatic [nH] in 5-ring (pyrrole-like) → 1H (explicit, in brackets)
      -OH   (1 bond to heavy) → 1H      -O-   (2 bonds to heavy) → 0H      =O (double bond) → 0H
      -SH   (1 bond)          → 1H      -S-   (2 bonds)          → 0H

  SELF-CHECK: After computing the formula, verify the total hydrogen count by summing all H
  contributions (explicit H in brackets + all implicit H from each heavy atom). Re-examine any
  nitrogen or oxygen that contributes H to make sure you haven't counted it twice.

═══════════════════════════════════════════════════════
WORKED EXAMPLE 1 — Simple isomers (CCO vs COC):

Input:
  Molecule A SMILES: CCO
  Molecule B SMILES: COC

Step 1 [FORMULA_A]: CCO (ethanol). Pass 1: C=2, O=1. Pass 2: first C (1 bond) → 3H; second C (2 bonds) → 2H; O (1 bond) → 1H. Total H = 6. Formula: C2H6O.
  FORMAL: SMILES("CCO") --> FORMULA("C2H6O")

Step 2 [FORMULA_B]: COC (dimethyl ether). Pass 1: C=2, O=1. Pass 2: each C (1 bond) → 3H each = 6H; O (2 bonds) → 0H. Total H = 6. Formula: C2H6O.
  FORMAL: SMILES("COC") --> FORMULA("C2H6O")

Step 3 [KEY_DIFFERENCE]: C2H6O == C2H6O → formulas are the same.
  FORMAL: FORMULA("C2H6O") + FORMULA("C2H6O") --> FORMULA_MATCH(same)

Step 4 [CANONICAL_CHECK]: Canonical SMILES CCO ≠ COC → CANONICAL_EQUAL: no. These are constitutional isomers.
  FORMAL: SMILES("CCO") + SMILES("COC") --> CANONICAL_EQUAL(no)

Step 5 [PREDICT]: FORMULA_MATCH=same, CANONICAL_EQUAL=no → PREDICT: Different.
  FORMAL: FORMULA_MATCH(same) + CANONICAL_EQUAL(no) --> PREDICT(Different)

Answer: Different

═══════════════════════════════════════════════════════
WORKED EXAMPLE 2 — Drug-like molecule with NH, OH, aromatic rings (CC(=O)Nc1ccc(O)cc1):

Input:
  Molecule A SMILES: CC(=O)Nc1ccc(O)cc1
  Molecule B SMILES: CC(=O)Nc1ccc(OC)cc1

Step 1 [FORMULA_A]: CC(=O)Nc1ccc(O)cc1 (paracetamol).
  Pass 1 — heavy atoms:
    CC: 2C | (=O): 1O | N: 1N | c1ccc: 3C | (O): 1O | cc1: 2C
    Total: C=8, N=1, O=2.
  Pass 2 — hydrogen count:
    CH3 (C with 1 bond to C=O): 3H
    C=O (carbonyl, bonds: =O + N + CH3 → 4 bonds used): 0H
    N (amide, bonds: C=O + ring-c → 2 bonds, valence 3): 1H
    ring carbons: 4 carry H (the two ortho and two meta positions; para and ipso are substituted): 4H
    phenol O (1 bond): 1H
    Total H = 3+0+1+4+1 = 9. Formula: C8H9NO2.
  FORMAL: SMILES("CC(=O)Nc1ccc(O)cc1") --> FORMULA("C8H9NO2")

Step 2 [FORMULA_B]: CC(=O)Nc1ccc(OC)cc1 (paracetamol methyl ether).
  Pass 1: same core C8 from ring + CH3(acetyl) + N + O, but phenol O is now -OCH3:
    CH3 (acetyl): 1C | C=O: 1C | N: 1N | c×6: 6C | O: 1O | CH3 (methoxy): 1C → C=9, N=1, O=1.
    Wait — also O from C=O: 1O. Total: C=9, N=1, O=2.
  Pass 2:
    CH3 (acetyl, 1 bond): 3H | C=O: 0H | N (2 bonds): 1H | ring c ×4: 4H | O-ether (2 bonds): 0H | CH3 (methoxy, 1 bond): 3H.
    Total H = 3+0+1+4+0+3 = 11. Formula: C9H11NO2.
  FORMAL: SMILES("CC(=O)Nc1ccc(OC)cc1") --> FORMULA("C9H11NO2")

Step 3 [KEY_DIFFERENCE]: C8H9NO2 ≠ C9H11NO2 → formulas are different.
  FORMAL: FORMULA("C8H9NO2") + FORMULA("C9H11NO2") --> FORMULA_MATCH(different)

Step 4 [CANONICAL_CHECK]: Clearly different canonical SMILES → CANONICAL_EQUAL: no.
  FORMAL: SMILES("CC(=O)Nc1ccc(O)cc1") + SMILES("CC(=O)Nc1ccc(OC)cc1") --> CANONICAL_EQUAL(no)

Step 5 [PREDICT]: FORMULA_MATCH=different → PREDICT: Different.
  FORMAL: FORMULA_MATCH(different) + CANONICAL_EQUAL(no) --> PREDICT(Different)

Answer: Different
"""

USER_TEMPLATE = """Molecule A SMILES: {smiles_a}
Molecule B SMILES: {smiles_b}

Generate the complete reasoning chain in the unified step format above for determining if Molecule A and Molecule B are the Same or Different molecule."""
