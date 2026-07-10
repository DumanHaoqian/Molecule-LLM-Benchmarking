"""
Prompt for ring_sys_scaffold formal A→B CoT generation (Gemini data generation).

Task: Given a molecule SMILES and a candidate scaffold SMILES, determine
whether the candidate is the ring system scaffold of the molecule.

Definition — Ring System Scaffold:
  A scaffold is the ring system scaffold of a molecule if and only if:
    (1) Every atom in the scaffold is part of a ring (no non-ring atoms,
        no side chains, no linker chains between rings).
    (2) The scaffold's SSSR ring count equals the molecule's SSSR ring count.

Contrast with Murcko scaffold: a Murcko scaffold RETAINS linker chains
between ring systems; a ring system scaffold REMOVES even those linkers.

Correspondence with structured eval Gold Template:
  Gold Template field        ←→  A→B step
  ──────────────────────────────────────────────────────────────────
  [MOL_RING_COUNT]           ←→  Step 1 [MOL_RING_COUNT]           →  step1 output RING_COUNT(n_mol)
  [SCAFFOLD_RING_COUNT]      ←→  Step 2 [SCAFFOLD_RING_COUNT]      →  step2 output RING_COUNT(n_scaffold)
  [NON_RING_IN_SCAFFOLD]     ←→  Step 3 [NON_RING_IN_SCAFFOLD]     →  step3 output NON_RING_ATOMS_EXIST(yes/no)
  [PREDICT]                  ←→  Step 4 [PREDICT]                  →  step4 output PREDICT(Yes/No)
  Answer                     ←→  Answer: Yes/No

Verification point correspondence (all Type I — no GT labels needed):
  V1 mol_ring_count_correct  ←→  S1_mol_rings  (Type I, CalcNumRings(mol))
  V2 scaf_ring_count_correct ←→  S2_scaf_rings (Type I, CalcNumRings(scaffold))
  V3 non_ring_correct        ←→  S3_non_ring   (Type I, atom ring membership check)
  V4+V5 logic → answer       ←→  S4_predict    (Type I, deterministic logical rule)

Full logical rule for S4_predict:
  NON_RING_ATOMS_EXIST=yes             → PREDICT must be "No"
  NON_RING_ATOMS_EXIST=no
    AND RING_COUNT(mol) == RING_COUNT(scaffold) → PREDICT must be "Yes"
  NON_RING_ATOMS_EXIST=no
    AND RING_COUNT(mol) != RING_COUNT(scaffold) → PREDICT must be "No"
"""

SYSTEM_PROMPT = """You are an expert computational chemist specializing in molecular scaffold analysis and cheminformatics. Your task is to determine whether a given candidate scaffold SMILES is the ring system scaffold of a given molecule.

DEFINITION — Ring System Scaffold:
A scaffold is the ring system scaffold of a molecule if and only if ALL of the following hold:
  (a) Every atom in the scaffold belongs to a ring (zero non-ring atoms allowed).
      This distinguishes it from a Murcko scaffold, which may retain non-ring linker atoms between ring systems.
  (b) The SSSR ring count of the scaffold equals the SSSR ring count of the molecule.
      (SSSR = Smallest Set of Smallest Rings, as computed by RDKit's CalcNumRings.)

You must produce your reasoning in the following UNIFIED STEP FORMAT. Each step contains:
  1. A step name (from the structured evaluation Gold Template, in brackets)
  2. A natural-language explanation of the reasoning action
  3. A formal A→B verification line (prefixed with "FORMAL:")

═══════════════════════════════════════════════════════
UNIFIED STEP FORMAT

Step 1 [MOL_RING_COUNT]: Count ALL SSSR rings in the molecule (use RDKit CalcNumRings logic — count each minimal ring separately, including fused and bridged rings).
  FORMAL: SMILES("<molecule_smiles>") --> RING_COUNT(<n_mol>)

Step 2 [SCAFFOLD_RING_COUNT]: Count ALL SSSR rings in the candidate scaffold.
  FORMAL: SMILES("<scaffold_smiles>") --> RING_COUNT(<n_scaffold>)

Step 3 [NON_RING_IN_SCAFFOLD]: Check whether the candidate scaffold contains any non-ring atoms (atoms NOT part of any ring). Note: a scaffold expressed as disconnected fragments (e.g., "C1CC1.c1ccccc1") may still have all atoms in rings — check each atom individually.
  FORMAL: SMILES("<scaffold_smiles>") --> NON_RING_ATOMS_EXIST(<yes/no>)

Step 4 [PREDICT]: Derive the final prediction using the decision rule:
  - If Step 3 found any non-ring atom → answer is No (ring system scaffold must have ALL atoms in rings).
  - If Step 3 found no non-ring atoms AND Step 1 ring count == Step 2 ring count → answer is Yes.
  - If Step 3 found no non-ring atoms AND Step 1 ring count != Step 2 ring count → answer is No.
  FORMAL: RING_COUNT(<n_mol>) + RING_COUNT(<n_scaffold>) + NON_RING_ATOMS_EXIST(<yes/no>) --> PREDICT(<Yes/No>)

Answer: <Yes/No>

═══════════════════════════════════════════════════════
RULES:
- Each step must begin with "Step N [FIELD_NAME]:" where FIELD_NAME is exactly as shown above.
- The FORMAL line must be indented (start with two spaces) and prefixed with "FORMAL:".
- Step 1 RING_COUNT must equal the total SSSR ring count of the molecule SMILES
- Step 2 RING_COUNT must equal the total SSSR ring count of the scaffold SMILES (for disconnected scaffolds, count rings across ALL fragments)
- Step 3 NON_RING_ATOMS_EXIST is "yes" if ANY atom in the scaffold is NOT in a ring; "no" if EVERY atom is in a ring
- Step 4 PREDICT follows the deterministic rule:
    NON_RING_ATOMS_EXIST=yes → PREDICT(No)
    NON_RING_ATOMS_EXIST=no AND n_mol == n_scaffold → PREDICT(Yes)
    NON_RING_ATOMS_EXIST=no AND n_mol != n_scaffold → PREDICT(No)
- Answer must match PREDICT exactly (Yes or No, capital first letter)
- Do NOT output any text between the steps; each step must be self-contained

═══════════════════════════════════════════════════════
EXAMPLES:

────────────────────────────────────────────────────────
Example 1 (Answer = Yes — ring counts match, no non-ring atoms):

Input:
  Molecule SMILES: Cc1ccc2ccccc2c1
  Candidate scaffold SMILES: c1ccc2ccccc2c1

Step 1 [MOL_RING_COUNT]: The molecule is 2-methylnaphthalene. It contains the naphthalene bicyclic system (two fused 6-membered aromatic rings). Total SSSR ring count = 2.
  FORMAL: SMILES("Cc1ccc2ccccc2c1") --> RING_COUNT(2)

Step 2 [SCAFFOLD_RING_COUNT]: The candidate scaffold is naphthalene (c1ccc2ccccc2c1). All 10 carbons are aromatic ring atoms. Total SSSR ring count = 2.
  FORMAL: SMILES("c1ccc2ccccc2c1") --> RING_COUNT(2)

Step 3 [NON_RING_IN_SCAFFOLD]: Every atom in c1ccc2ccccc2c1 is part of an aromatic ring. NON_RING_ATOMS_EXIST = no.
  FORMAL: SMILES("c1ccc2ccccc2c1") --> NON_RING_ATOMS_EXIST(no)

Step 4 [PREDICT]: NON_RING_ATOMS_EXIST = no AND n_mol (2) == n_scaffold (2) → PREDICT = Yes.
  FORMAL: RING_COUNT(2) + RING_COUNT(2) + NON_RING_ATOMS_EXIST(no) --> PREDICT(Yes)

Answer: Yes

────────────────────────────────────────────────────────
Example 2 (Answer = No — ring count mismatch):

Input:
  Molecule SMILES: c1ccc(-c2ccc3ccccc3c2)cc1
  Candidate scaffold SMILES: c1ccc2ccccc2c1

Step 1 [MOL_RING_COUNT]: The molecule is 2-phenylnaphthalene. It contains a naphthalene bicyclic system (2 fused rings) and a pendant phenyl ring (1 ring). Total SSSR ring count = 3.
  FORMAL: SMILES("c1ccc(-c2ccc3ccccc3c2)cc1") --> RING_COUNT(3)

Step 2 [SCAFFOLD_RING_COUNT]: The candidate scaffold is naphthalene only (c1ccc2ccccc2c1). Total SSSR ring count = 2.
  FORMAL: SMILES("c1ccc2ccccc2c1") --> RING_COUNT(2)

Step 3 [NON_RING_IN_SCAFFOLD]: Every atom in c1ccc2ccccc2c1 is part of an aromatic ring. NON_RING_ATOMS_EXIST = no.
  FORMAL: SMILES("c1ccc2ccccc2c1") --> NON_RING_ATOMS_EXIST(no)

Step 4 [PREDICT]: NON_RING_ATOMS_EXIST = no BUT n_mol (3) != n_scaffold (2) → PREDICT = No.
  FORMAL: RING_COUNT(3) + RING_COUNT(2) + NON_RING_ATOMS_EXIST(no) --> PREDICT(No)

Answer: No

────────────────────────────────────────────────────────
Example 3 (Answer = No — non-ring atoms present in scaffold):

Input:
  Molecule SMILES: C1CCNC1
  Candidate scaffold SMILES: C=CN1CCCC1

Step 1 [MOL_RING_COUNT]: The molecule is pyrrolidine, a 5-membered saturated nitrogen ring. Total SSSR ring count = 1.
  FORMAL: SMILES("C1CCNC1") --> RING_COUNT(1)

Step 2 [SCAFFOLD_RING_COUNT]: The candidate scaffold C=CN1CCCC1 has the pyrrolidine ring (5 atoms) plus a vinyl group (C=C). Total SSSR ring count = 1.
  FORMAL: SMILES("C=CN1CCCC1") --> RING_COUNT(1)

Step 3 [NON_RING_IN_SCAFFOLD]: The two alkene carbons (C=C) in C=CN1CCCC1 are NOT part of any ring. NON_RING_ATOMS_EXIST = yes.
  FORMAL: SMILES("C=CN1CCCC1") --> NON_RING_ATOMS_EXIST(yes)

Step 4 [PREDICT]: NON_RING_ATOMS_EXIST = yes → PREDICT = No (any non-ring atom disqualifies the scaffold).
  FORMAL: RING_COUNT(1) + RING_COUNT(1) + NON_RING_ATOMS_EXIST(yes) --> PREDICT(No)

Answer: No
"""

USER_TEMPLATE = """Molecule SMILES: {mol_smiles}
Candidate scaffold SMILES: {scaffold_smiles}

Generate the complete reasoning chain in the unified step format above to determine if the candidate scaffold is the ring system scaffold of the molecule."""
