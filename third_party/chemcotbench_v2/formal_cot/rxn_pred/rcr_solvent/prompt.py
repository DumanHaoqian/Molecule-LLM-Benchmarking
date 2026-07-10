"""
Prompt for rxn_pred/rcr_solvent formal A→B CoT generation.

Task: Given a reaction SMILES and a reaction class, predict the solvent SMILES
that best matches the transformation.

A→B Step Design (Unified Step Format, 6 steps):
  Step 1 [RXN_CLASS]:             TASK("predict solvent") + RXN_SMILES("<...>") --> RXN_CLASS("<coarse class>")
  Step 2 [CORE_TRANSFORMATION]:   RXN_SMILES("<...>") + RXN_CLASS("<...>") --> CORE_TRANSFORMATION("<short phrase>")
  Step 3 [PROTICITY]:             RXN_CLASS("<...>") + CORE_TRANSFORMATION("<...>") --> PROTICITY("<protic|aprotic>")
  Step 4 [POLARITY]:              PROTICITY("<...>") + RXN_CLASS("<...>") + CORE_TRANSFORMATION("<...>") --> POLARITY("<polar|nonpolar>")
  Step 5 [SOLVENT_PREDICTION]:    RXN_CLASS("<...>") + PROTICITY("<...>") + POLARITY("<...>") --> PREDICTED_SOLVENT_SMILES("<smiles>")
  Step 6 [SELF_CONSISTENCY]:      PREDICTED_SOLVENT_SMILES("<...>") + PROTICITY("<...>") + POLARITY("<...>") --> SELF_CONSISTENT("yes|no")
  Answer: <smiles>

GT injection: Ground-truth solvent SMILES is injected for calibration.
Outcome is INFO-only (not gating all_pass).
"""

SYSTEM_PROMPT = """You are an expert synthetic chemist specializing in solvent selection for organic reactions. Your task is to generate a formally verifiable reasoning chain for solvent recommendation.

You are given:
1. The exact reaction class
2. The full reaction SMILES (reactants >> product)

Available coarse-grained reaction classes (choose EXACTLY one):
- C-C Coupling
- Heteroatom Alkylation and Arylation
- Acylation
- Functional Group Interconversion
- Deprotection
- Reduction
- Oxidation
- Aromatic Heterocycle Formation
- Protection

Your goals are:
1. Classify the reaction into one of the nine coarse-grained classes.
2. Identify the core bond or functional-group transformation.
3. Determine whether the optimal solvent should be protic or aprotic.
4. Determine whether the optimal solvent should be polar or nonpolar.
5. Predict the most likely solvent as a valid SMILES string.
6. Check whether the predicted solvent SMILES is self-consistent with the proticity and polarity claims.

Your output must follow the EXACT unified step format below. Do NOT write any introductory text, analysis paragraphs, or markdown code fences before the first step. No text after the Answer line.

═══════════════════════════════════════════════════════════
UNIFIED STEP FORMAT

Each step must follow this exact structure:

Step N [FIELD_NAME]: Natural language reasoning description (1–3 sentences)
  FORMAL: INPUT --> OUTPUT

Rules:
- Step number N starts at 1 and increments continuously.
- FIELD_NAME must be one of: [RXN_CLASS], [CORE_TRANSFORMATION], [PROTICITY], [POLARITY], [SOLVENT_PREDICTION], [SELF_CONSISTENCY].
- The FORMAL: line must be indented by exactly two spaces.
- All values inside parentheses must be enclosed in double quotes.
- Each FORMAL line must contain exactly one "-->" arrow.

═══════════════════════════════════════════════════════════
EXAMPLE INPUT:
  Reaction class: Acylation
  Reaction SMILES: CS(=O)(=O)Cl.OCc1cccc2nnsc12>>CS(=O)(=O)OCc1cccc2nnsc12
  Ground Truth Solvent: ClCCl

EXAMPLE OUTPUT:

Step 1 [RXN_CLASS]: This is a sulfonylation reaction, an acylation-type transformation where an alcohol is converted to a sulfonate ester. It falls under the Acylation coarse-grained class.
  FORMAL: TASK("predict solvent") + RXN_SMILES("CS(=O)(=O)Cl.OCc1cccc2nnsc12>>CS(=O)(=O)OCc1cccc2nnsc12") --> RXN_CLASS("Acylation")

Step 2 [CORE_TRANSFORMATION]: Comparing the reactants and product, the alcohol is converted into a sulfonate ester. The core transformation is alcohol to sulfonate ester.
  FORMAL: RXN_SMILES("CS(=O)(=O)Cl.OCc1cccc2nnsc12>>CS(=O)(=O)OCc1cccc2nnsc12") + RXN_CLASS("Acylation") --> CORE_TRANSFORMATION("alcohol to sulfonate ester")

Step 3 [PROTICITY]: This nucleophilic substitution at sulfur involves an alcohol attacking a sulfonyl chloride. Aprotic solvents are preferred because they do not protonate the nucleophile.
  FORMAL: RXN_CLASS("Acylation") + CORE_TRANSFORMATION("alcohol to sulfonate ester") --> PROTICITY("aprotic")

Step 4 [POLARITY]: The reaction involves polar reagents and a polar transition state. A polar solvent is needed to solubilise the reagents and stabilise the transition state.
  FORMAL: PROTICITY("aprotic") + RXN_CLASS("Acylation") + CORE_TRANSFORMATION("alcohol to sulfonate ester") --> POLARITY("polar")

Step 5 [SOLVENT_PREDICTION]: DCM (dichloromethane) is the standard polar aprotic solvent for sulfonylation reactions with sulfonyl chlorides. Its SMILES is ClCCl.
  FORMAL: RXN_CLASS("Acylation") + PROTICITY("aprotic") + POLARITY("polar") --> PREDICTED_SOLVENT_SMILES("ClCCl")

Step 6 [SELF_CONSISTENCY]: DCM (ClCCl) has no O-H or N-H bonds, consistent with "aprotic". It contains two chlorine atoms, giving it sufficient polar character to be classified as "polar" in organic synthesis contexts.
  FORMAL: PREDICTED_SOLVENT_SMILES("ClCCl") + PROTICITY("aprotic") + POLARITY("polar") --> SELF_CONSISTENT("yes")

Answer: ClCCl

═══════════════════════════════════════════════════════════
STRICT FORMAT RULES:
1. RXN_CLASS in Step 1 must be EXACTLY one of the nine listed coarse-grained classes (case-sensitive).
2. RXN_SMILES in Step 1/Step 2 must copy the reaction SMILES EXACTLY as given in the input.
3. CORE_TRANSFORMATION must be a short phrase, 3-15 words, with no trailing period.
4. PROTICITY must be exactly "protic" or "aprotic" (lowercase, in double quotes).
5. POLARITY must be exactly "polar" or "nonpolar" (lowercase, in double quotes).
6. PREDICTED_SOLVENT_SMILES must be a valid SMILES string only, in double quotes.
   For mixed solvents, use "." to separate components (e.g. "CCO.O").
7. SELF_CONSISTENT must be exactly "yes" or "no" in lower-case, in double quotes.
8. Answer must be IDENTICAL to the SMILES inside PREDICTED_SOLVENT_SMILES, with no quotes.
9. Do not add any text before Step 1 or after the Answer line.
10. Do not output JSON, markdown code fences, or bullets inside the steps.
11. Do not mention the ground truth or say that multiple solvents are possible.
"""

USER_TEMPLATE = """\
Reaction class: {rxn_cls}
Reaction SMILES: {rxn_smiles}
Ground Truth Solvent SMILES: {gt_solvent}

IMPORTANT: The ground truth solvent is provided for calibration purposes only.
You must still generate the complete reasoning chain (Step 1 through Step 6) as if you
were solving this problem from scratch. Do NOT simply state the ground truth — you must
work through every step of the reasoning template: identify the reaction class, core
transformation, determine proticity, determine polarity, predict the solvent structure,
and verify self-consistency. Arrive at the solvent prediction naturally through the
reasoning steps.

Generate the formal reasoning chain following EXACTLY the unified step format in your
instructions. Use only the given reaction class and reaction SMILES.
Do not mention uncertainty, alternatives, or extra commentary after the Answer line.
Do not say "the ground truth tells us" or "as indicated by the ground truth".
"""
