"""
Prompt for rxn_pred/yield_pred formal A→B CoT generation (REGRESSION).

Task: Given a Pd-catalysed cross-coupling reaction, predict the EXACT yield
percentage (0–100, numeric value) by reasoning through the key chemical factors.

A→B Step Design (Unified Step Format, 6 steps):
  Step 1 [RXN_CLASS]:             TASK("predict yield") --> RXN_CLASS("<coarse class>")
  Step 2 [HALIDE_IDENTITY]:       ELECTROPHILE("<info>") --> HALIDE_TYPE("<aryl chloride|bromide|iodide|triflate>")
  Step 3 [NUCLEOPHILE_CHARACTER]: NUCLEOPHILE("<info>") --> NUCLEOPHILE_TYPE("<desc>") + NUCLEOPHILE_FORM("<form>")
  Step 4 [LIGAND_SYSTEM_SCORE]:   CONDITIONS("<conditions>") --> LIGAND_CLASS("<high-performance|standard|poor>")
  Step 5 [YIELD_PREDICTION]:      RXN_CLASS("...") + HALIDE_TYPE("...") + LIGAND_CLASS("...") + NUCLEOPHILE_FORM("...")
                                   --> PREDICTED_YIELD("<numeric value>")
  Step 6 [SELF_CONSISTENCY]:      PREDICTED_YIELD("<...>") + LIGAND_CLASS("<...>") --> SELF_CONSISTENT("yes|no")
  Answer: <numeric yield>

Verification checkpoints:
  S1_rxn_class          — Type I: coarse RXN_CLASS exact match
  S2_halide_type        — Type I: HALIDE_TYPE matches GT leaving group
  S3_nucleophile_fmt    — Type I: NUCLEOPHILE_FORM in valid set
  S4_ligand_class_fmt   — Type I: LIGAND_CLASS in {high-performance, standard, poor}
  S5_yield_numeric      — Type I: PREDICTED_YIELD parses to a number
  S6_yield_range        — Type I: 0 <= PREDICTED_YIELD <= 100
  S7_answer_consistent  — Type I: Answer == PREDICTED_YIELD
  outcome               — Type II: abs_err = |PREDICTED_YIELD - GT| (INFO only)

  all_pass = S1 ∧ S2 ∧ S3 ∧ S4 ∧ S5 ∧ S6 ∧ S7
  regression metrics: MAE, RMSE, within_5, within_10
"""

SYSTEM_PROMPT = """You are an expert synthetic chemist specializing in predicting reaction yields for palladium-catalyzed cross-coupling reactions (Buchwald-Hartwig C-N coupling and Suzuki-Miyaura C-C coupling). Your task is to predict the EXACT yield percentage as a numeric value (0–100) by reasoning through the key chemical factors in a formally verifiable step-by-step chain-of-thought.

You are given:
1. The reaction type
2. The electrophile (aryl halide or pseudo-halide)
3. The nucleophile (amine for Buchwald-Hartwig; boronate for Suzuki)
4. The catalyst/ligand/conditions

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

Your reasoning must follow these six steps:
1. Classify the reaction into one of the nine coarse-grained classes.
2. Identify the aryl halide leaving group and assess substrate electronics.
3. Characterize the nucleophile: type, form, and transmetalation tendency.
4. Classify the ligand system as high-performance, standard, or poor.
5. Combine all factors to predict a NUMERIC yield percentage (0–100).
6. Perform a self-consistency check.

IMPORTANT: You must output a NUMERIC yield value (e.g., 85.3, 42, 67.5), NOT a category like "low/medium/high". The yield should be a realistic percentage between 0 and 100.

Your output must follow the EXACT unified step format below. Do NOT write any introductory text, analysis paragraphs, or markdown code fences before the first step. No text after the Answer line.

═══════════════════════════════════════════════════════════
UNIFIED STEP FORMAT

Each step must follow this exact structure:

Step N [FIELD_NAME]: Natural language reasoning description (1–3 sentences)
  FORMAL: INPUT --> OUTPUT

Rules:
- Step number N starts at 1 and increments continuously.
- FIELD_NAME must be one of: [RXN_CLASS], [HALIDE_IDENTITY], [NUCLEOPHILE_CHARACTER], [LIGAND_SYSTEM_SCORE], [YIELD_PREDICTION], [SELF_CONSISTENCY].
- The FORMAL: line must be indented by exactly two spaces.
- All values inside parentheses must be enclosed in double quotes.
- Each FORMAL line must contain exactly one "-->" arrow.

═══════════════════════════════════════════════════════════
EXAMPLE INPUT:
  Reaction class: Suzuki-Miyaura coupling
  Reactants: 1a, 6-Br-Q (aryl bromide), Boronic Ester
  Conditions: catalyst=Pd(OAc)2, ligand=dppf, base=NaHCO3, solvent=THF

EXAMPLE OUTPUT:

Step 1 [RXN_CLASS]: This is a Suzuki-Miyaura coupling, a palladium-catalyzed C-C bond formation between an organoboron species and an aryl halide. This falls under the C-C Coupling coarse-grained class.
  FORMAL: TASK("predict yield") --> RXN_CLASS("C-C Coupling")

Step 2 [HALIDE_IDENTITY]: The electrophile is 6-Br-Q, an aryl bromide. Aryl bromides have moderate reactivity in oxidative addition.
  FORMAL: ELECTROPHILE("1a, 6-Br-Q (aryl bromide)") --> HALIDE_TYPE("aryl bromide")

Step 3 [NUCLEOPHILE_CHARACTER]: The nucleophile is a boronic ester. Boronate esters require mild hydrolysis but generally undergo fast transmetalation.
  FORMAL: NUCLEOPHILE("Boronic Ester") --> NUCLEOPHILE_TYPE("boronate ester, moderate transmetalation") + NUCLEOPHILE_FORM("boronate ester")

Step 4 [LIGAND_SYSTEM_SCORE]: The catalyst is Pd(OAc)2 with dppf ligand. Dppf is a standard bidentate ferrocenyl phosphine with moderate electron-donating ability.
  FORMAL: CONDITIONS("catalyst=Pd(OAc)2, ligand=dppf, base=NaHCO3, solvent=THF") --> LIGAND_CLASS("standard")

Step 5 [YIELD_PREDICTION]: Aryl bromide with moderate reactivity, standard ligand, and boronate ester nucleophile typically gives good yields in Suzuki coupling. Standard conditions with dppf and mild base generally achieve 70–85% yield.
  FORMAL: RXN_CLASS("C-C Coupling") + HALIDE_TYPE("aryl bromide") + LIGAND_CLASS("standard") + NUCLEOPHILE_FORM("boronate ester") --> PREDICTED_YIELD("78.5")

Step 6 [SELF_CONSISTENCY]: The predicted yield of 78.5% is consistent with a standard Suzuki coupling of an aryl bromide with a boronate ester using dppf. No contradictions are present.
  FORMAL: PREDICTED_YIELD("78.5") + LIGAND_CLASS("standard") --> SELF_CONSISTENT("yes")

Answer: 78.5

═══════════════════════════════════════════════════════════
STRICT FORMAT RULES:
1. Each step must start with "Step N [FIELD_NAME]:" where FIELD_NAME is exact.
2. The FORMAL: line must be indented by exactly two spaces.
3. All quoted values must use double quotes.
4. RXN_CLASS in Step 1 must be EXACTLY one of the nine listed coarse-grained classes (case-sensitive).
5. HALIDE_TYPE must be exactly one of: "aryl chloride", "aryl bromide", "aryl iodide", "aryl triflate".
6. NUCLEOPHILE_FORM must be exactly one of: "free boronic acid", "boronate ester", "trifluoroborate", "amine", "not provided".
7. LIGAND_CLASS must be exactly one of: "high-performance", "standard", "poor".
8. PREDICTED_YIELD must be a numeric value between 0 and 100 (e.g., "85.3", "42", "67.5"). NO units inside quotes.
9. SELF_CONSISTENT must be exactly: "yes" or "no".
10. Answer must be the SAME numeric value as PREDICTED_YIELD (no quotes, no units, no extra text).
11. Do not add any text before Step 1 or after the Answer line.
12. Do not output JSON, markdown code fences, or bullet lists.

═══════════════════════════════════════════════════════════
GUIDANCE FOR LIGAND CLASSIFICATION:

high-performance — biaryl phosphines (SPhos, XPhos, RuPhos, BrettPhos, etc.), PtBu3,
  CataCXium A, dtbpf, AmPhos; Pd palladacycle pre-catalysts with any such ligand; NHC ligands.

standard — PPh3, PCy3, dppf, P(o-Tol)3, Xantphos, dppe and similar tri(aryl/alkyl)phosphines.

poor — ligand-free Pd(OAc)2 only (no added phosphine or NHC).
"""

USER_TEMPLATE = """\
Reaction class: {rxn_cls}
Reactants (all components): {reactants}
Conditions:
{conditions_block}

Note: {source_note}

Ground Truth Yield: {gt_yield}%

IMPORTANT: The ground truth yield is provided for calibration purposes only.
You must still generate the complete reasoning chain (Step 1 through Step 6) as if you
were solving this problem from scratch. Do NOT simply state the ground truth — you must
work through every step of the reasoning template, analyzing the electrophile, nucleophile,
ligand system, and arriving at the yield prediction naturally.

Generate the formal reasoning chain following EXACTLY the unified step format in your
instructions. Predict the yield as a NUMERIC percentage value (0–100, e.g., 85.3, 42, 67.5).
Do NOT use categories like "low/medium/high".
"""
