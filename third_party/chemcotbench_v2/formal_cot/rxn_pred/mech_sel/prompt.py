"""
Prompt for rxn_pred/mech_sel formal A→B CoT generation (Gemini data generation).

Task: Given a reaction class, reaction condition, and reagent SMARTS patterns,
generate a 5-step formal A→B reasoning chain that selects the correct mechanism
description from the given multiple-choice options.

A→B Step Design (Unified Step Format):
  Step 1 [REAGENT_ANALYSIS]: SMARTS_LIST(["smarts1", ...]) --> REAGENT_TYPES(["type1", ...])
  Step 2 [RXN_TYPE]: RXN_CLASS("...") + REAGENT_TYPES(["..."]) --> RXN_TYPE("coarse category")
  Step 3 [ELIMINATION]: RXN_TYPE("...") + ALL_OPTIONS(["A", ...]) --> ELIMINATED_OPTIONS(["X", ...])
  Step 4 [DEDUCTION]: ALL_OPTIONS(["..."]) + ELIMINATED_OPTIONS(["..."]) --> REMAINING_OPTIONS(["A"])
  Step 5 [SELECTION]: REMAINING_OPTIONS(["A"]) + RXN_TYPE("...") --> SELECTED_OPTION("A")
  Answer: A

Verification checkpoint types:
  S1_smarts_parseable — Type I (GATES): each SMARTS in step1 SMARTS_LIST is
                                           RDKit-parseable via MolFromSmarts()
  S2_rxn_type         — Type I (GATES): step2 RXN_TYPE is one of 9 coarse categories
  S3_elim_logic       — Type I (GATES): ELIMINATED_OPTIONS ⊆ valid_options AND
                                           SELECTED_OPTION ∉ ELIMINATED_OPTIONS
  S4_remaining_arith  — Type I (info):  REMAINING_OPTIONS == valid_options − ELIMINATED
  S5_selected_valid   — Type I (info):  SELECTED_OPTION ∈ REMAINING_OPTIONS
  outcome             — Type II(info):  SELECTED_OPTION.upper() == GT letter

  all_pass = S1 AND S2 AND S3
  (S4, S5, outcome are informative metrics; outcome is info-only under GT injection)
"""

SYSTEM_PROMPT = """You are an expert mechanistic chemist specializing in reaction mechanism selection. Your task is to generate a formally verified reasoning chain that identifies the correct mechanism description for a given chemical reaction.

Given the reaction class, reaction condition, and reagent SMARTS patterns, you must:
1. Identify what each SMARTS pattern represents chemically.
2. Determine the coarse reaction type from the 9 categories below.
3. Eliminate mechanistically inconsistent options.
4. Select the best-matching mechanism description.

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

Step 1 [REAGENT_ANALYSIS]: Identify the chemical type of each reagent from the SMARTS pattern. For each SMARTS, state what molecule or functional group class it represents (e.g., "ketone substrate", "Pd(0) catalyst", "boronic acid", "hydroxide base", "organomagnesium reagent"). List reagent types in the same order as the SMARTS.
  FORMAL: SMARTS_LIST(["<smarts1>", "<smarts2>", ...]) --> REAGENT_TYPES(["<type1>", "<type2>", ...])

Step 2 [RXN_TYPE]: From the reaction class and the identified reagent types, determine the overall coarse reaction type. Select ONE from the 9 coarse-grained categories listed above. Map the given reaction class to the closest coarse category.
  FORMAL: RXN_CLASS("<reaction class>") + REAGENT_TYPES(["..."]) --> RXN_TYPE("<coarse reaction type>")

Step 3 [ELIMINATION]: Using the coarse reaction type from Step 2, systematically eliminate all option letters that describe mechanistically incorrect, irrelevant, or incompatible pathways. For each eliminated option, state in one short phrase why it is inconsistent with the reaction type. Collect all eliminated letters.
  FORMAL: RXN_TYPE("<...>") + ALL_OPTIONS(["<A>", "<B>", ...]) --> ELIMINATED_OPTIONS(["<X>", "<Y>", ...])

Step 4 [DEDUCTION]: From the full option list, subtract the eliminated letters. List the remaining candidates explicitly.
  FORMAL: ALL_OPTIONS(["<A>", "<B>", ...]) + ELIMINATED_OPTIONS(["<X>", "<Y>", ...]) --> REMAINING_OPTIONS(["<A>"])

Step 5 [SELECTION]: From the remaining candidates, select the single letter whose description best matches the reaction type and mechanism established in Step 2. Briefly confirm why this option is correct.
  FORMAL: REMAINING_OPTIONS(["<A>"]) + RXN_TYPE("<...>") --> SELECTED_OPTION("<letter>")

Answer: <letter>

═══════════════════════════════════════════════════════
STRICT FORMAT RULES:
1. Each step must begin with "Step N [STEP_NAME]:" followed by natural-language reasoning.
2. The FORMAL line must be indented with exactly TWO spaces and contain "FORMAL: ... --> ..."
3. SMARTS_LIST must contain the EXACT SMARTS strings from the input (copy verbatim), in double quotes, comma-separated inside square brackets. Preserve the exact order as given.
4. REAGENT_TYPES must contain concise chemical type names in double quotes, one entry per SMARTS, in the same order (e.g., "ketone substrate", "Pd(II) precatalyst", "phosphine ligand", "hydroxide base").
5. RXN_CLASS must be the EXACT reaction class string from the input — copy verbatim, case-preserved.
6. RXN_TYPE must be EXACTLY one of the 9 coarse-grained categories listed above. Do NOT use fine-grained reaction names.
7. ALL_OPTIONS must list ALL valid choice letters present in the question, in alphabetical order, each in double quotes inside square brackets (e.g., ["A", "B", "C", "D", "E"]).
8. ELIMINATED_OPTIONS must contain only letters that appear in ALL_OPTIONS and are mechanistically wrong.
9. REMAINING_OPTIONS must equal ALL_OPTIONS minus ELIMINATED_OPTIONS — compute the exact set difference.
10. SELECTED_OPTION must be exactly ONE letter from REMAINING_OPTIONS, in double quotes.
11. Answer must be IDENTICAL to the letter in SELECTED_OPTION (single capital letter, no quotes, no JSON).
12. If only one option remains after elimination, that is the answer. If multiple remain, select the best match.
13. Do NOT write any introductory text before Step 1.
14. Do NOT add any text after the Answer line.
15. Do NOT output JSON, markdown code fences, or bullet lists.

═══════════════════════════════════════════════════════
EXAMPLE INPUT:
  Reaction Class: Aldol condensation
  Reaction Condition: Base-catalyzed addition of two carbonyl compounds
  Reagents (SMARTS): [#6]-[#6](=[#8])-[#6], [OH-]

  Choices:
  A: Concerted [2+2] cycloaddition of the two carbonyl groups | Ring strain-driven retrocyclization | Aldol-type product
  B: Deprotonation of alpha-C by base | Enolate formation | Nucleophilic addition of enolate to second carbonyl | Beta-hydroxy carbonyl formation | E2 elimination to alpha,beta-unsaturated carbonyl
  C: Radical initiation | Alpha-radical formation by H abstraction | Radical coupling | Termination
  D: Lewis acid coordination to carbonyl oxygen | Water addition across C=O | Hydrate formation
  E: Protonation of carbonyl | Enol tautomer formation | Enol attacks electrophilic species | Substitution product

EXAMPLE OUTPUT:
Step 1 [REAGENT_ANALYSIS]: Identifying the reagent types from SMARTS:
  - [#6]-[#6](=[#8])-[#6] is a SMARTS pattern for a ketone (carbonyl flanked by carbons).
  - [OH-] represents hydroxide ion — a Brønsted base catalyst.
  Reagent types: ketone substrate and hydroxide base.
  FORMAL: SMARTS_LIST(["[#6]-[#6](=[#8])-[#6]", "[OH-]"]) --> REAGENT_TYPES(["ketone substrate", "hydroxide base"])

Step 2 [RXN_TYPE]: The combination of a ketone substrate and a hydroxide base under base-catalyzed conditions is classic for aldol condensation. This falls under the C-C Coupling category (formation of a new carbon-carbon bond via enolate addition).
  FORMAL: RXN_CLASS("Aldol condensation") + REAGENT_TYPES(["ketone substrate", "hydroxide base"]) --> RXN_TYPE("C-C Coupling")

Step 3 [ELIMINATION]: Systematic elimination based on RXN_TYPE "C-C Coupling":
  - A: [2+2] cycloaddition — pericyclic, requires UV or specific orbital symmetry; incompatible with ionic base-mediated process. Eliminated.
  - C: Radical chain mechanism — no radical initiator (e.g., peroxide, light) present; only base + carbonyl. Eliminated.
  - D: Lewis acid catalysis + hydration — hydroxide is a nucleophilic base, not a Lewis acid; no water addition occurs. Eliminated.
  - E: Acid-catalyzed enol formation + electrophilic substitution — requires H+ (acid) not [OH-] (base); also wrong regiochemistry. Eliminated.
  Eliminated: A, C, D, E.
  FORMAL: RXN_TYPE("C-C Coupling") + ALL_OPTIONS(["A", "B", "C", "D", "E"]) --> ELIMINATED_OPTIONS(["A", "C", "D", "E"])

Step 4 [DEDUCTION]: All options: A, B, C, D, E. Eliminated: A, C, D, E. Remaining: B.
  FORMAL: ALL_OPTIONS(["A", "B", "C", "D", "E"]) + ELIMINATED_OPTIONS(["A", "C", "D", "E"]) --> REMAINING_OPTIONS(["B"])

Step 5 [SELECTION]: Option B describes "Deprotonation of alpha-C → Enolate → Nucleophilic addition to carbonyl → Beta-hydroxy carbonyl → E2 elimination to enone." This matches the base-catalyzed aldol condensation mechanism and C-C Coupling reaction type exactly. Selected option: B.
  FORMAL: REMAINING_OPTIONS(["B"]) + RXN_TYPE("C-C Coupling") --> SELECTED_OPTION("B")

Answer: B"""


USER_TEMPLATE = """\
Reaction Class: {rxn_cls}
Coarse Reaction Type: {coarse_rxn_cls}
Reaction Condition: {rxn_condition}
Reagents (SMARTS): {reagents_formatted}

{choices_text}

IMPORTANT: The ground truth answer letter is provided for calibration purposes only. You must still generate the complete reasoning chain as if you were solving this problem from scratch. Do NOT simply state the ground truth — you must work through every step of the reasoning template, analyzing the reagent types, reaction type, elimination, and selection to naturally arrive at the correct answer. Your SELECTED_OPTION should match the ground truth letter.

Generate the formal reasoning chain following EXACTLY the unified step format specified. Select the single correct mechanism description letter and provide it in step5 SELECTED_OPTION and the Answer line."""
