"""
Prompt for rxn_pred/condition_ranking top-3 ranking CoT generation.

Task: Given a reaction and three candidate condition sets (labeled "1", "2", "3"
in GT rank order), produce a 6-step formal reasoning chain that compares all
C(3,2)=3 pairs and derives the top-3 ranking.

6-step format:
  Step 1 [RXN_CLASS]:        TASK("rank conditions") --> RXN_CLASS("<coarse class>")
  Step 2 [DECISION_FACTOR]:  RXN_CLASS("...") --> DECISION_FACTOR("<field>")
  Step 3 [PAIR_DIFFS]:       CONDITIONS(["1","2","3"]) --> PAIR_DIFFS(1/2:<f+f>; 1/3:<f+f>; 2/3:<f>)
  Step 4 [PAIRWISE_PREFS]:   DECISION_FACTOR("...") + PAIR_DIFFS(...) --> PAIRWISE_PREFS(1>2; 1>3; 2>3)
  Step 5 [RANKING]:          PAIRWISE_PREFS(...) --> RANKING(["1","2","3"])
  Step 6 [TOP2_SUPPORT]:     RANKING([...]) + PAIR_DIFFS(...) --> TOP2_SUPPORT(WINNER="1", LOSER="2", FIELD="<field>")
  Answer: ["1","2","3"]

GT injection: coarse_rxn_cls and gt_ranking are provided for calibration.
Outcome (answer matches gt_ranking) is informational only — NOT gating in all_pass.
"""

SYSTEM_PROMPT = """You are an expert synthetic chemist. Your task is to rank three candidate reaction condition sets (labeled "1", "2", "3") from best to worst predicted yield, by reasoning through a formally verifiable step-by-step chain.

You are given:
1. The reaction type (fine-grained and coarse-grained)
2. The reactants and product
3. Three candidate condition sets labeled 1, 2, and 3

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

Available condition fields for DECISION_FACTOR (choose EXACTLY one):
- catalyst
- ligand
- base
- reagent
- additive
- solvent

Your output must follow the EXACT unified step format below. Do NOT write any introductory text or markdown code fences before the first step. No text after the Answer line.

═══════════════════════════════════════════════════════════
UNIFIED STEP FORMAT

Each step must follow this exact structure:

Step N [FIELD_NAME]: Natural language reasoning description (1–3 sentences)
  FORMAL: INPUT --> OUTPUT

Rules:
- Step number N starts at 1 and increments continuously.
- FIELD_NAME must be one of: [RXN_CLASS], [DECISION_FACTOR], [PAIR_DIFFS], [PAIRWISE_PREFS], [RANKING], [TOP2_SUPPORT].
- The FORMAL: line must be indented by exactly two spaces.
- All values inside parentheses must be enclosed in double quotes (except list brackets).
- Each FORMAL line must contain exactly one "-->" arrow.

═══════════════════════════════════════════════════════════
EXAMPLE INPUT:
  Reaction class: Pd-catalyzed Suzuki cross-coupling
  Reactants: ClC1=CC=CC=C1 + OB(O)c1ccccc1
  Product: c1ccc(-c2ccccc2)cc1
  Condition set 1: catalyst=Pd(PPh3)4, solvent=EtOH/H2O
  Condition set 2: catalyst=Pd(OAc)2, ligand=SPhos, base=K2CO3, solvent=toluene
  Condition set 3: catalyst=PdCl2(dppf), base=Na2CO3, solvent=DMF/H2O
  Ground Truth Ranking: ["1","2","3"]

EXAMPLE OUTPUT:

Step 1 [RXN_CLASS]: This is a Suzuki cross-coupling forming a biaryl C-C bond between an aryl chloride and arylboronic acid, belonging to the C-C Coupling class.
  FORMAL: TASK("rank conditions") --> RXN_CLASS("C-C Coupling")

Step 2 [DECISION_FACTOR]: For Suzuki coupling with an aryl chloride (less reactive), the catalyst system—specifically the ligand—is the primary determinant of reactivity and yield.
  FORMAL: RXN_CLASS("C-C Coupling") --> DECISION_FACTOR("catalyst")

Step 3 [PAIR_DIFFS]: Comparing all three pairs: 1 vs 2 differ in catalyst, ligand, base, and solvent; 1 vs 3 differ in catalyst, base, and solvent; 2 vs 3 differ in catalyst, ligand, base, and solvent.
  FORMAL: CONDITIONS(["1","2","3"]) --> PAIR_DIFFS(1/2:catalyst+ligand+base+solvent; 1/3:catalyst+base+solvent; 2/3:catalyst+ligand+base+solvent)

Step 4 [PAIRWISE_PREFS]: Pd(PPh3)4 in protic EtOH/H2O is highly effective for Suzuki coupling; Pd(OAc)2/SPhos is excellent for aryl chlorides but aryl chlorides are challenging; PdCl2(dppf) with Na2CO3 in DMF/H2O is also good. Overall 1 beats 2, 1 beats 3, and 2 beats 3 based on solvent compatibility and catalyst activity.
  FORMAL: DECISION_FACTOR("catalyst") + PAIR_DIFFS(1/2:catalyst+ligand+base+solvent; 1/3:catalyst+base+solvent; 2/3:catalyst+ligand+base+solvent) --> PAIRWISE_PREFS(1>2; 1>3; 2>3)

Step 5 [RANKING]: From the three pairwise preferences (1>2, 1>3, 2>3), the consistent total order is 1 > 2 > 3.
  FORMAL: PAIRWISE_PREFS(1>2; 1>3; 2>3) --> RANKING(["1","2","3"])

Step 6 [TOP2_SUPPORT]: Condition set 1 (Pd(PPh3)4, EtOH/H2O) outperforms condition set 2 (Pd(OAc)2/SPhos, toluene) primarily because the aqueous protic co-solvent activates the boronic acid transmetalation step more effectively.
  FORMAL: RANKING(["1","2","3"]) + PAIR_DIFFS(1/2:catalyst+ligand+base+solvent) --> TOP2_SUPPORT(WINNER="1", LOSER="2", FIELD="catalyst")

Answer: ["1","2","3"]

═══════════════════════════════════════════════════════════
STRICT FORMAT RULES:
1. Each step must start with "Step N [FIELD_NAME]:" where FIELD_NAME is exact.
2. The FORMAL: line must be indented by exactly two spaces.
3. All quoted values (except the JSON list labels) must use double quotes.
4. RXN_CLASS in Step 1 must be EXACTLY one of the nine listed coarse-grained classes.
5. DECISION_FACTOR in Step 2 must be EXACTLY one of: catalyst, ligand, base, reagent, additive, solvent.
6. PAIR_DIFFS in Step 3 must cover all three pairs (1/2, 1/3, 2/3), separated by semicolons.
   Format: 1/2:<field1>+<field2>; 1/3:<field1>; 2/3:<field1>+<field2>
   Use "none" if a pair is identical across all condition fields.
7. PAIRWISE_PREFS in Step 4 must contain all three comparisons separated by semicolons.
   Each preference is X>Y where X and Y are "1", "2", or "3".
   The three preferences must be mutually consistent (no cycles).
8. RANKING in Step 5 must be a JSON array of the three labels in ranked order, consistent with PAIRWISE_PREFS.
9. TOP2_SUPPORT in Step 6: WINNER must equal RANKING[0], LOSER must equal RANKING[1].
   FIELD must be one of the valid condition fields.
10. Answer must be the ranking JSON array on its own line, matching RANKING in Step 5.
11. Do not add any text before Step 1 or after the Answer line.
12. Do not output JSON code fences, markdown, or bullet lists.
13. Do not mention observed yield numbers, percentages, or raw numerical scores.
14. IMPORTANT: The ground truth ranking is provided for calibration — you must still reason through all six steps naturally.
"""

USER_TEMPLATE = """\
[CALIBRATION — GT provided for reasoning calibration only]
Coarse reaction class: {coarse_rxn_cls}
Ground truth ranking: {gt_ranking}

Reaction class: {rxn_cls}
Reactants: {reactants}
Product: {product}
Condition set 1: {cond_1}
Condition set 2: {cond_2}
Condition set 3: {cond_3}

IMPORTANT: The ground truth information above is provided for calibration purposes only.
You must still generate the complete reasoning chain (Step 1 through Step 6) as if you
were solving this problem from scratch. Work through: classify the reaction (must match
the calibration class), identify the single most important decision factor, compare all
three pairs in PAIR_DIFFS, derive pairwise preferences, aggregate to a ranking, and
support the top-2 comparison — all must be consistent with the calibration ranking.

Generate the formal reasoning chain following EXACTLY the unified step format in your
instructions. End with the Answer ranking array.
"""
