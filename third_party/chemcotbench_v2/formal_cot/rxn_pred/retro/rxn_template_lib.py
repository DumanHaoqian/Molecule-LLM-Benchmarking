"""Forward reaction SMARTS templates for outcome_fwd_sim verification.

Maps reaction class names (Schneider-50k / retro pool classes) to RDKit
forward reaction SMARTS strings.  Used to validate retrosynthesis predictions
by running the forward reaction on predicted reactants and checking whether the
GT product is reproduced.

Key design principles:
  - SMARTS are intentionally GENERAL (no over-specific substituent matching) to
    avoid false negatives from structural variation.
  - Templates are applied in SOFT FALLBACK mode: outcome_fwd_sim is a secondary
    check only when outcome_exact is False.
  - get_templates() returns [] (not errors) when no match is found; this signals
    "inconclusive" (not "failed"), and outcome_pass falls back to outcome_exact.
  - For 2-reactant SMARTS, the caller must try all permutations of fragments.

SMARTS syntax notes (RDKit):
  - Use '.' to separate multiple reactant templates (not '+')
  - Use [N;!H0] for "nitrogen with at least one H" (not [N][H])
  - Use [O;!H0] for "alcohol/acid oxygen with H" (not [O][H])
  - Map key atoms with :N to preserve them in the product
  - Unmapped atoms in reactants are treated as leaving groups
"""

from __future__ import annotations

import re

# ── Protecting Group Equivalence Classes ─────────────────────────────────────
# Canonical substrings that identify each PG class (for soft matching).
PROTECTING_GROUP_EQUIV: dict[str, list[str]] = {
    "silyl_ether": ["[Si](C)(C)C", "[Si](CC)(CC)CC", "[Si](C(C)C)(C(C)C)", "O[Si]"],
    "boc":         ["C(=O)OC(C)(C)C"],
    "cbz":         ["C(=O)OCc1ccccc1"],
    "benzyl":      ["OCc1ccccc1"],
    "pmb":         ["OCc1ccc(OC)cc1"],
    "methyl_ester":     ["C(=O)OC", "C(=O)OCC"],
    "tert_butyl_ester": ["C(=O)OC(C)(C)C"],
}

# ── Forward reaction SMARTS templates ────────────────────────────────────────
# Keys are lowercase canonical names; values are lists of SMARTS tried in order.
# All SMARTS are written as forward reactions (reactants >> product).
_TEMPLATES: dict[str, list[str]] = {
    # ── Amide / carbamate bond formation ─────────────────────────────────────
    "n-acylation": [
        "[C:1](=[O:2])[Cl,Br,F].[N;!H0:3] >> [C:1](=[O:2])[N:3]",       # acyl halide + amine
        "[C:1](=O)[O;!H0].[N;!H0:2] >> [C:1](=O)[N:2]",                  # carboxylic acid + amine
    ],
    "acylation": [
        "[C:1](=[O:2])[Cl,Br,F].[N;!H0:3] >> [C:1](=[O:2])[N:3]",
    ],
    "schotten-baumann": [
        "[C:1](=[O:2])[Cl,Br,F].[N;!H0:3] >> [C:1](=[O:2])[N:3]",
    ],
    "imide schotten-baumann": [
        "[C:1](=[O:2])[Cl,Br,F].[N;!H0:3] >> [C:1](=[O:2])[N:3]",
    ],
    "phosphonamide schotten-baumann": [
        "[P:1](=O)[Cl,F].[N;!H0:2] >> [P:1](=O)[N:2]",
    ],
    "carbonylsulfonamide schotten-baumann": [
        "[C:1](=[O:2])[Cl,Br,F].[N;!H0:3][S] >> [C:1](=[O:2])[N:3][S]",
        "[C:1](=[O:2])[Cl,Br,F].[N;!H0:3] >> [C:1](=[O:2])[N:3]",
    ],
    "carboxylic acid + sulfonamide condensation": [
        "[C:1](=O)[O;!H0].[N;!H0:2][S] >> [C:1](=O)[N:2][S]",
        "[C:1](=O)[O;!H0].[N;!H0:2] >> [C:1](=O)[N:2]",
    ],
    "carboxy ester to carbamoyl": [
        "[C:1](=O)[O][C].[N;!H0:2] >> [C:1](=O)[N:2]",
    ],
    "cyano to carbamoyl": [
        "[C:1]#N >> [C:1](=O)N",
    ],
    "carbamate + amine reaction": [
        "[N:1]=C=O.[O;!H0:2] >> [N:1][C](=O)[O:2]",
    ],
    "isocyanate + alcohol reaction": [
        "[N:1]=C=O.[O;!H0:2] >> [N:1][C](=O)[O:2]",
    ],
    "isocyanate": [
        "[N:1]=C=O.[O;!H0:2] >> [N:1][C](=O)[O:2]",
    ],
    # isocyanate + amine → urea
    "urea formation": [
        "[N:1]=C=O.[N;!H0:2] >> [N:1][C](=O)[N:2]",
        "[N;!H0:1].[N:2]=C=O >> [N:1][C](=O)[N:2]",
        "[N;!H0:1].[C:2](=O)=[N:3] >> [N:1][C:2](=O)[N:3]",
    ],
    "urea synthesis": [
        "[N:1]=C=O.[N;!H0:2] >> [N:1][C](=O)[N:2]",
    ],

    # ── N-alkylation / N-arylation / cross-coupling (C–N) ────────────────────
    "n-alkylation": [
        "[C:1][Cl,Br,I].[N;!H0:2] >> [C:1][N:2]",
    ],
    "iodo n-alkylation": [
        "[C:1][I].[N;!H0:2] >> [C:1][N:2]",
    ],
    "n-arylation": [
        "[c:1][Cl,Br,I,F].[N;!H0:2] >> [c:1][N:2]",
    ],
    "fluoro n-arylation": [
        "[c:1][F].[N;!H0:2] >> [c:1][N:2]",
    ],
    "chloro n-arylation": [
        "[c:1][Cl].[N;!H0:2] >> [c:1][N:2]",
    ],
    "buchwald-hartwig amination": [
        "[c:1][Cl,Br,I,F].[N;!H0:2] >> [c:1][N:2]",
        "[C:1][Cl,Br,I].[N;!H0:2] >> [C:1][N:2]",
    ],
    "chloro buchwald-hartwig amination": [
        "[c:1][Cl].[N;!H0:2] >> [c:1][N:2]",
    ],
    "triflyloxy buchwald-hartwig amination": [
        "[c:1][O][S](=O)(=O).[N;!H0:2] >> [c:1][N:2]",
        "[c:1][Cl,Br,I,F].[N;!H0:2] >> [c:1][N:2]",
    ],
    "menshutkin reaction": [
        "[N:1].[C:2][Cl,Br,I] >> [N+:1][C:2]",
    ],
    "eschweiler-clarke methylation": [
        "[N;!H0:1] >> [N:1][CH3]",
    ],

    # ── C–C cross-coupling ────────────────────────────────────────────────────
    "suzuki coupling": [
        "[c:1][Cl,Br,I].[c:2][B] >> [c:1][c:2]",
        "[C:1][Cl,Br,I].[C:2][B] >> [C:1][C:2]",
    ],
    "iodo suzuki coupling": [
        "[c:1][I].[c:2][B] >> [c:1][c:2]",
    ],
    "stille reaction": [
        "[c:1][Cl,Br,I].[c:2][Sn] >> [c:1][c:2]",
        "[C:1][Cl,Br,I].[C:2][Sn] >> [C:1][C:2]",
    ],
    "chloro stille reaction": [
        "[c:1][Cl].[c:2][Sn] >> [c:1][c:2]",
    ],
    "bromo stille reaction": [
        "[c:1][Br].[c:2][Sn] >> [c:1][c:2]",
    ],
    "grignard reaction": [
        "[C:1](=[O:2]).[C:3][Mg] >> [C:1](-[O:2])[C:3]",
        "[C:1](=[O:2]).[C:3][Li] >> [C:1](-[O:2])[C:3]",
    ],
    "iodo grignard reaction": [
        "[C:1](=[O:2]).[C:3][Mg] >> [C:1](-[O:2])[C:3]",
    ],

    # ── Mitsunobu (alcohol inversion, various nucleophiles) ───────────────────
    "mitsunobu aryl ether synthesis": [
        "[c:1][O;!H0].[C:2][O;!H0] >> [c:1][O][C:2]",
        "[C:1][O;!H0].[c:2][O;!H0] >> [C:1][O][c:2]",
    ],
    "mitsunobu sulfonamide reaction": [
        "[C:1][O;!H0].[N;!H0:2][S] >> [C:1][N:2][S]",
        "[C:1][O;!H0].[N;!H0:2] >> [C:1][N:2]",
    ],
    "mitsunobu amine reaction": [
        "[C:1][O;!H0].[N;!H0:2] >> [C:1][N:2]",
    ],

    # ── Reductive transformations ─────────────────────────────────────────────
    "aldehyde reductive amination": [
        "[C;H1:1]=O.[N;!H0:2] >> [C;H1:1][N:2]",
        "[C:1]=O.[N;!H0:2] >> [C:1][N:2]",
    ],
    "reductive amination": [
        "[C:1]=O.[N;!H0:2] >> [C:1][N:2]",
    ],
    "secondary ketimine reduction": [
        "[C:1]=[N:2] >> [C:1][N:2]",
    ],
    "ketimine reduction": [
        "[C:1]=[N:2] >> [C:1][N:2]",
    ],
    "imine reduction": [
        "[C:1]=[N:2] >> [C:1][N:2]",
    ],
    "ketone to alcohol reduction": [
        "[C:1](=[O:2])[C:3] >> [C:1]([O:2])[C:3]",
    ],
    "reduction": [
        "[C:1](=[O:2])[C:3] >> [C:1]([O:2])[C:3]",
        "[C:1]=[N:2] >> [C:1][N:2]",
    ],

    # ── Oxidation ─────────────────────────────────────────────────────────────
    "ketone corey-kim oxidation": [
        "[C:1]([O;!H0])[C:2] >> [C:1](=O)[C:2]",
    ],
    "corey-kim oxidation": [
        "[C:1]([O;!H0])[C:2] >> [C:1](=O)[C:2]",
    ],
    "swern oxidation": [
        "[C:1]([O;!H0]) >> [C:1]=O",
    ],
    "dess-martin oxidation": [
        "[C:1]([O;!H0]) >> [C:1]=O",
    ],
    "oxidation": [
        "[C:1]([O;!H0])[C:2] >> [C:1](=O)[C:2]",
    ],

    # ── Elimination / dehydration ─────────────────────────────────────────────
    "alcohol elimination": [
        "[C:1]([O;!H0])[C:2] >> [C:1]=[C:2]",
    ],
    "dehydration": [
        "[C:1]([O;!H0])[C:2] >> [C:1]=[C:2]",
    ],
    "elimination": [
        "[C:1]([O;!H0])[C:2] >> [C:1]=[C:2]",
        "[C:1]([Cl,Br,I])[C:2] >> [C:1]=[C:2]",
    ],

    # ── Deprotection ─────────────────────────────────────────────────────────
    "o-tes deprotection": [
        "[O:1][Si] >> [O:1]",
    ],
    "o-tbs deprotection": [
        "[O:1][Si] >> [O:1]",
    ],
    "o-tms deprotection": [
        "[O:1][Si] >> [O:1]",
    ],
    "silyl ether deprotection": [
        "[O:1][Si] >> [O:1]",
    ],
    "boc deprotection": [
        "[N:1][C](=O)[O][C](C)(C)C >> [N:1]",
    ],
    "co2h-et deprotection": [
        "[C:1](=O)[O][CC] >> [C:1](=O)[O;H0]",
        "[C:1](=O)[O][C] >> [C:1](=O)[O;H0]",
    ],
    "deprotection": [
        "[O:1][Si] >> [O:1]",                               # silyl ether → free OH
        "[N:1][C](=O)[O][C](C)(C)C >> [N:1]",              # Boc → free amine
        "[C:1](=O)[O][C](C)(C)C >> [C:1](=O)[OH]",         # tBu ester → free acid
        "[C:1](=O)[O]Cc1ccccc1 >> [C:1](=O)[OH]",          # Benzyl ester → free acid
        "[C:1](=O)[O][C]([c])[c] >> [C:1](=O)[OH]",        # diphenyl/benzhydryl ester → free acid
        "[C:1](=O)[O][C] >> [C:1](=O)[OH]",                # general ester → free acid (broad fallback)
    ],

    # ── Hydrolysis ────────────────────────────────────────────────────────────
    "ester hydrolysis": [
        "[C:1](=O)[O:2][C] >> [C:1](=O)[O:2]",
    ],
    "hydrolysis": [
        "[C:1](=O)[O:2][C] >> [C:1](=O)[O:2]",
        "[C:1]#N >> [C:1](=O)N",
    ],

    # ── Olefination ───────────────────────────────────────────────────────────
    "julia olefination": [
        "[C:1][S](=O)(=O).[C:2]=O >> [C:1]=[C:2]",
        "[C:1][S](=O).[C:2]=O >> [C:1]=[C:2]",
    ],
    "julia-kocienski olefination": [
        "[C:1][S](=O)(=O).[C:2]=O >> [C:1]=[C:2]",
        "[C:1][n].[C:2]=O >> [C:1]=[C:2]",  # benzothiazole/tetrazole sulfone simplified
    ],
    "wittig reaction": [
        "[C:1]=[P].[C:2]=O >> [C:1]=[C:2]",
    ],
    "aza-wittig reaction": [
        "[N:1]=[P].[C:2]=O >> [N:1]=[C:2]",
        "[C:1]=[P].[N:2]=O >> [C:1]=[N:2]",
    ],

    # ── Condensation / aldol ─────────────────────────────────────────────────
    "benzoin condensation": [
        "[c:1][C:2](=[O:3]).[c:4][C:5](=[O:6]) >> [c:1][C:2]([O:3])[C:5](=[O:6])[c:4]",
    ],
    "claisen-schmidt condensation": [
        "[C:1](=O).[C:2][Cl,Br,I,O] >> [C:1]=[C:2]",  # simplified
    ],
    "mukaiyama aldol addition": [
        "[C:1](=[O:2]).[C:3][O][Si] >> [C:1]([O:2])[C:3]",
    ],
    "aldol": [
        "[C:1]=O.[C:2] >> [C:1]([O])[C:2]",
    ],

    # ── Miscellaneous ─────────────────────────────────────────────────────────
    "krapcho decarboxylation": [
        # Loses CO2 from a malonate-type ester: retro = monoester → diester (not simulated well)
        # Skip (return [])
    ],
    "carboxylic acid to acid chloride": [
        "[C:1](=O)[O;!H0] >> [C:1](=O)[Cl]",
    ],
    "iodo thioether synthesis": [
        "[C:1][I].[S;!H0:2] >> [C:1][S:2]",
    ],
    "thioether synthesis": [
        "[C:1][Cl,Br,I].[S;!H0:2] >> [C:1][S:2]",
    ],
    "chlorination": [
        "[C:1][O;!H0] >> [C:1][Cl]",
        "[C:1][H] >> [C:1][Cl]",
    ],
    "pummerer rearrangement": [],  # mechanistically complex, skip
    "oxa alder-ene reaction": [],  # pericyclic, skip
    "alder-ene reaction": [],      # pericyclic, skip
    "johnson-claisen rearrangement": [],  # pericyclic, skip
}

# ── Bond compatibility table ─────────────────────────────────────────────────
# Maps canonical reaction type name → frozenset of compatible step6 bond types.
# Bond types use lowercase format: "atom1-atom2" or "atom1=atom2".
#
# Convention: "bond broken" in retrosynthesis = bond FORMED in the forward reaction.
# This is the bond in the PRODUCT that we retrosynthetically disconnect.
# Example: N-acylation forward forms a C-N amide bond → step6 bond = "c-n"
#
# CRITICAL: Deprotection is NOT compatible with C-S or S-O.
# Those belong to Julia olefination / Pummerer / sulfonyl chemistry.
_BOND_COMPAT: dict[str, frozenset] = {
    # ── Amide / carbamate / urea (C-N forming) ────────────────────────────────
    "n-acylation":                        frozenset({"c-n", "n-c"}),
    "acylation":                          frozenset({"c-n", "n-c"}),
    "amide bond formation":               frozenset({"c-n", "n-c"}),
    "schotten-baumann":                   frozenset({"c-n", "n-c"}),
    "imide schotten-baumann":             frozenset({"c-n", "n-c"}),
    "phosphonamide schotten-baumann":     frozenset({"c-n", "n-c", "p-n"}),
    "carbonylsulfonamide schotten-baumann": frozenset({"c-n", "n-c", "s-n"}),
    "carboxy ester to carbamoyl":         frozenset({"c-n", "n-c"}),
    "cyano to carbamoyl":                 frozenset({"c-n", "n-c"}),
    "carboxylic acid + sulfonamide condensation": frozenset({"c-n", "n-c"}),
    "carbamate + amine reaction":         frozenset({"c-n", "n-c", "c-o"}),
    "urea formation":                     frozenset({"c-n", "n-c"}),
    "urea synthesis":                     frozenset({"c-n", "n-c"}),
    "isocyanate + alcohol reaction":      frozenset({"c-n", "c-o"}),

    # ── N-arylation / N-alkylation / cross-coupling C-N ───────────────────────
    "n-arylation":                        frozenset({"c-n", "n-c"}),
    "fluoro n-arylation":                 frozenset({"c-n", "n-c", "c-f"}),
    "chloro n-arylation":                 frozenset({"c-n", "n-c", "c-cl"}),
    "n-alkylation":                       frozenset({"c-n", "n-c"}),
    "iodo n-alkylation":                  frozenset({"c-n", "n-c", "c-i"}),
    "buchwald-hartwig amination":         frozenset({"c-n", "n-c"}),
    "chloro buchwald-hartwig amination":  frozenset({"c-n", "n-c"}),
    "triflyloxy buchwald-hartwig amination": frozenset({"c-n", "n-c"}),
    "menshutkin reaction":                frozenset({"c-n", "n-c"}),
    "nucleophilic aromatic substitution": frozenset({
        "c-n", "c-o", "c-s", "c-f", "c-cl", "c-br", "n-c", "o-c",
    }),
    "eschweiler-clarke methylation":      frozenset({"c-n", "n-c"}),

    # ── Reductive amination / imine reduction ─────────────────────────────────
    "reductive amination":                frozenset({"c-n", "n-c"}),
    "aldehyde reductive amination":       frozenset({"c-n", "n-c"}),
    "ketimine reduction":                 frozenset({"c-n", "n-c"}),
    "secondary ketimine reduction":       frozenset({"c-n", "n-c"}),
    "imine reduction":                    frozenset({"c-n", "n-c"}),
    "ketone to alcohol reduction":        frozenset({"c-o", "o-h"}),
    "reduction":                          frozenset({"c-h", "n-h", "o-h", "c-n", "c-o"}),

    # ── C-C bond forming ──────────────────────────────────────────────────────
    "grignard reaction":                  frozenset({"c-c"}),
    "iodo grignard reaction":             frozenset({"c-c"}),
    "grignard addition":                  frozenset({"c-c"}),
    "benzoin condensation":               frozenset({"c-c", "c-o"}),
    "claisen-schmidt condensation":       frozenset({"c-c", "c=c"}),
    "mukaiyama aldol addition":           frozenset({"c-c", "c-o"}),
    "krapcho decarboxylation":            frozenset({"c-c"}),
    "suzuki coupling":                    frozenset({"c-c"}),
    "iodo suzuki coupling":               frozenset({"c-c"}),
    "stille reaction":                    frozenset({"c-c"}),
    "chloro stille reaction":             frozenset({"c-c"}),
    "bromo stille reaction":              frozenset({"c-c"}),
    "heck reaction":                      frozenset({"c-c", "c=c"}),
    "iodo heck reaction":                 frozenset({"c-c", "c=c"}),
    "sonogashira coupling":               frozenset({"c-c"}),
    "iodo sonogashira coupling":          frozenset({"c-c"}),

    # ── C-O bond forming ──────────────────────────────────────────────────────
    "mitsunobu aryl ether synthesis":     frozenset({"c-o", "o-c"}),
    "mitsunobu sulfonamide reaction":     frozenset({"c-n", "n-c", "c-o"}),
    "mitsunobu amine reaction":           frozenset({"c-n", "n-c"}),
    "williamson ether synthesis":         frozenset({"c-o", "o-c"}),
    "etherification":                     frozenset({"c-o", "o-c"}),
    "ester formation":                    frozenset({"c-o", "o-c"}),
    "ester hydrolysis":                   frozenset({"c-o", "o-c", "o-h"}),
    "hydrolysis":                         frozenset({"c-o", "o-c", "o-h", "c-n"}),

    # ── Olefination (C=C forming) ─────────────────────────────────────────────
    # Julia: sulfone + aldehyde → alkene (via C-S and S-O bond involvement)
    "julia olefination":                  frozenset({"c=c", "c-s", "s-o", "c-c"}),
    "julia-kocienski olefination":        frozenset({"c=c", "c-s", "s-o", "c-c"}),
    "wittig reaction":                    frozenset({"c=c"}),
    "aza-wittig reaction":                frozenset({"c=n", "n=c", "c=c"}),
    "corey-fuchs reaction":               frozenset({"c-c", "c=c"}),

    # ── Elimination / dehydration (form C=C) ──────────────────────────────────
    "dehydration":                        frozenset({"c=c", "c-o", "c-h"}),
    "alcohol elimination":                frozenset({"c=c", "c-o", "c-h"}),
    "elimination":                        frozenset({"c=c", "c-o", "c-h", "c-cl", "c-br"}),

    # ── Oxidation ─────────────────────────────────────────────────────────────
    "ketone corey-kim oxidation":         frozenset({"c-h", "c-o", "c=o"}),
    "swern oxidation":                    frozenset({"c-h", "c-o", "c=o"}),
    "dess-martin oxidation":              frozenset({"c-h", "c-o", "c=o"}),
    "oxidation":                          frozenset({"c-h", "c-o", "c=o"}),

    # ── Deprotection ─────────────────────────────────────────────────────────
    # Standard deprotections: silyl ether → O-H, Boc → N-H, benzyl/Cbz → O-H, ester → O-H
    # Compatible bonds = those formed during deprotection (or equivalently,
    # the bond in the deprotected product that was not present in the protected form).
    # EXPLICITLY INCOMPATIBLE: C-S, S-O (those are sulfonyl/Julia chemistry, not deprotection).
    "deprotection":                       frozenset({"c-o", "o-h", "o-si", "c-n", "n-h", "c-h"}),
    "o-tes deprotection":                 frozenset({"c-o", "o-h", "o-si"}),
    "o-tbs deprotection":                 frozenset({"c-o", "o-h", "o-si"}),
    "o-tms deprotection":                 frozenset({"c-o", "o-h", "o-si"}),
    "silyl ether deprotection":           frozenset({"c-o", "o-h", "o-si"}),
    "boc deprotection":                   frozenset({"c-n", "n-h", "c-o"}),
    "co2h-et deprotection":               frozenset({"c-o", "o-h"}),

    # ── Misc ──────────────────────────────────────────────────────────────────
    "carboxylic acid to acid chloride":   frozenset({"c-cl", "c-o"}),
    "iodo thioether synthesis":           frozenset({"c-s"}),
    "thioether synthesis":                frozenset({"c-s"}),
    "chlorination":                       frozenset({"c-cl"}),
    "pummerer rearrangement":             frozenset({"c-s", "c-o", "s-o"}),
    "oxa alder-ene reaction":             frozenset({"c-c", "c-o"}),
    "alder-ene reaction":                 frozenset({"c-c"}),
    "johnson-claisen rearrangement":      frozenset({"c-c", "c-o"}),
}


def normalize_bond_type(bond_str: str) -> str:
    """Extract and normalize bond type from step6_bond_broken string.

    Converts "C-N bond broken", "C=C", "amide bond (C-N)", "O-H" etc.
    to lowercase two-atom notation: "c-n", "c=c", "o-h".

    Returns empty string if parsing fails.
    """
    if not bond_str:
        return ""
    # Match X-Y or X=Y where X, Y are 1-2 char element symbols (e.g. Si, Cl, Br)
    m = re.search(r'([A-Z][a-z]?)([=\-])([A-Z][a-z]?)', bond_str)
    if m:
        return f"{m.group(1).lower()}{m.group(2)}{m.group(3).lower()}"
    return bond_str.strip().lower()


def get_compatible_bonds(rxn_type: str) -> frozenset | None:
    """Return compatible step6 bond types for the given reaction type.

    Uses the same fuzzy matching strategy as get_templates():
      1. Direct key match in _BOND_COMPAT
      2. Alias dict → canonical key → _BOND_COMPAT
      3. Substring match
      4. Token-level overlap

    Returns None if the reaction type is unknown → no constraint (don't gate).
    Returns a frozenset of lowercase bond strings otherwise.
    """
    if not rxn_type:
        return None

    query = rxn_type.strip().lower()

    # 1. Direct match
    if query in _BOND_COMPAT:
        return _BOND_COMPAT[query]

    # 2. Alias → canonical → _BOND_COMPAT
    for alias in sorted(_ALIASES, key=len, reverse=True):
        if alias in query:
            canon = _ALIASES[alias]
            if canon in _BOND_COMPAT:
                return _BOND_COMPAT[canon]
            break  # alias matched but no compat entry registered → unknown

    # 3. Substring match
    for key in _BOND_COMPAT:
        if key in query or query in key:
            return _BOND_COMPAT[key]

    # 4. Token overlap
    stop_words = {"reaction", "coupling", "synthesis", "addition", "condensation",
                  "amination", "oxidation", "reduction", "rearrangement"}
    tokens = {t for t in re.split(r"[\s\-]+", query) if len(t) > 3 and t not in stop_words}
    for key in _BOND_COMPAT:
        key_tokens = set(re.split(r"[\s\-]+", key))
        if tokens & key_tokens:
            return _BOND_COMPAT[key]

    return None  # unknown reaction type → no constraint


# ── Alias / keyword mapping ───────────────────────────────────────────────────
# Alternate names / substrings → canonical key in _TEMPLATES.
_ALIASES: dict[str, str] = {
    "n-acylation":            "n-acylation",
    "acylation":              "n-acylation",
    "schotten":               "schotten-baumann",
    "schotten-baumann":       "schotten-baumann",
    "imide schotten":         "imide schotten-baumann",
    "phosphonamide":          "phosphonamide schotten-baumann",
    "carbonylsulfonamide":    "carbonylsulfonamide schotten-baumann",
    "sulfonamide condensation": "carboxylic acid + sulfonamide condensation",
    "carboxy ester to carbamoyl": "carboxy ester to carbamoyl",
    "cyano to carbamoyl":     "cyano to carbamoyl",
    "carbamate + amine":      "carbamate + amine reaction",
    "urea formation":         "urea formation",
    "urea synthesis":         "urea synthesis",
    "isocyanate":             "isocyanate + alcohol reaction",
    "n-alkylation":           "n-alkylation",
    "iodo n-alkylation":      "iodo n-alkylation",
    "n-arylation":            "n-arylation",
    "fluoro n-arylation":     "fluoro n-arylation",
    "chloro n-arylation":     "chloro n-arylation",
    "buchwald-hartwig":       "buchwald-hartwig amination",
    "buchwald":               "buchwald-hartwig amination",
    "chloro buchwald":        "chloro buchwald-hartwig amination",
    "triflyloxy buchwald":    "triflyloxy buchwald-hartwig amination",
    "menshutkin":             "menshutkin reaction",
    "eschweiler":             "eschweiler-clarke methylation",
    "suzuki":                 "suzuki coupling",
    "iodo suzuki":            "iodo suzuki coupling",
    "stille":                 "stille reaction",
    "chloro stille":          "chloro stille reaction",
    "bromo stille":           "bromo stille reaction",
    "grignard":               "grignard reaction",
    "iodo grignard":          "iodo grignard reaction",
    "mitsunobu aryl ether":   "mitsunobu aryl ether synthesis",
    "mitsunobu sulfonamide":  "mitsunobu sulfonamide reaction",
    "mitsunobu amine":        "mitsunobu amine reaction",
    "mitsunobu":              "mitsunobu aryl ether synthesis",
    "reductive amination":    "reductive amination",
    "aldehyde reductive":     "aldehyde reductive amination",
    "ketimine":               "ketimine reduction",
    "secondary ketimine":     "secondary ketimine reduction",
    "imine reduction":        "imine reduction",
    "ketone to alcohol":      "ketone to alcohol reduction",
    "reduction":              "reduction",
    "corey-kim":              "ketone corey-kim oxidation",
    "corey kim":              "ketone corey-kim oxidation",
    "swern":                  "swern oxidation",
    "dess-martin":            "dess-martin oxidation",
    "oxidation":              "oxidation",
    "alcohol elimination":    "alcohol elimination",
    "dehydration":            "dehydration",
    "elimination":            "elimination",
    "o-tes":                  "o-tes deprotection",
    "o-tbs":                  "o-tbs deprotection",
    "o-tms":                  "o-tms deprotection",
    "silyl":                  "silyl ether deprotection",
    "boc deprotect":          "boc deprotection",
    "co2h-et":                "co2h-et deprotection",
    "deprotect":              "deprotection",
    "ester hydrolysis":       "ester hydrolysis",
    "hydrolysis":             "hydrolysis",
    "julia olefination":      "julia olefination",
    "julia-kocienski":        "julia-kocienski olefination",
    "julia":                  "julia olefination",
    "kocienski":              "julia-kocienski olefination",
    "wittig":                 "wittig reaction",
    "aza-wittig":             "aza-wittig reaction",
    "benzoin":                "benzoin condensation",
    "claisen-schmidt":        "claisen-schmidt condensation",
    "mukaiyama":              "mukaiyama aldol addition",
    "krapcho":                "krapcho decarboxylation",
    "carboxylic acid to acid chloride": "carboxylic acid to acid chloride",
    "acid chloride":          "carboxylic acid to acid chloride",
    "thioether":              "iodo thioether synthesis",
    "iodo thioether":         "iodo thioether synthesis",
    "chlorination":           "chlorination",
    "pummerer":               "pummerer rearrangement",
    "oxa alder":              "oxa alder-ene reaction",
    "alder-ene":              "alder-ene reaction",
    "johnson-claisen":        "johnson-claisen rearrangement",
    "claisen rearrangement":  "johnson-claisen rearrangement",
}


def get_templates(rxn_type: str) -> list[str]:
    """Return forward reaction SMARTS list for the given reaction type string.

    Matching strategy (in priority order):
      1. Exact key match (lowercased, stripped)
      2. Alias dict lookup by any alias substring in rxn_type
      3. Any _TEMPLATES key is a substring of rxn_type (or vice versa)
      4. Token-level overlap between rxn_type words and template keys

    Returns [] when no match is found (inconclusive, not a failure).
    """
    if not rxn_type:
        return []

    query = rxn_type.strip().lower()

    # 1. Exact match
    if query in _TEMPLATES:
        return _TEMPLATES[query]

    # 2. Alias match (longest alias first to prefer specificity)
    matched_keys: list[str] = []
    for alias in sorted(_ALIASES, key=len, reverse=True):
        if alias in query:
            canon = _ALIASES[alias]
            if canon in _TEMPLATES and canon not in matched_keys:
                matched_keys.append(canon)
                break  # take the longest matching alias

    if matched_keys:
        templates: list[str] = []
        for k in matched_keys:
            templates.extend(_TEMPLATES[k])
        return templates

    # 3. Substring match: key ⊆ query  or  query ⊆ key
    sub_matches: list[str] = []
    for key in _TEMPLATES:
        if key in query or query in key:
            sub_matches.extend(_TEMPLATES[key])
    if sub_matches:
        return sub_matches

    # 4. Token overlap: any significant word in query matches a key
    stop_words = {"reaction", "coupling", "synthesis", "addition", "condensation",
                  "amination", "oxidation", "reduction", "rearrangement"}
    tokens = {t for t in re.split(r"[\s\-]+", query) if len(t) > 3 and t not in stop_words}
    token_matches: list[str] = []
    for key in _TEMPLATES:
        key_tokens = set(re.split(r"[\s\-]+", key))
        if tokens & key_tokens:
            token_matches.extend(_TEMPLATES[key])
    return token_matches


def run_forward_sim(
    pred_smi: str,
    rxn_type: str,
    product_smi: str,
) -> bool | None:
    """Forward-simulate predicted reactants through the reaction type template.

    Returns:
        True  — at least one forward simulation product matches the GT product
        False — templates found, simulation ran, but no product matched
        None  — no template available for this rxn_type (inconclusive)
    """
    from itertools import permutations as _permutations
    from rdkit.Chem import AllChem, MolFromSmiles, MolToSmiles, SanitizeMol

    templates = get_templates(rxn_type)
    if not templates:
        return None

    product_mol = MolFromSmiles(product_smi)
    if product_mol is None:
        return None
    prod_can_nostereo = MolToSmiles(product_mol, isomericSmiles=False)
    prod_can_stereo   = MolToSmiles(product_mol, isomericSmiles=True)

    frags = [f.strip() for f in pred_smi.split(".") if f.strip()]
    frag_mols = [(f, MolFromSmiles(f)) for f in frags]
    frag_mols = [(f, m) for f, m in frag_mols if m is not None]
    if not frag_mols:
        return False

    found_any_template = False
    for smarts in templates:
        if not smarts:
            continue
        try:
            rxn = AllChem.ReactionFromSmarts(smarts)
            if rxn is None:
                continue
            n_react = rxn.GetNumReactantTemplates()
            found_any_template = True

            if n_react == 1:
                combos = [(fm,) for fm in frag_mols]
            elif n_react == 2 and len(frag_mols) >= 2:
                combos = list(_permutations(frag_mols, 2))
            else:
                continue

            for combo in combos:
                mols = tuple(m for _, m in combo)
                try:
                    products = rxn.RunReactants(mols)
                    for p_set in products:
                        for p in p_set:
                            try:
                                SanitizeMol(p)
                                p_no_st = MolToSmiles(p, isomericSmiles=False)
                                if p_no_st and (p_no_st == prod_can_nostereo):
                                    return True
                                p_stereo = MolToSmiles(p, isomericSmiles=True)
                                if p_stereo and (p_stereo == prod_can_stereo):
                                    return True
                            except Exception:
                                pass
                except Exception:
                    pass
        except Exception:
            pass

    return False if found_any_template else None
