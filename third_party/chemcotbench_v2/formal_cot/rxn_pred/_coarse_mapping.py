"""Shared coarse-grained reaction class mapping for all rxn_pred tasks.

Maps 282 fine-grained reaction classes to 9 coarse categories.
"""

COARSE_CLASSES = [
    "C-C Coupling",
    "Heteroatom Alkylation and Arylation",
    "Acylation",
    "Functional Group Interconversion",
    "Deprotection",
    "Reduction",
    "Oxidation",
    "Aromatic Heterocycle Formation",
    "Protection",
]


def map_to_coarse(rxn_cls: str) -> str:
    """Map fine-grained reaction class to one of 9 coarse-grained categories."""
    rc = rxn_cls.lower().strip()

    # Direct match for already-coarse names (datasets may already use coarse labels)
    direct_map = {
        "c-c coupling": "C-C Coupling",
        "heteroatom alkylation and arylation": "Heteroatom Alkylation and Arylation",
        "acylation": "Acylation",
        "functional group interconversion": "Functional Group Interconversion",
        "deprotection": "Deprotection",
        "reduction": "Reduction",
        "oxidation": "Oxidation",
        "aromatic heterocycle formation": "Aromatic Heterocycle Formation",
        "protection": "Protection",
    }
    if rc in direct_map:
        return direct_map[rc]

    # --- Deprotection ---
    if "deprotection" in rc or "deprotect" in rc:
        return "Deprotection"
    if rc in ["methoxy to hydroxy", "tms ether hydrolysis", "ester hydrolysis"]:
        return "Deprotection"

    # --- Protection ---
    if "protection" in rc and "deprotection" not in rc:
        return "Protection"

    # --- C-C Coupling ---
    cc_keywords = [
        "suzuki", "sonogashira", "stille", "negishi", "heck", "kumada",
        "grignard", "cross metathesis", "ring-closing metathesis",
        "ring-opening metathesis", "miyaura boration", "miyaura",
        "carbonylative suzuki", "aldol", "michael addition", "mannich",
        "knoevenagel", "diels-alder", "pauson-khand", "wittig", "olefination",
        "horner-wadsworth-emmons", "julia", "baylis-hillman", "henry reaction",
        "ene reaction", "mukaiyama", "reformatsky", "simmons-smith",
        "stetter reaction", "stork enamine", "trost", "benzoin condensation",
        "keto alpha-alkylation", "hosomi-sakurai", "corey-chaykovsky",
        "darzens glycidic", "acetoacetic ester", "malonic ester synthesis",
        "atom transfer radical cyclization", "johnson-corey-chaykovsky",
        "pd-catalyzed c-h arylation", "rh-catalyzed c-h insertion",
        "petasis reaction",
    ]
    if any(k in rc for k in cc_keywords):
        return "C-C Coupling"

    # --- Heteroatom Alkylation and Arylation ---
    het_keywords = [
        "n-alkylation", "n-methylation", "n-arylation",
        "williamson ether", "snar ether", "thioether",
        "menshutkin", "mitsunobu",
        "sn1 reaction", "sn2 reaction",
        "buchwald-hartwig",
        "williamson", "heteroatom",
        "isocyanate + amine", "isocyanate + alcohol",
        "alcohol + amine condensation",
        "staudinger ligation", "gabriel synthesis",
        "meisenheimer", "cuaac click",
        "strecker aldehyde", "strecker amino acid", "strecker ketone",
    ]
    if any(k in rc for k in het_keywords):
        return "Heteroatom Alkylation and Arylation"

    # --- Reduction ---
    reduction_keywords = [
        "reduction", "reductive", "hydrogenation", "deoxygenation",
        "dehalogenation", "clemmensen", "wolff-kishner", "rosenmund",
        "stephen reduction", "dibal-h", "nabh4", "lialh4",
        "hydrosilylation", "eschweiler-clarke", "mcfadyen-stevens",
    ]
    if any(k in rc for k in reduction_keywords):
        return "Reduction"

    # --- Oxidation ---
    oxidation_keywords = [
        "oxidation", "epoxidation", "ozonolysis", "dess-martin",
        "corey-kim oxidation", "swern", "oppenauer", "baeyer-villiger",
        "dakin", "pcc", "wacker", "rubottom", "jones",
        "kornblum oxidation", "sharpless asymmetric dihydroxylation",
        "prilezhaev epoxidation", "mcpba epoxidation",
    ]
    if any(k in rc for k in oxidation_keywords):
        return "Oxidation"

    # --- Acylation ---
    acylation_keywords = [
        "esterification", "transesterification", "schotten-baumann",
        "weinreb", "amide synthesis", "amidation",
        "carboxylic acid + amide", "carboxylic acid + amine",
        "carboxylic acid + amidine", "carboxylic acid + imine",
        "carboxylic acid + thioamide", "carboxylic acid + sulfonamide",
        "carboxylic acid + thiol", "carboxylic ester + amine",
        "carbamate + amine", "carboxy to carbamoyl", "carboxy ester to carbamoyl",
        "carboxylic acid to acid chloride", "fischer-speier",
        "steglich", "friedel-crafts acylation",
        "passerini reaction", "ugi reaction", "ugi four-component",
    ]
    if any(k in rc for k in acylation_keywords):
        return "Acylation"

    # --- Functional Group Interconversion ---
    fgi_keywords = [
        "halogenation", "bromination", "chlorination", "iodination",
        "fluorination", "dehalogenation", "hydroxy to chloro",
        "appel", "dast", "sulfanyl to sulfonyl",
        "azido to amino", "cyano to carbamoyl",
        "nitration", "sulfonation",
        "epoxide ring opening", "acid-catalyzed epoxide",
        "hydroformylation", "oxymercuration-demercuration",
        "norris", "norish",
        "paterno-buchi", "ritter reaction",
        "jacobsen hydrolytic kinetic resolution",
        "schmidt glycosylation", "schmidt reaction",
        "bamford-stevens", "pinner reaction",
        "elimination", "krapcho decarboxylation", "decarboxylation",
        "bergman cyclization",
    ]
    if any(k in rc for k in fgi_keywords):
        return "Functional Group Interconversion"

    # --- Aromatic Heterocycle Formation ---
    het_cycle_keywords = [
        "furan synthesis", "pyrrole synthesis", "pyridine synthesis",
        "indole synthesis", "pyrazole synthesis", "tetrazole synthesis",
        "heterocycle", "paal-knorr", "hantzsch", "knorr",
        "biginelli", "gewald", "pictet-spengler",
        "1,3-dipolar cycloaddition",
    ]
    if any(k in rc for k in het_cycle_keywords):
        return "Aromatic Heterocycle Formation"

    # --- Rearrangements ---
    rearrangement_keywords = [
        "rearrangement", "claisen", "cope ", "beckmann", "curtius",
        "favorskii", "fries", "overman", "pummerer", "ramberg-backlund",
        "smiles rearrangement", "pinacol rearrangement", "polonovski",
        "wharton", "hofmann", "lossen",
    ]
    if any(k in rc for k in rearrangement_keywords):
        return "Functional Group Interconversion"

    return "OTHER"


def get_coarse_options_str() -> str:
    """Return the 9 coarse reaction classes as a formatted string for prompts."""
    return "\n".join(f"  {i+1}. {cls}" for i, cls in enumerate(COARSE_CLASSES))
