"""Layer 2 evaluator for MolEdit: template structure compliance + internal self-consistency.

Zero RDKit, zero GT comparison.
V-points check format adherence and self-consistency only.
"""

# ---------------------------------------------------------------------------
# Step field definitions per subtask (used for V1: structure completeness)
# ---------------------------------------------------------------------------

STEP_FIELDS = {
    "add_v2": [
        "step1_anchor_idx",
        "step1_anchor_element",
        "step1_leaving_smiles",
        "step2_frag_smiles",
        "step2_heavy_atoms",
        "step3_product_smiles",
        "step4_n_heavy_src",
        "step4_n_heavy_prod",
        "step4_heavy_delta",
        "step5_n_rings_src",
        "step5_n_rings_prod",
        "step5_ring_delta",
    ],
    "delete_v2": [
        "step1_anchor_idx",
        "step1_anchor_element",
        "step1_remove_group",
        "step2_remove_smiles",
        "step2_heavy_atoms",
        "step3_product_smiles",
        "step4_n_heavy_src",
        "step4_n_heavy_prod",
        "step4_heavy_delta",
        "step5_n_rings_src",
        "step5_n_rings_prod",
        "step5_ring_delta",
    ],
    "substitute_v2": [
        "step1_anchor_idx",
        "step1_anchor_element",
        "step1_remove_group_smiles",
        "step1_add_fragment_smiles",
        "step2_remove_heavy",
        "step3_add_heavy",
        "step4_product_smiles",
        "step5_n_heavy_src",
        "step5_n_heavy_prod",
        "step5_heavy_delta",
        "step6_n_rings_src",
        "step6_n_rings_prod",
        "step6_ring_delta",
    ],
}

# Expected step names per subtask (for V2)
STEP_NAMES = {
    "add_v2": [
        "ANCHOR_IDENTIFICATION",
        "FRAGMENT_IDENTIFICATION",
        "PRODUCT_CONSTRUCTION",
        "HEAVY_ATOM_VERIFICATION",
        "RING_VERIFICATION",
    ],
    "delete_v2": [
        "ANCHOR_IDENTIFICATION",
        "GROUP_SIZE_VERIFICATION",
        "PRODUCT_CONSTRUCTION",
        "HEAVY_ATOM_VERIFICATION",
        "RING_VERIFICATION",
    ],
    "substitute_v2": [
        "ANCHOR_IDENTIFICATION",
        "REMOVE_GROUP_SIZE",
        "ADD_FRAGMENT_SIZE",
        "PRODUCT_CONSTRUCTION",
        "HEAVY_ATOM_VERIFICATION",
        "RING_VERIFICATION",
    ],
}


def evaluate_record(record: dict, subtask: str) -> dict:
    """Evaluate a single MolEdit record for Layer 2 V-points."""
    step_fields = STEP_FIELDS[subtask]
    step_names = STEP_NAMES[subtask]
    edit_type = subtask.replace("_v2", "")

    raw_steps = record.get("raw_output_steps", []) or []

    # ── V1: all step fields present (non-None) ──
    v1 = all(record.get(f) is not None for f in step_fields)

    # ── V2: each step has correct Step Name in raw_output_steps ──
    v2 = True
    if raw_steps and len(raw_steps) >= len(step_names):
        for i, expected_name in enumerate(step_names):
            step_text = raw_steps[i] if i < len(raw_steps) else ""
            if f"[{expected_name}]" not in step_text:
                v2 = False
                break
    else:
        v2 = False

    # ── V3: each step contains "FORMAL:" line ──
    v3 = True
    if raw_steps and len(raw_steps) >= len(step_names):
        for i in range(len(step_names)):
            step_text = raw_steps[i] if i < len(raw_steps) else ""
            if "FORMAL:" not in step_text:
                v3 = False
                break
    else:
        v3 = False

    # ── V4: Answer line exists ──
    answer = record.get("answer_smiles")
    v4 = answer is not None and str(answer).strip() != ""

    # ── V5: product_smiles == answer (self-consistency) ──
    product_key = "step4_product_smiles" if subtask == "substitute_v2" else "step3_product_smiles"
    product = record.get(product_key)
    v5 = False
    if product is not None and answer is not None:
        v5 = str(product).strip() == str(answer).strip()

    # ── V6: heavy_delta arithmetic self-consistency ──
    src_key = "step5_n_heavy_src" if subtask == "substitute_v2" else "step4_n_heavy_src"
    prod_key = "step5_n_heavy_prod" if subtask == "substitute_v2" else "step4_n_heavy_prod"
    delta_key = "step5_heavy_delta" if subtask == "substitute_v2" else "step4_heavy_delta"

    n_src = record.get(src_key)
    n_prod = record.get(prod_key)
    delta = record.get(delta_key)
    v6 = False
    if all(isinstance(x, (int, float)) for x in (n_src, n_prod, delta)):
        v6 = (n_prod - n_src) == delta

    state_score = (int(v1) + int(v2) + int(v3) + int(v4) + int(v5) + int(v6)) / 6.0

    return {
        "V1": int(v1),
        "V2": int(v2),
        "V3": int(v3),
        "V4": int(v4),
        "V5": int(v5),
        "V6": int(v6),
        "state_score": round(state_score, 4),
    }
