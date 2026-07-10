"""Task registry and base configuration for all evaluation tasks."""
import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(os.environ.get("CHEMCOT_OUTPUT_DIR", PROJECT_ROOT / "results"))


@dataclass(frozen=True)
class TaskSpec:
    """Specification for a single task (e.g. mol_edit, rxn_pred, mol_und, mol_opt)."""
    name: str
    subtasks: set[str]
    # Subtask -> module name under formal_cot/{task}/ (may differ from subtask name)
    module_map: dict[str, str]
    # Subtask -> dataset filename override (default: clean_dataset.json)
    dataset_overrides: dict[str, str]
    # Primary record identity fields used for GT alignment and rerun deduplication.
    id_fields: list[str]
    # Whether this task uses formal_cot module structure for prompt/parser/verifier
    uses_formal_cot: bool = True
    # Whether this task uses the generic PromptBuilder pattern
    uses_prompt_builder: bool = True


TASK_REGISTRY: dict[str, TaskSpec] = {
    "mol_edit": TaskSpec(
        name="mol_edit",
        subtasks={"add_v2", "delete_v2", "substitute_v2"},
        module_map={
            "add_v2": "add_v2",
            "delete_v2": "delete_v2",
            "substitute_v2": "substitute_v2",
        },
        dataset_overrides={},
        id_fields=["orig_id", "id", "sample_id"],
    ),
     "rxn_pred": TaskSpec(
         name="rxn_pred",
         subtasks={
             "forward", "retro", "nepp", "byproduct",
             "condition_ranking",
             "rcr_catalyst", "rcr_reagent", "rcr_solvent",
             "rxn_template", "mech_sel", "yield_pred",
         },
         module_map={
             "forward": "forward",
             "retro": "retro",
             "nepp": "nepp",
             "byproduct": "byproduct_fixed",
             "condition_ranking": "condition_ranking",
             "rcr_catalyst": "rcr_catalyst",
             "rcr_reagent": "rcr_reagent",
             "rcr_solvent": "rcr_solvent",
             "rxn_template": "rxn_template",
            "mech_sel": "mech_sel",
            "yield_pred": "yield_pred",
        },
        dataset_overrides={
            "condition_ranking": "eval_dataset_shuffled.json",
        },
        id_fields=["pool_id", "id", "src_id", "sample_id"],
    ),
    "mol_und": TaskSpec(
        name="mol_und",
        subtasks={
            "fg_detect", "ring_count", "murcko_scaffold",
            "ring_sys_scaffold", "smiles_equivalent",
        },
        module_map={
            "fg_detect": "fg_detect",
            "ring_count": "ring_count",
            "murcko_scaffold": "murcko_scaffold",
            "ring_sys_scaffold": "ring_sys_scaffold",
            "mutated": "mutated",
            "permutated": "permutated",
            "smiles_equivalent": "smiles_equivalent",
        },
        dataset_overrides={},
        id_fields=["orig_idx", "idx", "id", "sample_id"],
    ),
    "mol_opt": TaskSpec(
        name="mol_opt",
        subtasks={
            "logp",
            "qed",
            "solubility",
            "drd",
            "gsk",
            "jnk",
            "logp_qed",
            "logp_solubility",
            "qed_solubility",
            "drd_logp",
            "drd_solubility",
            "gsk_logp",
        },
        module_map={},
        dataset_overrides={},
        id_fields=["idx", "id", "sample_id"],
        uses_formal_cot=False,
        uses_prompt_builder=False,
    ),
}


def get_task_spec(task: str) -> TaskSpec:
    if task not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task}. Must be one of {set(TASK_REGISTRY)}")
    return TASK_REGISTRY[task]


def resolve_module_name(task: str, subtask: str) -> str:
    spec = get_task_spec(task)
    return spec.module_map.get(subtask, subtask)


def resolve_dataset_name(task: str, subtask: str) -> str:
    spec = get_task_spec(task)
    return spec.dataset_overrides.get(subtask, "clean_dataset.json")


def resolve_gt_dataset_path(task: str, subtask: str) -> Path:
    """Resolve PRM GT dataset path for formal_cot-based tasks."""
    spec = get_task_spec(task)
    module_name = resolve_module_name(task, subtask)
    ds_name = resolve_dataset_name(task, subtask)
    return (
        PROJECT_ROOT
        / "results"
        / "formal_cot"
        / task
        / module_name
        / ds_name
    )


def resolve_output_dir(task: str, subtask: str, model_name: str) -> Path:
    safe_model = model_name.replace("/", "_")
    output_dir = OUTPUT_ROOT / "evaluation" / task / subtask / safe_model
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
