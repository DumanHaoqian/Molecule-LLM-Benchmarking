import unittest

from molbench.benchmarks.chemcotbench.benchmark import (
    ChemCoTV1Task,
    _parse_smiles,
    _query_source_smiles,
    _reference_count,
)
from molbench.metrics.chemcotbench_v1.metrics import (
    edit_score,
    molecule_pair,
    optimization_score,
    scaffold_pair,
)


class ChemCoTV1MetricsTest(unittest.TestCase):
    def test_molecule_exact_match_uses_chemical_identity(self):
        score = molecule_pair("C(C)O", "CCO")
        self.assertTrue(score["valid"])
        self.assertEqual(score["exact_match"], 1.0)
        self.assertEqual(score["morgan_sims"], 1.0)

    def test_edit_operations_use_functional_group_deltas(self):
        added = edit_score("CC", "CCC=O", "add", "aldehyde", None)
        deleted = edit_score("CC=O", "CC", "delete", None, "aldehyde")
        substituted = edit_score(
            "CC=O", "CC(=O)O", "sub", "carboxyl", "aldehyde"
        )
        self.assertEqual(added["success"], 1.0)
        self.assertEqual(deleted["success"], 1.0)
        self.assertEqual(substituted["success"], 1.0)

    def test_qed_optimization_is_scored_without_tdc(self):
        score = optimization_score("c1ccccc1", "Oc1ccccc1", "qed")
        self.assertTrue(score["valid"])
        self.assertIn("improvement", score)
        self.assertIn("scaffold_hard", score)

    def test_scaffold_reference_scores_hard_and_soft_match(self):
        score = scaffold_pair("c1ccccc1", "c1ccccc1")
        self.assertEqual(score["scaffold_hard"], 1.0)
        self.assertEqual(score["scaffold_soft"], 1.0)

    def test_task_postprocessors_accept_official_json_fields(self):
        count = ChemCoTV1Task("mol_und", "fg_count", "count", "/tmp")
        optimize = ChemCoTV1Task("mol_opt", "qed", "optimization", "/tmp")
        choice = ChemCoTV1Task("reaction", "mechsel", "choice", "/tmp")
        self.assertEqual(count.postprocess('{"count": 3}'), 3)
        self.assertEqual(
            optimize.postprocess('{"Final Target Molecule": "CCO"}'), "CCO"
        )
        self.assertEqual(choice.postprocess('{"answer": "G"}'), "G")

    def test_all_official_smiles_json_fields_are_parsed(self):
        self.assertEqual(_parse_smiles('{"Output Scaffold": "c1ccccc1"}', "molecule"), "c1ccccc1")
        self.assertEqual(_parse_smiles('{"pred_smi": "CCO"}', "molecule"), "CCO")
        self.assertEqual(_parse_smiles('{"SMILES": "N"}', "molecule"), "N")
        self.assertEqual(
            _parse_smiles('{"Reactants": "CCO.O"}', "molecule"), "CCO.O"
        )

    def test_source_smiles_are_normalized_from_official_queries(self):
        edit = "Input Molecule: CC=O, Functional Group to delete: aldehyde."
        optimize = "Source Molecule: Cn1ccnc1S."
        self.assertEqual(_query_source_smiles(edit, "mol_edit"), "CC=O")
        self.assertEqual(_query_source_smiles(optimize, "mol_opt"), "Cn1ccnc1S")

    def test_count_reference_accepts_released_gt_field(self):
        self.assertEqual(_reference_count({"gt": 3}, "fg_count"), 3)


if __name__ == "__main__":
    unittest.main()
