"""Regression tests for the per-suite Test Developer split.

A single Test Developer handed 20 dense specs across 4 numerical suites with a
flat 120-turn budget delivered ONE suite and exhausted its turns (2026-07-16);
the Inspector then failed the build for missing coverage. _run_test_dev_agent
now runs one agent per suite named in domain_test_descriptions, budgeted by
spec count. These tests pin the grouping and budgeting contract.

Run (from the repo root, under the project's interpreter):
    python -m unittest discover -s .claude/sdk/tests -p 'test_*.py'
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sdk.orchestrator import BuildOrchestrator


class GroupTestSpecsTest(unittest.TestCase):

    def test_specs_group_by_named_suite(self):
        specs = [
            "SUITE test_lensing_geometry.py -- CSV regression ...",
            "SUITE test_lensing_hyp1f1.py -- prefactor oracle ...",
            "SUITE test_lensing_geometry.py -- Morse census ...",
            "SUITE test_lensing_channels.py -- reconstruction ...",
        ]
        groups, cross = BuildOrchestrator._group_test_specs(specs)
        self.assertEqual(
            sorted(groups),
            ["test_lensing_channels.py", "test_lensing_geometry.py",
             "test_lensing_hyp1f1.py"],
        )
        self.assertEqual(len(groups["test_lensing_geometry.py"]), 2)
        self.assertEqual(cross, [])

    def test_suiteless_specs_become_cross_suite(self):
        specs = [
            "SUITE test_lensing_operator.py -- F_op oracle ...",
            "ALL FOUR SUITES -- style guard: 79 columns + ast.parse ...",
        ]
        groups, cross = BuildOrchestrator._group_test_specs(specs)
        self.assertEqual(sorted(groups), ["test_lensing_operator.py"])
        self.assertEqual(len(cross), 1)
        self.assertIn("style guard", cross[0])

    def test_all_suiteless_degrades_to_single_unscoped_run(self):
        specs = ["check thing A end to end", "check thing B end to end"]
        groups, cross = BuildOrchestrator._group_test_specs(specs)
        self.assertEqual(sorted(groups), ["(unscoped)"])
        self.assertEqual(len(groups["(unscoped)"]), 2)
        self.assertEqual(cross, [])

    def test_empty_specs_yield_nothing(self):
        groups, cross = BuildOrchestrator._group_test_specs([])
        self.assertEqual(groups, {})
        self.assertEqual(cross, [])

    def test_first_named_suite_wins_within_one_spec(self):
        # A spec that references a sibling suite in prose still belongs to
        # the suite it names first (its own).
        specs = [
            "SUITE test_lensing_channels.py -- reuse the guard idiom from "
            "test_lensing_gauge.py",
        ]
        groups, _ = BuildOrchestrator._group_test_specs(specs)
        self.assertEqual(sorted(groups), ["test_lensing_channels.py"])


class TestDevBudgetTest(unittest.TestCase):

    def test_budget_scales_with_spec_count(self):
        self.assertEqual(BuildOrchestrator._test_dev_budget(4), 140)
        self.assertEqual(BuildOrchestrator._test_dev_budget(6), 180)
        self.assertGreater(
            BuildOrchestrator._test_dev_budget(6),
            BuildOrchestrator._test_dev_budget(4),
        )

    def test_budget_is_capped(self):
        self.assertEqual(BuildOrchestrator._test_dev_budget(100), 250)

    def test_single_dense_suite_fits_the_observed_cost(self):
        # Empirics: ~120 turns bought ~one 6-spec suite. The budget for
        # 6 specs must clear that observation with headroom.
        self.assertGreaterEqual(BuildOrchestrator._test_dev_budget(6), 120)


if __name__ == "__main__":
    unittest.main()
