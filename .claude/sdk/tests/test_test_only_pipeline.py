"""Tests for the Coder-free compatibility-port execution route."""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path


CLAUDE_DIR = Path(__file__).resolve().parents[2]
if str(CLAUDE_DIR) not in sys.path:
    sys.path.insert(0, str(CLAUDE_DIR))

from sdk.gates import verify_plan
from sdk.orchestrator import BuildOrchestrator
from sdk.schemas import BuildMode, BuildReport, Plan


def _test_only_plan(**overrides) -> Plan:
    defaults = dict(
        summary="Port existing test fixtures to the current API.",
        work_packages=[],
        has_domain_tests=True,
        has_domain_changes=False,
        has_new_public_api=False,
        has_spec_update=False,
        files_affected=["cogwheel/tests/test_lensing_surrogate.py"],
        domain_test_descriptions=[
            "test_lensing_surrogate.py: port existing fixture construction; "
            "preserve its oracle and tolerance."
        ],
        simplifier_inputs=["No production change is required."],
        professor_inputs=["Preserve the existing numerical claim."],
        is_test_only=True,
    )
    defaults.update(overrides)
    return Plan(**defaults)


class TestOnlyPlanTests(unittest.TestCase):
    def test_test_only_plan_with_explicit_test_work_is_valid(self):
        failures, missing_turns = verify_plan(
            _test_only_plan(), require_professor=True
        )

        self.assertEqual(failures, [])
        self.assertEqual(missing_turns, [])

    def test_test_only_plan_requires_explicit_test_developer_work(self):
        plan = replace(
            _test_only_plan(), has_domain_tests=False,
            domain_test_descriptions=[],
        )

        failures, _ = verify_plan(plan)

        self.assertIn(
            "Test-only plan must provide explicit Test Developer descriptions.",
            failures,
        )

    def test_normal_empty_plan_remains_rejected(self):
        plan = replace(_test_only_plan(), is_test_only=False)

        failures, _ = verify_plan(plan)

        self.assertIn("Plan has no work packages.", failures)

    def test_parser_preserves_test_only_route(self):
        orchestrator = BuildOrchestrator.__new__(BuildOrchestrator)
        plan = orchestrator._parse_plan_from_dict({
            "summary": "test-only",
            "work_packages": [],
            "has_domain_tests": True,
            "has_domain_changes": False,
            "has_new_public_api": False,
            "has_spec_update": False,
            "files_affected": [],
            "domain_test_descriptions": ["test_example.py: port fixture."],
            "is_test_only": True,
        })

        self.assertTrue(plan.is_test_only)
        self.assertEqual(plan.work_packages, [])

    def test_test_only_dag_has_test_developer_then_inspector(self):
        orchestrator = BuildOrchestrator.__new__(BuildOrchestrator)
        orchestrator.plan = _test_only_plan()
        report = BuildReport(
            mode=BuildMode.FULL, work_packages_completed=0,
            work_packages_total=0,
        )

        dag = orchestrator._build_phase2a_dag(report)

        self.assertEqual([node.name for node in dag], ["test_dev", "inspector"])
        self.assertEqual(dag[0].depends_on, [])
        self.assertEqual(dag[1].depends_on, ["test_dev"])


if __name__ == "__main__":
    unittest.main()
