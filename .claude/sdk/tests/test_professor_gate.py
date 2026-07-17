"""Tests for the Professor-consult plan gate (gw planning-stage parity).

Run (from the repo root, under the project's interpreter):
    python -m unittest discover -s .claude/sdk/tests -p 'test_*.py'
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sdk.gates import verify_plan
from sdk.schemas import Plan, WorkPackage


def _plan(**kwargs):
    wp = WorkPackage(
        id="WP1", title="t", what="w", where=["cogwheel/x.py"], how="h",
        who="Coder", verification="v", max_turns=40,
    )
    defaults = dict(
        summary="s", work_packages=[wp],
        has_domain_tests=False, has_domain_changes=False,
        has_new_public_api=False,
        has_spec_update=False, files_affected=[],
        simplifier_inputs=["lean"],
    )
    defaults.update(kwargs)
    return Plan(**defaults)


class ProfessorGateTest(unittest.TestCase):

    def test_required_and_missing_fails(self):
        failures, _ = verify_plan(_plan(), require_professor=True)
        self.assertTrue(any("Professor" in f for f in failures))

    def test_required_and_cited_passes(self):
        failures, _ = verify_plan(
            _plan(professor_inputs=["tolerance must be 1.5 nats"]),
            require_professor=True)
        self.assertFalse(any("Professor" in f for f in failures))

    def test_not_required_missing_is_fine(self):
        failures, _ = verify_plan(_plan(), require_professor=False)
        self.assertFalse(any("Professor" in f for f in failures))

    def test_default_does_not_require(self):
        failures, _ = verify_plan(_plan())
        self.assertFalse(any("Professor" in f for f in failures))


if __name__ == "__main__":
    unittest.main()
