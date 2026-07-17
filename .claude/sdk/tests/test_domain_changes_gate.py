"""Tests for the `has_domain_changes` plan field and the gates it drives.

Restores A's `has_physics_changes` semantics under B's domain_* naming: a
domain-sensitive change with NO new tests must (a) be kept off the Foreman-Lite
fast path and (b) still trigger the post-build Professor review.

Run (from the repo root, under the project's interpreter):
    python -m unittest discover -s .claude/sdk/tests -p 'test_*.py'
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sdk.gates import is_fast_path_eligible
from sdk.orchestrator import BuildOrchestrator
from sdk.schemas import Plan, WorkPackage


def _plan(**kwargs):
    defaults = dict(
        summary="s", work_packages=[],
        has_domain_tests=False, has_domain_changes=False,
        has_new_public_api=False, has_spec_update=False,
        files_affected=["cogwheel/x.py"],
    )
    defaults.update(kwargs)
    return Plan(**defaults)


class FastPathEligibilityTest(unittest.TestCase):

    def test_baseline_is_eligible(self):
        self.assertTrue(is_fast_path_eligible(_plan()))

    def test_domain_change_without_tests_blocks_fast_path(self):
        # The core regression: a small, test-free, domain-sensitive change
        # (e.g. a likelihood/tolerance fix) must NOT auto-route to fast-path.
        self.assertFalse(is_fast_path_eligible(_plan(has_domain_changes=True)))

    def test_domain_tests_still_block_fast_path(self):
        self.assertFalse(is_fast_path_eligible(_plan(has_domain_tests=True)))

    def test_new_public_api_still_blocks(self):
        self.assertFalse(is_fast_path_eligible(_plan(has_new_public_api=True)))

    def test_too_many_files_blocks(self):
        self.assertFalse(is_fast_path_eligible(
            _plan(files_affected=["a.py", "b.py", "c.py"])))


class ParseRoundTripTest(unittest.TestCase):

    def test_parse_reads_has_domain_changes(self):
        o = BuildOrchestrator.__new__(BuildOrchestrator)
        data = {
            "summary": "fix likelihood normalization",
            "work_packages": [
                {"id": "WP1", "title": "t", "what": "w",
                 "where": ["cogwheel/likelihood.py"], "how": "h",
                 "who": "Coder", "max_turns": 40},
            ],
            "has_domain_tests": False,
            "has_domain_changes": True,
            "has_new_public_api": False,
            "has_spec_update": False,
            "files_affected": ["cogwheel/likelihood.py"],
        }
        plan = o._parse_plan_from_dict(data)
        self.assertTrue(plan.has_domain_changes)
        self.assertFalse(plan.has_domain_tests)

    def test_parse_defaults_domain_changes_false_when_absent(self):
        o = BuildOrchestrator.__new__(BuildOrchestrator)
        data = {"summary": "s", "work_packages": [], "files_affected": []}
        plan = o._parse_plan_from_dict(data)
        self.assertFalse(plan.has_domain_changes)


if __name__ == "__main__":
    unittest.main()
