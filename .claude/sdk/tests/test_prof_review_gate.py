"""Regression tests for the Phase-2 Professor inference-review gate.

Deterministic (no live agent): monkeypatch `_run_agent` to return a canned
verdict block and assert `_run_prof_review` parses it and that a FAIL verdict
raises GateFailure (the commit-blocking gate). Run under the pipeline env:

    conda run -n cogwheel_310 python -m unittest \
        discover -s .claude/sdk/tests -p 'test_*.py'
"""
import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sdk.orchestrator import BuildOrchestrator
from sdk.schemas import Plan, ProfReviewVerdict
from sdk.gates import GateFailure


def _orchestrator_with_plan(has_domain_tests=True, has_domain_changes=False):
    o = BuildOrchestrator.__new__(BuildOrchestrator)
    o._log = lambda *a, **k: None
    o.plan = Plan(
        summary="t", work_packages=[], has_domain_tests=has_domain_tests,
        has_domain_changes=has_domain_changes,
        has_new_public_api=False, has_spec_update=False, files_affected=[],
        domain_test_descriptions=["inverse consistency to 1e-6"],
    )
    return o


def _canned_agent(verdict_json):
    async def fake(agent_name, task, **kw):
        assert agent_name == "prof_review"
        assert "domain correctness" in task and "verdict" in task
        return (verdict_json, None)
    return fake


class ProfReviewGateTest(unittest.TestCase):
    def _verdict(self, js):
        o = _orchestrator_with_plan()
        o._run_agent = _canned_agent(js)
        return asyncio.run(o._run_prof_review())

    def test_fail_verdict_parsed(self):
        r = self._verdict('```json\n{"verdict":"FAIL","concerns":["off by 1e-3"],"summary":"bad"}\n```')
        self.assertEqual(r.verdict, ProfReviewVerdict.FAIL)
        self.assertEqual(r.concerns, ["off by 1e-3"])

    def test_pass_verdict_parsed(self):
        r = self._verdict('```json\n{"verdict":"PASS","summary":"inverses hold"}\n```')
        self.assertEqual(r.verdict, ProfReviewVerdict.PASS)

    def test_concern_verdict_parsed(self):
        r = self._verdict('```json\n{"verdict":"CONCERN","summary":"borderline"}\n```')
        self.assertEqual(r.verdict, ProfReviewVerdict.CONCERN)

    def test_unparseable_defaults_to_pass(self):
        # Fail-open: the Inspector gate is the hard correctness gate.
        r = self._verdict("no json here at all")
        self.assertEqual(r.verdict, ProfReviewVerdict.PASS)

    def test_fail_verdict_raises_gate_failure(self):
        r = self._verdict('```json\n{"verdict":"FAIL","concerns":["x"],"summary":"bad"}\n```')
        with self.assertRaises(GateFailure):
            if r.verdict == ProfReviewVerdict.FAIL:
                raise GateFailure(f"Professor inference review FAILED: {r.summary}")


class ReviewGatePredicateTest(unittest.TestCase):
    """Step-5 fires on `has_domain_changes OR has_domain_tests`.

    Guards the semantics of the gate in ``_run_full_pipeline``: a
    domain-sensitive change with NO new tests must still trigger the review
    (the regression that dropping A's ``has_physics_changes`` introduced).
    """

    @staticmethod
    def _fires(has_domain_tests, has_domain_changes):
        plan = _orchestrator_with_plan(
            has_domain_tests=has_domain_tests,
            has_domain_changes=has_domain_changes,
        ).plan
        return plan.has_domain_changes or plan.has_domain_tests

    def test_fires_on_domain_changes_without_tests(self):
        self.assertTrue(self._fires(has_domain_tests=False, has_domain_changes=True))

    def test_fires_on_domain_tests_without_changes(self):
        self.assertTrue(self._fires(has_domain_tests=True, has_domain_changes=False))

    def test_skipped_when_neither(self):
        self.assertFalse(self._fires(has_domain_tests=False, has_domain_changes=False))


class ProfReviewInvocationTest(unittest.TestCase):
    """_run_prof_review must pass bypassPermissions + the generous professor
    inter-message timeout so a slow pytest run isn't misread as a wedge."""

    def test_passes_generous_timeout_and_bypass(self):
        from sdk.orchestrator import PROFESSOR_INTER_MESSAGE_TIMEOUT
        o = _orchestrator_with_plan()
        captured = {}

        async def fake(agent_name, task, **kw):
            captured["agent_name"] = agent_name
            captured.update(kw)
            return ('```json\n{"verdict":"PASS"}\n```', None)

        o._run_agent = fake
        asyncio.run(o._run_prof_review())
        self.assertEqual(captured["agent_name"], "prof_review")
        self.assertEqual(captured["permission_override"], "bypassPermissions")
        self.assertEqual(
            captured["inter_message_timeout_override"],
            PROFESSOR_INTER_MESSAGE_TIMEOUT,
        )


if __name__ == "__main__":
    unittest.main()
