"""An unreadable Inspector result must TERMINATE the revision loop.

`ISSUES` with an empty findings list was an unbreakable loop: the gate keeps
it spinning while every exit requires a non-empty list to fire. The build
`analytic_caustic_reach` died there on 2026-07-30 (revision 3/2, no report, no
commit), and `gates.py` records the same shape at revision 8/2 two days
earlier — the guard added then kept the `bool(findings)` requirement, so the
hole survived inside its own fix.

These assert the VALUES the loop reads (verdict, findings, and what each
terminator returns), not which branch produced them.

Run (from the repo root, under the project's interpreter):
    python -m unittest discover -s .claude/sdk/tests -p 'test_*.py'
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sdk.gates import (
    check_inspector_gate,
    finding_signature,
    revision_budget_spent,
    should_escalate,
)
from sdk.orchestrator import MAX_REVISION_LOOPS, BuildOrchestrator
from sdk.schemas import EscalationLevel, InspectorVerdict


def _parse(text, round_number=0):
    """Call the real parser without constructing a whole orchestrator."""
    orch = BuildOrchestrator.__new__(BuildOrchestrator)
    orch._log = lambda *a, **k: None
    return BuildOrchestrator._parse_inspector_result(
        orch, text, round_number=round_number)


# The shape a wedge retry produces: a real review, no JSON verdict block.
UNPARSEABLE = (
    "I reviewed the diff and ran the caustic tests.\n"
    "The reach formula matches the brute-force scan.\n"
)


class ParseFailureTerminatesLoop(unittest.TestCase):

    def test_unparseable_output_yields_a_finding_not_an_empty_issues(self):
        result = _parse(UNPARSEABLE)
        self.assertEqual(result.verdict, InspectorVerdict.ISSUES)
        self.assertEqual(len(result.findings), 1)

    def test_the_finding_is_actionable_so_the_loop_can_escalate(self):
        # TRIVIAL would let revision_budget_spent mark the build PASS, which
        # ships work whose inspection was never readable.
        finding = _parse(UNPARSEABLE).findings[0]
        self.assertEqual(finding.severity, EscalationLevel.IMPLEMENTATION)

    def test_every_terminator_now_fires(self):
        findings = _parse(UNPARSEABLE).findings
        over = MAX_REVISION_LOOPS + 1
        self.assertTrue(should_escalate(findings, over))
        self.assertTrue(revision_budget_spent(findings, over))
        self.assertTrue(bool(finding_signature(findings)))

    def test_none_of_them_fired_on_the_old_empty_result(self):
        # The contrast control. Without it the assertions above only show that
        # a non-empty list works, not that the empty one was the defect.
        over = MAX_REVISION_LOOPS + 1
        self.assertFalse(should_escalate([], over))
        self.assertFalse(revision_budget_spent([], over))
        self.assertFalse(bool(finding_signature([])))

    def test_signature_is_round_invariant_so_repeats_are_detected(self):
        # finding_id embeds the round; the signature must not, or a recurring
        # parse failure looks like a new finding every revision and the
        # non-convergence check can never fire.
        self.assertEqual(finding_signature(_parse(UNPARSEABLE, 1).findings),
                         finding_signature(_parse(UNPARSEABLE, 2).findings))

    def test_an_explicit_pass_is_still_a_clean_pass(self):
        result = _parse("Everything checks out.\nPASS\n")
        self.assertEqual(result.verdict, InspectorVerdict.PASS)
        self.assertEqual(result.findings, [])
        self.assertTrue(check_inspector_gate(result))

    def test_a_json_verdict_still_wins_over_the_fallback(self):
        result = _parse('```json\n{"verdict": "PASS", "findings": []}\n```')
        self.assertEqual(result.verdict, InspectorVerdict.PASS)
        self.assertEqual(result.findings, [])


if __name__ == "__main__":
    unittest.main()
