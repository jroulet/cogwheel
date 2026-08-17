"""The plan gate must survive Python-style string concatenation in JSON.

Measured 2026-08-17 (tube_beat_free_representation, third launch): the
Architect emitted a COMPLETE plan (9k output tokens, end_turn) whose one
long `how` string was written as `"...part one. "\n + "part two..."` —
implicit-concatenation muscle memory from Python. `+` between literals is
not JSON; `_parse_plan` raised GateFailure and the build died with $9.80
of consultation on the floor. The repair collapses unescaped `" + "`
seams (unescaped quotes are always string delimiters in JSON) as a
last-resort fallback after plain parsing fails.

Run (from the repo root, under the project's interpreter):
    python -m unittest discover -s .claude/sdk/tests -p 'test_*.py'
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sdk.orchestrator import BuildOrchestrator


def _parse(text):
    orch = object.__new__(BuildOrchestrator)
    orch._log = lambda _msg: None
    return BuildOrchestrator._try_parse_json_plan(orch, text)


class ConcatenatedPlanStringsParse(unittest.TestCase):

    def test_the_real_failure_shape_parses_and_joins_the_value(self):
        # The measured seam: quote, newline, indentation, `+`, quote —
        # inside a fenced block, mid-value.
        text = (
            "Consultations complete. Here is the build plan.\n\n"
            "```json\n"
            "{\n"
            '  "summary": "recovery build",\n'
            '  "work_packages": [{"id": "WP1", "title": "t",\n'
            '    "what": "first piece. "\n'
            '        + "second piece.",\n'
            '    "where": "f.py", "how": "h", "who": "coder"}],\n'
            '  "has_domain_tests": false\n'
            "}\n"
            "```\n"
        )
        plan = _parse(text)
        self.assertIsNotNone(plan, "repaired plan must parse")
        # Value assertion: the seam joins into ONE string, content intact.
        self.assertEqual(plan.work_packages[0].what,
                         "first piece. second piece.")

    def test_escaped_quotes_and_prose_plus_are_not_mangled(self):
        # A value legitimately containing quoted code around a `+` — the
        # inner quotes arrive ESCAPED, so the repair must not touch them.
        text = (
            "```json\n"
            "{\n"
            '  "summary": "s",\n'
            '  "work_packages": [{"id": "WP1", "title": "t",\n'
            '    "what": "assert \\"a\\" + \\"b\\" stays literal",\n'
            '    "where": "f.py", "how": "h", "who": "coder"}],\n'
            '  "has_domain_tests": false\n'
            "}\n"
            "```\n"
        )
        plan = _parse(text)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.work_packages[0].what,
                         'assert "a" + "b" stays literal')


if __name__ == "__main__":
    unittest.main()
