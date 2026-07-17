"""Tests for the plan-gate depth banner (CLAUDE.md 'SDK Build Briefs').

Run (from the repo root, under the project's interpreter):
    python -m unittest discover -s .claude/sdk/tests -p 'test_*.py'
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sdk.gates import plan_depth_banner


def _plan(n_wps, filler=""):
    parts = ["## Mission\nFix the thing.\n"]
    for i in range(1, n_wps + 1):
        parts.append(f"### WP{i}: do part {i}\n- step\n")
    parts.append(filler)
    return "\n".join(parts)


class PlanDepthBannerTest(unittest.TestCase):

    def test_counts_wps(self):
        self.assertIn("3 WP(s)", plan_depth_banner(_plan(3)))

    def test_no_warning_at_or_below_three(self):
        for n in (1, 3):
            self.assertNotIn("WARNING", plan_depth_banner(_plan(n)))

    def test_warning_above_three(self):
        banner = plan_depth_banner(_plan(4))
        self.assertIn("4 WP(s)", banner)
        self.assertIn("WARNING", banner)
        self.assertIn("sequential builds", banner)

    def test_wp_mentions_in_prose_do_not_count(self):
        plan = _plan(2, filler="Refer to WP1 and ### WP-like prose but "
                               "not a header; also `## WP99` inside code "
                               "is preceded by backtick not line start.")
        self.assertIn("2 WP(s)", plan_depth_banner(plan))

    def test_reports_size_kb(self):
        self.assertRegex(plan_depth_banner(_plan(1)), r"\d+\.\d KB")


if __name__ == "__main__":
    unittest.main()
