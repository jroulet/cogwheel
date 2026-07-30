"""An unscoped domain-test spec must COUNT, not ride along for free.

F057 (1e-farfield, 2026-07-30): 10 of 11 domain-test descriptions did not
literally name a `test_<x>.py` file, so `_group_test_specs` routed them to
`cross_suite` — appended to the agent's prompt as real work, but invisible to
both the shard cap and the `60 + 20*n` budget. The sharder logged "1 spec(s)"
and handed an 11-spec load an 80-turn budget. Two Test Developers died at
error_max_turns with ZERO output, and the shard cap written to prevent exactly
that death never engaged, because the quantity it keys on was collapsed
upstream.

These assert the VALUES the sharder reads — group sizes, shard counts, turn
budgets — not which branch produced them.

Run (from the repo root, under the project's interpreter):
    python -m unittest discover -s .claude/sdk/tests -p 'test_*.py'
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sdk.orchestrator import BuildOrchestrator

GROUP = BuildOrchestrator._group_test_specs
SHARD = BuildOrchestrator._shard_specs
BUDGET = BuildOrchestrator._test_dev_budget

# The real shape: one spec quotes the suite file, the rest describe substantive
# tests without naming it.
NAMED = "byte-identity via a frozen golden table in test_lensing_surrogate.py"
UNSCOPED = [f"ACCEPTANCE ({c}) — a substantive test with no filename in it"
            for c in "abcdefghij"]


class UnscopedSpecsAreCounted(unittest.TestCase):

    def test_all_eleven_reach_the_one_suite_in_play(self):
        groups, cross = GROUP([NAMED] + UNSCOPED)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(next(iter(groups.values()))), 11)
        self.assertEqual(cross, [])

    def test_the_shard_cap_now_engages(self):
        groups, _ = GROUP([NAMED] + UNSCOPED)
        shards = SHARD(next(iter(groups.values())))
        self.assertEqual(len(shards), 4)          # 11 specs, cap 3/agent
        self.assertTrue(all(len(s) <= 3 for s in shards))

    def test_the_budget_reflects_the_real_load(self):
        # 80 turns was what the dead agents got; the load needs the cap.
        self.assertEqual(BUDGET(1), 80)
        self.assertEqual(BUDGET(11), 250)

    def test_a_genuinely_universal_requirement_stays_cross_suite(self):
        # Two named suites: an unscoped spec is ambiguous, so it must stay
        # cross-suite rather than being arbitrarily assigned to one of them.
        groups, cross = GROUP([
            "check A in test_alpha.py",
            "check B in test_beta.py",
            "ALL SUITES: no bare asserts",
        ])
        self.assertEqual(sorted(groups), ["test_alpha.py", "test_beta.py"])
        self.assertEqual(cross, ["ALL SUITES: no bare asserts"])

    def test_nothing_named_still_degrades_to_one_unscoped_group(self):
        groups, cross = GROUP(["do a thing", "do another"])
        self.assertEqual(list(groups), ["(unscoped)"])
        self.assertEqual(len(groups["(unscoped)"]), 2)
        self.assertEqual(cross, [])

    def test_universal_rule_stays_cross_suite_even_with_ONE_group(self):
        # The distinction is universal-vs-substantive, NOT named-vs-unnamed.
        # Folding a universal rule into the single group would apply it to one
        # shard instead of all of them — the first attempt at this fix did
        # exactly that and test_test_dev_split caught it.
        groups, cross = GROUP([
            NAMED,
            "ALL FOUR SUITES -- style guard: 79 columns + ast.parse",
            "ACCEPTANCE (a) — a substantive test with no filename",
        ])
        self.assertEqual(len(next(iter(groups.values()))), 2)   # named + (a)
        self.assertEqual(len(cross), 1)
        self.assertIn("style guard", cross[0])

    def test_the_predicate_separates_rules_from_tests(self):
        universal = BuildOrchestrator._is_universal_requirement
        for rule in ("ALL SUITES: no bare asserts",
                     "ALL FOUR SUITES -- style guard",
                     "every suite must import numpy"):
            self.assertTrue(universal(rule), rule)
        for test in ("ACCEPTANCE (a) — held-out eps insensitivity",
                     "DRY assertion — coordinate equals the primitive",
                     "EXCLUSION BALL measured in the (s,d) coordinate"):
            self.assertFalse(universal(test), test)

    def test_one_named_suite_alone_is_unaffected(self):
        groups, cross = GROUP(["check A in test_alpha.py"])
        self.assertEqual(groups, {"test_alpha.py": ["check A in test_alpha.py"]})
        self.assertEqual(cross, [])


class TheOldBehaviourWasTheDefect(unittest.TestCase):
    """Contrast control: reproduce the pre-fix grouping and show it starves.

    Without this, the assertions above only show the new path works — not that
    the old one was broken. A test that has never been seen to fail on a
    known-bad input has not been tested.
    """

    @staticmethod
    def _legacy_group(specs):
        """`_group_test_specs` as it stood before F057."""
        import re
        groups, cross = {}, []
        for spec in specs:
            m = re.search(r"\b(test_\w+\.py)\b", spec)
            if m:
                groups.setdefault(m.group(1), []).append(spec)
            else:
                cross.append(spec)
        if not groups and cross:
            groups["(unscoped)"] = list(cross)
            cross = []
        return groups, cross

    def test_legacy_saw_one_spec_where_there_were_eleven(self):
        groups, cross = self._legacy_group([NAMED] + UNSCOPED)
        self.assertEqual(len(next(iter(groups.values()))), 1)
        self.assertEqual(len(cross), 10)          # 10 invisible to the cap

    def test_legacy_budgeted_80_turns_for_a_250_turn_load(self):
        groups, cross = self._legacy_group([NAMED] + UNSCOPED)
        counted = len(next(iter(groups.values())))
        self.assertEqual(BUDGET(counted), 80)
        self.assertEqual(BUDGET(counted + len(cross)), 250)

    def test_legacy_never_sharded(self):
        groups, _ = self._legacy_group([NAMED] + UNSCOPED)
        self.assertEqual(len(SHARD(next(iter(groups.values())))), 1)


if __name__ == "__main__":
    unittest.main()
