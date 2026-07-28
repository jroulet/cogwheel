---
date: 2026-07-28
---

### Fixed: the Inspector revision loop had no termination path for trivial-only findings (.claude/ only)

`gates.should_escalate` computed `has_actionable` over IMPLEMENTATION and
DESIGN findings only, so a build with trivial-only findings past the revision
cap was permanently `False` — escalation did not misfire, there was no
termination path at all. Observed live: a build ran to `revision 8/2` against
a cap of 2, reproducing the identical `2 trivial, 0 impl, 0 design` from
revision 3 onward, ~26 minutes and ~$24 with zero implementation findings
outstanding.

`gates.revision_budget_spent` now terminates the loop at ANY severity once the
cap is spent, carrying remaining findings into the change report instead of
escalating them. `gates.finding_signature` stops the loop earlier when two
consecutive revisions produce an identical finding set (order-insensitive,
reword-sensitive), catching a stuck implementation finding too. The
orchestrator now logs finding text per revision, not just counts. These
changes are agent-infrastructure only, under `.claude/`, and are excluded from
the `main`-branch sync.
