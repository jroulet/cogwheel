---
date: 2026-07-28
section: Backlog
---

- **Inspector revision loop could not terminate on trivial-only findings** —
  FIXED. Observed live on the Born carrier build: the log counter itself
  showed `revision 8/2` against a cap of 2, with every pass from revision 3
  onward reporting the identical `2 trivial, 0 impl, 0 design`. ~26 minutes
  and ~$24 of Inspector + foreman_lite cycles, ZERO implementation findings
  outstanding across all eight reviews. The driver killed it and finished the
  build by hand.

  ROOT CAUSE, one predicate (`gates.should_escalate`):

      has_actionable = any(f.severity in (IMPLEMENTATION, DESIGN)
                           for f in findings)
      return has_actionable and loop_count > MAX_REVISION_LOOPS

  `TRIVIAL` is not in that set, so with trivial-only findings the expression
  is permanently False. Escalation did not misfire — there was NO TERMINATION
  PATH AT ALL. Inspector raises, foreman-lite fixes, Inspector re-raises,
  forever.

  The conceptual error: `should_escalate` answers "does a human need to
  decide?" while the loop needed "may we go round again?". Trivial findings
  answer NO to both, but only the first question was being asked. Excluding
  TRIVIAL from escalation is CORRECT — by their own definition ("style,
  missing tests, minor inefficiency — fix at convenience") they must never
  block a build or consume a human decision. The bug was reusing that
  predicate as the loop guard.

  FIXES:
  * `gates.revision_budget_spent(findings, loop_count)` — true when the loop
    must STOP at ANY severity. Trivial-only past the cap now terminates,
    proceeds, and carries the findings into the change report rather than
    escalating them.
  * `gates.finding_signature(findings)` — order-insensitive, reword-sensitive
    identity of a finding SET. Two consecutive revisions with the same
    signature means the fixer cannot or will not clear it, so the loop stops
    immediately instead of spending the remaining budget re-deriving it. This
    guard is severity-AGNOSTIC, so it also catches an implementation finding
    that would previously have burned the full budget before escalating.
  * The orchestrator now logs the finding TEXT
    (`[severity] id file: description`) at every revision. Counts alone cannot
    distinguish a converging loop from a stuck one — `2 trivial, 0 impl,
    0 design` eight times says nothing about WHAT — which is why this ran to
    eight passes before anyone opened a transcript.

  Verified by direct call: `should_escalate(trivial, 8)` stays False
  (unchanged, correct); `revision_budget_spent(trivial, 8)` is True and
  `(trivial, 1)` is False; `should_escalate(impl, 8)` stays True; an identical
  finding set re-raised yields an equal signature while a reworded one does
  not.

  RELATED, still open: [[sdk_inspector_librarian_routing_gap]] — a finding
  having no correct DISPOSITION, distinct from the loop not terminating.
  Both surfaced as builds burning budget on findings everyone agreed about.
