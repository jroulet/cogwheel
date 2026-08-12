---
date: 2026-08-12
section: Backlog
---

### Doc/spec findings defer to the Librarian at ANY severity

Closes `todo.d/sdk_inspector_librarian_routing_gap.md`.

The Inspector's own contract (check 2) makes `SPEC.md` and
`DATA_CONTRACTS.yaml` a bidirectional invariant it OWNS, and tells it to
"report the finding with both interpretations so it can be triaged upstream".
So on every build that ADDS a capability, it correctly reports that the spec
does not yet describe the new thing. It is obeying its contract.

F050 already handled this — but only for `TRIVIAL` findings:

    deferred = [f for f in trivial_findings
                if f.file.startswith('.claude/spec/')
                or 'librarian' in f.suggested_fix.lower()]

The identical finding classified `DESIGN` missed that filter entirely, and
DESIGN is a worse place to land: it costs an Architect triage every round AND
blocks the graceful exit, whose condition is
`revision_budget_spent(...) and not impl_findings and not design_findings`.
The only remaining exit is budget exhaustion followed by a human escalation.

**Measured, twice, identically.** 2026-07-28 (saddle lobe-serve, INS-S2-001)
and 2026-08-12 (deltoid exterior, INS-5-001): each burned 3/2 revision rounds
plus a driver escalation on ONE non-blocking finding whose own `suggested_fix`
began "Librarian scope:". In the 2026-08-12 case the Inspector explicitly
wrote "Non-blocking, no code defect" and the loop escalated anyway.

**Fix.** Hoist the F050 predicate above the severity split (Tier 0.4), so a
finding on a `.claude/spec/` file, or one routed to the Librarian by its own
`suggested_fix`, is deferred to the Librarian stage regardless of severity.
Severity is simply the wrong axis: a spec file is the Librarian's to edit no
matter how serious the divergence. The Tier 0.5 trivial-only block stays as a
backstop for findings whose severity is rewritten mid-loop.

This implements the fragment's option (b) — a disposition that records and
routes a valid finding without re-litigating it — WITHOUT the schema change it
proposed. No new severity value, no Inspector-contract edit, no change to what
the Inspector reports. The finding is still raised and still reaches the
Librarian; it just stops being blocking. Option (a) (teach the Inspector the
routing rule in `.claude/crew/inspector.md`) is now unnecessary for cost
reasons, though it would still reduce noise.

**Verified** against both historical findings plus controls: INS-5-001 and
INS-S2-001 defer; a librarian-routed non-spec file defers; a real code bug in
`cogwheel/lensing/surrogate.py` and a missing-test finding do NOT defer.
