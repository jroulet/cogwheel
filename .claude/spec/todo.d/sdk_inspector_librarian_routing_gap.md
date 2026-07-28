---
section: Backlog
---

- **Inspector has no way to file "valid, but another role owns it"**
  `[housekeeping]` — every additive-capability build burns its revision cap on
  the same doc-sync argument. Observed on the saddle lobe-serve build
  (2026-07-28): INS-S2-001 (SPEC.md / DATA_CONTRACTS.yaml stale w.r.t. the new
  `LobeInteriorChart` kind) was raised, overridden, re-raised, and overridden
  again — hitting revision 2/2 with ZERO implementation findings outstanding.

  This is not confusion about roles. The Inspector knew: its own finding says
  "Not a code bug; Librarian owns the sync." Three structural causes:

  1. **Its contract orders it to raise this.** `.claude/crew/inspector.md`
     check 2 makes SPEC/DATA_CONTRACTS a bidirectional invariant it OWNS:
     "If an artifact introduces something not described in the spec or
     contracts ... flag the inconsistency ... Do not assume which — report the
     finding with both interpretations so it can be triaged upstream." A new
     chart kind is exactly that. The Inspector is obeying its contract.
  2. **The severity taxonomy cannot express the resolution.** Only
     `bug` / `design` / `trivial`, all of which imply Coder rework. "Correct,
     but the docs trail it, and a different role syncs them" has no category,
     so it lands in `design` and triggers Architect triage.
  3. **The routing rule is invisible to the role that needs it.** The rule
     ("doc-sync/SPEC findings -> post-gate Librarian with exact replacement
     text") lives in `architect_knowledge`, the Architect's PRIVATE memory.
     `librarian` appears ZERO times in `inspector.md`.

  The loop is therefore deterministic, not stochastic: the diff does not
  change between passes, so the Inspector re-derives the same true fact and
  files it in the same category, while the Architect overrides it with a rule
  the Inspector cannot read. It terminates only by exhausting the cap.

  SAME BUG CLASS AS THE CODE DEFECTS THIS PROJECT KEEPS HITTING: a convention
  held at one site that another site cannot see. The delay frame lived at four
  sites before 8h-b7 (one wrong, worth a 174x-607x residual); `r_deltoid` was
  about to live at two before the 2026-07-28 plan gate caught it; this routing
  rule lives in one agent's memory while another agent is contractually
  obliged to trip over it.

  Fix, cheapest first:
  (a) Put the routing rule in `.claude/crew/inspector.md` itself. The
      Librarian's own contract ALREADY draws the line — `librarian.md` says
      the Inspector owns spec/contract ACCURACY as checkable invariants while
      the Librarian owns SYNC to downstream surfaces. The Inspector was simply
      never told the half that concerns it. State it in both contracts or in a
      shared fragment both read.
  (b) Add a severity/disposition such as `deferred` or `other-role` (with an
      `owner` field) so a valid finding can be recorded, routed, and NOT
      re-litigated. Overriding should suppress it for subsequent passes on an
      unchanged diff.

  Worth doing before the next additive build: the cost is one wasted Inspector
  pass plus an Architect triage per revision, on every build that adds a
  capability — which is most of them.
