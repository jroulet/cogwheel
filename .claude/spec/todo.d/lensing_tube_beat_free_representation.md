---
section: Backlog
---

- **TUBE BEAT-FREE REPRESENTATION — demodulate by BOTH fold carriers;
  the theta node count collapses** `[→ spec]` — owner direction
  2026-08-17, redirecting the graduation follow-up. The graduation
  build's measurement PROVED the coordinate was not the bottleneck:
  30 delay-uniformized nodes (eps ~0.145) barely beat 24 arc-length
  nodes (0.146), because the arc's beat structure was already
  near-uniform — no parametrization removes a beat. The stored envelope
  oscillates as cos(w * Delta_tau) because TWO carriers e^{i w tau_pm}
  are demodulated by ONE. Fix the REPRESENTATION: demodulate by both
  fold carriers (the switched-channel machinery already carries
  resolved pairs this way; Delta_tau is closed form from the cascade —
  fully analytic), storing a beat-free residual whose structure does
  not scale with w. Expected: theta nodes collapse to the
  smooth-variation scale (~5-10), passing the F083 bar (0.0237) at a
  FRACTION of the 48-node brute baseline — the no-explosion vision
  realized for the tube. Also owed in this build (the graduation
  build's Professor-flagged gaps, deferred at its commit): (a) the
  unguarded ValueError that crashes the graduated builder on every
  real production arc (verified four bands) — root-cause and fix;
  (b) the F083 falsification actually RUN (the accuracy half, not the
  tautological node-count half). The Nyquist-count machinery and the
  n_theta_cap become the SECONDARY safety net under the beat-free
  representation, not the primary mechanism; do not raise the cap to
  chase the bar. Schema: the two-carrier demodulation changes the
  stored envelope definition -> envelope-definition tag + contracts
  fragment (the FARFIELD_KERNEL_SUM_MINUS_GHOST precedent). Blocks
  tube training and the (f_max, f_floor) sweep.

  GRADUATION BUILD POST-MORTEM (2026-08-17; its tree-gate-red work is
  REVERTED, diff archived at
  `.claude/handoff/tube_graduation_salvage/working_tree.patch` — cherry-
  pick its schema/hard-refusal/coverage-reporting pieces, not its axis):
  (a) the crash's true name is "Tube delay map is not strictly
  increasing" — cumulative total variation is FLAT at Delta_tau's
  mid-arc extremum (|dDelta_tau/dtheta| = 0), so the discrete s' table
  has equal consecutive entries on every real arc and strict-
  monotonicity inversion fails; any axis built on TV(Delta_tau) must
  handle its stationary point (moot under the beat-free representation,
  which needs no such axis — a cautionary pin, not a task); (b) the
  measured 0.145 @ 30 s'-nodes vs 0.146 @ 24 s-nodes is the empirical
  proof the coordinate was not the bottleneck; (c) the gate also caught
  a lobe-path regression from the graduation edits
  (`LobeUCoorDBoundShiftMarginTestCase` eps-stability 0.84 vs 0.01) and
  a new absorber-guard hit in surrogate_training — both REVERTED with
  the tree; the beat-free build must leave the lobe path byte-identical
  and pre-clear any new constant against the part0 guard.
