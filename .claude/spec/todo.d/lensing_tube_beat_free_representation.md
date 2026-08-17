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
