---
date: 2026-08-14
section: Lensing serving
---

**Tube-chart serving folded into the D2 fundamental domain (owner
directive: "when there is a symmetry, we have to use it")** `[→ spec]` —
build `tube_d2_fold` + driver fix-forward. Tube queries now serve through
a D2 gauge-image search (`surrogate._tube_theta_inframe`: images theta,
pi-theta, -theta, pi+theta, identity first, first inside the chart frame
serves; fundamental-domain path bit-identical to the incumbent), closing
the residual F079 exposure — a trained arc's three D2 mirrors serve
through it on BOTH parities (mirror octants equal at rtol 1e-12; the
reflected float angle reaches the spline, so bit-exactness holds only on
the identity path). Astroid tube training restricted to the single
pi/4-bracketing arc (4 -> 1, ~4x training cut); saddle training keeps
`arcs[:max_tube_arcs]` (trim owed: `lensing_saddle_tube_fundamental_
training`). Census inherits through the production functions unchanged.

BUILD HISTORY, instructive: the original design (Professor-derived
sign-keyed fold of the gauge angle) had exactly-right ARITHMETIC on a
false PREMISE — the gauge<->source map is orientation-reversing (the
pi/4 gauge arc serves y1_eig < 0 sources) and near-cusp source regions
span gauge slivers of three arcs, so no single-arc sign fold can work
(measured 0/10 held-out served, eps NaN). The build's own bit-exact
octant pins PASSED against that broken design — self-consistent tests
cannot see a wrong premise; only the tree-wide gate could, and the
watchdog killed the build at gate start (the gate streams to a sidecar
log; 22-25 min under load vs the 1200 s staleness threshold — second
healthy-build kill of the day at Professor-PASS + ~1201 s). Fixes that
came out of it: orchestrator gate heartbeat (Popen + 60 s gate-log-growth
lines into the build log), and the fix-forward image-search design
(diagnosis and fix probe-verified driver-side; previously-failing
clusters 21/21 green, full gate green at commit). Escalation INS-2
(stale census mirror + dead max_tube_arcs knob) was fixed in-build; the
knob returned as the live saddle control in the final design.
