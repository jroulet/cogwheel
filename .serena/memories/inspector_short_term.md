# Inspector Short-Term Observations

## Build 6 C5 — Ghost Decay Gate, Pass 4 (2026-08-xx)

### Scope
Re-review after Coder's FOURTH attempt at WP1: add ghost decay gate to
`farfield_ghost_term` (F027) and update affected test suites. The production
code is correct (fixed constant 0.4, w-independent, physics-aligned). Coder
updated test_lensing_ghost_gate.py ADMIT_CONFIGS, test_lensing_born.py theta
range, and parts of test_lensing_exterior_windows.py — but left residual
test breakage.

### Assessment of Previous Findings

**INS-7-001 — PARTIALLY RESOLVED.** Down from 11 failures + 8 errors to
5 failures + 4 errors. The following remain:

1. `test_lensing_ghost_gate.py::test_raising_constant_to_two_refuses_an_admit_config`
   — ADMIT_CONFIGS[0]=(0.50,45.0,0.65) has separation=2.0116 (not "~1.98" as
   the inline comment claims), so patching MIN=2.0 doesn't refuse it (2.0116 >= 2.0
   is True). Reachable-red is inert.
2. `test_lensing_born.py::test_admitting_ghost_inflates_residual_and_node_count`
   — narrowed theta range (0.20,0.6) excludes the low-theta ghost-inflation
   region; now nodes_ghost == nodes_ppgo == 2.
3. `test_lensing_exterior_windows.py::GhostFrameCollapseTestCase` (2 tests +
   1 diagnostic plot): COLLAPSE_PROBES[0] Im(tau_c)=0.319 and [2] Im(tau_c)=0.187
   are below 0.4 → GhostDomainError from farfield_envelope_from_partition.
4. `test_lensing_exterior_windows.py::SelfFalsificationTestCase::test_raw_frame_ghost_leaves_residual_uncollapsed`
   — uses COLLAPSE_PROBES[0], same Im(tau_c)=0.319 failure.

**INS-7-002 — RESOLVED.** The cherry-picked "Im tau_c >= 0.9" docstring has
been replaced with the accurate "e.g. 0.69–0.87 at typical admitted test
configs; on-axis near-cusp configs have Im tau_c ~ 0.001".

### New Findings (this pass)

**INS-8-001 (BUG — test breakage)**: test_lensing_ghost_gate.py
`test_raising_constant_to_two_refuses_an_admit_config` fails because
ADMIT_CONFIGS[0]=(0.50, 45.0, 0.65) has actual separation=2.0116, not
"~1.98" as the inline comment says. Patching MIN=2.0 doesn't cause a refusal.
Fix: use a different ADMIT_CONFIG with sep between 0.7 and 2.0 (e.g. use
ADMIT_CONFIGS[1] which has sep=1.82), or raise the patched value to 2.1.

**INS-8-002 (BUG — test breakage)**: test_lensing_born.py
`test_admitting_ghost_inflates_residual_and_node_count` fails because
the narrowed theta range (0.20, 0.6) no longer contains the ghost-inflation
region that creates extra spline nodes. nodes_ghost == nodes_ppgo == 2.
Fix: either (a) widen theta range lower bound to include a region where the
ghost actually inflates the node count (possibly theta < 0.20 but above the
decay gate), or (b) restructure the test to check residual magnitude
inflation only (which does pass, as shown by the assertGreater on inflation
not failing).

**INS-8-003 (BUG — test breakage)**: test_lensing_exterior_windows.py
COLLAPSE_PROBES[0] (gamma=0.9, theta=45, offset=0.6, Im(tau_c)=0.319) and
COLLAPSE_PROBES[2] (gamma=0.9, theta=75, offset=0.6, Im(tau_c)=0.187) are
below the decay threshold. This breaks:
- GhostFrameCollapseTestCase.test_fixed_frame_collapses_where_raw_frame_is_wrong
- GhostFrameCollapseTestCase.test_collapse_diagnostic_plot
- SelfFalsificationTestCase.test_raw_frame_ghost_leaves_residual_uncollapsed

Fix: COLLAPSE_PROBES need configs with Im(tau_c) > 0.4. Since these tests
exercise the ghost FRAME COLLAPSE behavior (fixed vs raw frame), they need
configs the decay gate admits. Raise the offsets to push Im(tau_c) above 0.4,
or pick entirely new configs. The affected probes MUST have the ghost
admitted so the minus-ghost label assembles.

### Carried Forward
- INS-5-001: SPEC.md lines 53/97-137 still reference old annulus — Librarian.
- INS-5-003: DATA_CONTRACTS.yaml line 228 'caustic-frame annulus rho' — Librarian.

### Pattern
Same as last pass: the GATE-CONTRACT swap breaks sibling suites that encode
the old contract. The Coder fixed ~60% of the breakage but left the remaining
~40% — specifically the exterior-windows COLLAPSE_PROBES (unchanged), the
ghost_gate reachable-red test (wrong separation estimate for the new config),
and the born test's node-count assertion (theta range too narrow to exercise
the ghost inflation).

### Production Code Assessment
The production code (channels.py decay gate + geometry.py docstring) is
CORRECT and complete. All findings are test-only.

### Verdict
**ISSUES** — 5 test failures + 4 anti-vacuity errors remain. The decay gate
implementation itself is correct; the test fixture updates are incomplete.
