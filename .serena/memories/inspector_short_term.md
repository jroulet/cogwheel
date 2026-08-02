# Inspector Short-Term Observations

## Build 7 — _GHOST_SEPARATION_MIN Part 0 Resolution (2026-08-xx)

### Scope
Review of WP1: resolve the Part 0 question for `_GHOST_SEPARATION_MIN = 0.7`.
The plan determined the constant does NOT violate Part 0 (it's a lens-plane
quantity in Einstein-radius units tracing to geometry, not the prior box).
Work was: (1) add Part 0 resolution comment to `channels.py`, (2) update
`COVERAGE_DESIGN.md` table entry from SUSPECT to OK, (3) bonus: add a
5-test `GhostGateOrthogonalityWitnessTestCase` proving the decay and
separation gates are independent.

### Findings

**No new findings introduced by this build.**

The production code change is PURELY documentation (11 new comment lines;
value 0.7 unchanged, no behavioral change). The test addition is structurally
correct (5/5 new tests pass, gamma=5.0 saddle witness has Im(tau_c)=0.502
passing decay, separation=0.600 failing separation). The COVERAGE_DESIGN.md
table edit is factually consistent with FINDINGS.md F027 data (saddle admit_min
0.942 from [0.942, 2.421] sweep).

### Pre-existing issue carried forward

**INS-8-001 (pre-existing, NOT introduced by this build):**
`test_raising_constant_to_two_refuses_an_admit_config` fails because
ADMIT_CONFIGS[0]=(0.50, 45.0, 0.65) has separation=2.012 > 2.0. Patching
MIN=2.0 doesn't refuse it. This was introduced in the Build 6 C5 pass when
the offset was bumped 0.60→0.65 for decay-gate margin, and has been failing
since. Fix: use ADMIT_CONFIGS[1] (sep=1.82) or raise patched value to 2.1.

### Carried Forward (Librarian scope)
- INS-5-001: SPEC.md lines 53/97-137 still reference old annulus — Librarian.
- INS-5-003: DATA_CONTRACTS.yaml line 228 'caustic-frame annulus rho' — Librarian.

### Production Code Assessment
No behavioral change. The constant value 0.7 is unchanged. All 22 references
to the constant across other test files are unaffected. The comment block
accurately cites F027's driver measurement (refuse max 0.29 from in-test
configs, admit min 0.94 from the FINDINGS.md saddle sweep). The tripwire
constants (SEP_REFUSE_MAX=0.5, SEP_ADMIT_MIN=1.0) are correctly cited as live
assertions in the GhostGateBoundaryTestCase.

### Verdict
**PASS** — no new issues introduced. Pre-existing INS-8-001 remains but is
not caused by this build.
