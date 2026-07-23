# Professor short-term (Build 8h-b2 ghost-kernel INFERENCE REVIEW, 2026-07-23)

Reviewed `cogwheel/tests/test_lensing_ghost.py` (new, untracked) against the
8h-b2 ghost/complex-saddle spec. Env: `cogwheel-newlal` (python3.10). Ran
`pytest test_lensing_ghost.py -v`: **36 passed, 1 xfailed, 22s**.

## Verified PASS
- Oracle branch test non-circular: FORBIDDEN_ORACLE_NAMES covers all spec names
  (_ghost_kernel, _saddle_metric, _c1/_c2_polynomial, saddle_coefficients,
  image_kernel, delay, hessian, magnification, morse_index + extractors);
  AST-walked on Name.id/Attribute.attr (not substring); enforced on ALL
  _ORACLE_FUNCTIONS and self-falsified (tainted oracle flagged).
- Tolerances match pinned spec EXACTLY: tau |.|/arg 1e-6; amp analytic 1e-6,
  FD 1e-4 mag + 1e-4 rad; reconstruction 1e-10; far |C|<1e-3, ratio<0.5,
  Im tau_c>8; on-axis no-growth/unit-carrier 1e-12.
- Mutation/self-falsification suite all reachable-red: wrong log branch breaks
  delay gate; Morse double-count rotates amp phase by pi; det-floor removal ->
  garbage amp; argmin would pick GROWING member; growing conjugate blows up far
  carrier; byte-identity catches 1-ulp; anti-vacuity teardown fails a silent
  sweep. Gates are NOT vacuous.
- Real-image byte-identity: 4 tests pass -> ghost additions changed no real path.
- Reconstruction x_c.x_c=1/u_c holds (validates recon map, NOT branch — correct
  framing, matches my earlier ruling).

## One divergence (CONCERN-level, physically DEFENSIBLE — not a fail)
Spec wanted kernel to EVALUATE finitely EXACTLY on-axis (|Im tau_c|<1e-10, unit
carrier). Implementation REFUSES exactly on-axis (GhostDomainError): diagonal
source-aligned matrix collapses onto removable singularity u=a22, and a genuine
ghost cannot have |Im u|<root_tolerance(3e-7) (declassifies to real image).
Test reframes: certifies near-axis LIMIT (monotone Im tau_c->0, carrier->1 from
below, no spurious growth, decaying member) + pins exactly-on-axis refusal +
marks literal contract @expectedFailure (will XPASS when a future build adds the
on-axis limit treatment). PHYSICS: sound & safe — on-axis is a genuine
complex-saddle/real-axis coalescence; refusing (named exception) beats returning
garbage. Spec's own "gate that refuses to USE it is next build's job" is met more
conservatively (refuse to EVALUATE). Operator should note on-axis limit is
DEFERRED, not delivered.

## Deferred to operator (out of turn budget): heavy full-sampling / real-data
None triggered here — all ghost tests are single-point/short and fast.

VERDICT: PASS (with noted on-axis deferral).
