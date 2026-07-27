# Build 8h-b5 — Finish the 8h-b4 test port and verify the admission repair

## Mission

Build 8h-b4's PRODUCTION work is committed and sound (bc27d39); its
build died in test development with the suite red. This build finishes
the tests and VERIFIES the repair's central claim. No production
redesign — fix production only where a test exposes a genuine defect.

1. **Verify the admission repair's core claim (highest value).** The
   old scalar-reach exterior admission covers 0.944 / 0.632 / 0.271 /
   **0.000** of the true `{eta >= eta_max}` region for gamma bands
   (0.30,0.40) / (0.50,0.70) / (0.70,0.80) / (0.80,0.90) — driver-
   measured, 4000 quasi-uniform samples per band against the exact
   `geometry.nearest_caustic_point` oracle. Above gamma~0.85 the
   exclusion circle (5.74) exceeds the whole prior box (4.24) so ZERO
   exterior tiles were built though ~3040 sampled points per band are
   genuinely exterior. Write the acceptance test that the NEW
   per-column `_InteriorAdmission.admits_exterior` path recovers
   **>= 0.95** coverage in EVERY band including (0.80,0.90), plus the
   **exact-zero-false-admit** invariant (sample >= 5x5 across each
   admitted tile's interior; ZERO points with true caustic distance
   < eta_max) and the reachable-red (restore the scalar test -> the
   0.80-0.90 band must collapse to zero tiles). The oracle is CHEAP
   (0.09 ms/call, driver-measured): 1e4 probes x 4 bands ~ 6 s. This
   is a fast in-build test, not a sweep.
2. **Finish the test port.** `cogwheel/tests/test_lensing_exterior_
   windows.py` carries 14 failed + 12 errors; the other four suites
   carry 4 between them. Most of the failing tests in that file were
   AUTHORED BY THE DEAD BUILD (`SelfFalsificationTestCase`,
   `InteriorDirectionalAdmissionTestCase`) and were never green — they
   are incomplete, not regressed. Bring them to green against the
   CURRENT API, or, where a test's premise is genuinely retired,
   migrate its intent (deletions need a one-line justification each).

## Root-cause patterns already found (do not re-derive)

- **`_InteriorAdmission` has NO `rho_boundary` attribute** — the name
  appears only in a module comment. It stores PHYSICAL directional
  radii per band gamma (`radius_grid`, `(n_gamma, n_theta)`), and the
  interior coordinate is `rho = |y| / _caustic_reach`, so the
  band-conservative boundary is
  `radius_grid.min(axis=0) / _caustic_reach(gamma_mid)` (verified: min
  0.318 == the isotropic inradius the suite contrasts against, max
  0.858). ALREADY FIXED in `_rho_boundary`; check for other uses.
- **Band-edge gamma placement (a PATTERN, check every fixture).**
  `_train` derives a chart's `rho` range from the raw box corners at
  `gamma_mid` ONLY, but `rho = 1 + |y| - r_caustic(gamma, theta_c)`
  shifts with gamma — so a config at a band-edge gamma lands OUTSIDE
  the chart's range even with its `y1` inside the raw box, and the
  surrogate declines it. Fix by solving each config's `y1` so it lands
  at a chosen fraction of the chart's ACTUAL rho span AT ITS OWN
  gamma. ALREADY FIXED for `POS_CONFIGS`; the same trap applies
  anywhere a fixture places configs across a gamma band.
- **Notch semantics.** The first remaining failure asserts
  `rho_scalar < 1.0` and gets 1.027: exterior admission is
  scalar-`rho > 1 + margin` while the directional radius is smaller,
  so the "notch" annulus near cusps is physically exterior yet
  scalar-interior. Any test constructing a notch point must solve it
  against BOTH radii, not assume one.
- Also open: 2 `GhostDomainError` cases (`channels.py:898`) and one
  threshold assert (`0.72 >= 2.0`).

## Out of scope — hard fences

- NO production redesign; NO coordinate/axis changes (the Professor
  ruled exterior(+1)/interior/tube axes correct as shipped, and the
  saddle additive switch is already done). NO campaign, NO qd, NO Born
  rung, NO tolerance weakening.
- `EnvelopeReconstructionTestCase` positive-box eps: 12 of 13 held-out
  points pass (median 3.3e-2); ONE interior outlier at 2.6e-1 against
  a 0.2 budget-calibrated tolerance. Do NOT widen the tolerance to go
  green. Either find the local cause or leave it failing and REPORT —
  a documented red beats a silently loosened gate.

## Test-execution discipline (binding)

Run tests PER FILE with a bounded timeout; do NOT loop the full
five-suite battery to chase green (that is what exhausted the previous
build's agents — thirteen of them). The driver runs the final tally.

## Acceptance (two-tier)

1. In-build (FAST): the admission coverage test passes >= 0.95 in every
   band including (0.80,0.90) with the exact-zero-false-admit invariant
   and its reachable-red; `test_lensing_exterior_windows.py` green;
   the other four suites' 4 stragglers green or documented; tube
   byte-identity holds.
2. POST-BUILD (driver): tree gate, commit verified 8h-b4+b5, then the
   serving census.
