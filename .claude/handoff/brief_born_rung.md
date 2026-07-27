# Build 8h-c1: Born rung — analytic cover for the low-w far zone

## Mission

Add the Born (weak-deflection) analytic rung to the serving ladder so the
outer corner of the prior box is served WITHOUT quadrature. Certified-or-refuse
like every other rung: it ships with a measured validity boundary and refuses
outside it.

This is the last missing rung. Owner ruling 2026-07-23: NON-NEGOTIABLE, because
the low-w far zone varies on the EINSTEIN scale, so trained tiles out there are
prior-sized and therefore prior-dependent — a trained cover cannot be
prior-universal, and only an analytic rung can.

## Measured facts (driver-verified 2026-07-27; agents must NOT re-derive)

1. THE GAP. Prior support reaches the box corner `|y| = 3.0 * sqrt(2) = 4.2426`
   (`prior.py:77,82,103`: `_source_scale(m) = min(307/m, 3.0)`, saturated for
   `m <= 102.33 Msun`). Production exterior charts stop at
   `source_magnitude_max = _source_scale(m_lo) = 3.0`
   (`surrogate_training.py:3293,3321`). The uncovered annulus is
   `3.0 < |y| <= 4.2426`, entirely at low mass hence low `w`.
2. THE EXPANSION POINT IS NOT 1. `F(w -> 0) = sqrt(mu_macro) =
   1/sqrt((1-kappa)^2 - gamma^2)`, pinned to 7.85e-9 by
   `test_lensing_operator.py::MacroMagnificationLimitTestCase`
   (`FINDINGS.md:284-289,469-478`). The handoff phrase "the deep-diffraction
   floor where F -> 1" (`greenfield_audit.md:15`) is true ONLY at
   `gamma = kappa = 0`. A series expanded about 1 is wrong everywhere the
   shear is nonzero — i.e. everywhere this rung serves.
3. THE ORACLE IS AVAILABLE THROUGHOUT. The DD ceiling is `w * |y| <= 60`
   (`_hyp1f1.py:121`, 1e-10 accuracy to `w*|y| ~ 50`). At the box corner that
   permits `w <= 14.1`. The prior never approaches it: at `m = 10 Msun` where
   `Y` saturates, `w in [0.0248, 1.27]` over 20-1024 Hz, so `w*|y| <= 5.4`.
   Every Born claim in the target region is therefore CERTIFIABLE against the
   exact engine. Born is needed for prior-universality, NOT because the oracle
   fails.
4. THE SLOT. `likelihood.py::_surrogate_coefficients` (`:1471-1717`), after
   `surrogate.serve` returns falsy (`:1639`) and before `return None`. The
   surrogate intercept is `_amplification_coefficients` (`:1783-1787`).
5. `W_CEILING_SCHWINGER = 60.0` (`_schwinger.py:119`) — hard refuse above.
   `ASTROID_WALL = 443.7`, `SADDLE_WALL = 58.0` (`ppgo_map.py:184-185`).
6. Census attribution lives in `surrogate_census.classify_fallthrough`
   (`:214-280`); today this region lands in the `out-of-box` bucket.
7. Zero Born code and zero Born spec exist. `.claude/spec/` has never mentioned
   it; the only authority is `.claude/handoff/lensing/build8hb3_brief.md:42-53`.
   This build is the first time Born enters the spec.
8. NAMING HAZARD: `far-field` / `FarFieldChart` / `farfield_*` throughout this
   repo means "trained chart OUTSIDE the caustic in the far-field GAUGE", NOT
   the weak-deflection far field. Do not overload those names.

## In scope

- The Born amplification series itself: an analytic `F_born(w, y, gamma, kappa)`
  valid at large `|y|` / low `w`, expanded about `sqrt(mu_macro)` (fact 2).
  The Professor owns the derivation, its truncation order, and its error bound.
- Its MEASURED validity boundary in `(|y|, w, gamma)`: the region where the
  bound holds against the exact engine at a stated tolerance. This is a
  measurement, not a guess, and it must be a refusal boundary in code.
- Wiring into the serve path at the slot in fact 4, with a named refusal
  outside the validity region (match the existing refusal idiom:
  `LensDomainError` / `SchwingerCertificationError` / `GhostDomainError`).
- A census category in `classify_fallthrough` so the rung's contribution is
  attributable and the `out-of-box` bucket shrinks measurably.
- Fast tests: accuracy vs the exact oracle inside the validity region;
  refusal outside it; the macro-magnification limit (fact 2) recovered as
  `w -> 0`; and reachable-red controls for each.

## Out of scope (do NOT touch)

- The `w_min` ghost gate, the ghost/cusp/Pearcey questions, the uniform arms.
- Surrogate chart geometry, tiling, admission, training, `w`-range selection.
- Any campaign, sweep, pilot, or engine production run.
- `test_lensing_farfield_envelope.py` — it is ALREADY broken at HEAD (4 failed,
  21 errored: the tests pass `exclusion_radius=` but the production signature
  takes `exclusion_rho`). Pre-existing, unrelated, has its own fix. Do not
  repair it here and do not let its red mislead you.
- Extending the prior box, or changing `_source_scale` / `_Y_SCALE_CAP`.

## Acceptance

- `F_born` agrees with the exact engine to a STATED tolerance across a probe
  grid spanning the target annulus `3.0 < |y| <= 4.2426`, at the `w` the prior
  actually produces there, for `gamma` spanning `[0, 1.6]` (both parities).
- The validity boundary is measured, not asserted: a probe just OUTSIDE it must
  demonstrably breach the tolerance (reachable-red), and the code must refuse
  there by name.
- `w -> 0` recovers `1/sqrt((1-kappa)^2 - gamma^2)` (fact 2) to a stated
  tolerance.
- Serve path: a draw in the annulus that previously fell through to quadrature
  is now served by Born, demonstrated end-to-end through
  `_surrogate_coefficients`, with `lnL` agreement to a stated tolerance.
- No regression in the existing rungs: `test_lensing_channels.py`,
  `test_lensing_exterior_windows.py`, `test_lensing_ppgo_bandsplit.py` green.
- Full suite: driver-verified POST-BUILD, never in-build.

## Constraints

- HARD test-tier ceiling: any single test < 60 s, any test FILE < 5 min, fast
  tier only. No slow gates, no bulk sweeps, no hour-scale regressions in-build.
  The gate runs your tests AGAIN — every fast test is paid twice.
- Every oracle comparison must respect `w * |y| <= 60` (fact 3) and `w <= 60`
  (fact 5). A probe violating either is not a failure of Born.
- Accuracy dominates. Units/conventions per AGENTS.md. numba compatibility
  preserved on accelerated paths.
- Spec/TODO workflow applies: `todo.d` fragment, `completed.d` on completion,
  `changelog.d`, and a `spec_changelog.d` fragment — SPEC.md WILL change, since
  this rung does not exist in the spec at all (fact 7).
- Any new convention this rung introduces (a gauge, a normalisation, an
  expansion origin) MUST get ONE authoritative expression plus an assertion
  that its consumers agree. The delay-frame bug fixed in 8h-b7 was exactly a
  convention held implicitly at four sites; do not add a fifth.
- Branch `claude-dev` only. Never commit on `main`.
