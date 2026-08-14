---
date: 2026-08-14
section: Lensing serving
---

**c3-certificate admission for 2-image saddle exteriors** `[→ spec]` —
commit `1c90b3a`, build `symmetry_tie_c3_admission` (NEXT-SESSION ORDER
2/7; supersedes the dead eta-floor build). The scalar rho floor and the
`delta_tau > 0` tie hole are gone; admission is
`_SADDLE_FARFIELD_SAFETY(20) * ppgo_error_estimate(...) <=
_SADDLE_FARFIELD_CERT_BAR(1e-3)` at the band floor, None-refusal for
merging pairs, `_SADDLE_FARFIELD_MIN_IMAGE_SEP(0.05)` backstop;
symmetry-tied mirror pairs serve. Census mirror moved in the same build.

Driver-side pre-work that reshaped the brief (all measured, sha 9f331dd,
`scripts/calibration_pilot_followup.json`): the ghost term is unavailable
outside the shipped continuation's principal-log branch domain (Re z <= 0)
— including the ENTIRE gamma=1.2 worst-case ridge, the connecting region
and the transverse cone — while the non-image quartic pair there is a
GENUINE complex-conjugate stationary pair (residual 1e-14..1e-16); the
true remainder decays as `w^-3` (c3-shaped) everywhere (R^2 0.96-1.000);
c3's shortfall is bounded (<= 9.4x connecting / 5.9x generic pointwise),
which licenses the single scalar safety factor. Branch-corrected ghost
continuation deliberately NOT attempted (out of scope; future fragment if
ever needed).

Driver-completed after the build died at the tree gate: the ONLY red was
`test_lensing_part0_mechanical`'s absorber guard flagging the new
`_SADDLE_FARFIELD_SAFETY` — allowlisted beside `_PPGO_INTERIOR_SAFETY`
with the measured-margin justification (the guard fired exactly as
designed on a new `_SAFETY` constant). Inspector PASS, Professor PASS;
Phase 3 skipped, so Librarian/Dreamer ran driver-side post-commit.

Driver serve-path traces (report-only checkpoints from the brief): the
rung is reachable from production (`_amplification_coefficients` line
~2255, gamma > 1 intercept); the engine-level `_saddle_grid` consults the
full uniform-arm ladder including the ppGO+ghost arm via
`_uniform_arm_value`; the remaining `delta_taus > 0` tie-shape lives only
in `_ppgo_above_ceiling` (w > 150, both parities) — the separate F076
item.
