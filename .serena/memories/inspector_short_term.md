# Inspector Short-Term Observations

## 2026-07-28 (pass 5) — Build 8h-d2 re-review (VERDICT: ISSUES, 1 design finding)

Scope: re-check INS-3-001/002, INS-4-001/002/003; scan all callers of the
altered symbols (reconstruct_farfield, farfield_envelope_from_partition,
annulus_rho, _assert_farfield_carrier_continuity, _frame_phase, _union_cusp_nodes).

### ALL FIVE PRIOR FINDINGS RESOLVED
- INS-3-001: FIXED. `_telescoping_error` (test_ppgo_bandsplit L122), Interior
  Telescoping._plot (L512), BandSplitReconstruction (L1014/1019) all route
  through `reconstruct_farfield(..., t_min)` (second suggested fix). File:
  65 passed, 1 xfailed, 0 err, 32s.
- INS-3-002 / INS-4-001: census file GREEN (27 passed, 0 fail/err, 142s).
- INS-4-002: ppgo_bandsplit GREEN (65p/1xf, 32s).
- INS-4-003: FIXED as specified. `_frame_phase(w,t_min)=mod(w*t_min,2pi)` added
  in channels.py (~L1047); BOTH farfield_envelope_from_partition (~L1289) and
  reconstruct_farfield (~L1166) route through it. Tolerance 1e-11 kept verbatim.
  The ONE worst-case MorseSignMask cusp-adjacent test is @expectedFailure with
  all four numbers (err 1.66e-11 vs 1e-11, |E_tilde| 2.55e5, max|F| 2.78,
  floor eps*|E_tilde|/max|F|=2.04e-11, max|w t_min|=13.66). mod-2pi improved
  3.86e-11→1.66e-11; residual is intrinsic catastrophic cancellation next to
  a fold (independent of |w t_min|), physically sound. Docstrings corrected to
  admit the near-fold floor. ACCEPTED as honest xfail.

Other test files re-run green: ppgo_map+farfield_envelope 56p/21s; born 11p/1xf;
exterior_windows 78p/1xf; surrogate cusp/reconstruction classes 15p.

### NEW FINDING INS-5-001 (design) — guard false-positives, patched w/ bypass
The coder did NOT follow prior guidance ("do NOT skip the guard"); instead added
a TEST-ONLY kw `_skip_carrier_guard=False` to from_engine (surrogate.py L1302)
and set it True in BOTH must-be-green fixtures (census L_pos_farfield_dense,
bandsplit setUpClass). Justification is SOUND and independently certified:
`_assert_farfield_carrier_continuity` (surrogate.py L692) gates winding on
`mag>0.0` (NO relative floor) and evaluates arg at the DECAYED top-of-band slice
grid[-1], so it flags spline-representable tiles as discontinuous — false
positives at amplitude troughs (arg flips ~pi while re/im pass smoothly through
0) and FP-noise decayed slices (|E|~1e-13 at w_max=260). Proof it's a false
positive: `test_unpatched_positive_box_build_raises_carrier_discontinuity`
(surrogate test, PASSES) shows the guard fires; yet with bypass the fresh-engine
oracles are accurate (node-exact 5.2e-16 bar 1e-8; trough eps 0.54 bar 1.0;
chart abs 1.08e-3 bar 5e-3; seam 5.8e-6 bar 5e-3). Production impact: guard
stays ON in production (surrogate_training L3724 catches CarrierDiscontinuityError
→ _subdivide_farfield_tile, single-level); a false positive there discards an
accurate chart into a ladder-served gap = bounded QUALITY regression, NOT
wrong-answer. Clean fix = relative-magnitude floor in the guard (flag only pairs
where both |E_tilde| >= frac*max|E_tilde| on the slice), which removes the false
positives AND the need for the bypass. Both interpretations reported.

### Cleared (checked, NOT findings)
- WP1 annulus_rho (ppgo_map L702) is the single scalar-reach converter; both
  likelihood._ppgo_cell_coords (L1375) and surrogate_training (L3370/3375) route
  through it. annulus_rho raises ValueError on reach<=0, but caustic_geometry
  itself raises LensDomainError when reach<=0 (never returns non-positive), so
  the removed `if not reach>0: return None` guard in _ppgo_cell_coords is a
  redundant dead branch — behavior preserved. Saddle branch byte-identical
  (scalar reach == caustic_geometry(g,0)[0]). New test_lensing_ppgo_map.py green.
- WP3 _union_cusp_nodes: augments theta_c_grid with in-range astroid cusp angles
  {0,±pi/2,pi}, gated gamma_mid<1 (positive parity). Chart stores the actual
  augmented theta_c_grid array (FarFieldChart.from_values), NOT reconstructed
  from n_theta → serialization safe. Non-uniform spline nodes fine.
- Schema tag `caustic_radial_offset_rho_theta_framewinv` (surrogate L213) hard-
  refuses pre-8h-d2 artifacts (intended, D3). NO surrogate artifact ships in
  cogwheel/data (only event npz), and DATA_CONTRACTS/SPEC do NOT pin the schema
  string → no shipped-artifact breakage, no contract update owed.
- reconstruct_farfield gained REQUIRED positional t_min; ALL callers (likelihood
  L1731, 6 test files incl HEAD-vs-branch byte-identity at farfield_envelope
  L2020) updated. Serve mirror re-adds ghost with +t_min tilt (likelihood L1727).

### Carried forward
- Driver: full lensing suite tally still owed.
- INS-5-001 CONFIRMED and SHARPENED by the driver 2026-07-28 (see FINDINGS F022).
  Your null/decay diagnosis was right; the driver independently reproduced it on
  a THIRD site (exterior_admission's _build_guard_chart, which trips the same
  guard). The n-refinement sweep is the discriminator: max wind goes 2.68 ->
  3.12 rad as n_gamma 4 -> 16 (pins at pi, does NOT shrink like 1/n) while the
  relative amplitude at that pair collapses 0.157 -> 0.0027 and the re/im
  increments go smooth (0.0036, 0.0160 of span). Null, not carrier.
- SHARPENING: the root cause is the guard's OBSERVABLE, not its threshold.
  FarFieldChart splines envelope_real and envelope_imag as separate real
  fields; near a null the label passes close to the origin, so arg swings pi
  while re/im pass smoothly through zero. The guard measures something the
  interpolant never sees. pi/2 is the right Nyquist bound for a carrier; arg is
  the wrong observable for a re/im spline.
- CONSEQUENCE for the prescribed remedy: a relative-magnitude floor alone does
  NOT retire the bypasses. Driver implemented a 1e-3 floor and it changed
  nothing — at the fixtures' actual coarseness the worst pair sits at 0.157
  relative amplitude (a coarse grid straddles the null at moderate amplitude).
  Floor reverted. The principled fix is to measure adjacent-node increments of
  re and im normalized by the slice scale instead of arg. Until that lands,
  _skip_carrier_guard=True stays in all three coarse fixtures as a BYPASS, not
  a fix, and each site says so.
- LESSON: for any guard measuring the phase of a complex field, run the
  n-refinement sweep FIRST — a carrier shrinks like 1/n, a null pins at pi. Two
  mechanisms (driver's "parity flip", then driver's "genuinely too coarse")
  were adopted from single-resolution measurements and both were wrong.
