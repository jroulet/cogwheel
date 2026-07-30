# Coder Short-Term Observations

- WP1 caustic_geometry (ppgo_map.py): replaced 2x720 polar sweep over
  geometry.critical_point with closed-form (lam,e) extremisation. Candidate
  u-set: e<1 -> {1-e,1+e}; e>1 -> {1+e, (-1+sqrt(4e^2-3))/2 if >0,
  sqrt(e^2-1)}. Guards: u>0 strict, |cos2th|<=1+1e-12. reach=max
  sqrt(lam*((1-u)^2(1+2u)+e^2(2u-1))/u^2). Direction from winning u via
  eigenframe point (A*cos,B*sin), A=W-gamma,B=W+gamma, W=lam(1-u);
  canonicalized first-nonzero>=0. Removed n_theta kwarg (no caller passed it).
  Top refusal lam<=0 and EXACT abs(gamma)==lam. NOTE: |point|^2 != reach^2
  (differ by positive factor 1/(lam*u)) but direction (normalized) is
  unaffected — brief gives reach and direction as SEPARATE formulas by design.
  Verified: gamma=0.9 -> 5.692099788303083 dir[0,1]; nextafter(1,+-)->finite;
  (1.0,0),(0.5,0.5),lam<0 raise. surrogate.py untouched (pass-through kept).

- INS-1-001 fix (ppgo_map._measure_cell): the one-sided anisotropy fan
  angles=[0..pi/2] was quadrant-DEPENDENT — caustic_geometry canonicalizes
  direction only up to sign (2-fold), so the saddle-band reflection it
  returns differs from HEAD's retired scan, and the one-sided fan then
  probes a DIFFERENT arc -> over-certifies near-wall saddle (g=1.1,rho=0.2:
  30.0 vs HEAD 24.83). FIX = make the fan SYMMETRIC:
  angles=tuple(k*pi/8 for k in range(-4,5)) spanning [-pi/2,+pi/2].
  PAPER PROOF (no map regen run; that's an offline campaign = downstream):
  (a) symmetric set is invariant under the caustic's eigenaxis reflection R,
  and engine error(p)==error(R(p)) (documented 4-fold symmetry, what
  angle_bar tests assert) => min over fan is reflection-INVARIANT, so
  direction quadrant no longer matters; (b) two-sided fan is a SUPERSET of
  HEAD's one-sided fan (anchored at scan-dir, via R-invariance) => w_cert
  can only drop => NEVER less conservative than HEAD; (c) for positive
  parity (axis-aligned dir) the two sides are mirror images w/ identical
  error => min unchanged, so positive-parity cells are byte-identical to
  HEAD. Old angles are a subset of new, so test_lensing_ghost._anchor_source
  (mirrors _measure_cell placement) still matches for old angles.
  UNVERIFIED: actual certified_ppgo_map regeneration diff vs HEAD (offline
  artifact build) — left to Test Dev/Inspector. caustic_geometry direction
  canonicalization left as-is (now irrelevant under the invariant sweep).

- WP1 tube-chart arc-length: added frozen field `theta_to_s` (2,N_map) to
  TubeChart; from_values/_assemble take optional s_grid+theta_to_s (identity
  default s=theta-theta_lo). Serve maps query theta -> s via np.interp before
  contracting knots[3] (now built in s). NPZ stores/reads `chart{i}_theta_to_s`.
  _build_tube_chart places theta nodes as images of uniform s grid at
  rep_gamma=median via new _tube_arc_length_map (scipy cumulative_trapezoid of
  geometry.caustic_speed, branch-aware, 2001 pts, finite+strict-increasing raise).
- Identity default is NOT bit-identical to HEAD raw-theta spline: matches to
  ~5e-15 (B-spline translation invariance; knot arithmetic differs in last bits).
  Fine for tolerance-based suites; a bit-exact-vs-stored-HEAD assertion would drift.
- Old (pre-schema) tube npz records hard-refuse on load (KeyError on
  chart{i}_theta_to_s) — correct: an old chart's knots are in raw theta, so an
  identity-map fallback would serve at the wrong offset. Do NOT add a fallback;
  old artifacts must be retrained.
- Diagnostic s_map_gamma_endpoint_dev recorded in the build_tube CLOSURE report
  dict (the actual build report), NOT in _build_tube_chart's return, to avoid
  changing its 3-tuple contract (test unpacks it at test_..._training.py:1043).
