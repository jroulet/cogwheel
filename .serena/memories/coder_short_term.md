# Coder Short-Term Observations

- WP1 farfield-port RESTORE (Build 1e): `git checkout refs/sdk/farfield_port_wip
  -- cogwheel/lensing/ cogwheel/tests/ .claude/spec/DATA_CONTRACTS.yaml` laid
  down the (s,d) FarFieldChart. 7 files restored (worktree == ref, empty diff):
  surrogate.py, surrogate_census.py, surrogate_training.py, DATA_CONTRACTS.yaml,
  + 3 already-ported tests (exterior_windows, farfield_envelope, surrogate_census).
  import cogwheel.lensing.surrogate OK; pytest --collect-only == 1171, 0 errors.
  NOTE reconstruct_farfield lives in chang_refsdal/channels.py:1106 (7 args incl
  t_min), NOT surrogate.py. FarFieldChart.from_values is keyword-only w/ arc_map
  (_FarFieldArcMap, REQUIRED) + gamma_grid/s_grid/d_grid axes. select_chart adds
  y1_eig/y2_eig eigenframe kwargs. NO test body or impl edited by me — pure restore.

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

- Arc-map contract fix (2026-07-31): FarFieldChart._assemble now validates and normalizes _FarFieldArcMap before storing it, so both construction and _chart_from_npz hard-refuse any map whose gamma_nodes are not byte-equal to the chart gamma_grid. The validator also requires finite/increasing gamma and theta axes, correctly shaped finite cumulative s_table rows anchored at exact zero and strictly increasing, and coherent branch/endpoints. Focused FarFieldArcMapValidationTestCase passed (3 tests).
\n- 2026-07-31 far-field `(s, d)` port: old raw/caustic-fixed positive likelihood fixtures decline under current chart containment. Rebuild physical probes through `_from_farfield_smooth` at stored chart nodes. Macro-saddle far field is intentionally exact-engine-only; retain a non-vacuous exact-lnlike fallback pin. Node-exact far-field reconstruction across beta clears the existing 2e-9 label round-trip bar; broad tiny-chart cell midpoints are not a valid 1e-3 interpolation certificate (measured 31.55 down to 0.80).

- WP1 lobe-interior wedge-edge s-coordinate (Build 1e-lobe): REVIEWED AND
  DELIVERED as-is (no new code authored). Diff verified against all 7
  acceptance criteria: (a) from_lobe_engine builds monotonically-increasing
  s = sqrt(span) - sqrt(theta_max - theta) with exact endpoint clamping;
  (b) from_lobe_values routes spline to s_grid when both provided, raises
  on exactly-one-None; (c) _evaluate_chart maps theta_local -> s via
  np.interp when theta_to_s not None, identity fallback otherwise;
  (d) _chart_to_npz stamps _LOBE_AXIS_SCHEMA vs V1 based on theta_to_s
  presence; (e) _chart_from_npz V1 tolerates absent theta_to_s, current
  requires it (KeyError = hard-refuse); (f) _validate_theta_to_s checks
  (2,N) shape, finite, strict mono both rows, s[0]~0, theta[0]=grid[0];
  (g) _LOBE_ARC_MAP_SIZE=2001, schema tag has 'sqrtedge', both in
  _KNOWN_LOBE_AXIS_SCHEMAS. No _WEDGE_EPS in production code. No import
  changes needed. Tests referencing _LOBE_AXIS_SCHEMA still resolve to
  current schema (which is now the sqrtedge tag). ast.parse + import OK.