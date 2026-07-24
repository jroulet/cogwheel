# Architect Short-Term Observations

- Build 8h-b3-FIN (2026-07-23): CONTINUATION, plan BINDING, do-not-replan.
  In-tree partial S1-1 measured: geometry.r_caustic DONE; surrogate.py
  FarFieldChart->(rho,theta_c), _to/_from_caustic_fixed, serve mirror,
  save/load axis_schema tag DONE. INCOMPLETE = surrogate_training.py tiler:
  _farfield_tiles still Cartesian, _build_farfield_chart calls RETIRED
  from_engine(y1_range,y2_range) (new sig=rho_range/theta_c_range/n_rho/
  n_theta) -> training broken. S1-1-complete = finish tiler only, rename
  TrainingConfig n_y1/n_y2->n_rho/n_theta (thread through budget check +
  node_counts report), keep tile loop, don't rewrite done chart/serve.
  Prof R3/R4 NEW pins: (notch) scalar-rho>1 EXTERIOR admission is strict
  SUBSET of true exterior (directional r_caustic<=scalar max); notch annulus
  near cusps has scalar-rho<1 but is exterior -> INTERIOR (S2 directional)
  owns it; keep exterior at scalar-rho>1+margin (serve-mirror consistency
  paramount); theta_c=atan2 in (-pi,pi], no tile straddles ±pi cut; tiler
  MUST call same _caustic_reach as chart/serve. Diffractive-bottom bounded:
  ADD 0.3 lower bound (guard collapsed fit) 0.3<|obj|/max|F|<3. Crown
  gamma~0.90 SACR-C relaxed <=1e-1 (An-Evans quasi-symmetry floor), 1e-3
  for gamma<=~0.65. Tag test: ADD positive cross-serve falsification (serve
  A thru B's path -> WRONG F, diverges >>1e-3) + tag validated BEFORE
  numerics. S2-3 CLEAN SWAP no cusp carve-out (tau_c finite critical delay,
  unimodular demod, no 1/(tau_a-tau_c)/Im tau_c denom); pin tau_c path-
  continuity per tile (reseat on basin flip); a near-cusp exclusion test =
  false-red. Simplifier: all 6 WPs lean; S2-3 watch->resolved by R4.

- Build 8h-b2 (2026-07-23): single-WP ghost-kernel in geometry.py = WP3
  of build8hb_plan_full_v1.json verbatim. Professor pinned: bilinear
  (non-Hermitian, holomorphic) continuation is FORCED; log branch =
  principal clog(x_c.x_c) valid iff Re(x_c.x_c)>0 (else raise, no path
  integral); sqrt branch = pick root nearest real-saddle
  sqrt|mu|.e^{-i pi/2}, Morse absorbed (no morse_index call).
  Oracle: numpy.roots + Richardson-central-FD complex Hessian det,
  step h=1e-4 w/ h/2, floor h=1e-5; tol 1e-6 on analytic legs, 1e-4
  on FD leg. Anchors: |C|/|E_ff| within 10%, arg(E/C)<3.5deg. On-axis:
  |Im tau_c|<1e-10, ||e^{iw tau_c}|-1|<1e-12. Far rho=4: |C|<1e-3,
  ratio<0.5, Im tau_c>8. Degenerate guard: raise when
  |det H_c|<1e-8*(1+||A||_F)^2. Simplifier: single WP lean; REUSE
  _c1_polynomial/_c2_polynomial directly (pure arithmetic, complex-ok);
  do NOT call _saddle_metric/saddle_coefficients (hit norm/hessian
  real-only); write dedicated complex metric+Hessian helper.

- Build 8h-b3 (2026-07-23): caustic-fixed core, width-BINDING ->
  Option B two slices x3 WPs. Slice1 exterior: (S1-1) FarFieldChart
  coord migration raw-eigenframe -> caustic-fixed (rho,theta_c) both
  sides; (S1-2) w-windowed 3-class label + 3 envelope tags + serve
  mirror; (S1-3) fixed [w_floor,w_trust] windows replace mass strata
  (range check, NOT extended) + per-window LOO node reprovision.
  Slice2 interior WP6/7/8 verbatim except WP8 amended WHOLE-interior
  SACR-C (Prof Decision3: far-field label fails 6e-2 @ mid-gamma 0.40,
  disease generic not crown-only). Born rung DEFERRED (brief item5
  leaves to Prof; nothing in-box depends; prior-dependence recorded).
  Prof: w_floor(region)=min w s.t. w*min|tau_a-tau_b|(closest pair)>=2
  (half RHO_END=4, ghost-gate currency); PHYSICS const per-region.
  Window(iii) upper = w_trust capped by w_ceiling. Tag mismatch
  silently serves WRONG F = top risk. Seam bars both 1e-3; (i)/(ii)
  jump<1e-3*max|F| bc kernel enters GATED; reachable-red w_floor>=1
  not >=2 -> O(1e-2) jump. Bounded obj 0.3<|obj|/max|F|<3 on
  [0.03,w_floor]. Interior grid {0.40,0.65,0.90} A-fails/B-passes,
  crown SACR-C <=1e-1 not 1e-3. Containment subset 1e-12 + strata-
  removal BYTE-IDENTITY. Reprovision eps[0.5e-3,1e-3]@N_rec,>1e-3@N-1,
  y array_equal. Simplifier: S1-2/S1-3/S2-3 watch; extend
  _KNOWN_FARFIELD_DEFINITIONS AND dispatch atomically.
