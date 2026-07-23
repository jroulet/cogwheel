# Architect Short-Term Observations

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
