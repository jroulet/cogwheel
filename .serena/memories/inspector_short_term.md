# Inspector Short-Term Observations

## 2026-08-20 — diffractive_wall_nearfold_chart build (pass-4, FINAL PASS)

Re-reviewed all uncommitted changes (low_w_diffractive_chart.py [untracked],
scripts/train_low_w_diffractive_chart.py [untracked], likelihood.py,
serve_route_census.py, test_lensing_low_w_diffractive_chart.py [untracked]).

MANDATORY RE-CHECK — INS-2-002 (declined_mask excluded from content hash):
RESOLVED at BOTH sites this pass.
  * `low_w_diffractive_chart.py` load() line ~357 now hashes 8 fields:
    `_content_hash(gamma_prime_grid, rho_grid, theta_grid, log_w_grid,
    real_coeffs, imag_coeffs, derate, declined_mask)`.
  * `scripts/train_low_w_diffractive_chart.py` line ~601 now hashes 8 fields:
    `_content_hash(chart.gamma_prime_grid, chart.rho_grid,
    chart.theta_grid, chart.log_w_grid, chart.real_coeffs,
    chart.imag_coeffs, chart.derate, chart.declined_mask)`.
  Test helper `_save_chart_artifact` already hashed 8 fields; all three now
  agree on identical float64 bytes. `test_round_trip_is_bit_identical`,
  `test_rehashed_tamper_loads_cleanly`,
  `test_rehashed_declined_mask_tamper_loads_cleanly` all GREEN; the tamper
  test `test_tampered_declined_mask_hard_refuses` is now NON-vacuous (premise
  asserts `declined_mask.any()` and `not .all()`). Full changed test file
  green: 36 passed. serve_route_census test file green: 42 passed.
  Train script round-trip self-check also now asserts
  `np.array_equal(loaded.declined_mask, declined_mask)`.

NEW-CHANGE AUDIT (this pass's diff beyond the hash fix):
  * likelihood.py `_low_w_diffractive_chart_serve`: reconstruction identity
    VERIFIED against the test oracle `_engine_reference_kappa` — `sqrt(mu_pure)
    = lam * sqrt_mu` cancels the `1/lam`, so `F = mass_sheet_phase *
    prefactor_c(w) * sqrt_mu_full * r_pure` holds exactly. mass_sheet_phase
    `exp(0.5j*w*(log(lam)-kappa*s))` with `s=|y|^2/lam` matches the oracle.
    Reconstruction tail (demod by t_min -> reconstruct_farfield ->
    _reduce_dense_kernels -> _image_delays) is byte-identical to
    `_low_w_diffractive_serve`. Guards: `_reduced_shear` raises -> None;
    gamma_prime==0 -> None; s<=0 -> None; covers(dense_w) refuses off-log-w.
  * `_AUTO_LOW_W_CHART` sentinel + get_init_dict handling mirrors the
    born_residual_chart pattern exactly (pop-default / None-verbatim /
    raise-in-memory). Absent shipped npz -> OSError caught -> None + warn,
    byte-identical fall-through.
  * serve_route_census.py: 12th route `low_w_diffractive_chart`; kappa=beta=0
    mirror faithful (s=y1^2+y2^2, theta=atan2(y2,y1), rho_dir=caustic_rho
    (abs(gamma'),s,theta) — fresh local, does NOT rebind the outer scalar
    `rho` gauge). SERVE_ROUTES 11->12 widening is laggard-safe: the census
    invariant test uses dynamic `src.SERVE_ROUTES` (no hardcoded 11);
    `_ProductionModules` has exactly one construction site (keyword-based).

STILL OPEN -> Librarian (INS-1-003, doc staleness, NOT a Coder defect):
  SPEC.md LOW-W DIFFRACTIVE RUNGS still says the near-fold shell is DECLINED
  (code now SERVES it via LowWDiffractiveChart); DATA_CONTRACTS.yaml has no
  `low_w_diffractive_chart` entry. Carried to Librarian doc-sync.

ADVISORY: shipped `cogwheel/data/low_w_diffractive_chart.npz` still absent;
  full bake is a DRIVER post-build step (hash now covers declined_mask, so
  bake AFTER this commit is required).
