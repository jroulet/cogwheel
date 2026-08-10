# Professor short-term — 2026-08-10

## Session: F022 carrier-demodulation review

Reviewed proposed `carrier_rate` / `k_chart` residual carrier demodulation
design for the far-field envelope. Key observations:

1. The proposal is physics-sound: after per-node `t_min` demodulation,
   `E_tilde = E_ff * exp(+1j w t_min)` carries a RESIDUAL spatial carrier
   because `t_min` varies across chart tiles. The residual phase variation at
   spatial node A vs B is `w * (t_min_A - t_min_B)`, linearly growing with
   frequency. A median `k_chart` removes the linear trend of this residual.

2. `k_chart` is NOT `t_min` — it's a much smaller residual (likely ~0.01
   based on typical chart extents), giving ~0.2 rad residual at w_max~20.
   This is within spline tracking range.

3. The unwrapped-phase estimator is robust if the median is taken across
   many spatial nodes. Individual nodes near amplitude nulls would have
   contaminated `k_node` from the pi-jump, but the median rejects outliers.
   The F022 null problem and the residual-carrier problem are distinct:
   nulls cause ARG FLIPS (discontinuities), while the residual carrier is a
   SMOOTH spatial trend.

4. Power-law magnitude rescaling is NOT needed. The |E|~w^(-0.60) behaviour
   is identical at all spatial nodes — it's purely a 1D function in log_w
   that the spline captures. No between-node error comes from it.

5. The binding question is whether the residual carrier IS the dominant
   error source preventing 1e-3 eps. This depends on the spatial variation
   of t_min within typical chart tiles. RECOMMENDATION: measure before
   implementing. A refinement test (varying n_gamma or n_rho) before/after
   k_chart can discriminate: a carrier shrinks like 1/n, a smooth envelope
   shrinks like 1/n^4.

## Code-level observations

- `farfield_envelope_from_partition` (channels.py:1250) applies per-node
  `exp(+1j w t_min)` demodulation via `_frame_phase` (channels.py:1124).
  The serve mirror `reconstruct_farfield` (channels.py:1168) re-modulates
  by `exp(-1j w t_min_query)`. The spline interpolates `E_tilde` across
  spatial nodes; any residual carrier from t_min variation is an
  interpolation-error source.

- `ExteriorPolarChart` stores `real_coeffs` / `imag_coeffs` as separate
  cubic B-spline tensors. `_evaluate_chart` contracts via
  `_contract_tensor_spline` and returns `real + 1j*imag`. Adding
  `carrier_rate` to the class means storing it in NPZ meta and applying
  `exp(+1j * carrier_rate * w)` after evaluation.

- `_frame_phase` reduces `w*t_min` modulo 2π — a `k_chart*w` demodulation
  should use the same reduction to preserve telescoping precision.

- Backward compat: charts without `carrier_rate` in NPZ → default 0.0 → no
  remodulation → byte-identical to current behavior. Clean migration.

## Build review — 2026-08-10 (carrier-demodulation implementation review)

All 106 fast-tier tests pass (14 carrier + 92 surrogate). Implementation verified:

### Node-exact round-trip (test_lensing_exterior_carrier.py)
- `NodeRoundTripTestCase`: |E_served - E_raw| < 1e-13 at all 64 training
  nodes — spline is interpolating (not-a-knot cubic), demodulation/re-modulation
  telescopes to floating-point precision. `carrier_rate=0` backward-compatible
  path also node-exact. Pass.

### Held-out accuracy
- `HeldOutAccuracyTestCase`: midpoint error below 5e-2 bar at the geometric
  centre spatial node. Diagnostic plot saved. The bar is 5e-2 (not 1e-3)
  because the test uses a deliberate 4×4×4 spatial grid — the error is
  interpolation-limited, not carrier-limited. Pass.

### Self-falsification
- `SelfFalsificationTestCase`: corrupted carrier_rate (Δk=0.1) drives error
  above bar and >10× correct. `zero carrier_rate` for genuinely modulated
  envelope also above bar. All teeth-check tests in
  `CarrierSelfFalsificationTestCase` prove the suite can go RED. Pass.

### Production path (from_engine)
- `FromEngineRoundTripTestCase`: `from_engine` → `from_values(carrier_rate=k_chart)`
  serves within held-out bar. `carrier_rate` is stored and finite. Pass.

### Schema migration (test_lensing_surrogate.py)
- `ExteriorPolarStaleSchemaHardRefusalTestCase`: old schemas
  ('exterior_polar_rho_theta_c', 'exterior_polar_rho_u_v1') raise ValueError.
  New schema 'exterior_polar_carrier_demod_v2' loads successfully.
  carrier_rate=0.5 preserved through NPZ round-trip.
  Missing carrier_rate key defaults to 0.0 (backward compat).
  NaN/Inf carrier_rate raises ValueError in _assemble. Pass.

### Source code verification
- `from_values` (surrogate.py:1609): single canonical demodulation site —
  `E * exp(-i*k*w)` before fitting, carrier_rate passed through to _assemble.
- `_evaluate_chart` (surrogate.py:2768): re-modulation `exp(+i*k*w)` applied
  only for ExteriorPolarChart with nonzero carrier_rate.
- `_chart_from_npz` (surrogate.py:4188): `meta.get('carrier_rate', 0.0)`.
- No double-demodulation bug (INS-15-001 guard in from_engine at line 3013).
