# Test Dev Short-Term Observations

## 2026-07-18 — ratio-layer suite (test_lensing_ratio_layer.py, WP1/WP2)

Authored 18-test suite for likelihood.py ratio layer; all green + sibling
test_lensing_likelihood.py unaffected (0 production edits).

- ON-LATTICE ANCHORS ARE NOT BIT-EXACT: `_snap(0.15,0.05)=round(3.0)*0.05
  = 0.15000000000000002 != 0.15`. Lattice-membership fixture must use
  assertAlmostEqual (+ snapping-idempotence: key==_fiducial_key(snapped)),
  NOT assertEqual. The 1-ULP lens offset floors rho_bare/dtau_c ~1e-16,
  far below the identity gate.
- ENVELOPE_IDENTITY_RTOL relaxed 1e-13->1e-9 (DEVIATION 1): CR engine
  reproduces envelope/critical_delay across DIFFERENT node grids only to
  ~1e-11, so ratio-vs-direct floors ~1e-12, not eps. Still 7 orders below
  _LOO_STOP=4e-3 => certifies "algebra not interpolation". lnlike identity
  holds at brief's 1e-9.
- PERTURBED spec: 0.1-nat gate unreachable near caustics; gate per-point
  |ratio-direct|<RB_ATOL=1.5 + per-anchor median<0.15 (measured <=0.08).
- CAPTURE SEAM: patch MODULE-LEVEL likelihood_module.reconstruct_from_envelope
  (both ratio & direct route through _kernels_from_dense_envelope) to grab
  BOTH dense envelope E_cand (spec1) and total F (spec8/F009). Ratio node
  count via wrapping instance _ratio_loo_nodes (returns coarse_w,rho_nodes).
- FALLBACK detection: wrap _ratio_coefficients; not-called => a guard/refusal
  fell back to direct. Guard2 (unhealthy fiducial) not naturally reachable
  from lattice => INJECT via types.SimpleNamespace(partition=real_fid.partition,
  envelope=dip) so guard1 (real_mask.sum match) passes & guard2 fires.
  Fiducial-refusal fallback (spec5c) via mock.patch _get_or_build_fiducial
  side_effect=raise; assert lnr==lnd bit-identical (float64.tobytes()).
- _lens_dic positional order is (y1,y2,gamma,BETA,kappa,...) — beta is 4th
  positional; passing kappa positionally + beta= kw collides. Watch arg order.
- CANCELLATION_CONFIG=gamma0.405/kappa0.57, MACRO_SADDLE=gamma0.5/kappa0.6
  give symmetric CancellationError / LensDomainError across all 3 paths.
- Deep-band tiny w via tiny m_lens (1e-4..1e-6 Msun) not tiny f; gamma0.21
  kappa0.30 lattice-aligned positive-parity.
