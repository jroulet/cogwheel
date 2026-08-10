# Build Brief: Exterior 2D (rho, u) fold-carrier

## Mission

Extend the shipped 1D rho-fold-carrier (b061103) to a 2D `(rho, u)` fold-carrier so the u-axis phase winding is also removed. The probe (killed at 14 charts, all failing) showed the 1D rho-carrier fixed rho but left 11.66 rad of u-axis phase winding (off-grid u eps ~0.52). The fix is a 2D carrier `Re(tau_c(rho, u))` tabulated on the spline's actual axes, with serve re-modulation at the interpolated u.

## Physics (validated 2026-08-10)

- Re(tau_c) is LINEAR in rho (slope 2.4-2.7 at gamma=0.5; the slope varies with the angular coordinate — this is the u-dependence that the 1D rho-median averaged away).
- Re(tau_c) is LINEAR in u (dRe/du ~ -1.45, nearly constant across u in [0.07, 0.46]) — the ideal carrier form, confirming u is the right spline-axis variable.
- The exterior rho is the ADDITIVE form (rho = 1 + |y| - r_caustic, drho/d|y| = 1) — the linear-in-rho carrier is consistent with the caustic scaling.
- A 2D (rho, u) fold-carrier flattens the per-rho phase span in u from 11.66 -> <= 1.63 rad (measured), splineable at 4 nodes/axis.
- The u-axis total phase span is coordinate-independent (11.66 rad in both u and theta_c), but u reduces the max gradient (48 vs 82 per unit) — still insufficient alone; the carrier is needed.

## Work

1. **Extend the carrier to 2D (rho, u)**: replace the 1D `rho_carrier` (per-rho-node median over theta_c) with a 2D array `Re(tau_c(rho, u))` — one value per (rho, u) spline node. Compute via `geometry.ghost_kernel(...).delay.real` at each (rho, u) node: map the u node back to theta_c through the chart's `theta_to_u` map (inverse interp), then to a source via `_from_caustic_fixed`. Nodes where the ghost doesn't exist (GhostDomainError) get a conservative value (e.g. interpolate from neighbors, or None -> fall back to no carrier for that chart).
2. **Demodulate**: in `from_values`, demodulate the envelope by `exp(-1j * w_grid[:,None,None,None] * rho_u_carrier[None,None,:,:])` before the existing carrier_rate demodulation (composition order: rho_u_carrier demod, then carrier_rate demod, then fit).
3. **Re-modulate at serve**: in `_evaluate_chart` exterior branch, after the carrier_rate re-modulation, interpolate the query's (rho, u) into the 2D carrier via `np.interp` on each axis and re-modulate `exp(1j * w_query * carrier_interp)`. CRITICAL: interpolate u (the spline axis, after theta_c -> u map), never raw theta_c. The rho coordinate is the RAW rho (before log(rho-1) transform, same as the existing 1D rho_carrier handling).
4. **Serialization**: write the 2D carrier to NPZ (conditional, absent = None = no carrier); read via `.get()` for backward compat. Schema: this is an additive field on the existing 'exterior_polar_rho_log_carrier_v1' tag — decide whether to keep the tag (additive field) or bump. Prefer keeping the tag with the field optional (None = old behavior), so b061103 artifacts load.
5. **Training**: in `from_engine(fold_carrier=True)`, compute the 2D carrier per (rho, u) node instead of the 1D per-rho median; pass to from_values. The continuity check and k_chart estimation run on the 2D-demodulated envelope.
6. **Verify**: probe -> ~70 charts, all eps < 1e-3; u-axis off-grid eps below bar; round-trip machine precision; re-modulation at interpolated u verified (a test where the carrier is a known function of u and the re-modulated phase at off-grid rho depends on u, not theta_c).

## Measured facts (re-probe at HEAD before coding)
- 11.66 rad u-axis phase span; max dphase/du = 48, dphase/dtheta = 82
- Re(tau_c) linear in rho: slopes 2.69 (theta=0.1), 2.40 (theta=0.25)
- Re(tau_c) linear in u: dRe/du ~ -1.45, nearly constant over u in [0.07, 0.46]
- 2D (rho,u) carrier: per-rho phase span in u <= 1.63 rad (validated)
- Exterior rho is ADDITIVE (rho = 1 + |y| - r_caustic); theta_to_u map in the chart; `geometry.ghost_kernel(w, source, matrix).delay`
- Prior: 1D rho_carrier shipped in b061103 (per-rho median), re-modulates at RAW rho
- Probe: `scripts/probe_exterior_recursion.py` (4x4x4, w 4/decade, engine 80)

## Constraints
- Fast tests. Follow AGENTS.md.
- COORDINATE/PHASE transform — NOT node-density increase.
- Carrier is 2D (rho, u) on the spline axes; re-modulate at interpolated u, never raw theta_c.
- Do NOT demodulate by Im(tau_c) (e^{+w·Im} explosive).
- Compose with w-carrier + rho_log + ghost-exclusion; round-trip machine precision.
- Keep the existing 'exterior_polar_rho_log_carrier_v1' tag if the 2D carrier is an additive optional field; bump only if the stored object's semantics change.
- Plan-gate requirement: each `domain_test_descriptions` spec names exactly ONE primary `test_*.py`; no spec may reference another spec's primary file.

## Design note from the driver
The user (domain expert) caught two coordinate errors in my analysis: (1) the spline axis is u, not theta_c, so the carrier must be expressed and re-modulated in u; (2) the exterior rho is the additive form, not multiplicative. Both are incorporated: the carrier is 2D (rho, u), re-modulated at interpolated u. Validate that the phase flattening holds on the ACTUAL spline axes (rho, u), and that the serve re-modulation uses the interpolated u.
