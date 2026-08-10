# Build Brief: Exterior envelope rho-axis conditioning (log/(rho-1) coordinate)

## Mission

Reparameterize the exterior polar chart's rho axis so the ~4.5-decade envelope growth toward the caustic (rho=1) is splineable at the probe's node count (4 nodes/axis). This is the follow-on to the carrier-demodulation build (which fixed the w-axis): the spatial rho axis now dominates the residual. The fix is a COORDINATE transform, not added resolution. It must be coherent with the w-carrier demodulation and keep the serve/reconstruction path machine-precision consistent.

## Background (measured 2026-08-10, after carrier-demod build f4652e7)

- Carrier demodulation fixed w-axis: node eps 1e-17, w-midpoint eps ~1e-4.
- Remaining failures (off-grid): rho eps ~0.04, theta eps ~0.009, vs the 1e-3 bar.
- rho structure: |E| = 0.95 at rho=1.02 down to 2.6e-5 at rho=2.5 (gamma=0.5, theta=0.4, w=12) — ~4.5 decades. Power-law-like in (rho-1): exponent ~-1.7 (over rho-1 in [0.02,0.79], R²=0.75) to ~-3.2 (full range, R²=0.78) — the exponent STEEPENS near rho=1, so it is NOT a clean single power law. Classic log(rho-1) conditioning case.
- theta_c: ~20x smooth variation across the quadrant (|E| 0.0037 at theta=0.15 -> 0.107 at theta=1.42, U-shaped peaking at the cusps). The cusp-adapted u-coordinate (previous build) likely handles the cusp edges; verify.
- gamma: ~6x mild variation (0.0026 at gamma=0.3 -> 0.015 at gamma=0.9) — probably fine at 4 nodes; verify.

## Work

1. **Characterize** (design input): sweep |E| and phase vs (rho-1) at several (gamma, theta_c, w) to confirm the log(rho-1) structure is consistent across the tile/band, and measure the phase behavior in rho (is there a rotating phase in rho, like the w-carrier?). Determine the best coordinate: `u_r = log(rho-1)`, or `(rho-1)^q` for a tuned q (the exponent steepens near rho=1, so a single power may not fully linearize — test both; consider `log(rho-1)` which handles the variable exponent naturally).
2. **Design the representation** (deliverable): how to fit the COMPLEX envelope in the new rho coordinate at 4 nodes/axis clearing the 1e-3 bar, including:
   - The rho=1 boundary: `log(rho-1) -> -inf`. How to handle the caustic edge (the chart's rho range starts above 1; the boundary node can be placed at rho = 1 + epsilon, or the coordinate can be `log(rho-1)` with the lowest node at a finite offset). The exact serving of near-caustic draws is already the tube chart's domain (rho >= 1 + eta_overlap_min), so the exterior chart may not need rho=1 exactly.
   - The complex phase: is there a rotating phase in rho that needs a carrier-like treatment (like the w-carrier), or does the real/imag ordinate suffice once the magnitude is conditioned?
   - Consistency with the w-carrier demodulation: the two transforms (w carrier + rho coordinate) must compose; the stored object must round-trip to machine precision.
3. **Implement** in `ExteriorPolarChart` (from_values/_assemble/_evaluate_chart/npz/axis-schema), the `_build_farfield_chart` training path, and the serve/reconstruction (`reconstruct_farfield` + likelihood consumption). Consider whether the rho transform is an axis-coordinate change (like theta_to_u) needing a `rho_to_ur` map + schema bump, or a preprocessing of the envelope (like the carrier).
4. **Verify**: node-exact round-trip (machine precision), held-out eps < 1e-3 at 4 nodes/axis at off-grid points in rho AND theta (and confirm theta is handled by the existing u-coordinate or needs its own treatment), and the full exterior probe produces ~70 charts with all eps under the bar.

## Measured facts (re-probe at HEAD before coding)
- |E|: 0.95 @ rho=1.02, 2.6e-5 @ rho=2.5 (gamma=0.5, theta=0.4, w=12); ~4.5 decades
- (rho-1) power-law exponent: ~-1.7 (R²=0.75, rho-1 in [0.02,0.79]) to ~-3.2 (R²=0.78, full)
- theta_c: |E| 0.0037 @ 0.15 -> 0.107 @ 1.42, U-shape; gamma: 0.0026 @ 0.3 -> 0.015 @ 0.9
- Off-grid residuals after carrier fix: rho ~0.04, theta ~0.009 (eps vs 1e-3 bar)
- Relevant code: `ExteriorPolarChart.from_values` (~1590), `_evaluate_chart` exterior branch (~2722), `_build_farfield_chart` (surrogate_training.py ~2714), `reconstruct_farfield` (channels.py), carrier_rate machinery just added (surrogate.py)
- Envelope: `farfield_envelope_from_partition(partition, FARFIELD_KERNEL_SUM)`
- Probe: `scripts/probe_exterior_recursion.py` (4x4x4, w 4/decade, engine 80)

## Constraints
- Fast tests. Follow AGENTS.md.
- Coordinate/scale transform — NOT node-density increase.
- Must compose with the w-carrier demodulation (already shipped) and round-trip to machine precision.
- Complex envelope phase handled correctly.
- The rho=1 boundary: the exterior chart serves rho >= 1 + eta_overlap_min (tube takes over closer in); do not need rho=1 exactly unless serving requires it.
- Plan-gate requirement: each `domain_test_descriptions` spec names exactly ONE primary `test_*.py`; no spec may reference another spec's primary file.

## Design note from the driver
The user (domain expert) predicted this: "Once you do log|F| to fix the frequency axis, the spatial axes will also have to be fixed because you can't just fix the ordinate." The w-carrier fix is done; this build does the rho axis. Treat it as a coherent reparameterization consistent with the w-carrier approach. If theta_c also needs conditioning beyond the existing u-coordinate, scope it here or flag it as a clear follow-on.
