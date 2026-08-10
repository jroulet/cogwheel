# Build Brief: Exterior envelope w-axis power-law conditioning (design the full representation)

## Mission

Design and implement the correct coordinate/scale representation for the exterior surrogate envelope so it clears the 1e-3 held-out eps bar at the probe's node count (4x4x4 = 4 nodes/axis) WITHOUT resolving the steep decay by added node density alone. The envelope is complex with ~1000x dynamic range in w; the fix must be a coherent reparameterization, not a resolution increase. This is a DESIGN build — the coordinate transform may touch multiple axes and the serve/reconstruction path.

## Background (measured, Professor + Coder investigation 2026-08-09)

The exterior label `E_ff(w) = F(w) - sum_{a real} H_a(w) exp(1j*w*(tau_a - t_min))`, demodulated by exp(+1j*w*t_min), is SMOOTH and correctly non-oscillatory (the full kernel subtraction works; an earlier "beat" report was a measurement artifact from a bad w-grid perturbation — do NOT chase a beat). Verified:
- At gamma=0.481, rho=1.679, theta_c=0.419: |E| is monotone decreasing, 0 peaks, over w in [1, 19.3].
- |E(w)| ~ w^(-0.60) (power-law fit slope, R^2=0.996); log|E| is linear in log w.
- The spline is EXACT (1e-17) at all 7 training w-nodes, but the w-midpoint errors are 0.037-0.08 in absolute terms against |E| falling 0.38 -> 0.00044 (relative error 12% at low w to ~1200% at high w). eps = max|E_spl - E_eng| / max|F| (max|F| ~ 1.4) is breached because the absolute spline error is a fixed ~0.02-0.08 while |E| decays 1000x.
- The label spans ~3 orders of magnitude over 1.34 decades of w.

## The core problem to design around

A cubic spline fits smooth functions well, but a ~1000x power-law decay with only 7 knots (4/decade) leaves large BETWEEN-node curvature errors. The user's engineering principle: SPLINE SMOOTH THINGS — do not out-resolve steep ones. The magnitude is a clean power law, so a log-scale or power-law-rescaling coordinate should linearize it. BUT:

1. The envelope is COMPLEX (real + imag with a rotating phase in w). A naive log|E| discards phase. The build must design a representation that handles both magnitude (power-law, needs log/rescale) and phase (rotating, needs unwrapping or a rotating-frame carrier).
2. The user's note: "once you do log|F| to fix the frequency axis, the spatial axes will also have to be fixed because you can't just fix the ordinate." The transform is NOT just an ordinate change — if the w-axis coordinate changes, the spline's cross-axis structure (how E varies in rho/theta_c at each w) changes too, and the spatial axes may need matching treatment (e.g. rho has its own power-law or boundary structure near rho=1).
3. The serve path must stay consistent: `reconstruct_farfield` must reproduce the exact F from the stored (possibly transformed) coefficients. Any transform must round-trip to machine precision.

## Work

1. **Characterize the full structure** (design input): measure how the envelope magnitude AND phase vary over ALL axes at the probe configuration and a sweep of (gamma, rho, theta_c). Determine:
   - The w-power-law exponent as a function of (gamma, rho, theta_c) — is w^(-0.60) universal or does the exponent vary?
   - The phase behavior: is it linear in w (a rotating carrier) and at what rate?
   - The rho/theta_c/gamma dependence: are there similar power-law or boundary structures (e.g. near rho=1, near cusps)?
2. **Design the representation** (the deliverable): propose and PROTOTYPE the coordinate transform. Candidates to evaluate:
   - Fit log|E| (or a scaled E, e.g. E * w^p for the measured p) with real/imag parts each scaled, with the phase carried separately (e.g. unwrapped phase spline, or a rotating-frame demodulation by the measured carrier).
   - A fully log-domain complex representation (log-polar: ln|E| + i*phase) if the phase splines cleanly.
   - Rescaling the w-axis coordinate itself (e.g. u_w = w^(1/3) or log w already used) in combination with a magnitude scaling.
   - Whether rho (and theta_c) need a matched transform for consistency.
   Validate the prototype by fitting at 4 nodes/axis and measuring the held-out eps at off-grid points (should clear 1e-3).
3. **Implement** the chosen representation in `ExteriorPolarChart` (from_values, _assemble, _evaluate_chart, npz serialization, axis-schema), `_fit_tensor_spline`/`_contract_tensor_spline` usage, and the serve/reconstruction path (`reconstruct_farfield` + the likelihood consumption). Keep it backward-compatible or bump the exterior axis schema if the stored object changes.
4. **Verify**: node-exactness (round-trip to machine precision), held-out eps < 1e-3 at 4 nodes/axis (or document the minimum node count needed), and the full exterior probe produces ~70 charts with all eps under the bar.

## Measured facts (re-probe at HEAD before coding)
- |E(w)| ~ w^(-0.60), R^2=0.996 at gamma=0.481, rho=1.679, theta_c=0.419 (chart /tmp/probe_exterior_recursion/chart_astroid_b0_s0_ff_1_1.npz)
- Spline exact at nodes (1e-17); w-midpoint abs error 0.02-0.08; max|F| ~ 1.4; eps bar 1e-3
- Envelope label: `farfield_envelope_from_partition(partition, FARFIELD_KERNEL_SUM)` (channels.py ~1251), demodulated by exp(+1j*w*t_min)
- Serve: `_contract_tensor_spline` (surrogate.py ~1292), `_evaluate_chart` exterior branch (~2722), `reconstruct_farfield` (channels.py)
- Training: `ExteriorPolarChart.from_values` (~1590), `from_engine` (~2916), `_build_farfield_chart` (surrogate_training.py ~2714)
- Probe: `scripts/probe_exterior_recursion.py` (4x4x4, w 4/decade, engine 80, n_heldout 100)

## Constraints
- Fast tests. Follow AGENTS.md.
- The fix is a COORDINATE/SCALE transform — NOT a node-density increase. Adding w-nodes to out-resolve the decay is explicitly NOT acceptable as the primary fix.
- The complex envelope's phase must be handled correctly (round-trip to machine precision); a naive log-magnitude that loses phase is insufficient.
- If the spatial axes need matched transforms (rho near rho=1, theta_c, gamma), design and implement them consistently.
- The serve path and reconstruction must round-trip to machine precision.
- Plan-gate requirement: each `domain_test_descriptions` spec names exactly ONE primary `test_*.py`; no spec may reference another spec's primary file.

## Design note from the driver
The user (domain expert) flagged: "Once you do log|F| to fix the frequency axis, the spatial axes will also have to be fixed because you can't just fix the ordinate." So treat the coordinate transform as a COHERENT reparameterization of the whole exterior chart, not an isolated ordinate change. Design for the general case; a clean solution that also helps rho/theta_c/gamma is preferred over a w-only patch.
