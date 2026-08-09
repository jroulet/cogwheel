# Build Brief: Exterior cusp-exclusion cut (both parities)

## Mission

Fix the exterior tile-count explosion (~500 charts instead of ~70) by placing the correct cusp-exclusion cut so tiles never straddle the near-cusp envelope cancellation band. The cut must cover BOTH the positive-parity astroid cusps and the macro-saddle deltoid cusps.

## Background (measured, driver probe 2026-08-08)

A 4x4x4 exterior probe with recursion live was killed at 477 charts (tracking to ~500+, same as the OLD baseline) — the cusp-adapted `u = d**(2/3)` coordinate from the previous build did NOT reduce tile count. On-disk chart analysis + direct engine probes found the real cause:

- The FARFIELD_KERNEL_SUM envelope has a **near-cusp zero cancellation band**: `|E| = 0.00000` for `d_angular ≲ 0.024` rad from the π/2 cusp, jumping to ~0.09 at `d_angular = 0.028`. The two-image kernel-sum destructively interferes to ~0 on the cusp ray (image count stays 2 throughout — NOT an image-count flip).
- The zero-region is **rho-dependent**: at `rho=1.54` (tile center) the envelope is zero across the WHOLE tile theta range; at `rho≈1.51` (near the tile's inner edge) it turns on sharply.
- The tiler's `_exclude_near_cusp` uses `_CUSP_EXCLUSION_DISTANCE = 0.2` source-plane units from a cusp vertex. Measured failing π/2-cusp tiles have nearest-corner distances **as low as 0.132** (corner 0.206 admitted and fails). The exclusion radius is too tight for the envelope's structure.
- Failure concentration: near_0_cusp 41/100 fail, mid-theta 8/44 fail, near_π/2_cusp **123/179 fail** (median eps 0.0039, max 274).
- `_exclude_near_cusp` (surrogate_training.py:1676) checks **astroid cusps only** — saddle deltoid cusps are explicitly skipped ("not relevant for the caustic-centre-fixed exterior polar tiling").

## Work

1. **Measure the envelope turn-on structure**: for a sweep of `gamma` across the exterior band (e.g. 0.2, 0.4, 0.6, 0.8, 0.92) and for both parities, measure the source-plane distance from each cusp vertex at which `|FARFIELD_KERNEL_SUM|` turns on (crosses a small threshold, e.g. 1e-3 of the far-field max). Produce a table of turn-on distance vs gamma for astroid cusps; do the same for saddle deltoid cusps. This gives the correct exclusion radius as a function of gamma (or a single conservative constant).

2. **Set the correct exclusion cut**: update `_CUSP_EXCLUSION_DISTANCE` (or make it gamma-dependent) so no admitted tile's nearest corner falls inside the cancellation band. Verify the failing probe tiles (nearest corners 0.132–0.206) would now be excluded.

3. **Cover saddle cusps**: extend the near-cusp exclusion to the macro-saddle deltoid cusps. The saddle exterior uses a scalar `rho` (no directional r_caustic); determine the correct source-plane exclusion for the deltoid cusp geometry. Also check the deltoid-lobe interior path (the retired `_LOBE_CUSP_EXCLUSION_DISTANCE` case) — confirm the cusp-adapted u-coordinate there means no carve-out is needed, or add the correct cut if it is.

4. **Verify tile count collapses**: run the same 4x4x4 exterior probe (one band, recursion live) and confirm it produces ~70 charts (not 500+), with no tile straddling a cusp window and all held-out eps under the 1e-3 bar.

## Measured facts (re-probe at HEAD before coding)
- `_CUSP_EXCLUSION_DISTANCE = 0.2` at `surrogate_training.py:124`
- `_exclude_near_cusp` at `surrogate_training.py:1676`, called at `:2007` in the exterior tiler
- Exterior tiler `_farfield_exterior_tiles` around `:1940-2010`; `_cusp_aligned_theta_tiles` nearby
- `r_caustic(gamma, phi)` in `cogwheel/lensing/chang_refsdal/geometry.py`; `_from_caustic_fixed` in `surrogate.py`
- Probe: `scripts/probe_exterior_recursion.py` (4x4x4, w 4/decade, engine 80, n_heldout 100); outdir `/tmp/probe_exterior_recursion` (477 charts on disk from the killed run — inspect for the failing tile geometry)
- Envelope label: `FARFIELD_KERNEL_SUM` via `farfield_envelope_from_partition`

## Constraints
- Fast tests. Follow AGENTS.md.
- The cut must be measured, not guessed — the acceptance is a real probe run showing ~70 charts and no cusp-window straddle.
- Both parities must be covered (astroid + saddle deltoid).
- Keep the u-coordinate; it is correct and harmless.
- Plan-gate requirement: each `domain_test_descriptions` spec names exactly ONE primary `test_*.py`; no spec may reference another spec's primary file.
