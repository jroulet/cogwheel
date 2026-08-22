# Driver finding: shell chart off-grid accuracy — needs denser gamma' near the wall

## Measured (after the full bake, shipped artifacts)

The low-w shell chart (demodulated-difference, macro-lead carrier) is
NODE-EXACT (1e-17 at on-grid points) with NO poles — the representation is
correct. But the OFF-GRID served accuracy fails the 1e-4 bar in the
high-gamma' region:

- worst off-grid gamma' interpolation served rel-err = 2.0e-1 (20%) at
  gamma'_mid = 0.994, rho = 0.96, theta = 1.57, w = 0.10
- typical off-grid error 8e-5 to 9e-3 depending on cell
- grid: 14 gamma' x 10 rho x 16 theta x 16 w over gamma' in [0.05, 0.995],
  rho in [0.6, 1.4], w in [0.02, 1.0]

## Diagnosis

The residual R = f_schwinger - born_lead_carrier varies fastest near the
parity wall (gamma' -> 1, the sqrt(mu_macro) collapse region), and the
gamma' grid's linear spacing (np.linspace over [0.05, 0.995]) under-samples
it there. The worst cell (gamma'_mid=0.994) is between the two highest
gamma' grid nodes (0.9888, 0.9925) and (0.9950) — a wide gap in a region
where the residual changes fast. This is a NODE-DISTRIBUTION issue, not a
representation failure: the demodulated-difference representation is correct
(node-exact, no poles); the 1e-4 bar needs denser gamma' sampling toward the
wall.

## Options (Professor to rule, or driver decision)

1. Denser gamma' grid near the wall: replace the linear gamma' spacing with
   a wall-concentrated spacing (e.g. cluster nodes toward gamma' = 1, where
   log(1-gamma') is linear), so the residual's fastest-varying region is
   resolved. Re-bake (tens of minutes).
2. Accept a documented accuracy relaxation near the wall: the shell chart's
   high-gamma' cells (gamma' > ~0.95) decline to the exact engine (which
   works there), with the chart serving only gamma' <= ~0.95 where it meets
   1e-4. Simpler, but cedes the near-wall shell to the engine.
3. Combine: denser gamma' near the wall + decline the extreme wall sliver
   (gamma' > ~0.98).

Given the saga's lesson (never force a representation; the engine works), and
that the high-gamma' shell is a small population, OPTION 2 (decline gamma' >
~0.95 to the engine) is the pragmatic minimal fix — the chart covers where
it's accurate, the engine covers the near-wall sliver, both correct.

## Impact on the demand map

The near-wall shell (gamma' > ~0.95) is a small fraction of the wall-band
residual; declining it to the engine is a minor coverage loss vs the current
(20%-wrong) serve, which is unacceptable at 1e-4. The post-serve census must
reflect the gamma' > 0.95 engine fall-through.
