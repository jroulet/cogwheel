# Saddle forensics audit — Professor domain review (2026-08-08)

## Q1 — Lobe normalized-radius disease
Confirmed pathological: r_deltoid vanishes at deltoid cusps by |dtheta|^(1/3), same power law as astroid cusps. The rho_lobe coordinate therefore loses radial resolution near cusp directions. Milder because deltoid lobes have smaller angular extent (~pi/3 vs ~pi/2 per astroid quadrant). Cure: either cusp-adapted u = d^(2/3) coordinate (wedge pattern) or cusp carve-out + subdivision (pragmatic but loses coverage). Test: transverse rho_lobe cut at fixed theta_local ~2deg from cusp ray, error should grow rapidly as rho_lobe -> 1.

## Q2 — Ghost kernel parity gate
The Morse reference exp(-0.5j*pi) is CORRECT for both parities. The fold is parity-blind (Fermat-potential reflection symmetry). The merged saddle at any fold has the same Berry phase -pi/2 regardless of whether the merging pair is (min, saddle) or (saddle, saddle). NO parity-dependent branch needed. Verification: compare ghost-kernel phase to engine residual R = F_op - ppGO at a saddle config near fold; phase alignment within few degrees confirms correct reference.

## Q3 — Lobe cusp carve-out constant
Recommend physical y-unit exclusion (~0.1-0.15 y-units from cusp vertex) mirroring the exterior polar chart's 0.2 y-units, scaled for the deltoid's smaller extent. The separation-gate connection: near-cusp is where |tau_a - tau_c| -> 0, making E(w) non-smooth — both the spline and the SACR-C construction degrade there.

## Q4 — Saddle exterior polar chart
Scalar-reach rho = 1 + |y| - caustic_reach(gamma) is functionally correct as a coordinate frame (envelope is smooth, drho/d|y| = 1) but geometrically approximate: rho does NOT align with the deltoid boundary directionally. Sources at same rho but different theta_c may have very different physical proximity to the caustic. May need higher angular resolution. Test: served-vs-engine accuracy sweep with angular uniformity check across theta_c bins; monotonic decay of envelope magnitude along radial rays.

## Build review 2026-08-08 (lobe subdivision + cusp carve-out + ghost saddle + carrier flip)
Fast suite: 121 passed, 1 skipped, 1 xfailed in 30s. All four spec areas verified:
- LobeSubdivisionTestCase: additive keys present, children have strictly lower eps, packed >= 1, xfail on sparse grid, self-falsification PASSED.
- LobeCuspProximityTestCase: near-cusp tile refused, far-from-cusp tile not refused by proximity, mutation check (_LOBE_CUSP_EXCLUSION_DISTANCE=1e-9) admits the tile — self-falsification PASSED.
- GhostKernelSaddleTestCase: finite |kernel| > 0 on saddle parity, source outside fold, self-falsification (empty source, far interior) PASSED.
- LobeCarrierFlipRefusalTestCase: carrier_flip=True, ladder_served_gap=True, max_achieved_depth=0, packed=0, self-falsification (normal build packs) PASSED.
No concerns. Heavy full-sampling validation is operator-deferred.