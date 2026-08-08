---
section: Backlog
depends_on: [lensing_exterior_followup_four_items]
---

- **SADDLE FORENSICS: audit the macro-saddle charts for the same defects found
  in the astroid interior** `[→ spec]` — owner-directed 2026-08-06. Sequenced
  after the exterior work (see this fragment's `depends_on:`). Six
  questions; what is already established is marked.

  a. **Are the deltoid interior charts similarly ill-adapted?** VERY LIKELY.
     `LobeInteriorChart` interpolates on
     `rho_lobe = |y - centroid| / r_deltoid(theta_local)` — the SAME
     normalised-radius pattern as the wedge. A deltoid has THREE cusps, so
     `r_deltoid` carries `theta^(2/3)` at each and the normalisation drags it
     to every radius, `w`-independently. PREDICTED, NOT YET MEASURED: run the
     same 1-D transverse cut that settled the wedge (`s` vs raw `theta` vs
     `d^(2/3)` toward the nearer cusp) on a lobe tile.

  b. **Do they adaptively subdivide?** YES — since 2026-08-08.
     `_subdivide_lobe_tile` (a thin wrapper over the shared `_subdivide_tile`,
     the splitter/builder/gate triple, not a third copy) is wired into the
     gated-lobe branch of `_train_band_charts`. A gated lobe tile is
     recursively halved in lobe-local `(rho_lobe, theta_local)` (midpoint
     splits) until it clears `interior_eps_max` or hits
     `MAX_SUBDIVISION_DEPTH`; a carrier-flip (`CarrierDiscontinuityError`)
     child is recorded as a ladder-served gap and never recursed
     (subdivision cannot fix critical-basin phase discontinuities).

  c. **Are the cusps cut out?** YES — resolved 2026-08-08.
     `_lobe_interior_tiles(admission, cusp_angles, n_per_side)` DOES
     cusp-ALIGN via `_cusp_aligned_theta_tiles`, so no tile straddles a cusp
     ray or the lobe-local `+-pi` seam; and the carve-out question is
     ANSWERED: `_SaddleLobeAdmission.admits`'s existing `eta_max` tube-shell
     nearest-distance exclusion already rejects tiles within ``eta_max`` of a
     deltoid cusp — the cusp vertices are points of the caustic cloud the
     test probes against — so no separate cusp-ball carve-out is needed
     (Professor ruling; the redundant constant `_LOBE_CUSP_EXCLUSION_DISTANCE
     = 0.1` records the decision, sized downward from the exterior
     `_CUSP_EXCLUSION_DISTANCE = 0.2`).

  d. **Is the region exterior to the deltoids ill-posed?** LIKELY WORSE than
     the positive-parity exterior. It is charted by `FarFieldChart` in
     `(s, d)`, whose foot degeneracy is already measured
     ([[2026-08-07_polar_rechart]]) — and with TWO
     deltoids there are two separate caustic curves, so a source can be
     near-equidistant from feet on DIFFERENT curves, not merely on different
     arcs of one. Measure `tie_ratio` over the saddle exterior.

  e. **The inter-lobe corridor.** Already an open region in
     [[lensing_coverage_map]] ("Inter-lobe corridor (region 2). Settle by
     probe"), and the lobe frame already carries `corridor_half`. This is the
     region with NO natural centroid — neither lobe's polar frame is right —
     so it may want its own coordinate or explicit ladder service.

  f. **Is the ghost path valid for the saddle at all?** MECHANICALLY yes,
     UNVALIDATED in fact, and its branch pin is positive-parity reasoning.
     `ghost_kernel` has NO parity gate — it works off the macro matrix,
     extracts the complex-conjugate quartic-root pair, and refuses only when
     none exists. The first saddle-parity smoke test landed 2026-08-08
     (`GhostKernelSaddleTestCase` in `test_lensing_lobe_subdivision.py`;
     structural only — finite, non-zero kernel, no crash/NaN at `gamma > 1`),
     so "no test calls `ghost_kernel` at `gamma > 1`" no longer holds — but
     the branch VALUE is still unpinned. The `+-sqrt` branch is pinned by an
     explicitly positive-parity argument (`geometry.py:2343-2344`): "the real
     merged saddle, which the two real images continue into across the fold
     ... has Morse index 1, i.e. amplitude phase `exp(-0.5j*pi)`". For a macro
     saddle the merged pair's Morse indices differ, so that pin may be WRONG —
     and a wrong branch is a SIGN ERROR in the subtracted term, not a small
     inaccuracy.

     CONSEQUENCE for [[lensing_exterior_followup_four_items]] item 1: do NOT
     assume a wired `MINUS_GHOST` carries to the saddle. Re-derive the branch
     selection for `det A < 0` and pin it with a saddle-gamma test before
     enabling it there. Currently harmless ONLY because the label is never
     stamped for either parity.

  PROVENANCE worth keeping: the wedge path was a DEGRADED COPY of the lobe
  path. The brief said "transcribe the lobe path"; the plan gate then trimmed
  the cusp alignment the lobe actually has. So the lobe is better than the
  wedge was, and still carries the normalised-radius disease.
