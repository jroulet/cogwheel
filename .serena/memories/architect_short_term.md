# Architect Short-Term Observations

## 2026-08-17 build (c3_band_split_zero_refusal) — planning notes

- SHARED band-split mask: Born/c3/above-ceiling ALL zero the envelope
  ABOVE the split (`envelope[~below_mask]=0`, below_mask=dense_w<=split);
  only the BELOW-split populator differs (Born=chart, c3=engine,
  above-ceiling=engine). Simplifier: share ONLY
  `_band_split_mask(dense_w, split)->(band_split, below_mask)` (3 lines) +
  ONE null-split pin; do NOT extract a "serve-below" callable (leaky).
- c3 split point closed form: est(w)=C/w**3, C=sum sqrt|mu||c3|;
  w_split=(S*C/bar)**(1/3), S=20 bar=1e-3. est None (merging pair) =
  whole-draw refuse (coalescence discriminator), NEVER enters a split.
  Null-split: w_split<=w_lo -> whole-band analytic (today's serve);
  w_split>=w_hi OR w_split>150 -> return None (fall through, today's
  refuse). Serve split only w_lo<w_split<w_hi(<=150). Below-split engine
  reachable (w_split<=~60<150).
- above-ceiling per-node: current gate keys on band FLOOR (w_lo tiny ->
  0.00%). Fix: split at W_CEILING=150 — engine below 150, fold_ppgo above
  150 — gated on `150*min_delta_tau >= RHO_END` (all above-ceiling nodes
  resolved). min_delta_tau is a per-DRAW geometric constant; only w varies
  per node. Unresolved above-150 corner (RHO_END/min_delta_tau>150) stays
  refused -> that IS the deferred 2b residual.
- Professor DEFER 2b (tau_c re-gauge / envelope-negligible): unsettled
  empirical Qs (tau_c-as-fn-of-source, region-handover continuity, 54-min
  zero-output risk). Not load-bearing: Q2 clears the resolved high-w bulk.
  Enumerate the unresolved-near-caustic w*delta_min<4 residual as the
  acceptance's permitted "measure-zero named-refusal set". wave_refused
  reads small nonzero (within acceptance if enumerated + provenance cited).
- Escalation guard: analytic-above-split must match engine in overlap
  [w_split,60] to bar 1e-3; miss there falsifies calibration -> STOP.
  NO engine comparison above 150 (no oracle by construction).
- Existing test files to re-point/extend: test_lensing_ppgo_bandsplit.py,
  test_lensing_ppgo_above_ceiling.py, test_lensing_saddle_serve_gate.py,
  test_lensing_born_analytic_reachability.py.
