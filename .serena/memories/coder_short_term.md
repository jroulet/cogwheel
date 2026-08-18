# Coder Short-Term Observations

## 2026-08-17 (WP3 above-ceiling ceiling-split serving, likelihood.py)
- Rewrote `_ppgo_above_ceiling` (~L2227) from whole-band fold_ppgo serve to
  a ceiling-split serve: exact engine below W_CEILING_SCHWINGER_QD(150) via
  `_engine_envelope_below_split`, fold_ppgo above 150, stitched over the
  FULL dense_w under FARFIELD_KERNEL_SUM. Kept the entry guard w_max>150 and
  the real-delay (>=2 real delays, positive deltas) computation.
- GATE REKEYED to the ceiling: replaced `w_lo*min_delta_tau < RHO_END` with
  `W_CEILING_SCHWINGER_QD*min_delta_tau < RHO_END` (named constant, NOT a
  hardcoded 150.0). Admits only when the LOWEST above-ceiling node (~150) is
  resolved => every >150 node resolved => fold accurate. Dropped `w_lo`
  local (only the old gate used it). Conservative in the whole-band-above
  case (keying on 150 <= w_lo => stricter than the old w_lo gate there); an
  unresolved above-ceiling corner returns None -> engine ->
  SchwingerCertificationError (the enumerated deferred-2b residual).
- CRITICAL EDGE the plan did NOT enumerate — WHOLE-BAND-ABOVE (w_lo>=150,
  the deep-massive-lens asymptote this rung most needs to serve): the entry
  guard only fixes w_hi>150, NOT w_lo<150. `_band_split_mask(dense_w,150)`
  returns band_split=False + below_mask ALL-TRUE for an inactive split
  (split<=w_lo, since w_hi>150 always). That all-True fallback is tuned for
  BELOW-populator rungs (trusted floor ABOVE the band); this ceiling rung
  has INVERTED polarity (engine populates BELOW). Using it directly would
  route every >150 node into `_engine_envelope_below_split` ->
  _evaluate_envelope on all-above-150 nodes -> SchwingerCertificationError
  (a hard REGRESSION vs HEAD, which served these via whole-band fold). FIX:
  `if not band_split: below_mask = np.zeros(dense_w.shape, dtype=bool)` +
  skip the engine call (`engine_below = zeros`) when not band_split. Then
  fold_envelope[below_mask]=0.0 is a no-op and envelope = zeros + fold =
  HEAD's whole-band fold envelope BYTE-IDENTICAL (0+finite complex is exact).
  band_split<=>w_lo<150<w_hi<=>below_mask.any(), so band_split drives BOTH
  the mask inversion and the engine skip (single source).
- STITCH: engine_below (zeros above split, from `_engine_envelope_below_
  split`) + fold_envelope (zeros below split via [below_mask]=0.0). Inverted
  analog of the _band_split_mask docstring's `envelope[~below_mask]=0.0`
  convention (fold is the ABOVE populator here, not below). No THIRD inline
  copy of the split-point arithmetic (w_lo<split<w_hi / dense_w<=split) —
  that lives solely in `_band_split_mask`; only the polarity inversion and
  the standard populator-zeroing are inline. Preserved the finite_mask guard
  on fold_ppgo output verbatim.
- FOR INSPECTOR: (1) tests in test_lensing_ppgo_above_ceiling.py (GateBorders
  test_d_below_rho_end, GateFallthrough, NoSurrogate, SingleImage) were
  written against the OLD w_lo gate + whole-band serve; Test Developer must
  re-derive expectations against the ceiling gate (150*mdt vs RHO_END) and
  the split. GateBorders `test_b2_resolved_gate_passes`/`test_d_below_rho_
  end` pin the OLD w_lo threshold — will need updating. (2) Byte-identity
  claim for whole-band-above (w_lo>=150) is REASONED (0+x exact), UNVERIFIED
  by execution here. (3) The split-case (w_lo<150<w_hi) engine-below path is
  a NEW accuracy improvement (exact engine, not fold, below 150) — needs a
  boundary-continuity check at the 150 stitch that only the Test Dev/Prof
  can run.

## 2026-08-17 (WP2 saddle c3 band-split serving, likelihood.py)
- Rewrote `_saddle_farfield_analytic` (~L2370) from whole-band admit/refuse
  to per-draw band-split: analytic ZERO envelope ABOVE the c3 certificate
  split `w_split`, exact Schwinger engine BELOW, stitched over the FULL
  dense_w under FARFIELD_KERNEL_SUM (INS-2-001 full-length: saddle_kernels
  rows == dense_w.size, never sub-slice w). Gate `_saddle_farfield_analytic
  _serves` kept as the FIRST call: True == HEAD admit (whole-band zeros, no
  engine, byte-identical); False decomposes into est-None + separation
  refusals (return None == HEAD refuse) vs. the new splittable middle.
- New module helper `_saddle_c3_split_point(real_images, source, matrix)`:
  `w_split = w_ref*(S*est/bar)**(1/3)`, w_ref=1.0, S=_SADDLE_FARFIELD_SAFETY
  (20), bar=_SADDLE_FARFIELD_CERT_BAR(1e-3); returns None when est is None
  (merging pair -> WHOLE-DRAW refuse, never a split). Exact cube-root
  inversion of the certificate, w_ref-independent under the pure C/w**3 law
  (Professor ruling). NOT hardcoded.
- New module helper `_saddle_min_image_sep(real_images)` factors the
  min-pairwise-separation backstop so the gate AND the split rung share the
  IDENTICAL >= _SADDLE_FARFIELD_MIN_IMAGE_SEP test; DRY'd the gate to use it.
  Added an explicit `if min_sep is None: return False` guard in the gate
  (documented-unreachable; >=2 images guaranteed above) to clear a Pyright
  `>=`-on-None false positive introduced by the refactor -- NOT a bare
  type:ignore.
- New METHOD `_engine_envelope_below_split(self, lens, dense_w, below_mask)`:
  full-length complex envelope, exact engine on below_mask nodes, 0.0 above.
  DEVIATION 1 (gauge, physics override of the plan's literal wording): the
  plan said reuse `_evaluate_envelope`'s 2nd return, but that is the SACR-C
  `partition.envelope` (critical_delay carrier) which `reconstruct_farfield`
  does NOT invert -- used `farfield_envelope_from_partition(partition,
  FARFIELD_KERNEL_SUM)` instead (matched reduced-`_frame_phase` demod/re-mod
  -> machine-precision round trip; mirrors Born/F069). DEVIATION 2 (pad
  node): plan's `pad_w=dense_w[below_mask].max()` collapses a size-1 sub-band
  (unique([node,node])==[node]); used `pad_w=float(dense_w.max())` -- a
  distinct node strictly above the split (w_split<w_hi), dropped by `keep`,
  harmless when sub-band >=2 (w-independent geometry). `keep =
  searchsorted(partition.w, sub_w)` mirrors `_evaluate_envelope`'s own
  internal index scatter, so it is index-consistent.
- SPLIT-BRANCH PROOF: gate False + separation OK ==> accuracy bar failed at
  w_lo ==> w_lo < w_split strictly; with w_split<w_hi and <=150 the split is
  STRICTLY inside (w_lo,w_hi), so `_band_split_mask` always returns
  band_split=True there. Null-split boundaries: w_split<=w_lo -> gate True
  path (byte-identical HEAD admit); w_split>=w_hi or >150 -> None
  (byte-identical HEAD refuse).
- FOR INSPECTOR: (1) census mirror in surrogate_census.py (~L535) still
  calls the WHOLE-BAND `_saddle_farfield_analytic_serves` counting gate --
  it counts admits, NOT band-splits; out of this likelihood.py-only WP's
  scope, candidate follow-up WP to count the split serves. (2) ESCALATION
  contract unused: no evidence the analytic zero above w_split misses the
  engine within [w_split,60] -- that would falsify the c3 calibration, not
  the plumbing; left for the Test Developer / Professor to exercise.

## 2026-08-17 (band_split_mask shared helper, WP1)
- Factored the band-split arithmetic into module-level
  `_band_split_mask(dense_w, split) -> (band_split, below_mask)` in
  cogwheel/lensing/likelihood.py (placed right after
  `_saddle_farfield_analytic_serves`, ~L634). Convention documented in its
  docstring: band_split iff split STRICTLY inside (w_lo,w_hi); below_mask
  all-True when inactive; ALL rungs zero the envelope ABOVE split via
  envelope[~below_mask]=0.0 — only the below-split populator differs, so
  the helper shares arithmetic ONLY (no serve-below callable — Simplifier
  ruling).
- Refactored `_born_residual_analytic` to consume it: dropped the now-dead
  local `w_lo` (helper computes it), KEPT `w_hi` (Born-only eff_ceiling
  guard `w_hi>eff_ceiling -> w_trust=None` still needs it, stays in Born).
  Unpack as `_band_split, below_mask = ...` — band_split is unused
  downstream in Born (only below_mask is), underscore-prefixed to signal
  intentional non-use. Served coefficients byte-identical to HEAD.
- SCOPE NOTE for Inspector: the surrogate-path serve rung at L1813/1835
  still has an INLINE copy of the same `band_split = w_trust is not None
  and w_lo < w_trust < w_hi` / below_mask arithmetic. WP1 was Born-only by
  design; consolidating that second rung onto `_band_split_mask` is a
  candidate follow-up but was deliberately NOT touched here.
