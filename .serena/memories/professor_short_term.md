# Professor short-term — Build 8h-a/b ppGO ceiling review (2026-07-23)

Reviewed `cogwheel/tests/test_lensing_ppgo_bandsplit.py` (66 tests, all PASS, 15s,
env cogwheel-newlal). Verdict PASS. Coverage of the 5 specs:
1. Truncation-on-refusal (TruncationOnRefusalTestCase): stubbed _measure_cell with
   monotone saddle refusal above w*(angle); cell CERTIFIED, w_ceiling = min-over-angles
   accepted-prefix endpoint (independent oracle from `_w_nodes`+w* law, 1e-12), w_cert =
   sup-over-w floor on prefix. Reachable-red: `_max_accepted_prefix` swapped for no-trunc
   -> STATUS_INVALID. Physically correct: prefix stays valid; whole-cell invalidation
   would over-refuse, serving beyond w* would be unsound.
2. Cell-ceiling band-split guard: VERIFIED production `_surrogate_coefficients`
   (likelihood.py ~L1580) genuinely applies eff_ceiling=min(wall, cell_ceiling); w_hi>eff
   -> no split -> whole-band refuse. Test helper `_dispatch_band_splits` reimplements the
   decision but uses REAL `_ppgo_band_split`/`_ppgo_cell_ceiling` reads. Reachable-red both
   ways (honor_ceiling True/False).
3. Loader hard-refuse: exercises REAL CertifiedPpgoMap.load / use_certified_ppgo_map;
   ceiling-less -> KeyError, tampered value -> hash ValueError; both return False, global
   stays None, all queries UNKNOWN.
4. Strata trim respects ceiling: beyond-ceiling stratum KEPT (tail stays charted/refused,
   not handed to ppGO); reachable-red dropping ceiling wrongly trims.
5. Outer-annulus rho cap: real accessors; inclusive at rho_measured_max, UNKNOWN at
   nextafter; consumer routes beyond-measured to 'keep'/refuse; reachable-red inf-twin
   wrongly certifies. Sound: one finite sample can't certify [4,inf).

Diagnostic PNGs fresh (12:21): ppgo_truncation_per_angle_endpoint, outer_annulus_rho_cap
_step, ppgo_band_split_node_match, ppgo_sup_over_w_floor. (Could not render images — no
Read/image tool in this agent's toolset; validated via the independent-oracle numeric
asserts that back the same plotted quantities.)

Minor note (CONCERN-adjacent, not blocking): band-split guard test reproduces the
decision line in-test rather than calling _surrogate_coefficients end-to-end; fidelity
relies on that reimplementation staying in sync with production (which I confirmed by
reading the method). Heavy full-sampling/PP validation is operator-deferred.
