# Inspector Short-Term Observations

## 2026-08-14 — wire_serving_artifacts (F077) RE-REVIEW pass 3 — VERDICT: ISSUES (1 design; INS-2-001 RESOLVED)

Scope: re-check INS-2-001 (band-split shape crash) + full uncommitted-tree review of the
reachability-lift build. Files: likelihood.py (_born_residual_analytic L2237-2410,
_amplification_coefficients L2412-2569), born_residual_chart.py (covers w-guard, load),
marginalized_likelihood.py, surrogate_census.py (band-split mirror), scripts/train_born_residual.py,
DATA_CONTRACTS.yaml (WP-G), data/born_residual_chart.npz, 4 test files.
Fast suite: born_analytic_reachability + born_residual_wiring + ppgo_bandsplit =
129 passed / 4 skipped / 31s.

### INS-2-001 — RESOLVED (verified by hand + tests)
`_born_residual_analytic` now serves carrier + residual + ppgo over the FULL `dense_w`; the
band-split is applied by ZEROING the reconstructed envelope above w_trust
(`envelope[~below_mask] = 0.0`) then `reconstruct_farfield(dense_w, envelope, ...)` over the full
band — mirroring the buried _surrogate_coefficients twin. `chart_w = dense_w[below_mask]` is used
ONLY for the trained-band `covers(gamma, rho, chart_w)` refusal check, never fed to the geometry.
No shape ValueError. The old tripwire test `test_band_split_serve_raises_shape_defect` is GONE;
MapBandSplitTraceTestCase now pins the split PREMISES (w_trust strictly in-band, beyond-wall cell
keeps whole band) + NullSplitIdentity byte-exact. Class closed.
NOTE (not a finding, test-coverage gap owned by test author, acknowledged in the suite docstring):
no test yet exercises a FIRING band-split serve for finiteness/`max|k_split-k_nomap|>0`; only the
premise + null-split identity are pinned. Architect assigned the positive-trace invariant to the
test author separately.

### FINDING INS-3-001 (design / spec-invariant, Inspector-owned) — DATA_CONTRACTS false coverage claim
DATA_CONTRACTS.yaml born_residual_chart entry (L361, REWRITTEN this build by WP-G). WP-G correctly
fixed the false "attaches at construction time" claim (now describes `_AUTO_BORN_CHART` sentinel
auto-attach with refuse-to-None). BUT the rewrite RETAINS "covering the far exterior (rho > 2) on
both parities". The SHIPPED artifact is ASTROID-ONLY: gamma_grid = [0.05,0.081,0.131,0.212,0.343,
0.556,0.9] (all < 1.0, no saddle gamma>1 node), rho_grid=[2..4] (5), log_w_grid -> w in [5,60] (10).
7x5x10. Bidirectional: retrain to add saddle nodes (spec right, artifact incomplete) OR narrow text
to astroid parity gamma<=0.9 (artifact right, spec wrong). Route to triage; Librarian propagates
after direction chosen. Secondary (fold into same finding): the gate description says
`born_chart.covers(gamma, rho)` but code uses `covers(gamma, rho, chart_w)` (the w-band trained
refusal guard is omitted from the prose). CODE IS SAFE regardless — `covers()` refuses gamma>0.9 so
a saddle draw always falls through to the exact engine; this is a doc-accuracy defect, not a
serving bug. (This is the surviving half of last pass's INS-2-002; the band-split half of INS-2-002
is now moot since the split works.)

### CONFIRMED CORRECT (not findings)
- WP-A load(): shipped npz loads; schema/hash verified; covers() w-guard checks log(w) in
  [log_w_grid[0], log_w_grid[-1]].
- WP-C auto-attach sentinel; load anomaly -> None + RuntimeWarning (pure engine).
- WP-D get_init_dict round-trip both classes; WP-E marginalized threads sentinel. Tests green.
- WP-F census band-split mirror: NOT re-keyed to new arithmetic — comment documents that WP-B's
  Born intercept reuses the SAME three likelihood methods (_ppgo_cell_coords, _ppgo_band_split,
  _ppgo_cell_ceiling) verbatim, so the existing mirror (beyond-ceiling guard
  `w_hi <= eff_ceiling and w_lo < w_trust < w_hi`) already reflects it. Matches production negation
  of `w_hi > eff_ceiling`. Consistent.
- Dispatcher ordering sound: surrogate -> ppGO-above-ceiling -> saddle far-field (gamma>1) ->
  Born residual (WP-B, gate refuses gamma>0.9) -> seed/fiducial/ratio engine. No double-serve;
  gamma>1 draws skip Born via covers().

### CARRIED (pre-existing doc-staleness -> Librarian): exterior_polar V5 2D carrier tag staleness;
saddle exterior raw-theta text; region vocabulary (lobe_exterior/interior/wedge) absent from SPEC.
