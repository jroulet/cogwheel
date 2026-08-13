# Build Brief: On-axis cusp serving — fix the fold-arm pair selection + Pearcey on-axis control degeneracy

## Mission

Make the cusp arm serve interior cusp sources on the SYMMETRY AXIS (the
"teardrop neck"), where it currently refuses on both parities.  This is the
measure-zero-but-principled gap documented in
`.claude/spec/todo.d/lensing_fold_pair_drops_third_cusp_image.md`.

## Measured facts (at HEAD c24dee4)

- Interior source on the astroid cusp symmetry axis (gamma=0.5, src=(0.7,0),
  rho=0.495) has THREE coalescing images: tau=1.182 (min), 1.193 (saddle),
  1.193 (saddle — the degenerate symmetric pair).  This is the CUSP
  catastrophe (3 comparable images), not a fold.
- `_airy_fold._merging_fold_pair` picks the delay-adjacent min/saddle pair
  (1.182, 1.193) and SILENTLY DROPS the third image (the symmetric saddle at
  the same tau).  The fold arm then certifies a 2-image Airy form against a
  3-image reality; the uniform-error estimate blows up (measured 12.5 vs bar
  0.05 at w=200) and it refuses.
- The cusp arm ALSO refuses on-axis: `delta_parallel = 0` makes the Pearcey
  control x = 0, mapping the source to the 1-stationary EXTERIOR regime
  (n_stat=1), so the interior calibration bypass (len==3) never fires.
- MORAL FINDING (driver, verified): forcing a serve at x=0 gives |F|=4.33 vs
  exact 3.52 at w=40 (23% error) and the cluster-matching FAILS (0/4 images
  match the single stationary phase -1.641 vs cluster phases
  -0.14/+0.27/+0.27).  So the cusp arm's refusal is CORRECT given the
  current control mapping; the MAPPING is the defect.
- The grid (`operator.F_op`) REFUSES the on-axis interior source at all w
  (no rung serves).
- Off-axis (0.7, 0.05) serves via both arms — generic interior serving works.

## The fix direction (Professor must adjudicate the exact mechanism)

Two coupled defects:
1. **Fold arm**: `_merging_fold_pair` must DETECT a 3-image cusp cluster
   (the symmetric pair at equal delay) and decline (fall through to the cusp
   arm) rather than certify a wrong 2-image Airy form.  This is a fold-arm
   pair-selection defect in `_airy_fold.py`.
2. **Cusp arm**: the on-axis control degeneracy (`delta_parallel=0` → x=0 →
   1-stationary exterior).  A NON-DEGENERATE on-axis control is needed — the
   cusp-adapted angular coordinate `u = d^{2/3}` used by the surrogate's
   wedge/lobe charts (`_wedge_cusp_axis_map` / `_lobe_cusp_axis_map`), or a
   rotated/2nd-order control that keeps x off zero when delta_parallel ~ 0.

The Professor should decide whether these are one build or two, and whether
the on-axis control fix should reuse the surrogate's `u = d^{2/3}` machinery
(the likely right answer — it already exists and is gamma-universal).

## Acceptance
1. An interior source on the cusp symmetry axis (both parities, e.g. astroid
   (0.7,0) gamma=0.5; deltoid analogue) is served by a FAST path (cusp arm
   with the corrected control, or fold arm correctly routing to cusp) with
   the same tolerance as the generic off-axis case — NOT refused, NOT exact
   engine.
2. The fold arm no longer certifies a 2-image Airy form against a 3-image
   cusp cluster (it must detect and decline).
3. No regression: `test_lensing_airy_fold.py` (all classes incl.
   ServedValue*, InteriorCuspServing, Ppgo, DirectCuspVertex,
   ServingLadderDeterminism) green; `test_lensing_fast_path.py` green;
   `test_lensing_operator.py` green (the one-home pin).
4. Refusal-conservative; no live quadrature in the hot path; the exact
   engine is never the serving rung in the cusp neighbourhood (driver
   mandate).

## Constraints
- Fast tests only. Refusal-conservative.
- Reuse existing machinery where possible (`u=d^{2/3}` cusp-adapted axis,
   `_merging_fold_pair`, the arm certification gates).  Do NOT weaken the
   byte-identity / envelope-bar contracts.
- If the fold-arm fix is separable and the cusp-arm control fix is large,
   the Architect may split into two WPs — but the acceptance (on-axis serves)
   must be met by the end of the build.
