# Build: distance-gate the near-caustic serves; stop non-merging-pair folds; consume the ghost

## Mission

F075: exterior to the caustic the merging pair is COMPLEX (the ghost pair);
the two real images are the far (min, saddle) pair. `_airy_fold.fold_amplification`
selects that far pair and applies the Airy merging correction to it, and the
result SHIPS for `60 < w <= 150` (`operator._positive_parity_grid` offers the
arm before `f_schwinger`, no accuracy gate) at 15-216% relative error.

Fix both sides: (1) the fold arm refuses when the selected pair is not a real
merging pair; (2) a new `ppGO + ghost` rung serves the exterior band in the
ARM LADDER (the chart path's mid-band serving already consumes the ghost via
the gated channels wrapper; the ladder does not). Then audit the
training-path mirrors so the (deferred) surrogate training uses the
corrected gates.

WHY THIS PRECEDES TRAINING: `channels.evaluate().exact_total` is BOTH the
runtime fallback and the LABEL ORACLE for every tabulated residual
(`E_ks = truth - ppGO`). In the contaminated band the tabulation would
record `arm - ppGO`, not `truth - ppGO`. Fixing the ladder fixes the labels.

## Measured facts (SHA 87e62bb unless noted; oracle = f_schwinger via the
## reconstruction in /tmp/ppgo_cert/oracle.py, validated 1.4e-16, F069-safe)

1. Fold arm error at exterior 2-image configs: flat in w, does NOT vanish
   with distance — |y|/r_caustic = 1.40, w=70: ppGO errs 1.6e-4, fold errs
   4.5e-1. Extent (108 pts, gamma {0.3,0.5,0.7}, |y|/rc 1.05-1.40): 100% of
   served points over 1e-2, 94-97% over 1e-1, worst 2.16.
2. It ships: `ChangRefsdalChannels.evaluate().exact_total` equals the fold
   value EXACTLY (residual 0.0 after `t_min` demodulation) at tested configs,
   w=70/100. `w <= 60` is safe (exact DD batch). Above 150, `select_branch`
   can divert first.
3. Ghost convention validated: `+ kernel * exp(1j*w*tau_c)` (the `-` and
   conjugate variants are worse). `ppGO + ghost` closes the exterior residual
   to 1e-3..4e-7 for |y|/rc >= 1.15 — 10x to 1e5x better than the served
   fold value at EVERY measured point.
4. CAVEAT BAND: |y|/rc = 1.05 (w*Im tau_c <~ 0.3): ghost overshoots (5.2e-1
   -> 1.6e1). Both stationary-phase forms fail; that band must refuse.
5. `geometry.GhostAbsentError` (87e62bb) is raised iff four real images
   prove no ghost exists; bare `GhostDomainError` means unavailable ->
   refuse, never zero.
6. The admission hole: `nearest.distance < _ETA_MAX_FOLD (= 0.3)` is
   satisfiable arbitrarily far exterior; no leg checks the selected pair is
   real-merging. A 2-image census ALWAYS yields a (min, saddle) "pair".

## Scope

IN:
- `_airy_fold.fold_amplification` / `_merging_fold_pair`: refuse when the
  census has fewer than 4 real images (no real merging pair exists). The
  interior 4-image path is untouched.
- New exterior rung in `operator._uniform_arm_value` (or a sibling the
  ladder tries after fold/cusp): `geometric_amplification + ghost_kernel`
  with carrier `exp(1j*w*tau_c)`, gated on `w * Im(tau_c) >=` a floor you
  derive/calibrate so the caveat band (fact 4) refuses — state the floor and
  its measured margin. `GhostDomainError` -> refuse. Do not serve the ghost
  rung on the interior (GhostAbsentError -> the rung does not apply; raw
  ppGO's interior certificate was built separately today).
- Training-path mirror audit: `surrogate_training`, `surrogate_census`
  (`characterize_sample` re-gate is already filed in todo.d as
  lensing_census_mirror_regate — do it here), `scripts/train_lens_surrogate.py`:
  find every mirror of a serving gate that today's changes (interior rung
  re-gate, fold refusal, ghost rung) make stale; re-gate them to match, or
  report any you cannot with the reason. The owner has conditioned the
  future surrogate training run on this audit.
- Fast tests with derived fixtures; per-routing-decision pins live in the
  file that owns the predicate.

OUT:
- `_pearcey_cusp.py` (patched separately, F074), `likelihood.py`'s interior
  rung (just rebuilt), any surrogate retraining, slow tiers.

## Acceptance

- Exterior 2-image configs: fold arm refuses; the ghost rung serves with
  measured max error under the 1e-2 arm bar on the |y|/rc >= 1.15 band at
  w in (60, 150] against the F069-safe oracle (REPORTED numbers, not a
  permanent slow test); caveat band refuses to the engine (count reported).
- Interior fold behavior byte-identical (the 4-image path): existing
  airy_fold + fold_ppgo_handoff suites green unmodified except pins that
  asserted the exterior misbehavior.
- The mirror audit report: every stale training-path gate listed as
  re-gated or explained.
- RETROACTIVE LABEL CHECK: determine whether `certified_ppgo_map.npz`
  (trained 2026-08-03) and `born_residual_chart.npz` (2026-08-04) drew any
  labels through `_positive_parity_grid` in `60 < w <= 150` at exterior
  cells (read the training scripts' oracle route; if it went through
  `F_op`/`exact_total` in that band, say WHICH cells are suspect). Report
  only — flag cells for retraining, do not retrain.

## Constraints

- Branch claude-dev. Spec/TODO workflow applies; retire what you complete.
- Assert VALUES against the oracle with tolerances, not code paths.
- NEVER use `channels.evaluate().exact_total` as a reference without pairing
  it (t_min demodulation) — see FINDINGS F075's phantom column and memory
  `pair-frames-before-scoring`. Pairing gate first, always.
