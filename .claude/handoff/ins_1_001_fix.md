# INS-1-001 — stale fold-arm tests regress under WP-1 (F075) — TEST DEVELOPER handoff

**Scope owner:** Test Developer (this is test-authorship: fixture swap / assertion
inversion / docstring correction in `cogwheel/tests/test_lensing_airy_fold.py`).
The Coder does not write tests; WP-1's production guard is correct and already
landed. Everything below is verified by fresh execution against HEAD
(`/home/tejaswi/anaconda3/envs/cogwheel-newlal/bin/python`, gamma=0.5,
`macro_matrix(0.5,0,0)`), not inferred.

## Why the 3 tests are red
WP-1 added `if len(images) != 4: refuse` at every fold-correction entry point.
The three failing tests assert the pre-F075 buggy fold serve on fixtures that
are actually **2-real-image EXTERIOR** configs mislabeled "interior".

Measured image counts (fresh):
- `_CUSP_TIE_SOURCE_OFF_AXIS = [0.7, 0.05]` -> **2 images (EXTERIOR)**. Mislabeled
  "off-axis interior" at L5228-5230.
- `_CUSP_TIE_SOURCE = [0.7, 0.0]` -> 4 images (genuine interior, tie-refusal — CORRECT, leave as-is).
- `_on_axis_cusp_source()` = `[0.2, 1.4142]` -> **2 images (EXTERIOR)**. Docstring
  L5619 "on-axis ... interior source" and class docstring L5628-5636 "BOTH the
  fold and cusp arms serve" are physically FALSE.

Root geometry fact: at gamma=0.5 the astroid cusp (index 1) sits at the top tip
`(0, 1.4142)` with `soft_axis` along y1. Moving `+-dp` along soft_axis stays
EXTERIOR (2-image) in BOTH directions — so **no interior on-axis-cusp source is
reachable via that construction**. The "both arms serve" premise cannot be
salvaged with a fixture tweak; it must be inverted.

## Fix 1 — `FoldCuspTieRefusalTestCase` (fixture swap, preserves intent)
Replace the exterior fixture with a verified **interior off-axis 4-image** source
that has a valid non-tied merging fold pair and serves finite:

    _CUSP_TIE_SOURCE_OFF_AXIS = np.array([0.15, 0.14])   # interior (4-image), off-axis

Fresh verification of `[0.15, 0.14]` at gamma=0.5:
- `find_images` -> 4 (interior).
- `_merging_fold_pair` -> not None; `tau_minus - tau_plus = 0.255 > 0`
  (satisfies `test_merging_fold_pair_returns_pair_off_axis` +
  `assertGreater(tau_minus, tau_plus)` at L5285).
- `fold_amplification(w=500, .., gamma=0.5, kappa=0.0)` -> finite complex
  (satisfies `test_fold_amplification_serves_off_axis` L5289-5301).

Also correct the comment at L5228-5229 from "Off-axis interior source ... where
the merging fold pair is valid (no tied saddles)" — keep "interior" (now TRUE)
but drop the specific `[0.7, ...]` framing if present in prose.

NOTE (do NOT use these): sources near the original `[0.7, y2]` axis do NOT work —
`[0.7,0.01..0.03]` are 4-image with a fold pair but the fold **error gate**
refuses (`serves=False`); `[0.7,0.04]` is already 2-image exterior. Second clean
alternative if desired: `[0.2, 0.3]` (4-image, gap 0.109, serves finite).

## Fix 2 — `OnAxisServingLadderDeterminismTestCase` (invert to true behavior)
`_on_axis_cusp_source()` = `[0.2, 1.4142]` is EXTERIOR (2-image); post-F075 the
fold arm REFUSES here. Measured ladder behavior at that source, w=200:
- `fold_amplification` -> None (refuses).
- `cusp_amplification` -> `0.478105 - 1.367105j` (serves).
- `operator._uniform_arm_value` -> equals the cusp value **byte-identically**
  (`abs(ladder - cusp) == 0`).

Rewrite the two failing tests to assert the real post-F075 contract (keep the
determinism/reproducibility spirit):
- `test_both_arms_serve_and_ladder_uses_fold_priority`: fold refuses (None), cusp
  serves finite, and the ladder returns the **cusp** value byte-identically.
  (Rename to e.g. `test_fold_refuses_and_ladder_falls_to_cusp`.)
- `test_fold_arm_tried_first_at_on_axis_node`: the spy order is `['fold','cusp']`
  (fold tried first, refuses, THEN cusp) — i.e. `assertIn('cusp', order)` and
  `order == ['fold', 'cusp']`, replacing the old `assertNotIn('cusp', order)`.
- `test_ladder_is_reproducible_same_node_twice` (L5670) already passes (cusp is
  deterministic) — leave as-is; it now exercises the cusp-served path, still valid.
- Self-falsification tests below (L5710+) mock fold->None already and remain valid.

Correct docstrings: `_on_axis_cusp_source` L5619 and the class docstring
L5628-5636 must read "exterior (2-image) on-axis-cusp source; fold refuses, cusp
serves" instead of "interior ... both arms serve".

Alternatively these two may be RETIRED as redundant — exterior fold refusal +
ladder-fallthrough is already covered by `test_lensing_fold_ghost_exterior.py`.
Prefer inversion (keeps the on-axis-cusp determinism coverage) unless redundant.

## Acceptance
After the edits, the whole file must be green (the tree-wide commit gate runs it):
prior full run was 3 failed / 125 passed / 7 skipped / 2 xfailed; target 0 failed.
