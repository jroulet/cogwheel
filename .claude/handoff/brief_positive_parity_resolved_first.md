# Build brief — one authoritative geometric-vs-wave gate in the operator grids

## Mission

There are THREE different geometric-vs-wave decisions in the tree, and they
disagree:

| site | condition |
|---|---|
| `channels._exact_total` (~658) | `operator.select_branch`: resolved AND `L > L_MAX` |
| `operator._saddle_grid` (~951) | resolved AND `w > W_CEILING_SCHWINGER` |
| `operator._positive_parity_grid` (~1524) | none — every above-ceiling node goes to the uniform arms |

`select_branch`'s own docstring calls it "the authoritative wave/geometric
branch gate", and the training-label layer uses it. The two serving grids do
not. Make both grids route their geometric-vs-wave decision through
`select_branch` so the predicate has ONE home, the way `RHO_END` / `L_MAX`
already do (there is an existing test `test_thresholds_have_one_home` pinning
the CONSTANTS; the PREDICATE never got the same treatment).

This is a correctness fix. F028 measured `_positive_parity_grid` serving the
fold Airy arm at 60%–267% relative error on well-resolved configs where
geometric optics is accurate; and the driver measured that `_saddle_grid`'s
substitute condition (`w > 60` in place of the cancellation test) is not
equivalent to the authoritative gate.

## Measured facts (driver-supplied; do NOT re-derive, do NOT re-run these)

* `W_CEILING_SCHWINGER = 60.0`, `RHO_END = 4.0`, `L_MAX = 48`.
* `select_branch(w, delta_min, L)` returns `'geometric'` iff
  `w * delta_min >= RHO_END` AND `L > L_MAX`, where
  `L = cancellation_exponent(w, y, gamma, kappa)`.
* `F_op` is NOT an independent oracle above `w = 60`: it serves THROUGH
  `_uniform_arm_value`, so `|F_op - F_arm|` is identically 0 there (F028).
  Accuracy assertions above the ceiling must use `geometric_amplification`;
  below it, the quadrature.
* The uniform fold arm, served above the ceiling on resolved positive-parity
  configs, measured against `geometric_amplification` (itself cross-checked to
  `1e-5` against the quadrature on the same configs at `w = 45..60`):

  | gamma | w | w*Dtau | \|F_arm/F_geo\| | rel err |
  |---|---|---|---|---|
  | 0.70 | 70 | 35.2 | 0.348 | 7.5e-1 |
  | 0.70 | 500 | 251.6 | 1.846 | 2.7e+0 |
  | 0.90 | 500 | 564.2 | 0.192 | 9.4e-1 |

  The error GROWS with `w`. The arm's own certificate reads `1.2e-2`–`4.7e-2`
  against `envelope_bar = 0.05` — optimistic by 20x–100x. Root cause (F028,
  context only, NOT in scope): the arm sets the `Ai'` amplitude `q = 0`, a
  symmetric-fold assumption that cannot represent a two-image sum with unequal
  magnifications.
* Geometric-optics accuracy vs the quadrature, positive parity, binned by
  `w * delta_min` over `w in [1, 60]`: at the `resolved`-only gate
  (`>= RHO_END`) the error distribution is median `3.8e-3`, p90 `4.6e-1`,
  max `2.8e+02`. `resolved` ALONE is not an accuracy gate — `RHO_END` is a
  resolution threshold. This is why the second (`L > L_MAX`) condition exists.
* GATE COMPARISON, `w in [55, 60]` (the closest band to the production
  `w > 60` regime in which the quadrature still answers), 1500 positive-parity
  draws, error of `geometric_amplification` vs the quadrature:

  | admission gate | n | p90 | p99 | max |
  |---|---|---|---|---|
  | `resolved` only (`_saddle_grid`'s rule) | 1349 | 1.6e-2 | 1.05 | 7.4e+1 |
  | `L > L_MAX` only | 1013 | 3.5e-3 | 1.01 | 7.4e+1 |
  | both (`select_branch`) | 990 | 1.4e-3 | 7.1e-1 | 7.4e+1 |

  The two-condition gate is ~10x better at p90 than the resolved-only rule.
  That is the justification for unifying on `select_branch`.
* KNOWN RESIDUAL, must NOT be papered over: roughly 1% of the nodes the
  AUTHORITATIVE gate admits still carry O(1) error (p99 `7.1e-1`, max `74`).
  Raising the `w * delta_min` threshold to 100 does not move that tail
  (p99 stays `7.3e-1`), so it is not a resolution deficit, and
  `_certify_geometric_census` does not catch it — every sampled point passed
  the census. Diagnosing that tail is OUT OF SCOPE for this build. What is IN
  scope: do not describe the geometric branch as certified or exact in any
  docstring, comment, spec text, or test name you write. It is "the best
  available serve under the authoritative gate, with a measured ~1% O(1)
  tail (driver sweep, 2026-07-28)". Record it in `.claude/spec/FINDINGS.md`
  as a new finding so the tail is not rediscovered from scratch.

## In scope

* `cogwheel/lensing/chang_refsdal/operator.py`, the node loops in
  `_positive_parity_grid` and `_saddle_grid`: replace the two hand-rolled
  conditions with `select_branch`.
  - `delta_min` and the cancellation exponent must be computed at most ONCE per
    grid call, and not at all when no node exceeds the ceiling —
    `_real_delay_min_separation` solves the image quartic. `L` depends on `w`,
    so cache the `w`-independent part (`|y'|`) and scale per node.
  - Keep the uniform arms as the fallback for above-ceiling nodes the gate
    sends to `'wave'`, and keep the named refusal when both arms refuse. NO
    legacy fallback catch.
* Tests for the unified routing, authored by the Test Developer.
* `.claude/spec/SPEC.md`: its serving-ladder description presents the uniform
  arms as a certified rung; correct it to match the measurement.

## Out of scope — do not touch

* `_airy_fold.py` internals. The `q`/`b4` asymmetry refinement is a separate,
  later build. Do NOT attempt to improve the arm's accuracy here.
* The cusp Pearcey arm's accuracy.
* `RHO_END`, `L_MAX`, `W_CEILING_SCHWINGER`, `envelope_bar` VALUES — this build
  unifies the predicate, it does not retune thresholds.
* `channels._exact_total` — already correct; it is the reference, not a target.
* `surrogate.py`, `surrogate_training.py`, the likelihood layer, chart training.

## Known blast radius — existing tests encode the CURRENT routing

These assert that an above-ceiling node is served by the uniform arm, which is
exactly what changes for nodes the authoritative gate sends to `'geometric'`:

* `test_lensing_schwinger.py` (~1237, ~1255, ~1277)
* `test_lensing_saddle_geometry.py` (~1095, ~1118, ~1138)
* `test_lensing_surrogate.py` (~398)
* `test_lensing_airy_fold.py` (~73, ~1807), `test_lensing_waveform.py` (~327,
  ~589), `test_lensing_fast_path.py` (~1506)

Re-point them at the corrected contract. Do NOT weaken an assertion to make it
pass and do NOT delete a test that still encodes a true claim. Every test that
changes gets its reason in the docstring, citing F028. A silent expectation
flip is a build failure.

## Acceptance

1. `w <= W_CEILING_SCHWINGER`: serve values BYTE-IDENTICAL to the pre-build tree
   over a config grid spanning several `gamma` and `|y|`, both parities. This
   path must not move at all.
2. The geometric-vs-wave predicate has exactly ONE home: a test asserts both
   grids agree with `select_branch` node-for-node over a config grid, in the
   spirit of the existing `test_thresholds_have_one_home`.
3. Above the ceiling where `select_branch` says `'geometric'`: served by
   `geometric_amplification`. On the F028 table configs the served value is that
   call exactly, replacing the measured 60%–267% arm error.
4. Above the ceiling where it says `'wave'`: arms offered in the existing order,
   named refusal if both refuse. Exercise all three outcomes.
5. Refusal identity preserved: lowest-index refuser, authentic `f_schwinger`
   message.
6. `delta_min` computed at most once per grid call and not at all below the
   ceiling (a counting spy is acceptable).
7. Full fast suite green — driver-verified POST-build, not an in-build test.

## Constraints

* Fast tests only. Analytic or few-evaluation oracles; no brute-force sweeps,
  no timing assertions. `COGWHEEL_BRUTE_ACCURACY` / `COGWHEEL_STRICT_TIMING`
  are pinned empty in every agent env — a test needing them is a build-killer.
* Never write a committed test that imports a module from a git revision (F022).
* Use Serena. Before moving or renaming any symbol run
  `find_referencing_symbols` AND a `search_for_pattern` grep — the LSP misses
  cross-file refs silently.
* Spec/TODO workflow applies (behaviour change in `cogwheel/`). The fragment
  `.claude/spec/todo.d/lensing_fold_arm_serves_wrong_values.md` has two
  defects; this build closes the FIRST (admission routing) and NOT the second
  (`q = 0`). Rewrite that fragment to carry only the remaining `q`/`b4` work
  rather than deleting it, add the `completed.d/` entry, and add the
  `spec_changelog.d/` fragment for the SPEC.md edit.
* Accuracy dominates: a fast wrong serve is worthless. If the unified gate
  sends nodes to a named refusal that are served (wrongly) today, that is the
  CORRECT outcome — report the coverage change, do not widen the gate to
  preserve coverage.
