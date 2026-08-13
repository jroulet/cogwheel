# Build Brief: wire `lobe_exterior` into the regions filter

## Mission

The `deltoid_exterior_geometry_fix` build (uncommitted, in the working tree)
added a new training region `lobe_exterior` for the macro-saddle deltoid
exterior. It packs charts with `region='lobe_exterior'` but never wired that
region into the `regions=` filter. The tree gate is RED on 4 tests in two
suites that were outside the build's shards. Finish the wiring.

Do NOT re-open the coordinate design — `LobeExteriorChart`, the lobe-local
`(rho_lobe, u=d^(2/3))` coordinate, the corridor decision, and the cusp
exclusion are all settled and passing their own suites.

## Measured facts (at working tree, tree gate run 2026-08-12 16:17-16:30)

Gate result: **4 failed, 1931 passed, 255 skipped, 4 xfailed, 4 errors**
(783 s). All four failures are saddle tests in suites the build did not touch:

    test_lensing_regions_filter.py::RegionExclusivityTestCase::test_exterior_only_saddle
    test_lensing_regions_filter.py::RegionsDefaultEqualsAllTestCase::test_saddle_default_equals_explicit_all
    test_lensing_regions_filter.py::RegionsFilterMatchesFullRunTestCase::test_lobe_only_matches_full_restricted_saddle
    test_lensing_ppgo_midw_and_minus_ghost.py::MinusGhostServeRoundtripSelfFalsificationTestCase::test_missing_ghost_recovery_differs

Distinct errors observed:

    AssertionError: 0 not greater than 0 : anti-vacuity: this test asserted nothing (zero comparisons).
    AssertionError: vacuous: the test made no comparison
    AttributeError: type object 'MinusGhostServeRoundtripTestCase' has no attribute 'chart'
    KeyError: 'm_lo'   (cogwheel/lensing/surrogate_training.py:5385)

### Defect 1 — ALREADY FIXED BY THE DRIVER, do not redo

`surrogate_training.py:5385` reads `tile['m_lo'], tile['m_hi']` by hard index.
The new `lobe_exterior` tile dict omitted both, while every sibling tiler
(lines 5033, 5049, 5222) sets them from `m_lo_region, m_hi_region` (defined at
4877-4879). The driver added them to the `lobe_exterior` tile dict. The
`MinusGhost...` error is a `setUpClass` cascade from this KeyError and should
clear with it — VERIFY that, do not assume.

### Defect 2 — THE WORK OF THIS BUILD

`lobe_exterior` is absent from the regions vocabulary AND is not gated at all:

- `surrogate_training.py:3884` — `regions = regions or ("tube", "exterior",
  "wedge_interior", "lobe_interior")`
- `surrogate_training.py:4757` — the same tuple again
- `git grep "'lobe_exterior' in regions"` returns NOTHING: the new exterior
  block at ~5090-5135 runs unconditionally, so `regions=(r,)` cannot exclude
  it and per-region exclusivity is broken.

`test_lensing_regions_filter.py` pins the contract (see its module docstring):
DEFAULT = ALL, per-region exclusivity, and a restricted run equals the full
run's report with other regions filtered out.

## Work

Wire `lobe_exterior` into the filter the same way `lobe_interior` is wired:
add it to the default/all region tuple(s), gate the exterior block on
membership, and update `test_lensing_regions_filter.py`'s explicit "all"
tuple so DEFAULT == EXPLICIT-ALL still holds.

The Architect must decide and state ONE thing explicitly: whether the
macro-saddle exterior is a NEW region name (`lobe_exterior`, requiring the
vocabulary change above) or should be packed under the existing `exterior`
name. The prior build's plan chose `lobe_exterior`; if you keep that, the
vocabulary MUST grow, because a region a filter cannot name is a region a
filter cannot exclude. Say which and why.

## Acceptance

1. All 4 named tests green.
2. `regions=('lobe_exterior',)` builds ONLY lobe-exterior charts; every other
   single-region run excludes them. No region runs unconditionally.
3. DEFAULT (`regions=None`) == explicit all-regions tuple, for BOTH parities.
4. Astroid (parity==1) training output unchanged.
5. Full suite: no NEW failures against the 11 pre-existing known-red in
   `.claude/sdk/known_failures.txt`.

## Constraints

- Branch `claude-dev`. Fast tests only; no training run.
- Every domain-test description MUST begin with its SHARD letter and target
  suite FILE PATH, and shards must be DISJOINT (one file per shard) — F057.
  A plan was rejected at this gate today for omitting exactly this.
- CHECK THE FULL CONSUMER SET before finishing: this red gate exists because
  the previous build changed the saddle training path and neither of these two
  suites was in any shard. `git grep` for the region names and for
  `_train_band_charts` consumers.
- Keep the WP count at or below 2.
