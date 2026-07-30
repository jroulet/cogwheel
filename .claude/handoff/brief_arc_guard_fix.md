# Build brief — finish the interrupted 1b revision + fix the arc guard

## Situation: a half-revised tree from a crashed build

Build 1b (retire the six numerical estimators) died on an API outage
MID-REVISION. The worktree is UNCOMMITTED and in a KNOWN-PARTIAL state. Do
not assume any part is finished — this brief enumerates exactly what is done,
undone, and unverified. Trust this inventory over the appearance of the tree.

### DONE and driver-verified — do NOT redo
- `cogwheel/lensing/surrogate_training.py` production code: all six
  estimators retired (`_min_curvature_radius`, `_branch_speed_profile`,
  `_find_cusps` now an analytic root, `_probe_arc_side`/`_PROBE_ETA` gone,
  `_caustic_inradius` closed-form, `_InteriorAdmission` exact with
  `_CLOUD_MARGIN_FRAC` deleted). Module imports; the `eta_max > 0.5*r_min`
  decision flips on no band; arc labels are `(-1, 4)` from geometry. This is
  the completed main-WP output. Leave it, EXCEPT the guard below.

### UNDONE — the crash interrupted these
1. **THE GUARD BUG** (`_make_arc`, line ~690: `if abs(dot) <= 0.1: continue`).
   This shipped in the main WP — it was in the approved plan and is wrong.
   Full detail below; it is the one PRODUCTION change.
2. **INS-1b-002 unfixed**: `cogwheel/tests/test_lensing_surrogate.py` has a
   stale positional `_find_cusps(...)` caller (~line 1068). `_find_cusps` now
   takes REQUIRED keyword-only `gamma` and `branch`, so this fails at RUN
   (it collects clean — do not trust collection).
3. **INS-1b-003 unfixed**: `cogwheel/tests/test_lensing_surrogate_training.py`
   has stale `_find_cusps` callers (~lines 1006, 1660, 1661), same cause.
4. **The acceptance test 1b never wrote** (below). Its absence is HOW the
   guard regression reached Inspector unflagged.

### INHERITED — driver-verified post-crash (full run 2026-07-29, 504s)
120 passed, 37 skipped, 1 failed + 1 error (both the SAME test — the stale
caller in #2 above). So:
- `cogwheel/tests/test_lensing_caustic_cusps.py` (NEW, ~1182 lines) PASSES in
  full — the inherited cusp suite is sound; keep it, do not re-derive it.
- The revised `test_lensing_exterior_admission.py` PASSES — the
  `_CLOUD_MARGIN_FRAC` finding fix (INS-1b-001) is good; keep it.
- The ONLY real breakage is `test_lensing_surrogate.py` line 1069
  (`_find_cusps()` missing `gamma`/`branch`), which also trips its
  anti-vacuity teardown at line 688 (zero comparisons because the body
  aborted). Fixing the caller fixes both report lines.
Still: a test that pins a removed constant or a pre-fix behavior is DELETED,
not preserved because it exists.

## The guard bug (F041) — the one production change

`dot = fold_opening_direction . serve_normal` decides which fold side carries
the image pair. The SIGN is what matters and is never in doubt — swept over
the prior the minimum `|dot|` is 4.4e-3, twelve orders above float64 noise.
But `|dot|` scales with the caustic:

| gamma | 0.02 | 0.06 | 0.10 | 0.30 | 0.90 |
|---|---|---|---|---|---|
| `\|dot\|` | 0.030 | 0.090 | 0.150 | 0.441 | 0.994 |

`|dot| ~ 1.5*gamma`, so `abs(dot) <= 0.1` kills EVERY arc below `gamma ~ 0.067`.
Measured: `stable_gamma_bands((0.01, 0.30), +1)` returns two bands with ZERO
arcs, `(0.01, 0.0462)` and `(0.0462, 0.0644)` — a REGRESSION, and the same
category error as the `_PROBE_ETA` this build deleted: an absolute threshold
on a quantity that scales with the caustic.

`|dot|` is the WRONG QUANTITY, not a mis-scaled one — it measures how
TRANSVERSE the fold opening is (legitimate gamma-dependent geometry). What the
guard wanted is CUSP PROXIMITY, a dimensionless ratio the cascade already
gives: `theta` is dimensionless, so `|y'|` and `|y''|` both carry length and
`|y'|/|y''|` IS the angular distance to the cusp (`y'=0`). It is gamma-STABLE
(0.307, 0.313, 0.324, 0.319 across `gamma=0.02..0.9`) where `|dot|` swings 33x.

Two acceptable fixes — Architect/Professor pick, but it MUST be one, never a
smaller absolute number:
1. **Delete the guard.** Defensible: arc bounds are already cusp-window
   trimmed, `|y'|/|y''|` over the arc half-span stays `>= 0.39` everywhere
   sampled, and the sign has 12 orders of margin. Keep the fallback-fraction
   loop for `LensDomainError`; take `sign` from the first evaluable fraction.
2. **Replace with the dimensionless ratio**, `|y'|/|y''|` over arc half-span
   vs a dimensionless O(1) constant. State the constant and why.
Prefer (1) unless a concrete cusp-proximity failure is shown that (1) misses.

## Acceptance — MANDATORY; assert VALUES in the file that owns the predicate

1. `stable_gamma_bands((0.01, 0.30), +1)` (n_samples=200, min_width=0.02):
   ZERO dropped slivers AND every returned band has `len(arcs) > 0`. Both — a
   band that exists but yields no arc is unserved exactly like a dropped one.
2. The orientation quantity the fix uses is gamma-STABLE: it does not vary
   more than O(1) across `gamma in {0.02, 0.1, 0.3, 0.9}` at matched arc
   position (the property `|dot|` failed). If the guard is deleted instead,
   assert arcs build for every gamma down to 0.02 where a band exists.
3. Arc labels unchanged where arcs already built: `inward_sign` and
   `image_count == 4` match the pre-fix tree on `gamma >= 0.1` bands — the fix
   only ADDS small-gamma arcs, it must not move existing labels.
4. All four affected suites — `test_lensing_surrogate.py`,
   `test_lensing_surrogate_training.py`, `test_lensing_exterior_admission.py`,
   `test_lensing_caustic_cusps.py` — RUN green (not merely collect).

## Decomposition constraint (a prior plan tripped the gate on this)

The test work spans TWO distinct suites that the pipeline requires be authored
by SEPARATE Test-Developer shards — each test FILE has exactly one author:
- `test_lensing_surrogate.py` — the stale `_find_cusps` caller at line 1069.
- `test_lensing_surrogate_training.py` — the sliver/arcs acceptance test
  (acceptance 1; this is where `stable_gamma_bands` lives) plus any
  byte-identity retirement.
A single WP that writes both files fails plan verification
("Test-suite write-ownership conflict"). Decompose so no shard targets more
than one test suite. (`test_lensing_exterior_admission.py` and
`test_lensing_caustic_cusps.py` already PASS — see the inventory above — so
they need no author shard at all.)

## Constraints
- Assert VALUES against a tolerance, not code paths. A test that pins a
  removed constant (`_PROBE_ETA`, `_CLOUD_MARGIN_FRAC`, `_CUSP_SPEED_REL_FRAC`)
  is DELETED, not re-pointed.
- Slow tests never run in-build; `COGWHEEL_BRUTE_ACCURACY` /
  `COGWHEEL_STRICT_TIMING` stay empty.
- `SDK_CONDA_ENV` from repo-root `.env` (`cogwheel-newlal`); interpreter as
  `$(conda info --base)/envs/$SDK_CONDA_ENV/bin/python`, never `conda run -n`.
- 1b's own doc-sync never ran (it died): SPEC.md row 55 and COVERAGE_DESIGN.md
  still describe the retired estimators. That is the post-gate Librarian's job
  for the driver — flag it, do not attempt it in a WP.
- Do NOT weaken acceptance 1. If arcs still fail to build, report the residual
  cause; never add a fence or a smaller constant.
