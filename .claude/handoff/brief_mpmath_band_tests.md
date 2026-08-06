# Build — get the four mpmath-band tests out of the fast tier

## Mission

Four fast-tier tests call `f_schwinger` above `w = 60`, where a SINGLE
evaluation costs ~85-120 s instead of ~0.2 s. They do not fail; they pin xdist
workers until something else gives up. Bring them back under the ceiling (or
move them to a slow tier where their true cost is budgeted honestly), so the
full suite can COMPLETE.

## Measured facts (F061 — do not re-derive, do not re-measure)

Calling the shipping `f_schwinger` at `y_eig = (0.30, 0.15)`, `gamma' = 0.42`:

    w = 10.0   double-double    0.172 s
    w = 40.0   double-double    0.187 s
    w = 59.0   double-double    0.336 s
    w = 61.0   mpmath          84.536 s     <-- 250x cliff
    w = 70.0   mpmath         111.352 s

Routing, `cogwheel/lensing/chang_refsdal/_schwinger.py:940`:
  - `w <= 60` (`W_CEILING_SCHWINGER`)      -> double-double, fast
  - `60 < w <= 150` (`W_CEILING_SCHWINGER_QD`) -> `_f_schwinger_mpmath`, slow
  - `w > 150`                               -> unconditional refuse

THE TRAP IS THE MIDDLE BAND. The ceiling at 150 reads as "everything below is
affordable". Nothing refuses in `(60, 150]` — it SERVES, slowly. The DD
product cap (`w * |y| < 58`) is a DIFFERENT threshold and can permit `w_max`
well above 60, so a fixture that derives its w-range from the DD cap alone
walks straight into the expensive band.

## The four tests

| test | file:line |
|---|---|
| `DDWCeilingTestCase` (whole class, dies in `setUpClass`) | `test_lensing_wedge_dd_arclength.py:119` |
| `test_prior_draws_are_finite_or_exact_neg_inf` | `test_lensing_marginalized_likelihood.py:839` |
| `test_mutation_narrowing_except_turns_neginf_red` | `test_lensing_prior.py:1064` |
| `test_band_limit_refusal_precedes_coherent_score` | `test_lensing_saddle_likelihood.py:463` |

All four were named by py-spy, sitting in `mpmath.quad` under
`_raw_integral_mp` -> `_f_schwinger_mpmath`.

`DDWCeilingTestCase` is the worked example — its own docstring states the
error:

    the DD cap gives w_max ~ 121.6, which is ... above the Schwinger ceiling
    (~60).  Most refusals at the capped w_max are Schwinger-related (not DD)
    ...
    Cost: 4x4x4 = 64 nodes x ~13 w-points x 30ms ~ 25s.

The author expected `w in (60, 150]` to REFUSE cheaply and budgeted 30 ms per
evaluation. It serves at ~85-120 s. A fixture budgeted at 25 s is hours.

Critically, that class asserts the DD-cap FORMULA, not a success rate — its
own docstring says so ("we verify the FORMULA not the success rate"). The
formula assertions are geometry-independent, so a geometry whose DD cap lands
BELOW 60 preserves exactly what the class tests while making it fast. That is
the preferred fix there.

## Scope

IN — the four tests above; their fixture w-ranges/geometries; their cost
comments; tier decisions where a fix is genuinely impossible.

OUT — `f_schwinger` itself and both ceilings (`W_CEILING_SCHWINGER`,
`W_CEILING_SCHWINGER_QD`) are CORRECT and must not change. The 11 red
serving-ladder guards are a SEPARATE, already-recorded problem
(`todo.d/lensing_serving_ladder_guards_are_red.md`) — do not touch them, do
not "fix" them, and do not treat their redness as your failure. Gate
plumbing is already done (both gates now pass `--timeout`); do not re-do it.

## How to decide, per test

1. If the test's purpose does NOT require `w > 60` (the common case): bring
   the fixture's w-range under the ceiling. Prefer changing the GEOMETRY or
   the requested w-range over weakening an assertion.
2. If the test's purpose IS the mpmath band: gate it behind the file's
   existing slow-tier mechanism and CORRECT its cost comment to the real
   per-evaluation cost (~100 s), so the next reader budgets honestly.
3. Never delete an assertion to make a test fast. If coverage must move
   tiers, say so explicitly in the docstring with the reason.

## Acceptance

1. All four run to completion in the fast tier (or are correctly slow-tiered),
   with NO test exceeding 120 s.
2. A direct guard: a test that patches `_f_schwinger_mpmath` and FAILS if it
   is called during a default-tier run of the four affected files. Put it in
   `cogwheel/tests/test_lensing_schwinger.py` (it owns the dispatch
   predicate — one canonical home, per
   `test_thresholds_have_one_home`). This is the assertion that keeps the
   problem from coming back.
3. Every fixture cost comment touched in this build states a per-evaluation
   cost consistent with F061.
4. `.claude/sdk/run_full_suite.sh` COMPLETES and prints a tally. It will
   still report the 11 pre-existing serving-ladder failures — that is
   EXPECTED and is not a regression; report the count and move on.

## Constraints

- Branch `claude-dev`.
- **Every domain-test description MUST name its target suite file**
  (`test_<x>.py`) — F057.
- Keep the WP count at or below 3.
- Slow tiers stay empty in-build; no training run.
- Assert VALUES against an oracle and a tolerance, never which branch produced
  them. No `git show HEAD` oracle.
- Do not re-measure the F061 table; it is above, and re-measuring it costs
  ~5 minutes per row.
