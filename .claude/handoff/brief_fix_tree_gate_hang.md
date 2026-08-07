# Build Brief: Fix tree-gate teardown hang (fast-tier mpmath)

## Mission

The tree gate (`python -m pytest cogwheel/tests/ -n 4 --dist loadscope`) hangs
at teardown after ~96% of the suite completes, never printing the summary or
failure names. The master process is left alive at 0% CPU. This is the
`lensing_fast_tier_hangs_in_mpmath` blocker: 4 tests hang in the arbitrary-
precision `_f_schwinger_mpmath` path, holding xdist workers so the full-suite
gate sits at the end for hours and never completes.

A build cannot commit with a hung gate. This is the repo-wide prerequisite
for the polar re-chart and every subsequent build.

## Measured facts

- Gate command: `python -m pytest cogwheel/tests/ -q -p no:cacheprovider -n 4
  --dist loadscope -k 'not Timing and not timing'`
- The run reached ~96% (progress dots) then the pytest master stayed alive at
  0% CPU indefinitely, no summary printed. 12 failures (F) appeared in the
  progress line before the hang.
- conftest.py sets a 900s per-test ceiling when all slow tiers are OFF
  (pytest-timeout, SIGALRM). The hang is either a test ignoring the signal or
  an xdist teardown deadlock.
- Known: 4 tests reach `f_schwinger` above w=60 where one mpmath evaluation
  costs ~85-120s (F061). The conftest ceiling should kill these — investigate
  why it does not (e.g., the timeout plugin not installed, SIGALRM ignored in
  the mpmath call, or the hang is in xdist teardown not a test).

## Work

1. Install `pytest-timeout` in the SDK conda env if absent; verify the
   conftest ceiling actually fires (a 900s SIGALRM).
2. Reproduce the hang locally on ONE worker with `-x` and identify the
   specific test(s) that hang. Use `-k` to bisect.
3. Determine the hang mechanism:
   - If a test ignores SIGALRM (mpmath in C, signal blocked): shrink the
     fixture or make the fast tier skip it.
   - If it's an xdist teardown deadlock after a worker crashed: address the
     worker (loadfile grouping, worker restart) rather than the test.
4. The 4 offending tests should either be (a) moved behind a slow tier
   (COGWHEEL_TRAIN_TIER / BRUTE_ACCURACY), (b) have their w-range shrunk
   below the 60 ceiling, or (c) use a `pytest.mark.timeout` with SIGKILL.

## Acceptance

- The tree gate completes in a bounded time (under ~20 min) and prints the
  summary with failure names.
- No test hangs the master indefinitely.
- The previously-invisible failures (the 11 red serving guards, or whatever
  the 12 F's are) are now surfaced so they can be triaged.

## Constraints

- This is infra/test-hygiene, not physics. Fast focus.
- Follow AGENTS.md and the spec/TODO workflow.
