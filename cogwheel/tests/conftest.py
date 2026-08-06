"""Test-suite defaults.

Sets a per-test wall-clock ceiling for FAST-TIER runs so that a test which
stops terminating fails loudly and names itself, instead of pinning a worker
until something further up the stack gives up.

Motivating incident (F061, 2026-08-06): four tests reached ``f_schwinger``
above ``w = 60``, where one evaluation costs ~85-120 s instead of ~0.2 s.
They did not fail -- they held xdist workers while the full-suite gate sat at
99% for six hours, and a later build's tree gate burned its entire 3600 s
ceiling and STRANDED a build that had already passed Inspector and Professor.
Neither run named a single test; both needed a py-spy autopsy.

Deliberately narrow, so it cannot cost coverage:

* It applies ONLY when every slow tier is off. Slow-tier tests are allowed to
  be long -- that is what the tier is for -- so the ceiling lifts entirely
  when one is requested.
* It is a DEFAULT, not an override: an explicit ``--timeout`` on the command
  line wins (the SDK gates pass their own).
* It is a no-op when ``pytest-timeout`` is absent, so the suite still runs for
  anyone who has not installed it. The plugin is a convenience here, never a
  requirement.
"""
import os

#: Envs that mark an opt-in slow tier. Any of them set means long tests are
#: expected and the ceiling does not apply.
_SLOW_TIER_VARS = (
    'COGWHEEL_BRUTE_ACCURACY',
    'COGWHEEL_TRAIN_TIER',
    'COGWHEEL_STRICT_TIMING',
    'COGWHEEL_RUN_TIMING_SMOKE',
)

#: Fast-tier per-test ceiling [s]. Far above any healthy fast test (the
#: slowest legitimate ones run in tens of seconds) and far below the hours a
#: runaway mpmath sweep takes, so it separates the two without judgement
#: calls about individual tests.
_FAST_TIER_TIMEOUT = 900


def pytest_configure(config):
    """Apply the fast-tier per-test ceiling, if it applies and is possible."""
    if any(os.environ.get(var) for var in _SLOW_TIER_VARS):
        return

    if not config.pluginmanager.hasplugin('timeout'):
        return

    if getattr(config.option, 'timeout', None) is None:
        config.option.timeout = _FAST_TIER_TIMEOUT
        # signal, not thread: SIGALRM fails the offending test and lets the
        # run continue. thread kills the whole worker process, taking the
        # rest of its scope with it.
        config.option.timeout_method = 'signal'
