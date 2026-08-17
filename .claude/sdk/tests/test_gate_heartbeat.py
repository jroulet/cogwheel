"""A build waiting on a human decision must not read as stale.

F059 (2026-07-30): the escalation gate polled silently for a decision file.
The watchdog's only liveness signal is log mtime, so a healthy build blocked
on the driver looked identical to a wedge, and the watchdog killed
1e_farfield_port mid-decision after exactly 1200s. `launch_build.sh` had
warned about it in prose for weeks — "Respond promptly: the watchdog staleness
clock runs during the wait" — which asks a human to be fast instead of fixing
the conflict.

These assert the VALUE the watchdog actually reads: whether anything was
written to stdout during the wait.

Run (from the repo root, under the project's interpreter):
    python -m unittest discover -s .claude/sdk/tests -p 'test_*.py'
"""
import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sdk import gates


def _run_gate(seconds, poll=5, beat=240):
    """Drive _gate_wait for `seconds` of simulated waiting; return stdout."""
    ticks = {"n": 0}

    def fake_sleep(_):
        ticks["n"] += 1

    buf = io.StringIO()
    with mock.patch.object(gates.time, "sleep", fake_sleep), \
            redirect_stdout(buf):
        for waited in gates._gate_wait("a decision", None, poll=poll,
                                       beat=beat):
            if waited >= seconds:
                break
    return buf.getvalue()


class GateEmitsAHeartbeat(unittest.TestCase):

    def test_a_long_wait_writes_to_the_log(self):
        # 1200s is the watchdog's default staleness threshold. The build must
        # have said something well before reaching it.
        out = _run_gate(1200)
        self.assertTrue(out.strip(), "gate produced NO output over 1200s — "
                                     "the watchdog would kill this build")

    def test_the_first_beat_lands_well_inside_the_kill_threshold(self):
        out = _run_gate(1200)
        first = int(out.splitlines()[0].split("(")[1].split("m")[0])
        self.assertLess(first * 60, 1200,
                        "first heartbeat must precede the 1200s kill")

    def test_beats_repeat_so_a_long_wait_keeps_the_log_moving(self):
        out = _run_gate(1200)
        self.assertGreaterEqual(len(out.strip().splitlines()), 2)

    def test_the_beat_BACKS_OFF_on_a_very_long_wait(self):
        # A fixed 4-minute beat emitted 270 notifications across an 18-hour
        # wait. Back off — but the ceiling must stay BELOW the watchdog's
        # 1200s staleness threshold: the earlier hourly ceiling let the beat
        # gap exceed 1200s and the watchdog killed a healthy build
        # mid-escalation-wait (measured 2026-08-16). ~74 beats over 18h is
        # the price of staying alive; 270 was the flood.
        out = _run_gate(18 * 3600)
        n = len(out.strip().splitlines())
        self.assertLess(n, 100, f"{n} beats over 18h is a notification flood")
        self.assertGreater(n, 8, f"{n} beats over 18h is too quiet to trust")

    def test_no_beat_gap_ever_exceeds_the_watchdog_threshold(self):
        # THE invariant the 2026-08-16 kill exposed: whatever the backoff
        # does, consecutive beats must land closer together than the
        # watchdog's 1200s staleness threshold, or a long wait dies.
        out = _run_gate(18 * 3600)
        minutes = [int(line.split("(")[1].split("m")[0])
                   for line in out.strip().splitlines()]
        gaps = [b - a for a, b in zip(minutes, minutes[1:])]
        self.assertTrue(gaps, "need at least two beats to measure a gap")
        self.assertLess(max(gaps) * 60, 1200,
                        f"max beat gap {max(gaps)}m would outlast the "
                        f"1200s watchdog")

    def test_the_flood_is_what_the_old_cadence_did(self):
        # Contrast control: the pre-backoff rule, at the same duration.
        fixed = (18 * 3600) // 240
        self.assertEqual(fixed, 270)
        self.assertLess(len(_run_gate(18 * 3600).strip().splitlines()), fixed)

    def test_the_beat_says_the_build_is_alive_not_hung(self):
        # The line exists to be READ by a driver who is deciding whether to
        # investigate; "alive, not stale" is the whole message.
        self.assertIn("alive", _run_gate(600))

    def test_a_short_wait_stays_quiet(self):
        # A gate answered promptly must not spam the log.
        self.assertEqual(_run_gate(120), "")


class TheOldSilentLoopWasTheDefect(unittest.TestCase):
    """Contrast control: a poll loop with no heartbeat, at the same duration."""

    @staticmethod
    def _legacy_wait(seconds, poll=5):
        buf = io.StringIO()
        with redirect_stdout(buf):
            waited = 0
            while waited < seconds:
                waited += poll          # time.sleep(poll) in the real loop
        return buf.getvalue()

    def test_legacy_gate_was_silent_for_the_whole_kill_window(self):
        self.assertEqual(self._legacy_wait(1200), "",
                         "the pre-F059 loop printed nothing — which is "
                         "exactly why the watchdog could not tell it apart "
                         "from a wedge")


if __name__ == "__main__":
    unittest.main()
