"""Value-level regression tests for quiet, milestone-only Codex callbacks."""

from __future__ import annotations

import fcntl
import os
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

CLAUDE_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = CLAUDE_DIR.parent
if str(CLAUDE_DIR) not in sys.path:
    sys.path.insert(0, str(CLAUDE_DIR))

from sdk.gates import _notify_codex_driver


class CodexEventResumeTests(unittest.TestCase):
    def _fake_codex(self, directory: Path) -> Path:
        executable = directory / "codex"
        executable.write_text(
            "#!/usr/bin/env bash\n"
            "printf '%s %s %s\n' \"$1\" \"$2\" \"$3\" "
            ">> \"$CODEX_TEST_CAPTURE\"\n"
        )
        executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
        return executable

    def _env(self, tmp: Path, thread: str = "thread-123") -> dict[str, str]:
        self._fake_codex(tmp)
        return {
            **os.environ,
            "PATH": f"{tmp}:{os.environ['PATH']}",
            "CODEX_THREAD_ID": thread,
            "CODEX_TEST_CAPTURE": str(tmp / "calls"),
            "CODEX_RESUME_STATE_DIR": str(tmp / "state"),
            "CODEX_RESUME_RETRY_DELAYS": "0",
        }

    def test_same_event_occurrence_resumes_exact_thread_once(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            env = self._env(tmp)
            command = [str(REPO_ROOT / ".codex" / "resume_driver.sh"),
                       "build_terminal", "exit_status=0", "build-001"]
            subprocess.run(command, env=env, check=True)
            subprocess.run(command, env=env, check=True)
            self.assertEqual((tmp / "calls").read_text().splitlines(),
                             ["exec resume thread-123"])

    def test_same_text_in_distinct_threads_resumes_each(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            command = [str(REPO_ROOT / ".codex" / "resume_driver.sh"),
                       "build_terminal", "exit_status=0", "build-001"]
            subprocess.run(command, env=self._env(tmp, "thread-one"), check=True)
            subprocess.run(command, env=self._env(tmp, "thread-two"), check=True)
            self.assertEqual(len((tmp / "calls").read_text().splitlines()), 2)

    def test_contended_event_waits_instead_of_being_dropped(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            env = self._env(tmp)
            state = tmp / "state"
            state.mkdir()
            thread_key = subprocess.check_output(
                ["sha256sum"], input=b"thread-123").decode().split()[0]
            lock_path = state / f"thread-{thread_key}.lock"
            with lock_path.open("w") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                process = subprocess.Popen(
                    [str(REPO_ROOT / ".codex" / "resume_driver.sh"),
                     "build_escalation", "same paths", "escalation-002"],
                    env=env,
                )
                time.sleep(0.1)
                self.assertIsNone(process.poll())
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            process.wait(timeout=10)
            self.assertEqual(process.returncode, 0)
            self.assertEqual(len((tmp / "calls").read_text().splitlines()), 1)

    @patch("sdk.gates.subprocess.Popen")
    def test_escalation_callback_has_an_occurrence_identity(self, popen):
        with patch.dict(os.environ, {
            "AGENT_PROVIDER": "codex", "CODEX_THREAD_ID": "thread-123",
        }, clear=False):
            self.assertTrue(_notify_codex_driver("build_escalation", "paths"))
        command = popen.call_args.args[0]
        self.assertEqual(command[:3], [
            str(REPO_ROOT / ".codex" / "resume_driver.sh"),
            "build_escalation", "paths",
        ])
        self.assertTrue(command[3])
        self.assertTrue(popen.call_args.kwargs["start_new_session"])

    @patch("sdk.gates.subprocess.Popen")
    def test_claude_does_not_launch_a_codex_callback(self, popen):
        with patch.dict(os.environ, {
            "AGENT_PROVIDER": "claude", "CODEX_THREAD_ID": "thread-123",
        }, clear=False):
            self.assertFalse(_notify_codex_driver("build_escalation", "paths"))
        popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
