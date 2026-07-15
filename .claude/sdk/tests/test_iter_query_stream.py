"""Regression tests for the queue-drain agent-stream wrapper.

_iter_query_with_timeout must drain the SDK stream in ONE dedicated task
(claude_agent_sdk holds anyio cancel scopes across yields; resuming its
generator from per-message asyncio.wait_for tasks made mid-stream errors
exit a cancel scope in a foreign task — the build-killing RuntimeError of
2026-07-16). These tests pin the wrapper's contract: relay messages,
relay mid-stream exceptions, convert wedges to TimeoutError, clean up on
early caller exit.

Run: conda run -n cogwheel_310 python -m unittest \
    discover -s .claude/sdk/tests -p 'test_*.py'
"""
import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import sdk.orchestrator as orchestrator_module
from sdk.orchestrator import BuildOrchestrator


def _orch():
    o = BuildOrchestrator.__new__(BuildOrchestrator)
    o._log = lambda *a, **k: None
    return o


class IterQueryStreamTest(unittest.TestCase):
    def setUp(self):
        self._saved_timeout = orchestrator_module.INTER_MESSAGE_TIMEOUT_SECONDS

    def tearDown(self):
        orchestrator_module.INTER_MESSAGE_TIMEOUT_SECONDS = self._saved_timeout

    def test_normal_stream_relayed_in_order(self):
        async def gen():
            for i in range(5):
                yield i

        async def main():
            return [m async for m in _orch()._iter_query_with_timeout(gen(), "t")]

        self.assertEqual(asyncio.run(main()), [0, 1, 2, 3, 4])

    def test_mid_stream_exception_relayed_after_messages(self):
        async def gen():
            yield 1
            yield 2
            raise RuntimeError("cancel scope teardown")

        async def main():
            got = []
            with self.assertRaises(RuntimeError):
                async for m in _orch()._iter_query_with_timeout(gen(), "t"):
                    got.append(m)
            return got

        self.assertEqual(asyncio.run(main()), [1, 2])

    def test_wedge_becomes_timeout_error(self):
        orchestrator_module.INTER_MESSAGE_TIMEOUT_SECONDS = 1

        async def gen():
            yield 1
            await asyncio.sleep(3600)
            yield 2

        async def main():
            got = []
            with self.assertRaises(asyncio.TimeoutError):
                async for m in _orch()._iter_query_with_timeout(gen(), "t"):
                    got.append(m)
            return got

        self.assertEqual(asyncio.run(main()), [1])

    def test_early_caller_exit_cleans_up(self):
        async def gen():
            for i in range(100):
                yield i

        async def main():
            agen = _orch()._iter_query_with_timeout(gen(), "t")
            async for _ in agen:
                break
            await agen.aclose()
            return True

        self.assertTrue(asyncio.run(main()))


if __name__ == "__main__":
    unittest.main()
