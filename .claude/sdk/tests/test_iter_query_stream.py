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
import types
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

    def test_per_call_override_applies_when_global_disabled(self):
        # Global disabled (None), but a per-call override still enforces a wedge.
        orchestrator_module.INTER_MESSAGE_TIMEOUT_SECONDS = None

        async def gen():
            yield 1
            await asyncio.sleep(3600)
            yield 2

        async def main():
            got = []
            with self.assertRaises(asyncio.TimeoutError):
                async for m in _orch()._iter_query_with_timeout(
                        gen(), "t", timeout=1):
                    got.append(m)
            return got

        self.assertEqual(asyncio.run(main()), [1])

    def test_none_timeout_falls_back_to_global(self):
        # timeout=None must use the module global (short wedge ceiling here).
        orchestrator_module.INTER_MESSAGE_TIMEOUT_SECONDS = 1

        async def gen():
            yield 1
            await asyncio.sleep(3600)
            yield 2

        async def main():
            got = []
            with self.assertRaises(asyncio.TimeoutError):
                async for m in _orch()._iter_query_with_timeout(
                        gen(), "t", timeout=None):
                    got.append(m)
            return got

        self.assertEqual(asyncio.run(main()), [1])

    def test_generous_override_suppresses_short_global_wedge(self):
        # The prof_review case: a generous per-call override must NOT wedge on a
        # gap the tiny global would have killed.
        orchestrator_module.INTER_MESSAGE_TIMEOUT_SECONDS = 0.2  # tiny global

        async def gen():
            yield 1
            await asyncio.sleep(0.5)   # > global, < override
            yield 2

        async def main():
            return [m async for m in _orch()._iter_query_with_timeout(
                gen(), "t", timeout=5)]

        self.assertEqual(asyncio.run(main()), [1, 2])


class StallResumeTest(unittest.TestCase):
    """The double-stall killer (builds 2b x2, 2d): when the built-in-tools
    fallback leg ALSO wedges (asyncio.TimeoutError), `_run_agent` must
    attempt ONE bounded stall-resume of the agent's session before dying —
    a service-side stream death kills the request, not the session.
    """

    def setUp(self):
        self._saved_query = orchestrator_module.query
        self._saved_build = orchestrator_module.build_agent_options

    def tearDown(self):
        orchestrator_module.query = self._saved_query
        orchestrator_module.build_agent_options = self._saved_build

    def _make_orch(self):
        o = _orch()
        o.verbosity = orchestrator_module.Verbosity.QUIET
        o.use_serena = True
        o.project_root = "/tmp"
        o._serena = None
        o._specs_text = ""
        o._agent_count = 0
        o._agents_that_ran = []
        o._handle_message = lambda *a, **k: ("done", None)
        return o

    def test_fallback_stall_triggers_one_session_resume(self):
        calls = []
        opts = types.SimpleNamespace(resume=None)

        async def fake_build_opts(**kw):
            return opts

        orchestrator_module.query = lambda **kw: None
        orchestrator_module.build_agent_options = fake_build_opts
        o = self._make_orch()

        async def spy(async_iter, agent_id, timeout=None):
            calls.append(opts.resume)
            if len(calls) == 1:
                raise ValueError("simulated MCP failure")   # -> fallback leg
            if len(calls) == 2:
                raise asyncio.TimeoutError()                 # -> stall-resume
            return
            yield  # noqa: unreachable — marks this an async generator

        o._iter_query_with_timeout = spy

        async def main():
            return await o._run_agent(
                "coder", "do the thing", resume_session="sess-123",
                _denial_retry=False,
            )

        asyncio.run(main())
        self.assertEqual(len(calls), 3)
        # Legs 2 (fallback) and 3 (stall-resume) must resume the session.
        self.assertEqual(calls[1], "sess-123")
        self.assertEqual(calls[2], "sess-123")

    def test_second_stall_is_fatal(self):
        calls = []
        opts = types.SimpleNamespace(resume=None)

        async def fake_build_opts(**kw):
            return opts

        orchestrator_module.query = lambda **kw: None
        orchestrator_module.build_agent_options = fake_build_opts
        o = self._make_orch()

        async def spy(async_iter, agent_id, timeout=None):
            calls.append(1)
            if len(calls) == 1:
                raise ValueError("simulated MCP failure")
            raise asyncio.TimeoutError()   # stalls on EVERY later leg
            return
            yield  # noqa: unreachable

        o._iter_query_with_timeout = spy

        async def main():
            return await o._run_agent(
                "coder", "do the thing", resume_session="sess-123",
                _denial_retry=False,
            )

        with self.assertRaises(asyncio.TimeoutError):
            asyncio.run(main())
        self.assertEqual(len(calls), 3)   # main + fallback + ONE resume only

    def test_no_session_stall_stays_fatal(self):
        calls = []
        opts = types.SimpleNamespace(resume=None)

        async def fake_build_opts(**kw):
            return opts

        orchestrator_module.query = lambda **kw: None
        orchestrator_module.build_agent_options = fake_build_opts
        o = self._make_orch()

        async def spy(async_iter, agent_id, timeout=None):
            calls.append(1)
            if len(calls) == 1:
                raise ValueError("simulated MCP failure")
            raise asyncio.TimeoutError()
            return
            yield  # noqa: unreachable

        o._iter_query_with_timeout = spy

        async def main():
            return await o._run_agent(
                "coder", "do the thing", _denial_retry=False,
            )

        with self.assertRaises(asyncio.TimeoutError):
            asyncio.run(main())
        self.assertEqual(len(calls), 2)   # nothing to resume -> no third leg


class RunAgentTimeoutPropagationTest(unittest.TestCase):
    """`_run_agent` must forward `inter_message_timeout_override` to
    `_iter_query_with_timeout` on EVERY leg — the main stream and the
    generic MCP-failure retry leg — so a heavy phase (prof_review's pytest
    run, override=1800) never false-wedges at the global 300s ceiling on a
    retry. Guards the retry-leg propagation fix.
    """

    def setUp(self):
        self._saved_query = orchestrator_module.query
        self._saved_build = orchestrator_module.build_agent_options

    def tearDown(self):
        orchestrator_module.query = self._saved_query
        orchestrator_module.build_agent_options = self._saved_build

    def _make_orch(self):
        o = _orch()
        o.verbosity = orchestrator_module.Verbosity.QUIET
        o.use_serena = True          # so the generic retry leg is reachable
        o.project_root = "/tmp"
        o._serena = None
        o._specs_text = ""
        o._agent_count = 0
        o._agents_that_ran = []
        o._handle_message = lambda *a, **k: ("", None)
        return o

    def test_override_forwarded_on_main_and_retry_legs(self):
        seen_timeouts = []

        async def fake_build_opts(**kw):
            return types.SimpleNamespace(resume=None)

        # query() is only evaluated as an argument to the (spied) iterator.
        orchestrator_module.query = lambda **kw: None
        orchestrator_module.build_agent_options = fake_build_opts

        o = self._make_orch()

        async def spy(async_iter, agent_id, timeout=None):
            seen_timeouts.append(timeout)
            if len(seen_timeouts) == 1:
                # Force the main leg to fail so control reaches the generic
                # MCP-failure retry leg (the one whose propagation regressed).
                raise ValueError("simulated MCP failure")
            return
            yield  # noqa: unreachable — marks this an async generator

        o._iter_query_with_timeout = spy

        async def main():
            return await o._run_agent(
                "coder", "do the thing",
                inter_message_timeout_override=1800,
            )

        asyncio.run(main())
        self.assertEqual(seen_timeouts, [1800, 1800])

    def test_default_none_forwarded_when_no_override(self):
        seen_timeouts = []

        async def fake_build_opts(**kw):
            return types.SimpleNamespace(resume=None)

        orchestrator_module.query = lambda **kw: None
        orchestrator_module.build_agent_options = fake_build_opts

        o = self._make_orch()

        async def spy(async_iter, agent_id, timeout=None):
            seen_timeouts.append(timeout)
            return
            yield  # noqa: unreachable — marks this an async generator

        o._iter_query_with_timeout = spy

        async def main():
            return await o._run_agent("coder", "do the thing")

        asyncio.run(main())
        self.assertEqual(seen_timeouts, [None])


if __name__ == "__main__":
    unittest.main()
