"""Regression tests for the build-scoped Codex Serena lifecycle."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

CLAUDE_DIR = Path(__file__).resolve().parents[2]
if str(CLAUDE_DIR) not in sys.path:
    sys.path.insert(0, str(CLAUDE_DIR))

from sdk.agents import SerenaManager
from sdk.runtime_codex import (
    CODEX_ROLE_MODELS,
    ClaudeAgentOptions,
    _append_serena_config,
    _effective_prompt,
    _model_for,
    _reasoning_for,
)


class CodexRoleRoutingTests(unittest.TestCase):
    def test_scientific_roles_use_frontier_model(self):
        for role in ("architect", "coder", "inspector", "professor",
                     "prof_review"):
            with self.subTest(role=role), patch.dict(os.environ, {}, clear=True):
                options = ClaudeAgentOptions(agent_name=role)
                self.assertEqual(_model_for(options), "gpt-5.6-sol")
                self.assertEqual(_reasoning_for(options), "high")

    def test_support_roles_use_balanced_model(self):
        for role in ("foreman_lite", "test_dev", "librarian", "tidier",
                     "dreamer", "simplifier"):
            with self.subTest(role=role), patch.dict(os.environ, {}, clear=True):
                options = ClaudeAgentOptions(agent_name=role)
                self.assertEqual(_model_for(options), "gpt-5.6-terra")
                self.assertEqual(_reasoning_for(options), "medium")

    def test_role_override_precedes_global_override(self):
        options = ClaudeAgentOptions(agent_name="test_dev")
        with patch.dict(
            os.environ,
            {
                "CODEX_MODEL": "global-model",
                "CODEX_MODEL_TEST_DEV": "test-model",
                "CODEX_REASONING_EFFORT": "low",
                "CODEX_REASONING_EFFORT_TEST_DEV": "high",
            },
            clear=True,
        ):
            self.assertEqual(_model_for(options), "test-model")
            self.assertEqual(_reasoning_for(options), "high")

    def test_unknown_role_inherits_normal_codex_config(self):
        options = ClaudeAgentOptions(agent_name="unknown")
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_model_for(options), "")
            self.assertEqual(_reasoning_for(options), "")

    def test_role_map_covers_every_codex_agent_definition(self):
        agents_dir = CLAUDE_DIR.parent / ".codex" / "agents"
        configured = {path.stem for path in agents_dir.glob("*.toml")}
        self.assertEqual(configured, set(CODEX_ROLE_MODELS))


class CodexSerenaConfigTests(unittest.TestCase):
    def test_build_uses_required_shared_server_and_disables_interactive(self):
        command: list[str] = []
        with patch.dict(
            os.environ,
            {"CODEX_SERENA_URL": "http://localhost:8324/sse"},
            clear=False,
        ):
            os.environ.pop("AGENT_DISABLE_SERENA", None)
            _append_serena_config(command)

        self.assertIn("mcp_servers.serena.enabled=false", command)
        shared = next(
            value for value in command
            if value.startswith("mcp_servers.serena_build=")
        )
        self.assertIn('url="http://localhost:8324/sse"', shared)
        self.assertIn("required=true", shared)

    def test_no_serena_suppresses_shared_server(self):
        command: list[str] = []
        with patch.dict(
            os.environ,
            {
                "CODEX_SERENA_URL": "http://localhost:8324/sse",
                "AGENT_DISABLE_SERENA": "1",
            },
            clear=False,
        ):
            _append_serena_config(command)

        self.assertEqual(
            command, ["-c", "mcp_servers.serena.enabled=false"]
        )

    def test_build_prompt_uses_shared_server_tool_prefix(self):
        options = ClaudeAgentOptions(
            system_prompt="Use mcp__serena__read_file.",
            permission_mode="default",
        )
        with patch.dict(
            os.environ,
            {"CODEX_SERENA_URL": "http://localhost:8324/sse"},
            clear=False,
        ):
            os.environ.pop("AGENT_DISABLE_SERENA", None)
            prompt = _effective_prompt(
                "Then call mcp__serena__find_symbol.", options
            )

        self.assertNotIn("mcp__serena__read_file", prompt)
        self.assertIn("mcp__serena_build__read_file", prompt)
        self.assertIn("mcp__serena_build__find_symbol", prompt)


class SerenaManagerContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_codex_server_uses_distinct_port_and_context(self):
        manager = SerenaManager(
            "/repo",
            port=8324,
            context="codex",
        )
        manager._wait_for_ready = AsyncMock()

        process = MagicMock()
        process.poll.return_value = None
        with patch("sdk.agents.subprocess.Popen", return_value=process) as popen:
            await manager.start()

        argv = popen.call_args.args[0]
        self.assertEqual(argv[argv.index("--port") + 1], "8324")
        self.assertEqual(argv[argv.index("--context") + 1], "codex")
        self.assertEqual(manager.url, "http://localhost:8324/sse")


if __name__ == "__main__":
    unittest.main()
