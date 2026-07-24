"""Provider boundary for the shared build orchestrator.

Claude remains the default and uses the Claude Agent SDK without adaptation.
Set ``AGENT_PROVIDER=codex`` (normally via ``.codex/build``) to use the
Codex CLI compatibility layer instead.
"""

from __future__ import annotations

import os

RUNTIME_PROVIDER = os.environ.get("AGENT_PROVIDER", "claude").strip().lower()
if RUNTIME_PROVIDER not in {"claude", "codex"}:
    raise ValueError(
        "AGENT_PROVIDER must be 'claude' or 'codex', "
        f"not {RUNTIME_PROVIDER!r}"
    )

if RUNTIME_PROVIDER == "codex":
    from .runtime_codex import (
        AgentDefinition,
        AssistantMessage,
        ClaudeAgentOptions,
        HookJSONOutput,
        HookMatcher,
        PermissionMode,
        ResultMessage,
        SandboxSettings,
        SystemMessage,
        TextBlock,
        ToolUseBlock,
        query,
    )
else:
    from claude_agent_sdk import (  # pyright: ignore[reportMissingImports]
        AgentDefinition,
        AssistantMessage,
        ClaudeAgentOptions,
        HookJSONOutput,
        HookMatcher,
        PermissionMode,
        ResultMessage,
        SandboxSettings,
        SystemMessage,
        TextBlock,
        ToolUseBlock,
        query,
    )

__all__ = [
    "AgentDefinition",
    "AssistantMessage",
    "ClaudeAgentOptions",
    "HookJSONOutput",
    "HookMatcher",
    "PermissionMode",
    "RUNTIME_PROVIDER",
    "ResultMessage",
    "SandboxSettings",
    "SystemMessage",
    "TextBlock",
    "ToolUseBlock",
    "query",
]
