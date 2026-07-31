"""Small Claude-SDK-compatible facade backed by ``codex exec --json``.

The orchestrator predates the Codex backend and deliberately keeps its
provider-neutral state machine.  These types expose only the subset of the
Claude Agent SDK surface that the orchestrator consumes.  Codex itself owns
tool execution, project hooks, MCP startup, authentication, and conversation
state; this module translates its JSONL event stream into the shared message
types.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal, Optional

HookJSONOutput = dict[str, Any]
PermissionMode = Literal["default", "acceptEdits", "plan", "bypassPermissions"]
SandboxSettings = dict[str, Any]

# Provider-specific cost/quality routing.  Claude keeps its established
# AGENT_MODELS map in agents.py; these defaults apply only when the shared
# orchestrator is running through ``codex exec``.
CODEX_ROLE_MODELS = {
    "architect": "gpt-5.6-sol",
    "coder": "gpt-5.6-terra",
    "inspector": "gpt-5.6-terra",
    "professor": "gpt-5.6-sol",
    "prof_review": "gpt-5.6-terra",
    "foreman_lite": "gpt-5.6-terra",
    "test_dev": "gpt-5.6-terra",
    "librarian": "gpt-5.6-terra",
    "tidier": "gpt-5.6-terra",
    "dreamer": "gpt-5.6-terra",
    "simplifier": "gpt-5.6-terra",
}

CODEX_ROLE_REASONING_EFFORTS = {
    "architect": "high",
    "coder": "high",
    "inspector": "high",
    "professor": "high",
    "prof_review": "high",
    "foreman_lite": "medium",
    "test_dev": "high",
    "librarian": "medium",
    "tidier": "medium",
    "dreamer": "medium",
    "simplifier": "medium",
}
DEFAULT_CODEX_JSON_STREAM_LIMIT = 8 * 1024 * 1024


@dataclass
class AgentDefinition:
    description: str = ""
    prompt: str = ""
    tools: list[str] = field(default_factory=list)
    model: Optional[str] = None


@dataclass
class HookMatcher:
    matcher: str = ""
    hooks: list[Any] = field(default_factory=list)


class ClaudeAgentOptions:
    """Permissive options bag matching the fields used by this repository."""

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


@dataclass
class TextBlock:
    text: str


@dataclass
class ToolUseBlock:
    name: str
    input: dict[str, Any]


@dataclass
class AssistantMessage:
    content: list[Any]


@dataclass
class SystemMessage:
    subtype: str


@dataclass
class ResultMessage:
    subtype: str
    result: str
    total_cost_usd: Optional[float] = None
    session_id: Optional[str] = None


def _role_override(prefix: str, role: str) -> str:
    """Return a role-specific environment override, if configured."""

    if not role:
        return ""
    key = f"{prefix}_{role.upper()}"
    return os.environ.get(key, "").strip()


def model_for_role(
    role: str,
    explicit_override: Optional[str] = None,
) -> str:
    """Resolve the effective Codex model for one build role."""

    return (
        (explicit_override or "").strip()
        or _role_override("CODEX_MODEL", role)
        or os.environ.get("CODEX_MODEL", "").strip()
        or CODEX_ROLE_MODELS.get(role, "")
    )


def _model_for(options: ClaudeAgentOptions) -> str:
    """Resolve Codex model with explicit, environment, then default precedence."""

    return model_for_role(
        getattr(options, "agent_name", ""),
        getattr(options, "codex_model_override", None),
    )


def _json_stream_limit() -> int:
    """Return the reader limit for newline-delimited Codex JSON events."""

    raw = os.environ.get("CODEX_JSON_STREAM_LIMIT", "").strip()
    if not raw:
        return DEFAULT_CODEX_JSON_STREAM_LIMIT
    limit = int(raw)
    if limit < 65_536:
        raise ValueError(
            "CODEX_JSON_STREAM_LIMIT must be at least 65536 bytes"
        )
    return limit


def _reasoning_for(options: ClaudeAgentOptions) -> str:
    """Resolve Codex reasoning effort with the same override precedence."""

    role = getattr(options, "agent_name", "")
    return (
        _role_override("CODEX_REASONING_EFFORT", role)
        or os.environ.get("CODEX_REASONING_EFFORT", "").strip()
        or CODEX_ROLE_REASONING_EFFORTS.get(role, "")
    )


def _effective_prompt(prompt: str, options: ClaudeAgentOptions) -> str:
    sections: list[str] = []
    system_prompt = getattr(options, "system_prompt", "")
    if system_prompt:
        sections.append(system_prompt)

    permission_mode = getattr(options, "permission_mode", "default")
    allowed_tools = getattr(options, "allowed_tools", None)
    if permission_mode == "plan":
        sections.append(
            "This is a read-only planning/review role. Do not modify files."
        )
    if allowed_tools == []:
        sections.append(
            "Do not call tools in this turn; return the requested structured "
            "answer from the supplied context."
        )

    sections.append(
        "Runtime note: you are running through the Codex backend of the shared "
        "Cogwheel agent pipeline. Follow AGENTS.md and the role contract in "
        "this prompt. Project hooks and Serena are configured by .codex/."
    )
    sections.append(prompt)
    effective = "\n\n".join(section for section in sections if section)
    if (
        os.environ.get("CODEX_SERENA_URL")
        and os.environ.get("AGENT_DISABLE_SERENA") != "1"
    ):
        # The project config reserves `serena` for the interactive stdio
        # server. Builds add the shared SSE endpoint as `serena_build`, so
        # translate the established crew prompts to the actual Codex tool
        # prefix without maintaining a second set of role contracts.
        effective = effective.replace(
            "mcp__serena__", "mcp__serena_build__"
        )
        sections = [
            "Before any native project tool use, call tool_search to discover "
            "mcp__serena_build__initial_instructions, then call it. "
            "The build-role native-tool gate opens only after that call.",
            effective,
        ]
        effective = "\n\n".join(sections)
    return effective


def _sandbox_for(options: ClaudeAgentOptions) -> str:
    if getattr(options, "permission_mode", "default") == "plan":
        return "read-only"
    return "workspace-write"


def _append_serena_config(command: list[str]) -> None:
    """Apply the build-scoped Serena overrides to a Codex CLI command."""

    serena_url = os.environ.get("CODEX_SERENA_URL", "").strip()
    if os.environ.get("AGENT_DISABLE_SERENA") == "1":
        command.extend(["-c", "mcp_servers.serena.enabled=false"])
    elif serena_url:
        # Project config defines the interactive stdio server as `serena`.
        # Disable it for build subprocesses and add the one warm SSE server
        # owned by the long-lived Python orchestrator. A separate name avoids
        # an invalid deep merge of stdio `command` and HTTP `url` fields.
        serena_config = (
            "{url="
            + json.dumps(serena_url)
            + ",startup_timeout_sec=180,required=true}"
        )
        command.extend([
            "-c",
            "mcp_servers.serena.enabled=false",
            "-c",
            f"mcp_servers.serena_build={serena_config}",
        ])


def _tool_block(item: dict[str, Any]) -> Optional[ToolUseBlock]:
    item_type = item.get("type", "")
    if item_type == "command_execution":
        return ToolUseBlock(
            name="Bash",
            input={"command": item.get("command", "")},
        )
    if item_type == "mcp_tool_call":
        server = item.get("server", "mcp")
        tool = item.get("tool", item.get("name", "tool"))
        arguments = item.get("arguments", item.get("input", {}))
        return ToolUseBlock(
            name=f"mcp__{server}__{tool}",
            input=arguments if isinstance(arguments, dict) else {"value": arguments},
        )
    if item_type in {"file_change", "web_search", "reasoning"}:
        return ToolUseBlock(name=item_type, input=item)
    return None


async def _read_stderr(stream: asyncio.StreamReader) -> str:
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = await stream.read(4096)
        if not chunk:
            break
        if size < 20_000:
            remaining = 20_000 - size
            chunks.append(chunk[:remaining])
            size += min(len(chunk), remaining)
    return b"".join(chunks).decode(errors="replace").strip()


async def query(
    *,
    prompt: str,
    options: ClaudeAgentOptions,
) -> AsyncIterator[Any]:
    """Run one Codex turn and yield the shared orchestrator message types."""

    codex = shutil.which("codex")
    if codex is None:
        raise RuntimeError(
            "Codex backend selected, but the 'codex' executable is not on PATH"
        )

    resume_id = getattr(options, "resume", None)
    command = [codex, "exec"]
    if resume_id:
        command.extend([
            "resume",
            "--json",
            "--dangerously-bypass-hook-trust",
        ])
    else:
        command.extend([
            "--json",
            "--sandbox",
            _sandbox_for(options),
            "-c",
            'approval_policy="never"',
            "--dangerously-bypass-hook-trust",
        ])

    model = _model_for(options)
    if model:
        command.extend(["--model", model])
    reasoning = _reasoning_for(options)
    if reasoning:
        command.extend(["-c", f'model_reasoning_effort="{reasoning}"'])
    _append_serena_config(command)

    cwd = os.path.abspath(getattr(options, "cwd", os.getcwd()))
    if not resume_id:
        command.extend(["-C", cwd])
    if resume_id:
        command.append(str(resume_id))
    command.append("-")

    child_env = os.environ.copy()
    child_env.update(getattr(options, "env", {}) or {})
    child_env["AGENT_PROVIDER"] = "codex"
    if (
        os.environ.get("CODEX_SERENA_URL")
        and os.environ.get("AGENT_DISABLE_SERENA") != "1"
    ):
        child_env["CODEX_SERENA_READY_KEY"] = uuid.uuid4().hex

    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=cwd,
        env=child_env,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=_json_stream_limit(),
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None

    process.stdin.write(_effective_prompt(prompt, options).encode())
    await process.stdin.drain()
    process.stdin.close()

    stderr_task = asyncio.create_task(_read_stderr(process.stderr))
    session_id: Optional[str] = None
    final_text = ""
    result_subtype = "success"
    saw_terminal_event = False

    try:
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type", "")
            if event_type == "thread.started":
                session_id = event.get("thread_id")
                yield SystemMessage(subtype="init")
                continue

            if event_type == "item.started":
                block = _tool_block(event.get("item", {}))
                if block is not None:
                    yield AssistantMessage(content=[block])
                continue

            if event_type == "item.completed":
                item = event.get("item", {})
                if item.get("type") == "agent_message":
                    final_text = item.get("text", final_text)
                continue

            if event_type == "turn.completed":
                saw_terminal_event = True
                result_subtype = "success"
                continue

            if event_type in {"turn.failed", "error"}:
                saw_terminal_event = True
                result_subtype = "error"
                error = event.get("error", event.get("message", "Codex turn failed"))
                final_text = (
                    error if isinstance(error, str) else json.dumps(error)
                )
    finally:
        if process.returncode is None:
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                process.terminate()
                await process.wait()

    stderr = await stderr_task
    if process.returncode != 0:
        result_subtype = "error"
        detail = stderr or f"codex exited with status {process.returncode}"
        final_text = f"{final_text}\n{detail}".strip()
    elif not saw_terminal_event:
        result_subtype = "error"
        final_text = (
            final_text
            or stderr
            or "Codex event stream ended without a terminal turn event"
        )

    yield ResultMessage(
        subtype=result_subtype,
        result=final_text,
        session_id=session_id,
    )
