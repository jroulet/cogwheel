"""Small Claude-SDK-compatible facade backed by ``opencode run --format json``.

The orchestrator predates the OpenCode backend and deliberately keeps its
provider-neutral state machine.  These types expose only the subset of the
Claude Agent SDK surface that the orchestrator consumes.  OpenCode itself owns
tool execution, project plugins, MCP startup, authentication, and conversation
state; this module translates its JSON event stream into the shared message
types.

OpenCode JSON event types (observed v1.18):
  - step_start:  {type, timestamp, sessionID, part: {type:"step-start", messageID, ...}}
  - text:        {type, timestamp, sessionID, part: {type:"text", text, time: {start, end}}}
  - tool_use:    {type, timestamp, sessionID, part: {type:"tool", tool, callID, state: {status, input, output, error, metadata}}}
  - step_finish: {type, timestamp, sessionID, part: {type:"step-finish", reason, tokens, cost}}
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

# ── Model routing ───────────────────────────────────────────────────────
#
# Two model providers are supported, selected by OPENCODE_MODEL_PROVIDER
# (unset/empty = AI Commons; "go" = OpenCode Go native models).  Each has
# its own role → model map so the orchestrator's agent-type assignments
# ("opus" vs "sonnet" roles) are cleanly separated from the provider.
#
# Roles follow the two-tier convention:
#   opus-tier:  architect, coder, inspector, professor, prof_review, test_dev
#   sonnet-tier: foreman_lite, librarian, tidier, dreamer, simplifier

# ── AI Commons ──

_AI_COMMONS_OPUS = "my-custom-provider/claude-v4.6-opus"
_AI_COMMONS_SONNET = "my-custom-provider/claude-v4.6-sonnet"

AI_COMMONS_ROLE_MODELS = {
    "architect":    _AI_COMMONS_OPUS,
    "coder":        _AI_COMMONS_OPUS,
    "inspector":    _AI_COMMONS_OPUS,
    "professor":    _AI_COMMONS_OPUS,
    "prof_review":  _AI_COMMONS_OPUS,
    "foreman_lite": _AI_COMMONS_SONNET,
    "test_dev":     _AI_COMMONS_OPUS,
    "librarian":    _AI_COMMONS_SONNET,
    "tidier":       _AI_COMMONS_SONNET,
    "dreamer":      _AI_COMMONS_SONNET,
    "simplifier":   _AI_COMMONS_SONNET,
}

AI_COMMONS_ROLE_VARIANTS = {
    "architect":    "high",
    "coder":        "high",
    "inspector":    "high",
    "professor":    "high",
    "prof_review":  "high",
    "foreman_lite": "medium",
    "test_dev":     "high",
    "librarian":    "medium",
    "tidier":       "medium",
    "dreamer":      "medium",
    "simplifier":   "medium",
}

_AI_COMMONS_CLAUDEMAP = {
    "claude-opus-4-8":   _AI_COMMONS_OPUS,
    "claude-opus-4":     _AI_COMMONS_OPUS,
    "claude-sonnet-5":   _AI_COMMONS_SONNET,
    "claude-sonnet-4-6": _AI_COMMONS_SONNET,
    "claude-haiku-3-5":  "my-custom-provider/claude-v4.5-haiku",
}

# ── OpenCode Go (open native models) ──

_GO_OPUS = "opencode-go/deepseek-v4-pro"
_GO_SONNET = "opencode-go/deepseek-v4-flash"

GO_ROLE_MODELS = {
    "architect":    _GO_OPUS,
    "coder":        _GO_OPUS,
    "inspector":    _GO_OPUS,
    "professor":    _GO_OPUS,
    "prof_review":  _GO_OPUS,
    "foreman_lite": _GO_SONNET,
    "test_dev":     _GO_OPUS,
    "librarian":    _GO_SONNET,
    "tidier":       _GO_SONNET,
    "dreamer":      _GO_SONNET,
    "simplifier":   _GO_SONNET,
}

# Variants don't apply to Go native models (DeepSeek uses reasoning tokens
# via the model itself, not a separate variant flag).  Keep an empty map so
# callers that iterate roles do not crash.
GO_ROLE_VARIANTS = {role: "" for role in GO_ROLE_MODELS}

# When the orchestrator hardcodes a Claude name (triage, skills, etc.),
# translate it to the corresponding Go model.  For non-matching names the
# caller's string is passed through unchanged.
_GO_CLAUDEMAP = {
    "claude-opus-4-8":   _GO_OPUS,
    "claude-opus-4":     _GO_OPUS,
    "claude-sonnet-5":   _GO_SONNET,
    "claude-sonnet-4-6": _GO_SONNET,
    "claude-haiku-3-5":  _GO_SONNET,
}

# ── DeepSeek (API-key provider) ──
#
# Same DeepSeek models as OpenCode Go, but through the user's own DeepSeek
# API key.  Use when opencode-go quota is exhausted (OPENCODE_MODEL_PROVIDER=
# deepseek) -- the models are billed to the API key, not the Go subscription.

_DS_OPUS = "deepseek/deepseek-v4-pro"
_DS_SONNET = "deepseek/deepseek-v4-flash"

DEEPSEEK_ROLE_MODELS = {
    "architect":    _DS_OPUS,
    "coder":        _DS_OPUS,
    "inspector":    _DS_OPUS,
    "professor":    _DS_OPUS,
    "prof_review":  _DS_OPUS,
    "foreman_lite": _DS_SONNET,
    "test_dev":     _DS_OPUS,
    "librarian":    _DS_SONNET,
    "tidier":       _DS_SONNET,
    "dreamer":      _DS_SONNET,
    "simplifier":   _DS_SONNET,
}

DEEPSEEK_ROLE_VARIANTS = {role: "" for role in DEEPSEEK_ROLE_MODELS}

_DS_CLAUDEMAP = {
    "claude-opus-4-8":   _DS_OPUS,
    "claude-opus-4":     _DS_OPUS,
    "claude-sonnet-5":   _DS_SONNET,
    "claude-sonnet-4-6": _DS_SONNET,
    "claude-haiku-3-5":  _DS_SONNET,
}

# ── Active selection ──
#
# Set OPENCODE_MODEL_PROVIDER=go in the environment (or in .env) to use
# the Go native models; =deepseek to use the API-key DeepSeek provider.
# Unset or empty retains the AI Commons mappings.

_PROVIDER = os.environ.get("OPENCODE_MODEL_PROVIDER", "").strip().lower()
_GO_MODE = _PROVIDER == "go"
_DS_MODE = _PROVIDER == "deepseek"

if _GO_MODE:
    OPENCODE_ROLE_MODELS = GO_ROLE_MODELS
    OPENCODE_ROLE_VARIANTS = GO_ROLE_VARIANTS
    _CLAUDE_TO_OPENCODE_MODEL = _GO_CLAUDEMAP
elif _DS_MODE:
    OPENCODE_ROLE_MODELS = DEEPSEEK_ROLE_MODELS
    OPENCODE_ROLE_VARIANTS = DEEPSEEK_ROLE_VARIANTS
    _CLAUDE_TO_OPENCODE_MODEL = _DS_CLAUDEMAP
else:
    OPENCODE_ROLE_MODELS = AI_COMMONS_ROLE_MODELS
    OPENCODE_ROLE_VARIANTS = AI_COMMONS_ROLE_VARIANTS
    _CLAUDE_TO_OPENCODE_MODEL = _AI_COMMONS_CLAUDEMAP

DEFAULT_OPENCODE_JSON_STREAM_LIMIT = 8 * 1024 * 1024
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
    """Resolve the effective OpenCode model for one build role."""

    override = (explicit_override or "").strip()
    if override:
        # Translate Claude model names to OpenCode equivalents.
        return _CLAUDE_TO_OPENCODE_MODEL.get(override, override)
    return (
        _role_override("OPENCODE_MODEL", role)
        or os.environ.get("OPENCODE_MODEL", "").strip()
        or OPENCODE_ROLE_MODELS.get(role, "")
    )


def _model_for(options: ClaudeAgentOptions) -> str:
    """Resolve OpenCode model with explicit, environment, then default precedence.

    Falls back to translating Claude model names via _CLAUDE_TO_OPENCODE_MODEL
    when no role-based resolution produces a result (ad-hoc calls like triage
    that pass model= directly without agent_name).
    """
    resolved = model_for_role(
        getattr(options, "agent_name", ""),
        getattr(options, "opencode_model_override", None),
    )
    if resolved:
        return resolved
    # No role-based resolution — translate the hardcoded Claude model name.
    claude_model = getattr(options, "model", "")
    return _CLAUDE_TO_OPENCODE_MODEL.get(claude_model, claude_model)


def _json_stream_limit() -> int:
    """Return the reader limit for newline-delimited OpenCode JSON events."""

    raw = os.environ.get("OPENCODE_JSON_STREAM_LIMIT", "").strip()
    if not raw:
        return DEFAULT_OPENCODE_JSON_STREAM_LIMIT
    limit = int(raw)
    if limit < 65_536:
        raise ValueError(
            "OPENCODE_JSON_STREAM_LIMIT must be at least 65536 bytes"
        )
    return limit


def _variant_for(options: ClaudeAgentOptions) -> str:
    """Resolve OpenCode model variant (reasoning effort) with override precedence."""

    role = getattr(options, "agent_name", "")
    return (
        _role_override("OPENCODE_VARIANT", role)
        or os.environ.get("OPENCODE_VARIANT", "").strip()
        or OPENCODE_ROLE_VARIANTS.get(role, "")
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
        "Runtime note: you are running through the OpenCode backend of the shared "
        "Cogwheel agent pipeline. Follow AGENTS.md and the role contract in "
        "this prompt. Project plugins and Serena are configured by .opencode/."
    )
    sections.append(prompt)
    effective = "\n\n".join(section for section in sections if section)
    if (
        os.environ.get("OPENCODE_SERENA_URL")
        and os.environ.get("AGENT_DISABLE_SERENA") != "1"
    ):
        # The project config reserves `serena` for the interactive MCP
        # server. Builds add the shared server as `serena_build`, so
        # translate the established crew prompts to the actual OpenCode
        # tool prefix without maintaining a second set of role contracts.
        effective = effective.replace(
            "mcp__serena__", "mcp__serena_build__"
        )
        sections = [
            "Before any native project tool use, call "
            "mcp__serena_build__initial_instructions. "
            "The build-role native-tool gate opens only after that call.",
            effective,
        ]
        effective = "\n\n".join(sections)
    return effective


def _agent_for(options: ClaudeAgentOptions) -> str:
    """Resolve the OpenCode agent name from the orchestrator role.

    The orchestrator's permission_mode maps to OpenCode agents:
    - plan → 'plan' agent (read-only, no edits)
    - bypassPermissions → 'build' agent (full access, auto-approve)
    """
    permission_mode = getattr(options, "permission_mode", "default")
    if permission_mode == "plan":
        return "plan"
    return "build"


def _tool_block(part: dict[str, Any]) -> Optional[ToolUseBlock]:
    """Convert an OpenCode tool_use event part to a ToolUseBlock."""
    tool = part.get("tool", "")
    state = part.get("state", {})
    tool_input = state.get("input", {})

    if not tool:
        return None

    # OpenCode tool names are lowercase (bash, read, edit, glob, grep, etc.)
    # MCP tools appear as mcp__server__tool_name
    # Map to the capitalized form the orchestrator expects for built-in tools
    TOOL_NAME_MAP = {
        "bash": "Bash",
        "read": "Read",
        "edit": "Edit",
        "write": "Write",
        "glob": "Glob",
        "grep": "Grep",
        "task": "Task",
        "webfetch": "WebFetch",
        "websearch": "WebSearch",
        "todowrite": "TodoWrite",
    }

    mapped_name = TOOL_NAME_MAP.get(tool, tool)
    return ToolUseBlock(
        name=mapped_name,
        input=tool_input if isinstance(tool_input, dict) else {"value": tool_input},
    )


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
    """Run one OpenCode turn and yield the shared orchestrator message types."""

    opencode = shutil.which("opencode")
    if opencode is None:
        raise RuntimeError(
            "OpenCode backend selected, but the 'opencode' executable is not on PATH"
        )

    resume_id = getattr(options, "resume", None)
    command = [opencode, "run", "--format", "json", "--auto"]

    model = _model_for(options)
    if model:
        command.extend(["--model", model])
    variant = _variant_for(options)
    if variant:
        command.extend(["--variant", variant])

    # Agent selection
    agent = _agent_for(options)
    command.extend(["--agent", agent])

    # Session resumption
    if resume_id:
        command.extend(["--session", str(resume_id), "--continue"])

    cwd = os.path.abspath(getattr(options, "cwd", os.getcwd()))
    command.extend(["--dir", cwd])

    # The prompt goes as positional arguments (message)
    # OpenCode reads the message from positional args or stdin
    # Using -- to separate flags from the message content
    command.append("--")
    command.append(_effective_prompt(prompt, options))

    child_env = os.environ.copy()
    child_env.update(getattr(options, "env", {}) or {})
    child_env["AGENT_PROVIDER"] = "opencode"

    # Build-scoped Serena override: disable the interactive stdio server
    # and point at the warm build server owned by the orchestrator.
    # Uses OPENCODE_CONFIG_CONTENT (inline JSON merge) since opencode run
    # has no -c flag for config overrides.
    mcp_servers = getattr(options, "mcp_servers", None)
    serena_url = os.environ.get("OPENCODE_SERENA_URL", "").strip()
    if not mcp_servers and serena_url:
        # Fallback mode: Serena crashed, orchestrator is retrying with
        # built-in tools. Re-enable native read/edit/glob/grep (the project
        # opencode.json hard-denies them) and disable Serena entirely.
        child_env["OPENCODE_CONFIG_CONTENT"] = json.dumps({
            "$schema": "https://opencode.ai/config.json",
            "mcp": {"serena": {"enabled": False}},
            "permission": {
                "read": "allow",
                "edit": "allow",
                "glob": "allow",
                "grep": "allow",
                "bash": "allow",
            },
        })
    elif os.environ.get("AGENT_DISABLE_SERENA") == "1":
        child_env["OPENCODE_CONFIG_CONTENT"] = json.dumps({
            "$schema": "https://opencode.ai/config.json",
            "mcp": {"serena": {"enabled": False}},
            "permission": {
                "read": "allow",
                "edit": "allow",
                "glob": "allow",
                "grep": "allow",
                "bash": "allow",
            },
        })
    elif serena_url:
        child_env["OPENCODE_CONFIG_CONTENT"] = json.dumps({
            "$schema": "https://opencode.ai/config.json",
            "mcp": {
                "serena": {"enabled": False},
                "serena_build": {
                    "type": "remote",
                    "url": serena_url,
                    "enabled": True,
                },
            },
        })

    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=cwd,
        env=child_env,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=_json_stream_limit(),
    )
    assert process.stdout is not None
    assert process.stderr is not None

    stderr_task = asyncio.create_task(_read_stderr(process.stderr))
    session_id: Optional[str] = None
    final_text = ""
    total_cost: float = 0.0
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
            event_session = event.get("sessionID", "")
            part = event.get("part", {})

            # Capture session ID from first event
            if not session_id and event_session:
                session_id = event_session
                yield SystemMessage(subtype="init")

            if event_type == "tool_use":
                block = _tool_block(part)
                if block is not None:
                    yield AssistantMessage(content=[block])
                continue

            if event_type == "text":
                text = part.get("text", "")
                if text:
                    # Each text event is a complete message (not a chunk) —
                    # but multi-step conversations produce multiple text events
                    # across steps. Keep only the LAST step's text as the final
                    # result (same as Codex: item.completed agent_message).
                    # However, within a single step the text is already complete,
                    # so overwriting is correct per-step. The orchestrator reads
                    # ALL text blocks via AssistantMessage/TextBlock yields below.
                    final_text = text
                yield AssistantMessage(content=[TextBlock(text=text)])
                continue

            if event_type == "step_finish":
                reason = part.get("reason", "")
                cost = part.get("cost", 0)
                if isinstance(cost, (int, float)):
                    total_cost += cost
                if reason == "stop":
                    saw_terminal_event = True
                    result_subtype = "success"
                elif reason == "error":
                    saw_terminal_event = True
                    result_subtype = "error"
                # reason == "tool-calls" means more steps coming
                continue

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
        detail = stderr or f"opencode exited with status {process.returncode}"
        final_text = f"{final_text}\n{detail}".strip()
    elif not saw_terminal_event:
        result_subtype = "error"
        final_text = (
            final_text
            or stderr
            or "OpenCode event stream ended without a terminal step_finish event"
        )

    yield ResultMessage(
        subtype=result_subtype,
        result=final_text,
        total_cost_usd=total_cost if total_cost > 0 else None,
        session_id=session_id,
    )
