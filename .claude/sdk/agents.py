"""Agent factory — creates SDK agents with the right model, prompt, tools,
and MCP configuration for each crew role.

Each agent is created on demand (not upfront) and gets only the instruction
sections and memories relevant to its role.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from .runtime import (
    AgentDefinition,
    ClaudeAgentOptions,
    HookJSONOutput,
    HookMatcher,
    PermissionMode,
    RUNTIME_PROVIDER,
    SandboxSettings,
)

logger = logging.getLogger(__name__)

from .memory import get_memory_names_for_agent, load_memories_text
from .prompts.sections import get_sections_for_agent


# ── SDK hooks (injected programmatically since setting_sources=['user']) ──────
#
# These are thin wrappers that call the actual shell scripts in .claude/hooks/,
# keeping those scripts as the single source of truth.


async def _run_hook_script(script_name: str, hook_input, _tool_use_id, _context) -> HookJSONOutput:
    """Run a .claude/hooks/ shell script as if it were a settings.json hook.

    Pipes the hook input JSON on stdin, parses the JSON output.
    Returns {} (allow) if the script produces no output.

    Stderr is captured and non-zero exits / parse errors are appended to
    `.claude/sdk/logs/hook_failures.log` so silent hook failures become
    diagnosable. Every invocation is also logged to `hook_trace.log` —
    the trace file is the primary diagnostic for "are hooks firing at
    all?" questions.
    """
    import json as _json
    # agents.py lives at <repo>/.claude/sdk/agents.py, so THREE dirnames reach
    # the repo root. It previously used two, landing on <repo>/.claude, and the
    # join below then yielded <repo>/.claude/.claude/hooks/... — a path that
    # never exists. isfile() failed, _run_hook_script returned {} (fail-open),
    # and so these SDK hooks NEVER fired and no hook_trace.log was ever written
    # (verified: true in gw_detection_ias too, which carries the same form).
    # Fixed 2026-07-16. This turns serena-redirection on for native
    # Read/Grep/Glob/Edit/Write/Bash, which is the intended design — the deny
    # messages are instructive ("USE SERENA ... pick the right tool") and agents
    # retry with the Serena equivalent. Note this was NOT the cause of the
    # 2026-07-16 zero-write builds; that was a role-scoping error in the plans
    # (Coder WPs must not author tests or run measurement campaigns — see
    # .claude/crew/architect.md and META_PLAN).
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    script = os.path.join(project_root, ".claude", "hooks", script_name)
    if not os.path.isfile(script):
        return {}
    # hook_input is a TypedDict (plain dict at runtime), despite the static
    # type hint suggesting attribute access. Using `hook_input.tool_name`
    # raises AttributeError which the SDK silently swallows — every hook
    # would then fire but crash immediately, tool would run unblocked,
    # no warning surfaces. Use subscript access.
    tool_name = hook_input["tool_name"] if isinstance(hook_input, dict) else getattr(hook_input, "tool_name", "")
    tool_input = hook_input.get("tool_input") if isinstance(hook_input, dict) else getattr(hook_input, "tool_input", None)
    input_json = _json.dumps({
        "tool_name": tool_name,
        "tool_input": tool_input or {},
    })

    def _log(filename: str, message: str) -> None:
        """Append a line to a log under .claude/sdk/logs/ — best-effort."""
        try:
            log_dir = os.path.join(project_root, ".claude", "sdk", "logs")
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, filename), "a") as f:
                f.write(message.rstrip() + "\n")
        except Exception:
            pass

    # Trace every invocation — primary diagnostic for "hooks not firing"
    _log(
        "hook_trace.log",
        f"{script_name} tool={tool_name} "
        f"input={input_json[:200]}",
    )

    proc = await asyncio.create_subprocess_exec(
        "bash", script,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate(input_json.encode())

    if proc.returncode != 0:
        _log(
            "hook_failures.log",
            f"{script_name} tool={tool_name} "
            f"exit={proc.returncode} stderr={stderr.decode(errors='replace').strip()[:200]}",
        )
        # Fail open: return allow rather than block on a broken hook.
        return {}

    if not stdout.strip():
        _log("hook_trace.log", f"  -> no stdout (allow)")
        return {}
    try:
        result = _json.loads(stdout)
        specific = result.get("hookSpecificOutput", {})
        if specific.get("permissionDecision") == "deny":
            _log(
                "hook_trace.log",
                f"  -> DENY ({specific.get('permissionDecisionReason', '')[:100]!r})",
            )
            return {
                "decision": "block",
                "reason": specific.get("permissionDecisionReason", "Blocked by hook."),
            }
        _log("hook_trace.log", f"  -> pass-through (allow)")
        return result
    except _json.JSONDecodeError:
        _log(
            "hook_failures.log",
            f"{script_name} tool={tool_name} "
            f"invalid JSON: {stdout.decode(errors='replace')[:200]}",
        )
        return {}


async def _serena_symbolic_hook(hook_input, tool_use_id, context) -> HookJSONOutput:
    """Delegate to .claude/hooks/use-serena.sh — block shell-as-grep."""
    return await _run_hook_script("use-serena.sh", hook_input, tool_use_id, context)


async def _use_serena_hook(hook_input, tool_use_id, context) -> HookJSONOutput:
    """Delegate to .claude/hooks/use-serena.sh — teach Serena best practices.

    Blocks native Read/Edit/Write/Grep/Glob and redirects to Serena
    equivalents. Safe for SDK agents because _build_sdk_hooks() is only
    used when use_serena=True; when Serena crashes and the retry uses
    use_serena=False, hooks=None and native tools pass through.
    """
    return await _run_hook_script("use-serena.sh", hook_input, tool_use_id, context)


def _build_sdk_hooks() -> dict:
    """Build hooks dict for ClaudeAgentOptions.

    Mirrors the hooks in .claude/settings.json but injected programmatically
    since SDK agents use setting_sources=['user'] (skipping project settings).
    The shell scripts in .claude/hooks/ are the single source of truth.

    Only used when use_serena=True. When Serena crashes and the orchestrator
    retries with use_serena=False, hooks=None — no hooks fire at all and
    native Read/Edit/Write/Bash all pass through as the fallback path.
    """
    return {
        "PreToolUse": [
            HookMatcher(
                matcher="mcp__serena__execute_shell_command",
                hooks=[_serena_symbolic_hook],
            ),
            HookMatcher(
                matcher="Read|Grep|Glob|Edit|Write|Bash",
                hooks=[_use_serena_hook],
            ),
        ],
    }


# ── Environment builder ────────────────────────────────────────────────────


def _build_env() -> dict[str, str]:
    """Build the environment dict for SDK subprocesses.

    Ensures the Python that launched the orchestrator is first on PATH
    so that bare ``python`` resolves correctly.
    """
    python_bin_dir = os.path.dirname(sys.executable)

    env = {
        "PATH": f"{python_bin_dir}:{os.environ.get('PATH', '')}",
        "HOME": os.environ.get("HOME", ""),
        "USER": os.environ.get("USER", ""),
        "CLAUDECODE": "",
        "AGENT_PROVIDER": os.environ.get("AGENT_PROVIDER", "claude"),
        # BUILDS ARE FAST — owner mandate (2026-07-21, stated three
        # times): the slow test tiers NEVER run inside a build.  These
        # gates are pinned EMPTY here so no in-build agent inherits or
        # accidentally enables them; the slow sweeps are the driver's
        # POST-BUILD parallel jobs (.claude/sdk/post_build_sweeps.sh).
        "COGWHEEL_BRUTE_ACCURACY": "",
        "COGWHEEL_STRICT_TIMING": "",
    }
    return env


# ── Shared Serena server manager ────────────────────────────────────────────


class SerenaManager:
    """Manage one shared Serena server for every role in a build."""

    def __init__(self, project_root: str, port: int | None = None,
                 external_url: str | None = None,
                 context: str = "claude-code",
                 transport: str = "sse"):
        if port is None:
            # Per-repo SSE port via the .env idiom (SDK_SERENA_PORT,
            # exported by launch_build.sh / .claude/build). Sibling
            # pipelines (gw) hardcode 8322 AND their watchdogs kill any
            # 8322 listener, so this repo's port must be movable.
            port = int(os.environ.get("SDK_SERENA_PORT", "8322"))
        self.project_root = project_root
        self.context = context
        self.transport = transport
        self._configured_port = port
        self.external_url = external_url
        if external_url:
            self.url = external_url
            self.port = 0
        else:
            self.port = port
            endpoint = "mcp" if transport == "streamable-http" else "sse"
            self.url = f"http://localhost:{port}/{endpoint}"
        self.process: subprocess.Popen | None = None

    async def start(self) -> str:
        """Start the configured Serena server and return its MCP URL.

        If external_url is set, probe the URL first. If unreachable, fall
        back to spawning our own Serena on the default port. Makes the
        skill template idempotent: callers can pass --serena-url to reuse
        an external Serena if one's warm; absence doesn't wedge the build.
        """
        if self.external_url:
            if await self._url_reachable(self.external_url):
                return self.url
            fallback_port = self._configured_port
            logger.warning(
                "Serena at %s unreachable; spawning own Serena on :%d",
                self.external_url, fallback_port,
            )
            self.external_url = None
            self.port = fallback_port
            endpoint = (
                "mcp" if self.transport == "streamable-http" else "sse"
            )
            self.url = f"http://localhost:{self.port}/{endpoint}"
        python_bin_dir = os.path.dirname(sys.executable)
        self.process = subprocess.Popen(
            [
                "uvx", "--from", "git+https://github.com/oraios/serena",
                "--with", "pyright[nodejs]",
                "serena", "start-mcp-server",
                "--transport", self.transport,
                "--port", str(self.port),
                "--project", self.project_root,
                "--context", self.context,
            ],
            env={
                **os.environ,
                "PATH": f"{python_bin_dir}:{os.environ.get('PATH', '')}",
            },
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            await self._wait_for_ready()
        except Exception:
            await self.stop()
            raise
        return self.url

    @staticmethod
    async def _url_reachable(url: str, timeout: float = 1.0) -> bool:
        """Return True if a TCP connection to the URL's host:port succeeds."""
        parsed = urlparse(url)
        if not parsed.hostname or not parsed.port:
            return False
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(parsed.hostname, parsed.port),
                timeout=timeout,
            )
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            return True
        except (asyncio.TimeoutError, ConnectionRefusedError, OSError):
            return False

    async def _wait_for_ready(self, timeout: float = 180.0,
                              settle: float = 3.0):
        """Wait for the Serena server to accept connections.

        IMPORTANT: do NOT open an SSE/HTTP session before the agent does —
        SSE resets project activation on each new client connection. A bare
        TCP connect (`_url_reachable`) sends no request and does not create
        an MCP session, so it is activation-safe; uvicorn only accepts TCP
        once the application is up, so TCP-accept is a faithful readiness
        signal.

        HISTORY (2026-07-20): this was a fixed 8 s sleep. On a loaded box
        the uvx cold start (git fetch + pyright + LSP init) can exceed it;
        the port then accepts nothing when the FIRST agent — always the
        Architect — connects, and that session runs tool-less for its whole
        life ('No such tool available' for every Serena tool). Three
        consecutive builds planned blind this way, while warm-server
        launches worked, masking the race as flakiness.
        """
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            if self.process and self.process.poll() is not None:
                raise RuntimeError(
                    f"Serena server exited during startup "
                    f"(rc={self.process.returncode})")
            if await self._url_reachable(self.url):
                break
            if asyncio.get_event_loop().time() > deadline:
                raise RuntimeError(
                    f"Serena server not accepting connections at "
                    f"{self.url} after {timeout:.0f}s")
            await asyncio.sleep(0.5)
        # Small margin between socket-accept and full MCP readiness.
        await asyncio.sleep(settle)

    def get_mcp_config(self) -> dict:
        """MCP server config dict for ClaudeAgentOptions."""
        return {"type": self.transport, "url": self.url}

    async def stop(self):
        if self.external_url:
            return
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None


# ── Model assignments ────────────────────────────────────────────────────────

AGENT_MODELS: dict[str, str] = {
    "architect":    "claude-opus-4-8",
    "professor":    "claude-opus-4-8",
    "simplifier":   "claude-sonnet-5",
    "foreman_lite": "claude-sonnet-5",
    "coder":        "claude-opus-4-8",
    "tidier":       "claude-sonnet-5",
    "test_dev":     "claude-sonnet-5",    # upgraded to opus if domain tests
    "inspector":    "claude-opus-4-8",
    "librarian":    "claude-sonnet-5",
    "dreamer":      "claude-sonnet-5",
    "prof_review":  "claude-opus-4-8",   # Phase 2: post-build inference review
}


# ── Tool sets per role ───────────────────────────────────────────────────────

_READ_TOOLS = ["Read", "Glob", "Grep"]
_EDIT_TOOLS = ["Read", "Glob", "Grep", "Edit", "Write", "Bash"]

# Editing tools for an agent that must NOT have any shell access.  The Tidier
# uses this: it is a pure style editor and has no business running git or any
# other shell command.  (gw 2026-06-16 incident: a Tidier ran
# `git checkout HEAD -- <file>`, wiping uncommitted Coder work, because it had
# Bash + Serena execute_shell_command.  A style pass needs neither.)
_EDIT_TOOLS_NO_SHELL = ["Read", "Glob", "Grep", "Edit", "Write"]

_SERENA_READ = [
    "mcp__serena__read_file",
    "mcp__serena__find_file",
    "mcp__serena__list_dir",
    "mcp__serena__get_symbols_overview",
    "mcp__serena__find_symbol",
    "mcp__serena__find_referencing_symbols",
    "mcp__serena__search_for_pattern",
    "mcp__serena__read_memory",
    "mcp__serena__list_memories",
    "mcp__serena__write_memory",
    "mcp__serena__edit_memory",
]

_SERENA_CODE_EDIT = [
    "mcp__serena__replace_content",
    "mcp__serena__replace_symbol_body",
    "mcp__serena__insert_at_line",
    "mcp__serena__insert_after_symbol",
    "mcp__serena__insert_before_symbol",
    "mcp__serena__delete_lines",
    "mcp__serena__replace_lines",
    "mcp__serena__rename_symbol",
    "mcp__serena__create_text_file",
    "mcp__serena__restart_language_server",
]

_SERENA_EDIT = _SERENA_READ + _SERENA_CODE_EDIT + [
    "mcp__serena__execute_shell_command",
]

# Serena editing tools WITHOUT shell access (no execute_shell_command).  The
# Tidier uses this so it cannot run git/shell at all.  Syntax verification is
# done via the language server (get_diagnostics_for_file) instead of a shell.
_SERENA_EDIT_NO_SHELL = _SERENA_READ + _SERENA_CODE_EDIT + [
    "mcp__serena__get_diagnostics_for_file",
]

AGENT_TOOLS: dict[str, dict[str, list[str]]] = {
    # Planning agents (architect/professor/simplifier) ALSO get the
    # built-in read tools as a belt-and-braces fallback: their sessions
    # are the FIRST to connect after a Serena SSE start, and a session
    # whose MCP handshake fails at startup has no MCP tools for its
    # whole life ("No such tool available"). Root-caused 2026-07-20:
    # the readiness wait was a fixed 8 s sleep that a loaded-box uvx
    # cold start can exceed (fixed in SerenaManager._wait_for_ready —
    # TCP-accept poll), after three consecutive builds planned blind;
    # warm-server launches worked, masking the race as flakiness.
    # NOT a plan-mode/MCP interaction (disproven: the 8c-cont
    # architect made successful serena calls in plan mode). Built-in
    # Read/Glob/Grep are read-only and plan-mode-permitted, so this
    # is additive and safe, and keeps planning functional under ANY
    # future MCP-session failure.
    "architect": {
        "serena": _SERENA_READ + _READ_TOOLS,
        "fallback": _READ_TOOLS + ["Agent"],
    },
    "professor": {
        "serena": _SERENA_READ + _READ_TOOLS,
        "fallback": _READ_TOOLS,
    },
    "prof_review": {
        # Phase 2: runs the domain tests, so it needs shell access.
        "serena": _SERENA_READ + ["mcp__serena__execute_shell_command"],
        "fallback": _READ_TOOLS + ["Bash"],
    },
    "simplifier": {
        "serena": _SERENA_READ + _READ_TOOLS,  # see plan-mode note above
        "fallback": _READ_TOOLS,
    },
    "coder": {
        "serena": _SERENA_EDIT,
        "fallback": _EDIT_TOOLS,
    },
    "foreman_lite": {
        "serena": _SERENA_EDIT,
        "fallback": _EDIT_TOOLS,
    },
    "tidier": {
        "serena": _SERENA_EDIT_NO_SHELL,
        "fallback": _EDIT_TOOLS_NO_SHELL,
    },
    "test_dev": {
        "serena": _SERENA_EDIT,
        "fallback": _EDIT_TOOLS,
    },
    "inspector": {
        "serena": _SERENA_READ + ["mcp__serena__execute_shell_command"],
        "fallback": _READ_TOOLS + ["Bash"],
    },
    "librarian": {
        "serena": _SERENA_EDIT,
        "fallback": _EDIT_TOOLS,
    },
    "dreamer": {
        "serena": _SERENA_READ + [
            "mcp__serena__write_memory",
            "mcp__serena__edit_memory",
            "mcp__serena__execute_shell_command",
        ],
        "fallback": _READ_TOOLS + ["Bash"],
    },
}

AGENT_PERMISSION_MODES: dict[str, PermissionMode] = {
    "architect":    "plan",
    "professor":    "plan",               # Phase 1 subagent: read-only
    "simplifier":   "plan",
    "coder":        "bypassPermissions",
    "foreman_lite": "bypassPermissions",
    "tidier":       "bypassPermissions",
    "test_dev":     "bypassPermissions",
    "inspector":    "bypassPermissions",
    "librarian":    "bypassPermissions",
    "dreamer":      "bypassPermissions",
    "prof_review":  "bypassPermissions",  # Phase 2: runs pytest
}


# ── Skill tool sets ──────────────────────────────────────────────────────────

SKILL_TOOLS: dict[str, dict[str, list[str]]] = {
    "tidier": {
        "serena": _SERENA_EDIT_NO_SHELL,
        "fallback": _EDIT_TOOLS_NO_SHELL,
    },
    "librarian": {
        "serena": _SERENA_EDIT,
        "fallback": _EDIT_TOOLS,
    },
    "simplifier": {
        "serena": [],
        "fallback": [],
    },
    "dreamer": {
        "serena": _SERENA_READ + [
            "mcp__serena__write_memory",
            "mcp__serena__edit_memory",
            "mcp__serena__execute_shell_command",
        ],
        "fallback": _READ_TOOLS + ["Bash"],
    },
}


# ── Crew prompt loading ─────────────────────────────────────────────────────

_CREW_FILE_MAP: dict[str, str] = {
    "architect":    "architect.md",
    "professor":    "professor.md",
    "simplifier":   "simplifier.md",
    "foreman_lite": "foreman_lite.md",
    "coder":        "coder.md",
    "tidier":       "tidy.md",
    "test_dev":     "test_dev.md",
    "inspector":    "inspector.md",
    "librarian":    "librarian.md",
    "dreamer":      "dreamer.md",
    "prof_review":  "prof_review.md",
}


def load_crew_prompt(agent_name: str, project_root: str) -> str:
    """Load the crew .md file for the given agent role."""
    crew_file = _CREW_FILE_MAP.get(agent_name)
    if not crew_file:
        return ""

    crew_path = Path(project_root) / ".claude" / "crew" / crew_file
    if not crew_path.exists():
        return f"(crew prompt for {agent_name} not found at {crew_path})"

    return crew_path.read_text(encoding="utf-8")


# ── System prompt assembly ───────────────────────────────────────────────────


async def build_system_prompt(
    agent_name: str,
    project_root: str,
    task_context: str = "",
    extra_instructions: str = "",
) -> str:
    """Assemble the full system prompt for an agent."""
    parts: list[str] = []

    crew = load_crew_prompt(agent_name, project_root)
    if crew:
        parts.append(crew)

    sections = get_sections_for_agent(agent_name)
    if sections:
        parts.append("# Project Instructions\n" + sections)

    memory_names = get_memory_names_for_agent(agent_name)
    if memory_names:
        memories_text = await load_memories_text(memory_names, project_root)
        parts.append("# Memories\n" + memories_text)

    if task_context:
        parts.append("# Task Context\n" + task_context)

    if extra_instructions:
        parts.append(extra_instructions)

    return "\n\n---\n\n".join(parts)


async def build_agent_options(
    agent_name: str,
    project_root: str,
    task_context: str = "",
    extra_instructions: str = "",
    use_serena: bool = True,
    mcp_config: Optional[dict] = None,
    max_turns: int = 75,
    model_override: Optional[str] = None,
    permission_override: Optional[PermissionMode] = None,
) -> ClaudeAgentOptions:
    """Build ClaudeAgentOptions for a given agent role."""
    model = model_override or AGENT_MODELS[agent_name]
    tool_config = AGENT_TOOLS[agent_name]
    permission_mode = permission_override or AGENT_PERMISSION_MODES[agent_name]

    system_prompt = await build_system_prompt(
        agent_name, project_root, task_context, extra_instructions,
    )

    _SERENA_REPLACEABLE = {"Read", "Glob", "Grep"}

    mcp_servers: dict = {}
    disallowed_tools: list[str] = [
        "ToolSearch", "ExitPlanMode", "EnterPlanMode",
        "AskUserQuestion",
        "mcp__serena__activate_project",
        "mcp__serena__initial_instructions",
        "mcp__serena__check_onboarding_performed",
        "mcp__serena__onboarding",
        "mcp__serena__get_current_config",
        "mcp__serena__switch_modes",
        "mcp__serena__prepare_for_new_conversation",
    ]
    if use_serena and mcp_config is not None:
        mcp_servers["serena"] = mcp_config
        allowed_tools = tool_config["serena"] + tool_config["fallback"]
        disallowed_tools += [t for t in tool_config["fallback"] if t in _SERENA_REPLACEABLE]
        if permission_mode == "plan":
            disallowed_tools += _SERENA_CODE_EDIT
            disallowed_tools.append("mcp__serena__execute_shell_command")
    else:
        allowed_tools = tool_config["fallback"]

    if permission_mode == "plan":
        for t in ("Edit", "Write", "Bash"):
            if t not in disallowed_tools:
                disallowed_tools.append(t)

    env = _build_env()

    # Do NOT add an ignoreViolations /tmp allowlist here to chase the
    # intermittent "The user doesn't want to take this action right now. STOP"
    # denial on coder /tmp writes. MEASURED 2026-07-16, detached (build
    # context), N=5 per arm, only the allowlist varied:
    #     with ignoreViolations {"file": ["/tmp/**", "/private/tmp/**"]}: 0/5 denied
    #     without it:                                                     0/5 denied
    # It makes no difference. The denial is real and does kill builds, but it is
    # NOT this: a minimal probe never reproduces it (0/10) while real build
    # coders hit it repeatedly, so the trigger is something the probe lacks —
    # a large system prompt (crew prompt + pre-read spec files + full WP text),
    # max_turns=90, and 7-10 prior tool calls. Suspect session depth/size, not
    # the sandbox. See META_PLAN; scratchpad/denial_rate.py is the harness.
    sandbox: SandboxSettings = {
        "enabled": True,
        "autoAllowBashIfSandboxed": True,
        "excludedCommands": ["git", "ssh", "scp", "rsync"],
        "allowUnsandboxedCommands": True,
        "network": {
            "allowLocalBinding": True,
            "allowAllUnixSockets": True,
        },
    }

    disallowed_set = set(disallowed_tools)
    allowed_tools = [t for t in allowed_tools if t not in disallowed_set]

    provider_options = (
        {"codex_model_override": model_override}
        if RUNTIME_PROVIDER == "codex"
        else {}
    )
    return ClaudeAgentOptions(
        agent_name=agent_name,
        model=model,
        system_prompt=system_prompt,
        allowed_tools=allowed_tools,
        disallowed_tools=disallowed_tools,
        permission_mode=permission_mode,
        max_turns=max_turns,
        mcp_servers=mcp_servers,
        # Agents-only permission allowlist (root-caused 2026-07-16):
        # setting_sources=["user"] is deliberate (keeps project settings.json
        # hooks out so the serena-crash fallback can run hook-free), but the
        # user scope carries no permissions block, so every shell call was
        # adjudicated by the auto-mode classifier, which FAILS CLOSED on its
        # own transient errors -- the intermittent bare "STOP" denial that
        # killed builds (106 denials / 59 sessions, depth-correlated because
        # the classifier prompt embeds the agent transcript). The explicit
        # settings file below carries ONLY portable allow rules -- no hooks,
        # no MCP keys (it can never spawn extra serenas) -- and allowlisted
        # tools take the classifier fast-path, never reaching it.
        settings=(lambda _p: _p if os.path.isfile(_p) else None)(
            os.path.join(project_root, ".claude", "settings.agents.json")),
        setting_sources=["user"],
        hooks=_build_sdk_hooks() if use_serena else None,
        cwd=project_root,
        env=env,
        sandbox=sandbox,
        **provider_options,
    )


async def build_phase1_subagents(
    project_root: str,
    use_serena: bool = True,
) -> dict[str, AgentDefinition]:
    """Build AgentDefinition entries for the Phase 1 consultation subagents."""
    agents: dict[str, AgentDefinition] = {}

    for name, model, description in [
        (
            "professor",
            "opus",
            "GW parameter-estimation expert — ask about likelihood / prior / "
            "sampler / marginalization physics and statistics, numerical-"
            "accuracy risks, convention pitfalls (IMRPhenomX, units), or test "
            "specifications. Multi-round: call multiple times for deep "
            "back-and-forth.",
        ),
        (
            "simplifier",
            "sonnet",
            "Complexity auditor — check if your proposed approach is "
            "over-engineered or if a simpler alternative exists.  Returns "
            "per-item verdicts: lean (fine) / watch (justified) / trim "
            "(too complex).",
        ),
    ]:
        parts: list[str] = []
        crew = load_crew_prompt(name, project_root)
        if crew:
            parts.append(crew)
        sections = get_sections_for_agent(name)
        if sections:
            parts.append("# Project Instructions\n" + sections)
        prompt = "\n\n---\n\n".join(parts)

        tool_config_entry = AGENT_TOOLS[name]
        tools = tool_config_entry["serena"] if use_serena else tool_config_entry["fallback"]

        agents[name] = AgentDefinition(
            description=description,
            prompt=prompt,
            tools=list(tools),
            model=model,
        )

    return agents
