"""Build pipeline orchestrator — the state machine that drives the
three-phase build pipeline.

This is pure Python control flow.  LLM calls happen only inside agents,
which are created on demand and destroyed when done.

Usage:
    orchestrator = BuildOrchestrator(
        task="Add feature X to module Y",
        project_root=".",
    )
    asyncio.run(orchestrator.run())
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from claude_agent_sdk import (  # pyright: ignore[reportMissingImports]  # install-time dep; this is a template repo
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolUseBlock,
    query,
)

from claude_agent_sdk import PermissionMode  # pyright: ignore[reportMissingImports]  # install-time dep; this is a template repo

from .agents import (
    _build_env, _build_sdk_hooks, build_agent_options, build_phase1_subagents,
    AGENT_MODELS, SerenaManager, load_crew_prompt, SKILL_TOOLS,
)
from .memory import (
    AGENT_MEMORIES,
    get_memory_names_for_agent,
    load_memories_text,
    SERENA_MEMORIES_DIR,
)
from .prompts.sections import get_sections_for_agent
# ── Codex outside inspector (optional) ──────────────────────────────────────
# Set to True during installation if the user opts in. The orchestrator
# runs the Codex adversarial reviewer in parallel with the Claude Inspector
# on round 0 only; round 1+ is Claude solo. See
# references/outside-inspector-integration.md for full details.
CODEX_ENABLED: bool = False
CODEX_TIMEOUT_SECONDS: int = 120

from .gates import (
    MAX_CLOSURE_RECHECKS,
    MAX_REVISION_LOOPS,
    GateFailure,
    EscalationNeeded,
    check_branch_safety,
    check_commit_allowed,
    check_inspector_gate,
    classify_findings,
    is_fast_path_eligible,
    merge_inspector_results,
    prompt_escalation_decision,
    prompt_user_approval,
    should_escalate,
    verify_plan,
)
from .schemas import (
    BuildMode,
    BuildReport,
    DAGNode,
    DreamerResult,
    EscalationLevel,
    Finding,
    InspectorResult,
    InspectorVerdict,
    Phase,
    Plan,
    ProfReviewResult,
    ProfReviewVerdict,
    TriageEntry,
    TriageResult,
    TriageVerdict,
    WorkPackage,
    build_dependency_graph,
)
from .state import write_state


SHORT_TERM_CONSOLIDATION_THRESHOLD = 1500


class Verbosity(Enum):
    QUIET = "quiet"
    NORMAL = "normal"
    VERBOSE = "verbose"


CHANGE_REPORT_INSTRUCTION = (
    "\n\n## Change report (REQUIRED)\n"
    "At the END of your response, include a **change report** block:\n"
    "```change-report\n"
    "SUMMARY: <one-line description of what you did>\n"
    "FILES: <comma-separated list of files you modified or created>\n"
    "PREFIX: <conventional commit prefix: feat|fix|refactor|style|test|docs|chore>\n"
    "```\n"
    "This is used to build the git commit message.  Be specific in SUMMARY — "
    "describe WHAT changed, not the task you were given."
)

SPEC_FILES = [
    ".claude/spec/SPEC.md",
    ".claude/spec/TODO.md",
    ".claude/spec/FINDINGS.md",
    ".claude/spec/DATA_CONTRACTS.yaml",
]


# ── Inter-message timeout ────────────────────────────────────────────────────
# When a Serena tool call wedges, the SDK's `query()` async generator stops
# yielding messages but does not raise. This per-message timeout converts
# that into a catchable `asyncio.TimeoutError`, which the existing fallback
# in `_run_agent` catches and retries with built-in tools (sans Serena).
#
# Set to 0 (or any non-positive value) to disable entirely.
_raw_inter_msg_timeout = int(
    os.environ.get("SDK_INTER_MESSAGE_TIMEOUT_SECONDS", "300"))
INTER_MESSAGE_TIMEOUT_SECONDS: Optional[int] = (
    _raw_inter_msg_timeout if _raw_inter_msg_timeout > 0 else None)


@dataclass
class BuildOrchestrator:
    """State machine that drives the three-phase build pipeline.

    Phase 1 (Planning): Architect + Simplifier → Plan
    Phase 2 (Execution): Coder(s) → [Tidier ∥ TestDev] → Inspector → Librarian
    Phase 3 (Dreaming): Dreamer consolidates memories
    """

    task: str
    project_root: str
    verbosity: Verbosity = Verbosity.NORMAL
    use_serena: bool = True
    fast_path: bool = False
    dry_run: bool = False
    auto_approve: bool = False
    approval_dir: str | None = None
    serena_url: str | None = None

    # Runtime state
    phase: Phase = field(default=Phase.PLANNING, init=False)
    plan: Optional[Plan] = field(default=None, init=False)
    build_report: Optional[BuildReport] = field(default=None, init=False)
    _total_cost: float = field(default=0.0, init=False)
    _agent_count: int = field(default=0, init=False)
    _agents_that_ran: list[str] = field(default_factory=list, init=False)
    _change_reports: list[dict] = field(default_factory=list, init=False)
    _serena: Optional[SerenaManager] = field(default=None, init=False)
    _specs_text: str = field(default="", init=False)
    _inspector_result: Optional[InspectorResult] = field(default=None, init=False)
    _architect_session: Optional[str] = field(default=None, init=False)
    _coder_sessions: dict = field(default_factory=dict, init=False)  # wp_id → session_id
    _triage_result: Optional[str] = field(default=None, init=False)

    async def _triage(self) -> str:
        """Classify task complexity with a 1-turn Sonnet call.

        Returns 'trivial', 'standard', or 'complex'.
        Trivial tasks auto-route to fast-path (skip Phase 1).
        """
        recent_log = subprocess.run(
            ["git", "log", "--oneline", "-10"],
            capture_output=True, text=True, cwd=self.project_root,
        ).stdout.strip()

        prompt = (
            f"Classify this development task for complexity.\n\n"
            f"Task: {self.task}\n\n"
            f"Recent commits:\n{recent_log}\n\n"
            f"Respond with exactly one word:\n"
            f"- trivial: typo, config tweak, single-file edit, docs-only change\n"
            f"- standard: code change with clear scope, no domain-critical implications\n"
            f"- complex: multi-module, architecture changes, new public API, "
            f"domain-critical logic changes\n"
        )

        options = ClaudeAgentOptions(
            model="claude-sonnet-4-6",
            max_turns=1,
            # No tools: a tool call under max_turns=1 cannot complete and
            # crashes the subprocess (exit 1). This call is text-only; an
            # @file brief could otherwise coax a tool call. Empty the
            # allowlist and hard-block built-ins.
            allowed_tools=[],
            disallowed_tools=[
                "Bash", "Read", "Edit", "Write", "Glob", "Grep", "Task",
                "WebFetch", "WebSearch", "ToolSearch", "NotebookEdit",
                "TodoWrite",
            ],
            system_prompt=(
                "You are a task complexity classifier. "
                "Respond with exactly one word: trivial, standard, or complex."
            ),
            permission_mode="bypassPermissions",
            cwd=self.project_root,
            env=_build_env(),
        )

        result = ""
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, ResultMessage):
                result = (message.result or "").strip().lower()
                if message.total_cost_usd:
                    self._total_cost += message.total_cost_usd

        if result not in ("trivial", "standard", "complex"):
            self._log(f"  Triage returned '{result}', defaulting to 'complex'")
            return "complex"

        return result

    async def _fill_max_turns(self, missing_wp_ids: list[str]) -> None:
        """Resume the Architect to fill in max_turns on WPs that lack it."""
        wp_list = ", ".join(missing_wp_ids)
        prompt = (
            f"Your plan was approved, but the following WPs are missing "
            f"`max_turns` estimates: {wp_list}.\n\n"
            f"Using the formula in .claude/crew/architect.md (5-8 turns "
            f"per file audited, 10-15 per fix commit, 3-5 per file edited, "
            f"~10 overhead), estimate max_turns for each.\n\n"
            f"Respond with ONLY a JSON object mapping WP id to max_turns, "
            f'e.g. {{"WP-2": 120, "WP-3": 90}}. Nothing else.'
        )

        options = ClaudeAgentOptions(
            model="claude-sonnet-4-6",
            max_turns=1,
            # No tools: a tool call under max_turns=1 cannot complete and
            # crashes the subprocess (exit 1). Turn-budget estimation is pure
            # arithmetic; resuming the plan-mode Architect could otherwise
            # re-grant tools, so empty the allowlist and hard-block built-ins.
            allowed_tools=[],
            disallowed_tools=[
                "Bash", "Read", "Edit", "Write", "Glob", "Grep", "Task",
                "WebFetch", "WebSearch", "ToolSearch", "NotebookEdit",
                "TodoWrite",
            ],
            system_prompt="You are a turn-budget estimator. Respond with only JSON.",
            permission_mode="bypassPermissions",
            cwd=self.project_root,
            env=_build_env(),
        )
        if self._architect_session:
            options.resume = self._architect_session

        result = ""
        # Best-effort: max_turns is only a budget, and there is a deterministic
        # len(where)*8+10 fallback below. This is a RAW query() (resume of the
        # Architect session) with none of _run_agent's timeout/retry handling,
        # and it has aborted whole builds on transient SDK subprocess failures
        # ("Fatal error in message reader: Command failed with exit code 1").
        # Wrap it so ANY query failure degrades to the fallback formula instead
        # of killing Phase 2.
        try:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, ResultMessage):
                    result = (message.result or "").strip()
                    if message.total_cost_usd:
                        self._total_cost += message.total_cost_usd
        except Exception as _e:
            self._log(
                f"  max_turns estimation query failed "
                f"({type(_e).__name__}: {_e}); falling back to the "
                f"len(where)*8+10 formula for all missing WPs"
            )
            result = ""

        try:
            estimates = json.loads(result)
        except (ValueError, TypeError):
            self._log(f"  Could not parse Architect turn estimates: {result!r}")
            self._log(f"  Falling back to len(where)*8+10 formula")
            estimates = {}

        assert self.plan is not None  # Phase 1 sets self.plan before Phase 2 entry
        id_to_wp = {wp.id: wp for wp in self.plan.work_packages}
        for wp_id in missing_wp_ids:
            wp = id_to_wp.get(wp_id)
            if wp is None:
                continue
            if wp_id in estimates and isinstance(estimates[wp_id], int):
                wp.max_turns = max(75, estimates[wp_id])
                self._log(f"  {wp_id}: Architect estimated {wp.max_turns} turns")
            else:
                estimated = len(wp.where) * 8 + 10
                wp.max_turns = max(75, estimated)
                self._log(
                    f"  {wp_id}: auto-estimated {wp.max_turns} turns "
                    f"({len(wp.where)} files × 8 + 10)"
                )

    def _pre_read_specs(self) -> str:
        """Read spec files once in Python, format for system prompt injection."""
        parts = []
        for rel_path in SPEC_FILES:
            full_path = Path(self.project_root) / rel_path
            if full_path.exists():
                content = full_path.read_text(encoding="utf-8")
                parts.append(f"### {rel_path}\n```\n{content}\n```")
            else:
                parts.append(f"### {rel_path}\n(file not found)")
        return "# Pre-loaded Spec Files\n\n" + "\n\n".join(parts)

    def _pre_read_task_files(self) -> str:
        """Opportunistically pre-read files mentioned in the task description.

        Scans the task string for relative file paths that exist in the repo,
        reads them under a per-file size limit, and injects into the prompt.
        Eliminates 3-7x redundant reads observed in build audits.
        """
        import re

        PER_FILE_LIMIT = 15_000
        TOTAL_LIMIT = 50_000

        # NOTE: `-` is explicitly included in the character class — file paths
        # with dates (e.g. `2026-04-19_name.md`) or hyphenated slugs must match
        # or the file won't be pre-read and the architect will waste turns
        # spawning sub-agents to read it.
        candidates = re.findall(
            r'(?:^|\s|["`\'(])([a-zA-Z_.][\w./-]*\.(?:py|md|yaml|yml|json|toml|sh))\b',
            self.task,
        )
        seen = set(SPEC_FILES)
        unique = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique.append(c)

        if not unique:
            return ""

        parts = []
        total = 0
        read_files = []
        for rel_path in unique:
            full_path = Path(self.project_root) / rel_path
            if not full_path.is_file():
                continue
            size = full_path.stat().st_size
            if size > PER_FILE_LIMIT:
                continue
            if total + size > TOTAL_LIMIT:
                break
            content = full_path.read_text(encoding="utf-8", errors="replace")
            parts.append(f"### {rel_path}\n```\n{content}\n```")
            total += size
            read_files.append(rel_path)

        if not parts:
            return ""

        self._log(f"Pre-read {len(read_files)} task-referenced files ({total // 1024}KB)")
        return "# Pre-loaded Task Files\n\n" + "\n\n".join(parts)

    async def run(self) -> BuildReport:
        """Execute the full build pipeline."""
        # Signal to post-commit hook that an SDK build is active.
        # The hook skips its Librarian launch when this is set, because
        # the orchestrator runs its own Librarian in the pipeline.
        os.environ["SDK_BUILD_ACTIVE"] = "1"

        self._log_phase("Starting build pipeline")
        self._log(f"Task: {self.task}")
        self._log(f"Project: {self.project_root}")

        # Safety gate
        branch = check_branch_safety(self.project_root)
        self._log(f"Branch: {branch}")

        # Start shared Serena SSE server
        if self.use_serena:
            self._serena = SerenaManager(
                self.project_root, external_url=self.serena_url
            )
            if self.serena_url:
                self._log(f"Using existing Serena SSE at {self.serena_url}")
            else:
                self._log("Starting Serena SSE server...")
            await self._serena.start()
            if not self.serena_url:
                self._log(f"Serena SSE ready at {self._serena.url}")

        # Pre-read spec files
        self._log("Pre-reading spec files...")
        self._specs_text = self._pre_read_specs()
        task_files_text = self._pre_read_task_files()
        if task_files_text:
            self._specs_text += "\n\n" + task_files_text

        try:
            # Phase 0: Triage (skip if --fast)
            if not self.fast_path:
                self._log_phase("Phase 0: Triage")
                self._triage_result = await self._triage()
                self._log(f"Triage: {self._triage_result}")
                if self._triage_result == "trivial":
                    self._log("  → fast-path, skipping Phase 1")
                    self.fast_path = True

            # Phase 1: Planning (skip if --fast or trivial)
            if not self.fast_path:
                self.phase = Phase.PLANNING
                self._log_phase("Phase 1: Planning")

                max_plan_attempts = 3
                user_feedback = ""
                for attempt in range(1, max_plan_attempts + 1):
                    self.plan = await self._run_phase_1(
                        revision_feedback=user_feedback,
                    )

                    failures, missing_turns = verify_plan(self.plan)
                    if failures:
                        self._log("Plan verification failed:")
                        for f in failures:
                            self._log(f"  - {f}")
                        raise GateFailure("Plan did not pass verification gate.")

                    # Fill in missing max_turns via Architect
                    if missing_turns:
                        self._log(f"  {len(missing_turns)} WP(s) missing max_turns")
                        await self._fill_max_turns(missing_turns)

                    for wp in self.plan.work_packages:
                        self._log(f"  {wp.id}: max_turns={wp.max_turns}")

                    approved, user_feedback = prompt_user_approval(
                        self._format_plan(self.plan),
                        auto_approve=self.auto_approve,
                        approval_dir=self.approval_dir,
                    )
                    if approved:
                        break
                    if attempt == max_plan_attempts:
                        self._log(f"Plan rejected {max_plan_attempts} times. Aborting.")
                        return self._empty_report()
                    self._log(f"Plan rejected (attempt {attempt}/{max_plan_attempts}).")
                    if user_feedback:
                        self._log(f"  User feedback: {user_feedback}")

                # Persist plan for crash recovery
                if self.approval_dir and self.plan:
                    plan_path = Path(self.approval_dir) / "plan.json"
                    self.plan.save(plan_path)
                    self._log(f"Plan saved to {plan_path}")

                if self.dry_run:
                    self._log("Dry run — stopping after plan approval.")
                    return self._empty_report()
            else:
                self._log("Fast-path mode — skipping Phase 1.")

            # Phase 2: Execution
            self.phase = Phase.EXECUTION
            self._log_phase("Phase 2: Execution")
            self.build_report = await self._run_phase_2()

            self.phase = Phase.DONE
            self._log_phase("Build Complete")
            self._log(f"Total agents spawned: {self._agent_count}")
            self._log(f"Total cost: ${self._total_cost:.4f}")

            self.build_report.total_cost = self._total_cost
            self._append_cost_ledger()
            return self.build_report

        except GateFailure as e:
            self._log(f"GATE FAILURE: {e}")
            raise
        except EscalationNeeded as e:
            self._log(f"ESCALATION: {e}")
            raise
        except KeyboardInterrupt:
            self._log("Build cancelled by user.")
            raise
        finally:
            # Phase 3 (Memory Consolidation) runs only when:
            #   (a) we got past planning, AND
            #   (b) the try block did NOT raise an exception.
            #
            # Condition (b) prevents the Dreamer from running against
            # incomplete state after a mid-Phase-2 failure, which has
            # historically caused silent wedges.
            import sys as _sys
            _exc_type = _sys.exc_info()[0]
            _exc_in_flight = _exc_type is not None
            if (not _exc_in_flight
                    and self.phase in (Phase.EXECUTION, Phase.DREAMING, Phase.DONE)):
                try:
                    self.phase = Phase.DREAMING
                    self._log_phase("Phase 3: Memory Consolidation")
                    await self._run_phase_3()
                except Exception as dream_err:
                    self._log(f"Dreamer failed (non-fatal): {dream_err}")
            elif _exc_type is not None:
                self._log(
                    f"Skipping Phase 3: build raised {_exc_type.__name__}. "
                    f"Memory consolidation against incomplete state risks "
                    f"silent wedges — skipping to preserve the failure signal."
                )
            if self._serena is not None:
                self._log("Stopping Serena SSE server...")
                await self._serena.stop()
                self._serena = None
            os.environ.pop("SDK_BUILD_ACTIVE", None)
            os.environ.pop("SDK_FAST_PATH", None)

    # ── Phase 1: Planning ────────────────────────────────────────────────

    async def _run_phase_1(self, revision_feedback: str = "") -> Plan:
        """Architect-driven planning with Simplifier subagent.

        If `revision_feedback` is non-empty AND `self._architect_session` is
        set, resumes the architect's existing session so the prior plan is
        in its context and only edits are produced. Otherwise falls back to
        the fresh-planning path.
        """
        # Revision path: resume the architect's session so the prior plan is
        # in its context. The architect produces an EDIT rather than a
        # regeneration, avoiding regression risk from re-planning with only
        # text feedback.
        if revision_feedback and self._architect_session is not None:
            return await self._run_phase_1_revision(revision_feedback)

        subagents = await build_phase1_subagents(
            project_root=self.project_root,
            use_serena=self.use_serena,
        )

        if revision_feedback:
            self._log("Architect revising plan with user feedback (no session to resume — fresh replan)")
        else:
            self._log("Architect planning (with Simplifier subagent)")

        revision_section = ""
        if revision_feedback:
            revision_section = (
                f"\n## REVISION REQUEST\n"
                f"Your previous plan was rejected.  Feedback:\n"
                f"> {revision_feedback}\n"
                f"> Incorporate this feedback into a revised plan.\n\n"
            )

        architect_task = (
            f"You are the lead planner.  Produce a build plan for the task below.\n"
            f"This is a non-interactive pipeline — you cannot ask questions.  "
            f"If the task is ambiguous, make reasonable assumptions and document "
            f"them in the plan summary.\n\n"
            f"## Task\n{self.task}\n\n"
            f"{revision_section}"
            f"## Your subagents\n"
            f"You have one subagent available via the **Agent** tool:\n\n"
            f"- **simplifier**: Complexity auditor (Sonnet).  Check if your "
            f"approach is over-engineered or if simpler alternatives exist.  "
            f"Returns per-item verdicts: lean / watch / trim.\n\n"
            f"**You MUST consult the Simplifier** at least once.\n\n"
            f"## Workflow\n"
            f"1. Orient on the codebase (read specs, navigate symbols).\n"
            f"2. Consult the Simplifier — depth should match task complexity.\n"
            f"3. For work packages that require domain-specific tests: "
            f"write test descriptions in a `domain_test_descriptions` field.\n"
            f"4. Draft work packages informed by consultation feedback.\n"
            f"5. Output the plan as a **raw JSON object** in your final text "
            f"message.  Do NOT write plan files, do NOT use Write or "
            f"ExitPlanMode.  Just output the JSON directly as text.\n\n"
            f"## Plan fields\n"
            f"Your final output must include:\n"
            f"- summary (str)\n"
            f"- work_packages (list of objects with: id, title, what, where, "
            f"how, who ['Coder' or 'Foreman-Lite'], depends_on, verification, "
            f"max_turns [int — estimated turn budget for this WP])\n"
            f"- has_domain_tests (bool) — true if new domain-specific tests\n"
            f"- has_new_public_api (bool)\n"
            f"- has_spec_update (bool) — true if SPEC.md needs updating\n"
            f"- files_affected (list of str)\n"
            f"- domain_test_descriptions (list of str — natural-language "
            f"test specs: setup, operation, expected result)\n"
            f"- simplifier_inputs (list of str — key points from the "
            f"Simplifier that shaped the plan)\n"
        )

        mcp_config = (
            self._serena.get_mcp_config()
            if self.use_serena and self._serena is not None
            else None
        )
        options = await build_agent_options(
            agent_name="architect",
            project_root=self.project_root,
            task_context=architect_task,
            use_serena=self.use_serena,
            mcp_config=mcp_config,
            extra_instructions=self._specs_text,
        )

        options.agents = subagents
        if "Agent" not in options.allowed_tools:
            options.allowed_tools = list(options.allowed_tools) + ["Agent"]

        self._agent_count += 1
        self._agents_that_ran.append("architect")
        agent_id = f"architect-{self._agent_count}"
        if self.verbosity != Verbosity.QUIET:
            self._log(f"[{agent_id}] spawning ({AGENT_MODELS['architect']})")

        result_text = ""
        all_text_blocks: list[str] = []
        session_id = None
        async for message in query(prompt=architect_task, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        all_text_blocks.append(block.text)
            result_text, session_id = self._handle_message(
                agent_id, message, result_text, session_id,
            )

        self._architect_session = session_id
        write_state(self.project_root, "architect", status="plan_delivered")

        all_text = "\n".join(all_text_blocks)
        plan = self._parse_plan(result_text, fallback_text=all_text)

        if not plan.simplifier_inputs:
            self._log("  WARNING: Architect did not consult Simplifier")

        self._log(f"  Plan received: {len(plan.work_packages)} WPs")
        return plan

    async def _run_phase_1_revision(self, revision_feedback: str) -> Plan:
        """Edit-mode revision: resume architect session, feedback as user turn.

        Sends ONLY the feedback as the next user-turn prompt and sets
        `options.resume = self._architect_session`. The architect's prior
        plan and planning context are already in the resumed session's
        history, so it edits rather than regenerates.

        Mirrors the resume pattern in `_fill_max_turns`.
        """
        mcp_config = (
            self._serena.get_mcp_config()
            if self.use_serena and self._serena is not None
            else None
        )
        options = await build_agent_options(
            agent_name="architect",
            project_root=self.project_root,
            task_context="",  # Session has original system prompt; don't re-inject.
            use_serena=self.use_serena,
            mcp_config=mcp_config,
            extra_instructions="",
        )
        options.resume = self._architect_session

        # Keep subagents + Agent tool available in case the architect wants
        # to re-consult Simplifier on a revised WP.
        subagents = await build_phase1_subagents(
            project_root=self.project_root,
            use_serena=self.use_serena,
        )
        options.agents = subagents
        if "Agent" not in options.allowed_tools:
            options.allowed_tools = list(options.allowed_tools) + ["Agent"]

        prompt = (
            f"The build pipeline rejected your plan. Feedback from the "
            f"reviewer:\n\n{revision_feedback}\n\n"
            f"Produce a REVISED plan as a raw JSON object in your final text "
            f"message — same schema as your previous plan. Edit ONLY the "
            f"parts the feedback names; keep every other work package "
            f"unchanged. At the top of the summary list the specific changes "
            f"you made (e.g. 'Changes from v1: ...').\n\n"
            f"Do NOT regenerate from scratch. Edit."
        )

        self._agent_count += 1
        self._agents_that_ran.append("architect")
        agent_id = f"architect-{self._agent_count}"
        if self.verbosity != Verbosity.QUIET:
            self._log(
                f"[{agent_id}] resuming session "
                f"{self._architect_session[:12] if self._architect_session else '?'}"
                f"... (edit mode)"
            )

        result_text = ""
        all_text_blocks: list[str] = []
        session_id = None
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        all_text_blocks.append(block.text)
            result_text, session_id = self._handle_message(
                agent_id, message, result_text, session_id,
            )

        write_state(self.project_root, "architect", status="plan_delivered")
        if session_id is not None:
            self._architect_session = session_id

        all_text = "\n".join(all_text_blocks)
        plan = self._parse_plan(result_text, fallback_text=all_text)

        self._log(f"  Revised plan received: {len(plan.work_packages)} WPs")
        return plan

    # ── Phase 2: Execution ───────────────────────────────────────────────

    async def _run_phase_2(self) -> BuildReport:
        """Execute work packages with code-enforced sequencing."""
        if self.fast_path:
            return await self._run_fast_path_raw()

        if self.plan and is_fast_path_eligible(self.plan):
            self._log("Plan qualifies for fast-path (auto-detected)")
            return await self._run_fast_path_from_plan()

        return await self._run_full_pipeline()

    async def _run_fast_path_raw(self) -> BuildReport:
        """Fast-path from --fast flag: Foreman-Lite gets raw task."""
        # Signal to post-commit hook: fast-path doesn't run its own Librarian,
        # so the hook should fire the Librarian if doc-relevant files changed.
        os.environ["SDK_FAST_PATH"] = "1"
        self._log("Using fast-path execution (--fast, Foreman-Lite)")

        task_context = (
            f"Execute this task directly (fast-path — trivial change):\n\n"
            f"Task: {self.task}\n\n"
            f"Requirements:\n"
            f"- Make the code change\n"
            f"- Run syntax check\n"
            f"- Run import check on changed modules\n"
            f"- Verify no cross-file callers are affected\n"
            f"- Quick doc check: scan `.claude/spec/SPEC.md` for staleness\n"
            f"- Do NOT commit"
            + CHANGE_REPORT_INSTRUCTION
        )

        result_text, _ = await self._run_agent("foreman_lite", task_context)
        self._collect_change_report(result_text)

        commit_msg = self._build_commit_message()
        commit_sha = self._git_commit_safe(commit_msg)

        write_state(self.project_root, "foreman_lite", last_commit=commit_sha, status="fast_path_complete")

        report = BuildReport(
            mode=BuildMode.FAST_PATH,
            work_packages_completed=1,
            work_packages_total=1,
        )
        if commit_sha:
            report.commits.append(commit_sha)
        return report

    async def _run_fast_path_from_plan(self) -> BuildReport:
        """Fast-path auto-detected from plan."""
        assert self.plan is not None
        # Signal to post-commit hook: fast-path doesn't run its own Librarian.
        os.environ["SDK_FAST_PATH"] = "1"
        self._log("Using fast-path execution (plan-based, Foreman-Lite)")

        wp_descriptions = "\n\n".join(
            f"### {wp.id}: {wp.title}\n"
            f"**What**: {wp.what}\n"
            f"**Where**: {', '.join(wp.where)}\n"
            f"**How**: {wp.how}\n"
            f"**Verification**: {wp.verification}"
            for wp in self.plan.work_packages
        )

        task_context = (
            f"Execute this plan directly (fast-path):\n\n"
            f"## Plan summary\n{self.plan.summary}\n\n"
            f"## Work packages\n{wp_descriptions}\n\n"
            f"## Requirements\n"
            f"- Implement all work packages\n"
            f"- Run syntax + import checks\n"
            f"- Verify no callers affected\n"
            f"- Quick doc check\n"
            f"- Do NOT commit"
            + CHANGE_REPORT_INSTRUCTION
        )

        result_text, _ = await self._run_agent("foreman_lite", task_context)
        self._collect_change_report(result_text)

        commit_msg = self._build_commit_message()
        commit_sha = self._git_commit_safe(commit_msg)

        write_state(self.project_root, "foreman_lite", last_commit=commit_sha, status="fast_path_complete")

        report = BuildReport(
            mode=BuildMode.FAST_PATH,
            work_packages_completed=len(self.plan.work_packages),
            work_packages_total=len(self.plan.work_packages),
        )
        if commit_sha:
            report.commits.append(commit_sha)
        return report

    async def _run_full_pipeline(self) -> BuildReport:
        """Full pipeline: Coder → [Tidier ∥ TestDev] → Inspector → commit → Librarian."""
        assert self.plan is not None
        report = BuildReport(
            mode=BuildMode.FULL,
            work_packages_completed=0,
            work_packages_total=len(self.plan.work_packages),
        )

        # Phase 2a DAG: Coder → [Tidier ∥ TestDev] → Inspector
        dag = self._build_phase2a_dag(report)
        await self._run_dag(dag)

        # Step 5: Professor inference review (after the Inspector gate passes,
        # before the commit). Only when the build has domain-specific tests for
        # the Professor to run; a FAIL blocks the commit.
        if self.plan.has_domain_tests:
            self._log("Step 5: Professor inference review")
            prof_result = await self._run_prof_review()
            report.prof_review_result = prof_result
            if prof_result.verdict == ProfReviewVerdict.FAIL:
                raise GateFailure(
                    f"Professor inference review FAILED: {prof_result.summary}\n"
                    f"Concerns: {prof_result.concerns}"
                )
            elif prof_result.verdict == ProfReviewVerdict.CONCERN:
                self._log(f"  Professor: CONCERN — {prof_result.summary}")
                self._log("  (proceeding with commit — concerns noted in report)")
            else:
                self._log("  Professor: PASS")
        else:
            self._log("Step 5: Professor inference review skipped (no domain tests)")

        # Commit (Inspector + inference-review gates passed)
        build_changed_files = self._git_changed_files()
        check_commit_allowed(self._inspector_result, BuildMode.FULL)
        self._log("Committing changes")
        commit_msg = self._build_commit_message()
        commit_sha = self._git_commit_safe(commit_msg)
        if commit_sha:
            report.commits.append(commit_sha)

        write_state(self.project_root, "foreman_lite", last_commit=commit_sha, status="committed")

        # Deterministic doc sync (if sync script exists)
        self._log("Step 5a: Deterministic doc sync")
        sync_script = Path(self.project_root) / "scripts" / "sync_derived_docs.py"
        if sync_script.exists():
            _sync_result = subprocess.run(
                [sys.executable, str(sync_script)],
                cwd=self.project_root, capture_output=True, text=True,
            )
            if _sync_result.stdout.strip():
                self._log(_sync_result.stdout.strip())

        # Librarian (narrative doc sync)
        # Build context from earlier pipeline stages so the Librarian knows
        # *what* changed, not just *which files*.
        self._log("Step 5b: Librarian doc sync")
        context_parts: list[str] = []

        if self._change_reports:
            lines = []
            for cr in self._change_reports:
                files = cr.get("files", "")
                lines.append(f"- {cr['summary']} (files: {files})")
            context_parts.append("Coder change reports:\n" + "\n".join(lines))

        if self._inspector_result:
            insp = self._inspector_result
            insp_lines = []
            if insp.summary:
                insp_lines.append(f"Inspector summary: {insp.summary[:500]}")
            if insp.findings:
                insp_lines.append("Inspector findings (all resolved):")
                for f in insp.findings:
                    insp_lines.append(f"- [{f.severity.value}] {f.file}: {f.description}")
            if insp_lines:
                context_parts.append("\n".join(insp_lines))

        pipeline_context = "\n\n".join(context_parts)

        librarian_task = (
            f"Audit documentation surfaces for staleness caused by the recent "
            f"code changes.  Update as needed.\n\n"
            f"Work packages completed: {[wp.title for wp in self.plan.work_packages]}\n\n"
            f"Files changed in this build:\n"
            + "\n".join(f"- {f}" for f in build_changed_files)
            + (f"\n\n{pipeline_context}" if pipeline_context else "")
            + f"\n\n**IMPORTANT**: If you edit any file under `docs/source/`, "
            f"run the Sphinx docs rebuild command before finishing."
        )
        await self._run_skill("librarian", librarian_task, max_turns=75)
        write_state(self.project_root, "librarian", status="completed")

        if self._has_uncommitted_changes():
            lib_sha = self._git_commit_safe("docs: update documentation after build")
            if lib_sha:
                report.commits.append(lib_sha)

        # Render canonical files from fragments (Librarian may have created
        # new changelog/todo/completed fragments that need assembly).
        render_script = Path(self.project_root) / "scripts" / "render_fragments.py"
        if render_script.exists():
            subprocess.run(
                [sys.executable, str(render_script)],
                cwd=str(self.project_root),
                capture_output=True,
            )
            if self._has_uncommitted_changes():
                render_sha = self._git_commit_safe(
                    "docs: render fragments after librarian")
                if render_sha:
                    report.commits.append(render_sha)

        return report

    # ── DAG engine ───────────────────────────────────────────────────────

    def _build_phase2a_dag(self, report: BuildReport) -> list[DAGNode]:
        """Build the pre-commit Phase 2a DAG."""

        async def run_coders() -> str:
            return await self._run_coders(report)

        async def run_tidier() -> str:
            return await self._run_tidier_skill()

        async def run_test_dev() -> str:
            return await self._run_test_dev_agent()

        async def run_inspector() -> str:
            await self._run_inspector_with_loop(report)
            return ""

        return [
            DAGNode(
                name="coder", kind="agent",
                depends_on=[],
                skip_when=lambda *_: False,
                run=run_coders,
            ),
            DAGNode(
                name="tidier", kind="skill",
                depends_on=["coder"],
                skip_when=lambda _, f: not any(x.endswith(".py") for x in f),
                run=run_tidier,
            ),
            DAGNode(
                name="test_dev", kind="agent",
                depends_on=["coder"],
                skip_when=lambda p, _: not p.has_domain_tests and not p.has_new_public_api,
                run=run_test_dev,
            ),
            DAGNode(
                name="inspector", kind="agent",
                depends_on=["coder", "tidier", "test_dev"],
                skip_when=lambda *_: False,
                run=run_inspector,
            ),
        ]

    async def _run_dag(self, nodes: list[DAGNode]) -> dict[str, str]:
        """Execute DAG nodes respecting dependencies, parallelising where possible."""
        completed: dict[str, str] = {}
        remaining = {n.name: n for n in nodes}

        while remaining:
            ready = [
                n for n in remaining.values()
                if all(d in completed for d in n.depends_on)
            ]
            if not ready:
                unresolved = {
                    name: [d for d in node.depends_on if d not in completed]
                    for name, node in remaining.items()
                }
                raise ValueError(f"Circular or unresolvable DAG deps: {unresolved}")

            changed_files = self._git_changed_files()
            to_run: list[DAGNode] = []
            to_skip: list[DAGNode] = []
            assert self.plan is not None  # Phase 2 runs only after planning succeeds
            plan = self.plan
            for node in ready:
                if node.skip_when(plan, changed_files):
                    to_skip.append(node)
                    self._log(f"  DAG: skipping {node.name} (skip condition met)")
                else:
                    to_run.append(node)

            if to_run:
                if len(to_run) > 1:
                    self._log(f"  DAG: running in parallel — {[n.name for n in to_run]}")
                results = await asyncio.gather(*[n.run() for n in to_run])
                for node, result in zip(to_run, results):
                    completed[node.name] = result

            for node in to_skip:
                completed[node.name] = ""
            for node in ready:
                del remaining[node.name]

        return completed

    # ── DAG node implementations ─────────────────────────────────────────

    async def _run_coders(self, report: BuildReport) -> str:
        """Run Coder for all work packages."""
        assert self.plan is not None
        self._log("Step 1: Coder executing work packages")
        batches = build_dependency_graph(self.plan.work_packages)

        for i, batch in enumerate(batches):
            self._log(f"  Batch {i + 1}/{len(batches)}: {[wp.id for wp in batch]}")
            if len(batch) == 1:
                await self._run_coder(batch[0])
                report.work_packages_completed += 1
            else:
                await asyncio.gather(*[self._run_coder(wp) for wp in batch])
                report.work_packages_completed += len(batch)

        return ""

    async def _run_tidier_skill(self) -> str:
        """Run Tidier skill on changed Python files."""
        all_files_changed = self._git_changed_files()
        py_files = [f for f in all_files_changed if f.endswith(".py")]
        self._log("Step 2: Tidier cleanup")
        tidier_task = (
            f"Clean up the files changed by recent code work.\n"
            f"Apply the canonical rubric: spacing, import ordering.\n"
            f"Files to check: {py_files or '(no Python files changed)'}"
            + CHANGE_REPORT_INSTRUCTION
        )
        # The Tidier is cosmetic cleanup (spacing, import ordering). A failure
        # here — notably error_max_turns when it grinds on a large changed file
        # — must NOT abort a build whose Coder + Test-Developer work is already
        # done and verified. Degrade gracefully: log and proceed to the
        # Inspector. Partial tidy edits are kept (they parse; the Inspector
        # reviews them); we do NOT roll back, because test_dev runs in parallel
        # on the same tree and that would discard the newly written tests.
        try:
            result = await self._run_skill("tidier", tidier_task, max_turns=75)
        except Exception as _e:
            self._log(
                f"  Tidier failed ({type(_e).__name__}: {_e}); cosmetic "
                f"cleanup is non-fatal — proceeding to Inspector with the "
                f"current (possibly partially-tidied) tree"
            )
            write_state(self.project_root, "tidy", status="failed")
            return ""
        self._collect_change_report(result)
        write_state(self.project_root, "tidy", status="completed")
        return result

    async def _run_test_dev_agent(self) -> str:
        """Run Test Developer to write tests."""
        assert self.plan is not None
        self._log("Step 3: Test Developer writing tests")
        model = "claude-opus-4-8" if self.plan.has_domain_tests else None
        test_specs = "\n".join(
            f"- {desc}" for desc in self.plan.domain_test_descriptions
        )
        test_task = (
            f"Write tests for the recent code changes.\n\n"
            f"Work packages: {[wp.id + ': ' + wp.title for wp in self.plan.work_packages]}\n\n"
            + (f"Domain test specifications from Architect:\n{test_specs}\n\n"
               if test_specs else "")
            + f"Run all tests after writing them.\n\n"
            + CHANGE_REPORT_INSTRUCTION
        )
        # Dynamic max_turns: domain tests get more budget
        max_turns = 120 if self.plan.has_domain_tests else 75
        result_text, _ = await self._run_agent(
            "test_dev", test_task,
            model_override=model,
            max_turns_override=max_turns,
        )
        self._collect_change_report(result_text)
        write_state(self.project_root, "test_dev", status="completed")
        return result_text

    async def _run_inspector_with_loop(self, report: BuildReport) -> None:
        """Run Inspector with revision loop."""
        assert self.plan is not None
        self._log("Step 4: Inspector review")

        round_number = 0
        inspector_result = await self._run_inspector(round_number=round_number)

        open_findings: dict[str, Finding] = {
            f.finding_id: f for f in inspector_result.findings if f.finding_id
        }

        revision_loops = 0
        closure_rechecks = 0

        while not check_inspector_gate(inspector_result) or open_findings:
            # Closure re-check path
            if check_inspector_gate(inspector_result) and open_findings:
                if closure_rechecks >= MAX_CLOSURE_RECHECKS:
                    self._log(
                        f"  WARNING: {len(open_findings)} finding(s) unresolved after "
                        f"{MAX_CLOSURE_RECHECKS} closure re-checks — proceeding: "
                        + ", ".join(open_findings)
                    )
                    break
                closure_rechecks += 1
                self._log(
                    f"  Inspector PASS but {len(open_findings)} open finding(s) — "
                    f"closure re-check {closure_rechecks}/{MAX_CLOSURE_RECHECKS}"
                )
                round_number += 1
                inspector_result = await self._run_inspector(
                    open_findings=list(open_findings.values()),
                    round_number=round_number,
                )
                for rid in inspector_result.resolved_ids:
                    open_findings.pop(rid, None)
                for f in inspector_result.findings:
                    if f.finding_id and f.finding_id not in open_findings:
                        open_findings[f.finding_id] = f
                continue

            # Normal revision loop
            revision_loops += 1
            classified = classify_findings(inspector_result.findings)

            if should_escalate(inspector_result.findings, revision_loops):
                raise EscalationNeeded(inspector_result.findings, revision_loops)

            trivial_findings = classified[EscalationLevel.TRIVIAL]
            impl_findings = classified[EscalationLevel.IMPLEMENTATION]
            design_findings = classified[EscalationLevel.DESIGN]

            self._log(
                f"  Inspector issues — revision {revision_loops}/{MAX_REVISION_LOOPS}"
                f" ({len(trivial_findings)} trivial, {len(impl_findings)} impl"
                f", {len(design_findings)} design)"
            )

            if not trivial_findings and not impl_findings and not design_findings and not open_findings:
                self._log("  Inspector: no actionable findings — treating as PASS.")
                # Flip the stored verdict so the post-loop assignment to
                # self._inspector_result carries PASS — otherwise the commit
                # gate (check_commit_allowed) rejects an ISSUES verdict even
                # though there is nothing actionable to fix.
                inspector_result.verdict = InspectorVerdict.PASS
                break

            # Architect triages DESIGN findings directly
            if design_findings:
                triage = await self._run_architect_triage(design_findings)
                user_findings: list[Finding] = []
                arch_rationale_parts: list[str] = []

                for entry in triage.entries:
                    if entry.verdict == TriageVerdict.CODER_FIX:
                        if entry.finding_id in open_findings:
                            f = open_findings[entry.finding_id]
                            f.severity = EscalationLevel.IMPLEMENTATION
                            f.suggested_fix = entry.coder_instructions
                            impl_findings.append(f)
                        self._log(f"    Architect: {entry.finding_id} → coder fix")
                    elif entry.verdict == TriageVerdict.OVERRIDE:
                        open_findings.pop(entry.finding_id, None)
                        self._log(f"    Architect: {entry.finding_id} → override ({entry.rationale})")
                    elif entry.verdict == TriageVerdict.ESCALATE:
                        if entry.finding_id in open_findings:
                            user_findings.append(open_findings[entry.finding_id])
                        arch_rationale_parts.append(f"{entry.finding_id}: {entry.rationale}")
                        self._log(f"    Architect: {entry.finding_id} → escalate to user")

                if user_findings:
                    decision, feedback = prompt_escalation_decision(
                        user_findings,
                        architect_rationale="\n".join(arch_rationale_parts),
                    )
                    if decision == "accept":
                        for f in user_findings:
                            open_findings.pop(f.finding_id, None)
                        self._log("  User: accepted findings, proceeding")
                    elif decision == "fix":
                        for f in user_findings:
                            f.severity = EscalationLevel.IMPLEMENTATION
                            f.suggested_fix = feedback
                            impl_findings.append(f)
                        self._log(f"  User: fix with instructions: {feedback[:80]}")
                    elif decision == "abort":
                        raise EscalationNeeded(user_findings, revision_loops)

            # Tier 1: Foreman-Lite fixes trivial findings
            if trivial_findings:
                trivial_text = "\n".join(
                    f"- [id: {f.finding_id}] [{f.file}] {f.description}"
                    + (f" → {f.suggested_fix}" if f.suggested_fix else "")
                    for f in trivial_findings
                )
                trivial_task = (
                    f"Fix these trivial Inspector findings directly:\n\n"
                    f"{trivial_text}\n\n"
                    f"Fix ALL of them — do not skip any finding ID."
                    + CHANGE_REPORT_INSTRUCTION
                )
                trivial_result, _ = await self._run_agent("foreman_lite", trivial_task)
                self._collect_change_report(trivial_result)

            # Tier 2: Coder fixes implementation findings
            if impl_findings:
                diff = self._git_diff_cached()
                wp_summaries = "\n".join(
                    f"- {wp.id}: {wp.title} — {wp.what}"
                    for wp in self.plan.work_packages
                )
                findings_text = "\n".join(
                    f"- [id: {f.finding_id}] [{f.file}] {f.description}"
                    + (f"\n  Suggested fix: {f.suggested_fix}" if f.suggested_fix else "")
                    for f in impl_findings
                )
                fix_task = (
                    f"## Context\n"
                    f"You are fixing Inspector findings from a build.\n\n"
                    f"### Original work packages\n{wp_summaries}\n\n"
                    f"### Inspector findings to fix\n{findings_text}\n\n"
                    f"Fix ALL findings. Do not skip any.\n\n"
                    f"### Recent changes (git diff)\n"
                    f"```\n{diff[:3000]}\n```\n\n"
                    + CHANGE_REPORT_INSTRUCTION
                )
                # Resume most recent Coder session if available — it already
                # knows the code it wrote and why.
                last_coder = (
                    list(self._coder_sessions.values())[-1]
                    if self._coder_sessions else None
                )
                fix_result, _ = await self._run_agent(
                    "coder", fix_task, resume_session=last_coder,
                )
                self._collect_change_report(fix_result)

            round_number += 1
            inspector_result = await self._run_inspector(
                open_findings=list(open_findings.values()),
                round_number=round_number,
            )
            for rid in inspector_result.resolved_ids:
                open_findings.pop(rid, None)
            for f in inspector_result.findings:
                if f.finding_id and f.finding_id not in open_findings:
                    open_findings[f.finding_id] = f

        self._log("  Inspector: PASS")
        write_state(self.project_root, "inspector", status="PASS")
        report.inspector_result = inspector_result
        report.revision_loops = revision_loops
        self._inspector_result = inspector_result

    async def _run_coder(self, wp: WorkPackage) -> tuple[str, Optional[str]]:
        """Run a Coder agent for a single work package."""
        task = (
            f"## Work Package {wp.id}: {wp.title}\n\n"
            f"**What**: {wp.what}\n"
            f"**Where**: {', '.join(wp.where)}\n"
            f"**How**: {wp.how}\n"
            f"**Verification**: {wp.verification}\n"
            + CHANGE_REPORT_INSTRUCTION
        )
        result_text, session_id = await self._run_agent(
            "coder", task,
            max_turns_override=wp.max_turns,
        )
        self._collect_change_report(result_text)
        write_state(self.project_root, "coder", status=f"completed_{wp.id}")
        if session_id:
            self._coder_sessions[wp.id] = session_id
        return result_text, session_id

    async def _run_codex_review(
        self, changed_files: list[str], plan_context: str,
    ) -> Optional[InspectorResult]:
        """Run Codex adversarial review. Returns InspectorResult or None on failure.

        Soft failure semantics: never blocks the pipeline. Any error returns None,
        and merge_inspector_results handles None gracefully.
        See references/outside-inspector-integration.md for full details.
        """
        try:
            diff = self._git_diff_cached()
            crew_path = Path(self.project_root) / ".claude" / "crew" / "outside_inspector.md"
            if not crew_path.exists():
                self._log("  Codex: outside_inspector.md not found, skipping")
                return None
            focus_text = crew_path.read_text(encoding="utf-8")

            import shutil
            if not shutil.which("codex"):
                self._log("  Codex: CLI not installed, skipping")
                return None

            prompt = (
                f"{focus_text}\n\n"
                f"## Plan context\n{plan_context}\n\n"
                f"## Changed files\n{', '.join(changed_files)}\n\n"
                f"## Diff\n```\n{diff[:8000]}\n```\n"
            )
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    "codex", "ask", "-q", prompt,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                    cwd=self.project_root,
                ),
                timeout=CODEX_TIMEOUT_SECONDS,
            )
            stdout, _ = await asyncio.wait_for(
                proc.communicate(), timeout=CODEX_TIMEOUT_SECONDS,
            )
            response = stdout.decode("utf-8", errors="replace").strip()

            # Parse findings from Codex JSON response
            json_str = self._extract_json_block(response) or response
            data = json.loads(json_str)
            findings = []
            if isinstance(data, list):
                for n, f in enumerate(data):
                    severity_map = {
                        "critical": EscalationLevel.DESIGN,
                        "high": EscalationLevel.IMPLEMENTATION,
                        "medium": EscalationLevel.IMPLEMENTATION,
                        "low": EscalationLevel.TRIVIAL,
                    }
                    findings.append(Finding(
                        severity=severity_map.get(f.get("severity", "medium"), EscalationLevel.IMPLEMENTATION),
                        file=f.get("file", "unknown"),
                        description=f.get("description", ""),
                        suggested_fix=f.get("suggested_fix"),
                        finding_id=f"CDX-0-{n + 1:03d}",
                        source="codex",
                    ))

            verdict = InspectorVerdict.ISSUES if findings else InspectorVerdict.PASS
            self._log(f"  Codex: {len(findings)} finding(s)")
            return InspectorResult(verdict=verdict, findings=findings)

        except Exception as e:
            self._log(f"  Codex: failed ({type(e).__name__}: {e}), skipping")
            return None

    async def _run_inspector(
        self,
        open_findings: Optional[list[Finding]] = None,
        round_number: int = 0,
    ) -> InspectorResult:
        """Run Inspector (+ Codex on round 0 if enabled) and merge results."""
        changed_files = self._git_changed_files()

        plan_context = ""
        if self.plan:
            plan_files = ", ".join(self.plan.files_affected)
            wp_summaries = "\n".join(
                f"- {wp.id}: {wp.title} (files: {', '.join(wp.where)})"
                for wp in self.plan.work_packages
            )
            plan_context = (
                f"## Approved plan\n"
                f"Summary: {self.plan.summary[:200]}\n\n"
                f"Work packages:\n{wp_summaries}\n\n"
                f"Files the plan expected to change: {plan_files}\n\n"
            )

        previous_findings_section = ""
        if open_findings:
            pf_lines = "\n".join(
                f'- id: "{f.finding_id}" | [{f.file}] {f.description}'
                + (f"\n  Previous suggestion: {f.suggested_fix}" if f.suggested_fix else "")
                for f in open_findings
            )
            previous_findings_section = (
                f"## Previously open findings (MANDATORY RE-CHECK)\n"
                f"You MUST re-examine each one — do NOT take the Coder's word.\n"
                f"List confirmed-resolved IDs in `resolved_ids`.\n\n"
                f"{pf_lines}\n\n"
            )

        next_round = round_number + 1
        task = (
            f"Review all uncommitted changes in the working tree.\n\n"
            f"{plan_context}"
            f"{previous_findings_section}"
            f"Files actually changed:\n"
            + "\n".join(f"- {f}" for f in changed_files) + "\n\n"
            f"## Required output format\n"
            f"```json\n"
            f'{{\n'
            f'  "verdict": "PASS" or "ISSUES",\n'
            f'  "resolved_ids": ["INS-0-001", ...],\n'
            f'  "findings": [\n'
            f'    {{\n'
            f'      "id": "INS-{next_round}-001",\n'
            f'      "severity": "trivial" or "implementation" or "design",\n'
            f'      "file": "path/to/file.py",\n'
            f'      "description": "what is wrong",\n'
            f'      "suggested_fix": "how to fix it"\n'
            f'    }}\n'
            f'  ]\n'
            f'}}\n'
            f"```\n"
        )
        # Dynamic max_turns: round 0 scales with changed file count
        # (~8 turns per file for read + trace callers + check imports +
        # verify spec, plus 15 for smoke test + import checks), clamped
        # [75, 200]. Revision rounds use a fixed 120.
        if round_number > 0:
            max_turns = 120
        else:
            n_files = len(changed_files)
            max_turns = max(75, min(n_files * 8 + 15, 200))

        if round_number == 0 and CODEX_ENABLED:
            # Round 0 with Codex: run both in parallel via asyncio.gather.
            # Claude failure is fatal; Codex failure is soft.
            async def _claude_inspector():
                r, _ = await self._run_agent("inspector", task, max_turns_override=max_turns)
                return self._parse_inspector_result(r, round_number=round_number)

            plan_ctx = plan_context if plan_context else ""
            claude_task = asyncio.ensure_future(_claude_inspector())
            codex_task = asyncio.ensure_future(
                self._run_codex_review(changed_files, plan_ctx)
            )
            try:
                results = await asyncio.gather(
                    claude_task, codex_task, return_exceptions=True,
                )
            except BaseException:
                # If gather itself is cancelled, clean up both tasks
                claude_task.cancel()
                codex_task.cancel()
                raise

            claude_result = results[0]
            codex_result = results[1]
            if isinstance(claude_result, BaseException):
                # Claude failed — cancel orphaned Codex task and propagate.
                # Narrow against BaseException (not Exception) so pyright
                # can rule out the full exception branch from asyncio.gather's
                # `T | BaseException` union on the remaining control flow.
                codex_task.cancel()
                raise claude_result
            if isinstance(codex_result, BaseException):
                self._log(f"  Codex: failed ({type(codex_result).__name__}), proceeding Claude-only")
                codex_result = None
            return merge_inspector_results(claude_result, codex_result)

        # Round 1+ or no Codex: Claude only
        result_text, _ = await self._run_agent(
            "inspector", task,
            max_turns_override=max_turns,
        )
        return self._parse_inspector_result(result_text, round_number=round_number)

    def _parse_inspector_result(self, result_text: str, round_number: int = 0) -> InspectorResult:
        """Parse Inspector result from JSON or text."""
        json_str = self._extract_json_block(result_text)
        if json_str:
            try:
                data = json.loads(json_str)
                verdict_str = data.get("verdict", "").upper()
                verdict = InspectorVerdict.PASS if verdict_str == "PASS" else InspectorVerdict.ISSUES

                findings = []
                for n, f in enumerate(data.get("findings", [])):
                    severity_str = f.get("severity", "implementation").lower()
                    try:
                        severity = EscalationLevel(severity_str)
                    except ValueError:
                        severity = EscalationLevel.IMPLEMENTATION
                    finding_id = f.get("id", f"INS-{round_number}-{n + 1:03d}")
                    findings.append(Finding(
                        severity=severity,
                        file=f.get("file", "unknown"),
                        description=f.get("description", ""),
                        suggested_fix=f.get("suggested_fix"),
                        finding_id=finding_id,
                    ))

                return InspectorResult(
                    verdict=verdict,
                    findings=findings,
                    resolved_ids=data.get("resolved_ids", []),
                    summary=result_text,
                )
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

        # Fallback: text-based
        verdict = InspectorVerdict.ISSUES
        for line in result_text.splitlines():
            stripped = line.strip().upper()
            if stripped in ("PASS", "**PASS**", "## PASS", "VERDICT: PASS"):
                verdict = InspectorVerdict.PASS
                break

        return InspectorResult(verdict=verdict, summary=result_text)

    # ── Escalation chain of command ──────────────────────────────────────

    async def _run_architect_triage(self, findings: list[Finding]) -> TriageResult:
        """Architect triages DESIGN findings: coder_fix / override / escalate."""
        assert self.plan is not None
        self._log("  Architect triaging design findings")

        wp_summaries = "\n".join(
            f"- {wp.id}: {wp.title} — {wp.what}"
            for wp in self.plan.work_packages
        )
        findings_text = "\n".join(
            f'- id: "{f.finding_id}" | [{f.file}] {f.description}'
            + (f"\n  Inspector suggested: {f.suggested_fix}" if f.suggested_fix else "")
            for f in findings
        )

        task = (
            f"You are triaging Inspector findings classified as DESIGN level.\n"
            f"You wrote the plan for this build, so you know the design intent.\n\n"
            f"Review each finding and decide:\n\n"
            f"- **coder_fix**: Actually an implementation issue.\n"
            f"- **override**: Inspector is wrong or this was intentional.\n"
            f"- **escalate**: Genuine architectural issue needing owner decision.\n\n"
            f"## Approved plan\n{self.plan.summary}\n\n"
            f"Work packages:\n{wp_summaries}\n\n"
            f"## Findings to triage\n{findings_text}\n\n"
            f"## Required output\n"
            f"```json\n"
            f'{{\n'
            f'  "verdicts": [\n'
            f'    {{"finding_id": "...", "verdict": "coder_fix|override|escalate", '
            f'"rationale": "...", "coder_instructions": "..."}}\n'
            f"  ]\n"
            f"}}\n"
            f"```\n"
        )
        result_text, _ = await self._run_agent(
            "architect", task,
            model_override="claude-sonnet-4-6",
        )
        return self._parse_triage_result(result_text)

    def _parse_triage_result(self, result_text: str) -> TriageResult:
        """Parse Architect triage JSON into TriageResult."""
        json_str = self._extract_json_block(result_text)
        if json_str:
            try:
                data = json.loads(json_str)
                entries = []
                for v in data.get("verdicts", []):
                    try:
                        verdict = TriageVerdict(v.get("verdict", "escalate"))
                    except ValueError:
                        verdict = TriageVerdict.ESCALATE
                    entries.append(TriageEntry(
                        finding_id=v.get("finding_id", ""),
                        verdict=verdict,
                        rationale=v.get("rationale", ""),
                        coder_instructions=v.get("coder_instructions", ""),
                    ))
                return TriageResult(entries=entries)
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

        self._log("  WARNING: Could not parse triage JSON — escalating all")
        return TriageResult()

    # ── Phase 3: Dreaming ────────────────────────────────────────────────

    async def _run_phase_3(self) -> DreamerResult:
        """Run Dreamer for memory consolidation, or skip if nothing to consolidate."""
        agents = sorted(set(self._agents_that_ran))
        substantial = self._has_substantial_short_term_memories()

        if not substantial:
            self._log(
                f"Skipping Dreamer — short-term content below "
                f"{SHORT_TERM_CONSOLIDATION_THRESHOLD} char threshold."
            )
            write_state(self.project_root, "dreamer", status="skipped")
            if self._has_uncommitted_changes():
                self._git_commit_safe("chore: update agent state after build")
            return DreamerResult(
                entries_processed=0, promoted=0, corrected=0,
                confirmed=0, discarded=0, summary="(skipped)",
            )

        agents_list = ", ".join(agents)
        task = (
            f"Consolidate all agents' short-term memories into long-term.\n\n"
            f"## Agents that participated\n{agents_list}\n\n"
            f"For each short-term entry:\n"
            f"- PROMOTE: new reusable pattern → add to long-term\n"
            f"- CORRECT: contradicts long-term → update\n"
            f"- CONFIRM: already known → discard\n"
            f"- DISCARD: session-specific → drop\n\n"
            f"## Participation gap check\n"
            f"Flag agents that participated but wrote nothing to short-term.\n\n"
            f"Clear short-term memories when done.\n"
            f"Commit memory changes.\n"
            f"Ensure working tree is clean."
        )
        write_state(self.project_root, "dreamer", status="consolidating")
        await self._run_skill("dreamer", task, max_turns=75)
        write_state(self.project_root, "dreamer", status="consolidated")

        if self._has_uncommitted_changes():
            self._git_commit_safe("chore: update agent state after build")

        return DreamerResult(
            entries_processed=0, promoted=0, corrected=0,
            confirmed=0, discarded=0, summary="(see Dreamer output)",
        )

    def _has_substantial_short_term_memories(self) -> bool:
        """Return True if accumulated short-term content warrants consolidation."""
        mem_dir = Path(self.project_root) / SERENA_MEMORIES_DIR
        total_chars = 0
        for agent_cfg in AGENT_MEMORIES.values():
            mem_name = agent_cfg.get("writes")
            if not mem_name:
                continue
            mem_path = mem_dir / f"{mem_name}.md"
            if not mem_path.exists():
                continue
            raw = mem_path.read_text(encoding="utf-8")
            real_lines = [
                line for line in raw.splitlines()
                if line.strip()
                and not line.startswith("#")
                and "(empty" not in line
                and "(last consolidated" not in line
            ]
            total_chars += sum(len(l) for l in real_lines)
        return total_chars >= SHORT_TERM_CONSOLIDATION_THRESHOLD

    # ── Skill runner ─────────────────────────────────────────────────────

    async def _run_skill(
        self, name: str, task_prompt: str,
        model: str = "claude-sonnet-4-6", max_turns: int = 5,
    ) -> str:
        """Run a constrained skill (few turns, focused prompt, shared Serena)."""
        self._agent_count += 1
        self._agents_that_ran.append(name)
        skill_id = f"skill:{name}-{self._agent_count}"
        self._log(f"[{skill_id}] running ({model}, max {max_turns} turns)")

        parts: list[str] = []
        crew = load_crew_prompt(name, self.project_root)
        if crew:
            parts.append(crew)
        sections = get_sections_for_agent(name)
        if sections:
            parts.append("# Project Instructions\n" + sections)
        memory_names = get_memory_names_for_agent(name)
        if memory_names:
            memories_text = await load_memories_text(memory_names, self.project_root)
            parts.append("# Memories\n" + memories_text)
        system_prompt = "\n\n---\n\n".join(parts)

        tool_config = SKILL_TOOLS.get(name, {"serena": [], "fallback": []})
        mcp_servers = {}
        disallowed: list[str] = [
            "ToolSearch", "ExitPlanMode", "EnterPlanMode", "AskUserQuestion",
        ]
        _SERENA_REPLACEABLE = {"Read", "Glob", "Grep"}

        if self.use_serena and self._serena is not None and tool_config["serena"]:
            mcp_servers["serena"] = self._serena.get_mcp_config()
            allowed = list(tool_config["serena"]) + list(tool_config["fallback"])
            disallowed += [t for t in tool_config["fallback"] if t in _SERENA_REPLACEABLE]
        else:
            allowed = list(tool_config["fallback"])

        disallowed_set = set(disallowed)
        allowed = [t for t in allowed if t not in disallowed_set]

        options = ClaudeAgentOptions(
            model=model,
            system_prompt=system_prompt,
            allowed_tools=allowed,
            disallowed_tools=disallowed,
            permission_mode="bypassPermissions",
            max_turns=max_turns,
            mcp_servers=mcp_servers,
            setting_sources=["user"],
            hooks=_build_sdk_hooks() if self.use_serena else None,
            cwd=self.project_root,
            env=_build_env(),
        )

        result_text = ""
        session_id = None
        async for message in query(prompt=task_prompt, options=options):
            result_text, session_id = self._handle_message(
                skill_id, message, result_text, session_id,
            )

        if not result_text.strip():
            self._log(f"  WARNING: [{skill_id}] returned empty output (possible dead agent)")

        return result_text

    # ── Agent runner ─────────────────────────────────────────────────────

    async def _iter_query_with_timeout(self, async_iter, agent_id):
        """Wrap an async iterable with a per-message timeout.

        Converts transport wedges into catchable TimeoutErrors.
        """
        iterator = async_iter.__aiter__()
        while True:
            try:
                if INTER_MESSAGE_TIMEOUT_SECONDS is None:
                    message = await iterator.__anext__()
                else:
                    message = await asyncio.wait_for(
                        iterator.__anext__(),
                        timeout=INTER_MESSAGE_TIMEOUT_SECONDS,
                    )
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError:
                self._log(
                    f"  [{agent_id}] no message for "
                    f"{INTER_MESSAGE_TIMEOUT_SECONDS}s — treating as "
                    f"transport wedge"
                )
                raise
            yield message

    async def _run_prof_review(self) -> ProfReviewResult:
        """Run the Professor's post-build inference review.

        The Professor runs the domain tests written by the Test Developer,
        inspects any diagnostic plots, and delivers a verdict (PASS / CONCERN /
        FAIL); a FAIL blocks the commit. Uses the prof_review crew prompt and
        runs with bypassPermissions (from AGENT_PERMISSION_MODES) so it can
        execute the tests.
        """
        assert self.plan is not None
        specs = "\n".join(f"- {d}" for d in self.plan.domain_test_descriptions)
        task = (
            "You are reviewing the domain correctness of the recent build.\n\n"
            "## Test specifications\n"
            f"{specs or '(no specific test specs provided)'}\n\n"
            "## What to do\n"
            "1. Run the project's fast domain tests (e.g. "
            "`python -m pytest cogwheel/tests/ -v`). Run ONLY fast tests — "
            "never a long sampling / real-data run.\n"
            "2. Inspect any diagnostic plots the tests produce.\n"
            "3. Verify results match first-principles / stated-tolerance "
            "expectations.\n\n"
            "## Verdict\n"
            "Report your verdict as a JSON block:\n"
            "```json\n"
            '{\n'
            '  "verdict": "PASS" or "CONCERN" or "FAIL",\n'
            '  "concerns": ["list of concerns if any"],\n'
            '  "summary": "brief explanation"\n'
            '}\n'
            "```\n"
        )
        result_text, _ = await self._run_agent("prof_review", task)
        return self._parse_prof_review_result(result_text)

    def _parse_prof_review_result(self, result_text: str) -> ProfReviewResult:
        """Parse the inference-review verdict (JSON first, then text fallback)."""
        json_str = self._extract_json_block(result_text)
        if json_str:
            try:
                data = json.loads(json_str)
                verdict_str = data.get("verdict", "").upper()
                if verdict_str == "FAIL":
                    verdict = ProfReviewVerdict.FAIL
                elif verdict_str == "CONCERN":
                    verdict = ProfReviewVerdict.CONCERN
                else:
                    verdict = ProfReviewVerdict.PASS
                return ProfReviewResult(
                    verdict=verdict,
                    concerns=data.get("concerns", []),
                    summary=data.get("summary", result_text),
                )
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
        # Fallback: text match, default PASS (fail-open — the Inspector gate is
        # the hard correctness gate; this is an advisory domain check).
        verdict = ProfReviewVerdict.PASS
        for line in result_text.splitlines():
            s = line.strip().upper()
            if s in ("FAIL", "**FAIL**", "VERDICT: FAIL"):
                verdict = ProfReviewVerdict.FAIL
                break
            if s in ("CONCERN", "**CONCERN**", "VERDICT: CONCERN"):
                verdict = ProfReviewVerdict.CONCERN
                break
        return ProfReviewResult(verdict=verdict, summary=result_text)

    async def _run_agent(
        self,
        agent_name: str,
        task_context: str,
        model_override: Optional[str] = None,
        resume_session: Optional[str] = None,
        permission_override: Optional[PermissionMode] = None,
        max_turns_override: Optional[int] = None,
    ) -> tuple[str, Optional[str]]:
        """Create and run a single agent, streaming output per verbosity."""
        self._agent_count += 1
        self._agents_that_ran.append(agent_name)
        agent_id = f"{agent_name}-{self._agent_count}"
        verb = "resuming" if resume_session else "spawning"

        if self.verbosity != Verbosity.QUIET:
            model = model_override or AGENT_MODELS[agent_name]
            self._log(f"[{agent_id}] {verb} ({model})")

        mcp_config = (
            self._serena.get_mcp_config()
            if self.use_serena and self._serena is not None
            else None
        )
        _SPEC_AGENTS = {"coder", "inspector"}
        extra = self._specs_text if agent_name in _SPEC_AGENTS else ""

        _build_opts_kwargs: dict = dict(
            agent_name=agent_name,
            project_root=self.project_root,
            task_context=task_context,
            extra_instructions=extra,
            use_serena=self.use_serena,
            mcp_config=mcp_config,
            model_override=model_override,
            permission_override=permission_override,
        )
        if max_turns_override is not None:
            _build_opts_kwargs["max_turns"] = max_turns_override
            self._log(f"[{agent_id}] max_turns={max_turns_override}")
        options = await build_agent_options(**_build_opts_kwargs)

        if resume_session:
            options.resume = resume_session

        result_text = ""
        session_id = None
        try:
            async for message in self._iter_query_with_timeout(
                    query(prompt=task_context, options=options), agent_id):
                result_text, session_id = self._handle_message(
                    agent_id, message, result_text, session_id,
                )
        except RuntimeError:
            raise
        except Exception as e:
            if self.use_serena:
                self._log(f"[{agent_id}] MCP failed ({type(e).__name__}: {e}), retrying with built-in tools")
                _retry_kwargs: dict = dict(
                    agent_name=agent_name,
                    project_root=self.project_root,
                    task_context=task_context,
                    extra_instructions=extra,
                    use_serena=False,
                    mcp_config=None,
                    model_override=model_override,
                    permission_override=permission_override,
                )
                if max_turns_override is not None:
                    _retry_kwargs["max_turns"] = max_turns_override
                options = await build_agent_options(**_retry_kwargs)
                if resume_session:
                    options.resume = resume_session
                async for message in self._iter_query_with_timeout(
                        query(prompt=task_context, options=options), agent_id):
                    result_text, session_id = self._handle_message(
                        agent_id, message, result_text, session_id,
                    )
            else:
                raise

        if not result_text.strip():
            self._log(f"  WARNING: [{agent_id}] returned empty output (possible dead agent)")

        return result_text, session_id

    @staticmethod
    def _fmt_tool(block: ToolUseBlock) -> str:
        """Format a tool call for logging."""
        name = block.name
        inp = block.input or {}

        for key in ("file_path", "relative_path", "path", "filename"):
            if key in inp:
                val = str(inp[key])
                for prefix in ("./",):
                    val = val.removeprefix(prefix)
                return f"{name}: {val}"

        for key in ("pattern", "query", "regex", "name_path"):
            if key in inp:
                return f"{name}: {inp[key]!r}"

        if "description" in inp:
            return f"{name}: {inp['description']}"

        if "command" in inp:
            return f"{name}: {inp['command'][:80]}"

        return name

    def _handle_message(
        self, agent_id: str, message, result_text: str, session_id: Optional[str],
    ) -> tuple[str, Optional[str]]:
        """Process a single streaming message from an agent."""
        if isinstance(message, AssistantMessage):
            if self.verbosity == Verbosity.VERBOSE:
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"  [{agent_id}] {block.text[:200]}")
                    elif isinstance(block, ToolUseBlock):
                        print(f"  [{agent_id}] tool: {self._fmt_tool(block)}")
            elif self.verbosity == Verbosity.NORMAL:
                for block in message.content:
                    if isinstance(block, ToolUseBlock):
                        print(f"  [{agent_id}] {self._fmt_tool(block)}")

        elif isinstance(message, SystemMessage):
            if self.verbosity == Verbosity.VERBOSE:
                print(f"  [{agent_id}] system: {message.subtype}")

        elif isinstance(message, ResultMessage):
            if message.total_cost_usd:
                self._total_cost += message.total_cost_usd
            if hasattr(message, "session_id"):
                session_id = message.session_id
            if message.subtype == "success":
                result_text = message.result or ""
                if self.verbosity != Verbosity.QUIET:
                    cost = f"${message.total_cost_usd:.4f}" if message.total_cost_usd else "?"
                    self._log(f"[{agent_id}] done ({cost})")
            else:
                result_text = message.result or result_text
                self._log(
                    f"[{agent_id}] FAILED: {message.subtype} "
                    f"(partial output: {len(result_text)} chars)"
                )
                raise RuntimeError(
                    f"Agent {agent_id} ended with status '{message.subtype}'."
                )

        return result_text, session_id

    # ── Helpers ──────────────────────────────────────────────────────────

    def _extract_json_block(self, text: str) -> Optional[str]:
        """Extract the first JSON block from text."""
        json_str = ""
        in_json = False
        for line in text.splitlines():
            if line.strip().startswith("```json"):
                in_json = True
                continue
            if line.strip() == "```" and in_json:
                break
            if in_json:
                json_str += line + "\n"

        if json_str.strip():
            return json_str.strip()

        brace_start = text.find("{")
        if brace_start >= 0:
            depth = 0
            for i in range(brace_start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[brace_start:i + 1]
                        try:
                            json.loads(candidate)
                            return candidate
                        except json.JSONDecodeError:
                            break
        return None

    @staticmethod
    def _as_str_list(value) -> list[str]:
        """Coerce a plan field to list[str].

        The schema declares ``where`` (and similar) as ``list[str]``, but the
        Architect occasionally emits a bare string. Passing that through makes
        ``", ".join(where)`` iterate characters (the plan then renders
        ``c, o, g, w, ...``). Normalize here so the renderer and every
        downstream consumer of ``wp.where`` get a real list.
        """
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value else []
        return [str(v) for v in value]

    def _parse_plan_from_dict(self, data: dict) -> Plan:
        """Convert a plan dict to a Plan."""
        work_packages = [
            WorkPackage(
                id=wp["id"],
                title=wp["title"],
                what=wp["what"],
                where=self._as_str_list(wp.get("where", [])),
                how=wp["how"],
                who=wp["who"],
                dependencies=wp.get("depends_on", wp.get("dependencies", [])),
                verification=wp.get("verification", ""),
                max_turns=wp.get("max_turns"),
            )
            for wp in data.get("work_packages", [])
        ]
        return Plan(
            summary=data.get("summary", ""),
            work_packages=work_packages,
            has_domain_tests=data.get("has_domain_tests", False),
            has_new_public_api=data.get("has_new_public_api", False),
            has_spec_update=data.get("has_spec_update", False),
            files_affected=data.get("files_affected", []),
            domain_test_descriptions=data.get("domain_test_descriptions", []),
            simplifier_inputs=data.get("simplifier_inputs", []),
        )

    def _try_parse_json_plan(self, text: str) -> Plan | None:
        """Try to extract a JSON plan from text."""
        try:
            data = json.loads(text)
            if "work_packages" in data:
                return self._parse_plan_from_dict(data)
        except (json.JSONDecodeError, ValueError):
            pass

        json_str = self._extract_json_block(text)
        if json_str:
            try:
                data = json.loads(json_str)
                if "work_packages" in data:
                    return self._parse_plan_from_dict(data)
            except json.JSONDecodeError:
                pass
        return None

    def _parse_plan(self, architect_output: str, fallback_text: str = "") -> Plan:
        """Extract the JSON plan from the Architect's output."""
        plan = self._try_parse_json_plan(architect_output)
        if plan:
            return plan

        if fallback_text and fallback_text != architect_output:
            plan = self._try_parse_json_plan(fallback_text)
            if plan:
                self._log("  (plan JSON found in earlier turn)")
                return plan

        raise GateFailure(
            f"Could not find JSON plan in Architect's output.\n"
            f"Raw output:\n{architect_output[:500]}"
        )

    def _format_plan(self, plan: Plan) -> str:
        """Format a Plan for display."""
        lines = [f"## {plan.summary}\n"]
        for wp in plan.work_packages:
            lines.append(f"### {wp.id}: {wp.title}")
            lines.append(f"  What: {wp.what}")
            lines.append(f"  Where: {', '.join(wp.where)}")
            lines.append(f"  How: {wp.how}")
            lines.append(f"  Who: {wp.who}")
            if wp.dependencies:
                lines.append(f"  Dependencies: {', '.join(wp.dependencies)}")
            lines.append(f"  Verification: {wp.verification}")
            if wp.max_turns:
                lines.append(f"  Max turns: {wp.max_turns}")
            lines.append("")

        if plan.domain_test_descriptions:
            lines.append("### Domain Test Descriptions")
            for desc in plan.domain_test_descriptions:
                lines.append(f"  - {desc}")
            lines.append("")

        lines.append(f"Domain tests: {'yes' if plan.has_domain_tests else 'no'}")
        lines.append(f"New public API: {'yes' if plan.has_new_public_api else 'no'}")
        lines.append(f"Spec update: {'yes' if plan.has_spec_update else 'no'}")
        lines.append(f"Files affected: {len(plan.files_affected)}")
        return "\n".join(lines)

    def _collect_change_report(self, agent_output: str) -> None:
        """Extract a change-report block from agent output."""
        report = self._parse_change_report(agent_output)
        if report:
            self._change_reports.append(report)

    @staticmethod
    def _parse_change_report(text: str) -> Optional[dict]:
        """Parse a change-report block."""
        in_block = False
        report: dict[str, str] = {}
        for line in text.splitlines():
            stripped = line.strip()
            if stripped == "```change-report":
                in_block = True
                continue
            if stripped == "```" and in_block:
                break
            if in_block:
                if stripped.upper().startswith("SUMMARY:"):
                    report["summary"] = stripped[len("SUMMARY:"):].strip()
                elif stripped.upper().startswith("FILES:"):
                    report["files"] = stripped[len("FILES:"):].strip()
                elif stripped.upper().startswith("PREFIX:"):
                    report["prefix"] = stripped[len("PREFIX:"):].strip().lower()

        if report.get("summary"):
            return report
        return None

    def _build_commit_message(self) -> str:
        """Build a commit message from agent change reports."""
        if self._change_reports:
            prefix = self._change_reports[0].get("prefix", "feat")
            subject_summary = self._change_reports[0]["summary"]
            max_subject = 72 - len(prefix) - 2
            if len(subject_summary) > max_subject:
                subject_summary = subject_summary[:max_subject - 3] + "..."
            subject = f"{prefix}: {subject_summary}"

            body_lines = []
            for i, report in enumerate(self._change_reports):
                summary = report["summary"]
                files = report.get("files", "")
                if i == 0:
                    if files:
                        body_lines.append(f"  Files: {files}")
                else:
                    body_lines.append(f"- {summary}")
                    if files:
                        body_lines.append(f"  Files: {files}")

            return subject + "\n\n" + "\n".join(body_lines) if body_lines else subject

        elif self.plan:
            summary = self.plan.summary.strip()
            max_subject = 72 - len("feat: ")
            if len(summary) > max_subject:
                summary = summary[:max_subject - 3] + "..."
            subject = f"feat: {summary}"
            body_lines = [f"- {wp.id}: {wp.title}" for wp in self.plan.work_packages]
            return subject + "\n\n" + "\n".join(body_lines)

        else:
            task_summary = self.task.strip()
            if len(task_summary) > 66:
                task_summary = task_summary[:63] + "..."
            return f"feat: {task_summary}"

    def _git_changed_files(self) -> list[str]:
        """Get files with uncommitted changes."""
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            capture_output=True, text=True, cwd=self.project_root,
        )
        files = set(result.stdout.strip().splitlines()) if result.returncode == 0 else set()

        result2 = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            capture_output=True, text=True, cwd=self.project_root,
        )
        if result2.returncode == 0:
            files.update(result2.stdout.strip().splitlines())

        return sorted(f for f in files if f)

    def _git_diff_cached(self) -> str:
        """Get current diff for context."""
        result = subprocess.run(
            ["git", "diff", "HEAD"],
            capture_output=True, text=True, cwd=self.project_root,
        )
        return result.stdout if result.returncode == 0 else ""

    def _has_uncommitted_changes(self) -> bool:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, cwd=self.project_root,
        )
        return bool(result.stdout.strip())

    def _git_commit_safe(self, message: str) -> Optional[str]:
        """Create a git commit, staging only tracked + safe new files."""
        if not self._has_uncommitted_changes():
            return None

        subprocess.run(["git", "add", "-u"], cwd=self.project_root, check=True)

        safe_dirs = [
            "cogwheel/", "docs/", "scripts/", "changelog.d/",
            ".claude/spec/", ".claude/agent_state/", ".serena/memories/",
        ]
        new_files = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            capture_output=True, text=True, cwd=self.project_root,
        )
        if new_files.returncode == 0 and new_files.stdout.strip():
            for f in new_files.stdout.strip().splitlines():
                if any(f.startswith(d) for d in safe_dirs):
                    subprocess.run(["git", "add", f], cwd=self.project_root, check=True)

        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=self.project_root,
        )
        if staged.returncode == 0:
            return None

        # Spec/doc discipline preflight: run the pre-commit hook NOW and
        # auto-remediate missing changelog fragments before the real commit,
        # so a completed build doesn't die opaquely at `git commit`.
        self._ensure_spec_doc_fragments(message)

        full_message = message + "\n\nCo-Authored-By: Claude <noreply@anthropic.com>"
        subprocess.run(
            ["git", "commit", "-m", full_message],
            cwd=self.project_root, check=True,
        )

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=self.project_root,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    def _ensure_spec_doc_fragments(self, message: str) -> None:
        """Preflight the spec/doc discipline hook; auto-stub missing
        changelog fragments (see sdk/commit_preflight.py)."""
        from .commit_preflight import ensure_spec_doc_fragments
        ensure_spec_doc_fragments(self.project_root, message, log=self._log)

    def _append_cost_ledger(self) -> None:
        """Append build cost entry to .claude/sdk/build_costs.jsonl."""
        from datetime import datetime, timezone
        ledger_path = Path(self.project_root) / ".claude" / "sdk" / "build_costs.jsonl"
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task": self.task[:120],
            "mode": self.build_report.mode.value if self.build_report else "unknown",
            "cost_usd": round(self._total_cost, 6),
            "agents_spawned": self._agent_count,
            "agents": self._agents_that_ran,
            "revision_loops": getattr(self.build_report, "revision_loops", 0),
        }
        try:
            with open(ledger_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            self._log("  WARNING: Could not write to build_costs.jsonl")

    def _empty_report(self) -> BuildReport:
        return BuildReport(mode=BuildMode.FULL, work_packages_completed=0, work_packages_total=0)

    def _log(self, msg: str):
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)

    def _log_phase(self, msg: str):
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{ts}] {'─' * 56}", flush=True)
        print(f"[{ts}]   {msg}", flush=True)
        print(f"[{ts}] {'─' * 56}", flush=True)
