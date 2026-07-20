"""CLI entry point for the SDK pipeline orchestrator.

Usage:
    python .claude/sdk/cli.py build "Add feature X to module Y"
    python .claude/sdk/cli.py build --fast "Fix typo in module.py"
    python .claude/sdk/cli.py build --plan-only "Refactor data loading"
    python .claude/sdk/cli.py build -v "Add new processor"
    python .claude/sdk/cli.py build --log build.log "Big refactor"
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

_SDK_DIR = os.path.dirname(os.path.abspath(__file__))
_CLAUDE_DIR = os.path.dirname(_SDK_DIR)
if _CLAUDE_DIR not in sys.path:
    sys.path.insert(0, _CLAUDE_DIR)


def main():
    parser = argparse.ArgumentParser(
        prog="sdk-build",
        description="SDK-based pipeline build orchestrator",
    )
    subparsers = parser.add_subparsers(dest="command")

    build_parser = subparsers.add_parser(
        "build",
        help="Run the full build pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Wedge-handling knobs (env vars / watchdog args):\n"
            "  SDK_INTER_MESSAGE_TIMEOUT_SECONDS\n"
            "      Per-message wedge timeout, in seconds. Default: 300.\n"
            "      Resets on every tool call / streamed message — only fires\n"
            "      on true silence wedges, not on long total task durations.\n"
            "      Set to 0 to disable entirely (use for tasks where the\n"
            "      Coder may legitimately think for >5 min between tool\n"
            "      calls — extremely rare in practice).\n"
            "\n"
            "  .claude/sdk/watchdog.sh <log_path> [stale_seconds]\n"
            "      Optional sidecar that SIGKILLs the orchestrator if its\n"
            "      log mtime hasn't advanced in stale_seconds. Default: 600.\n"
            "      Pass 0 to disable the staleness check while still\n"
            "      cleaning up the orchestrator subtree on natural exit.\n"
        ),
    )
    build_parser.add_argument("task", help="Description of the build task")
    build_parser.add_argument(
        "--fast", action="store_true",
        help="Fast-path: skip Phase 1 planning, Foreman-Lite handles directly",
    )
    build_parser.add_argument(
        "--plan-only", action="store_true",
        help="Run Phase 1 only (plan + approval), then stop",
    )
    build_parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show agent text output in addition to tool calls",
    )
    build_parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Show only phase transitions and final report",
    )
    build_parser.add_argument(
        "--no-serena", action="store_true",
        help="Disable Serena MCP (use built-in tools only)",
    )
    build_parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Auto-approve the Phase 1 plan (skip interactive [y/n/q] prompt). "
             "Required for non-TTY runs (subprocess, agent invocation, CI).",
    )
    build_parser.add_argument(
        "--log", metavar="FILE", default=None,
        help="Write all output to a log file",
    )
    build_parser.add_argument(
        "--project-root", default=None,
        help="Project root (default: git root or cwd)",
    )
    build_parser.add_argument(
        "--approval-dir", metavar="DIR", default=None,
        help="File-based plan approval: write plan to DIR/plan.json, wait for "
             "DIR/plan_approved or DIR/plan_rejected. Used by in-session /build.",
    )
    build_parser.add_argument(
        "--serena-url", metavar="URL", default=None,
        help="Connect to an existing Serena SSE server instead of spawning one. "
             "Used by in-session /build to reuse the session's Serena instance.",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "build":
        _run_build(args)


class _TeeWriter:
    """Write to both a real stream and a shared log file simultaneously."""

    def __init__(self, real_stream, log_file):
        self._real = real_stream
        self._log = log_file

    def write(self, text: str) -> int:
        self._real.write(text)
        self._log.write(text)
        self._log.flush()
        return len(text)

    def flush(self) -> None:
        self._real.flush()
        self._log.flush()


def _run_build(args):
    """Execute the build pipeline."""
    from sdk.orchestrator import BuildOrchestrator, Verbosity

    # Support @file syntax: if task starts with @, inject a short task
    # description that tells the Architect to read the file directly.
    # The full content is NOT inlined into the task string because the
    # Claude API transport HTML-escapes special characters (&, <, >, |)
    # in long system prompts, mangling markdown tables and code blocks.
    task = args.task
    if task.startswith("@"):
        task_path = task[1:]
        if not os.path.isfile(task_path):
            print(f"Error: task file not found: {task_path}", file=sys.stderr)
            sys.exit(1)
        task = (
            f"Read the full task specification from the file: {task_path} — "
            f"Use mcp__serena__read_file or Read to load it, then follow "
            f"its instructions exactly. Do NOT summarize — read the entire file."
        )

    # Sanitize: newlines in the task string crash the claude_agent_sdk's
    # subprocess IPC during triage (the CLI receives a multi-line prompt
    # that breaks argument passing). Replace with spaces.
    task = task.replace("\n", " ").replace("\r", " ")

    project_root = args.project_root or os.getcwd()

    if args.verbose:
        verbosity = Verbosity.VERBOSE
    elif args.quiet:
        verbosity = Verbosity.QUIET
    else:
        verbosity = Verbosity.NORMAL

    log_dir = os.path.join(project_root, ".claude", "sdk", "logs")
    os.makedirs(log_dir, exist_ok=True)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    auto_log = os.path.join(log_dir, f"build_{timestamp}.log")
    latest_link = os.path.join(log_dir, "latest.log")

    log_path = args.log or auto_log
    _log_file = open(log_path, "w", encoding="utf-8")
    tee_out = _TeeWriter(sys.stdout, _log_file)
    tee_err = _TeeWriter(sys.stderr, _log_file)
    sys.stdout = tee_out
    sys.stderr = tee_err

    abs_log_path = os.path.abspath(log_path)
    abs_log_dir = os.path.abspath(log_dir)
    if os.path.dirname(abs_log_path) == abs_log_dir:
        symlink_target = os.path.basename(abs_log_path)
    else:
        symlink_target = abs_log_path

    try:
        if os.path.islink(latest_link) or os.path.exists(latest_link):
            os.remove(latest_link)
        os.symlink(symlink_target, latest_link)
    except OSError:
        pass

    print(f"Log: {log_path}")
    print(f"Monitor: tail -f {latest_link}")
    # The build runs detached, so whoever launched it (a Claude Code agent)
    # must arm an EVENT-DRIVEN Monitor on the log rather than timer-polling.
    # Markers must cover EVERY terminal state, not just milestones: a
    # case-sensitive filter once missed the watchdog's uppercase "KILLED"
    # line and a build died silently. grep -i + phase markers make both
    # progress and death produce events, so long silence is itself a signal.
    _mon_markers = (
        "Inspector: (PASS|found issues)|Professor: (PASS|CONCERN|FAIL)|"
        "GATE FAILURE|Build (failed|complete)|Coder checkpoint|"
        "Phase [0-9]:|Traceback|TimeoutError|transport wedge|"
        "committed|KILLED|watchdog|Exit code|"
        # Decision-wait lines MUST be in the filter: a driver monitor
        # without them let an ESCALATION sit unanswered for 15+ minutes
        # while the watchdog staleness clock ran (2026-07-20, build8c_cont
        # — the driver was only saved by the owner poking). Every state
        # where the pipeline BLOCKS on a human/driver file decision has to
        # emit a monitor event, or an unattended overnight run dies quietly.
        "ESCALATION|escalation|plan_ready|Plan written|"
        "Waiting for a decision"
    )
    print(
        "AGENT: arm an event-driven Monitor now "
        "(do NOT timer-poll the log):\n"
        f"  Monitor(persistent=true, timeout_ms=3600000, "
        f"command='tail -f {log_path} | grep -iE --line-buffered "
        f"\"{_mon_markers}\"')\n"
        "  TaskStop the Monitor when the build ends.\n"
        "  Health check = log mtime advancing, NOT pgrep (the conda\n"
        "  wrapper survives a subtree kill and reads as alive)."
    )

    orchestrator = BuildOrchestrator(
        task=task,
        project_root=project_root,
        verbosity=verbosity,
        use_serena=not args.no_serena,
        fast_path=args.fast,
        dry_run=args.plan_only,
        auto_approve=args.yes,
        approval_dir=args.approval_dir,
        serena_url=args.serena_url,
    )

    try:
        report = asyncio.run(orchestrator.run())
        _print_report(report)
    except KeyboardInterrupt:
        print("\nBuild cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"\nBuild failed: {e}")
        import traceback
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
    finally:
        if _log_file:
            sys.stdout = tee_out._real
            sys.stderr = tee_err._real
            _log_file.close()


def _print_report(report):
    """Print the final build report."""
    print("\n" + "=" * 60)
    print("  BUILD REPORT")
    print("=" * 60)
    print(f"  Mode: {report.mode.value}")
    print(f"  Work packages: {report.work_packages_completed}/{report.work_packages_total}")
    if report.inspector_result:
        print(f"  Inspector: {report.inspector_result.verdict.value}")
    if report.revision_loops:
        print(f"  Revision loops: {report.revision_loops}")
    if report.escalations:
        print(f"  Escalations: {len(report.escalations)}")
    if report.commits:
        print(f"  Commits:")
        for c in report.commits:
            print(f"    {c}")
    print(f"  Total cost: ${report.total_cost:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
