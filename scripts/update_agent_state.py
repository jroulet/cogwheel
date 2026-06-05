#!/usr/bin/env python3
"""Update ``.claude/agent_state/<agent>.json`` after a skill-spawned agent run.

The SDK orchestrator (`.claude/sdk/build.py`) updates agent state natively
when it spawns agents. Skill-invoked agents (via `/inspect`, `/doc-sync`,
`/dream`, `/tidy`) use the Agent tool directly and bypass that machinery —
so their ``last_commit`` / ``last_run`` fields would otherwise never
advance, stranding the next run on a stale tracker that triggers "review
the whole month" diffs.

Usage:
    python scripts/update_agent_state.py <agent_name> [--status PASS|ISSUES|completed|failed]

Examples:
    python scripts/update_agent_state.py inspector --status PASS
    python scripts/update_agent_state.py librarian --status completed
    python scripts/update_agent_state.py dreamer --status completed

Writes:
    {
      "last_commit": "<current HEAD sha>",
      "last_run":    "<UTC ISO 8601 timestamp>",
      "status":      "<status>"
    }

Slash command definitions under `.claude/commands/` invoke this as the
final step after the Agent tool returns, so every agent pair advances
uniformly regardless of whether it was launched by the SDK orchestrator
or by a skill.

Ported from gw_detection_ias scripts/update_agent_state.py (df52c8c5).
KNOWN_AGENTS adapted for teja-force's 10-role crew (no Professor; adds
outside_inspector).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STATE_DIR = REPO / ".claude" / "agent_state"

KNOWN_AGENTS = {
    "architect", "coder", "dreamer", "foreman_lite", "inspector",
    "librarian", "outside_inspector", "simplifier", "test_dev",
    "tidy", "tidier",
}


def git_head(repo: Path) -> str:
    """Return the 40-char commit SHA of HEAD. Raises if not in a git repo."""
    out = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if len(out) != 40:
        raise RuntimeError(f"unexpected git rev-parse output: {out!r}")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "agent_name",
        help=f"Agent identifier; one of {sorted(KNOWN_AGENTS)} (or any name — "
             "the script is permissive, but a typo means a new state file)",
    )
    p.add_argument(
        "--status", default="completed",
        help='Status field (default: "completed"). Inspector may pass '
             '"PASS" or "ISSUES"; other agents typically leave default.',
    )
    p.add_argument(
        "--state-dir", type=Path, default=STATE_DIR,
        help="Override state directory (default: .claude/agent_state/).",
    )
    args = p.parse_args()

    if args.agent_name not in KNOWN_AGENTS:
        print(
            f"warning: {args.agent_name!r} is not in the known-agents set; "
            f"still writing a state file for it.",
            file=sys.stderr,
        )

    args.state_dir.mkdir(parents=True, exist_ok=True)
    state_file = args.state_dir / f"{args.agent_name}.json"

    state = {
        "last_commit": git_head(REPO),
        "last_run": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": args.status,
    }
    state_file.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(
        f"updated {state_file.relative_to(REPO)}: "
        f"last_commit={state['last_commit'][:7]}, status={state['status']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
