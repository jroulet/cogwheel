"""Agent state file management.

Each agent has a JSON state file in `.claude/agent_state/` tracking
`last_commit`, `last_run`, and optional status.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


STATE_DIR = Path(".claude/agent_state")


def _state_path(project_root: str, agent_name: str) -> Path:
    return Path(project_root) / STATE_DIR / f"{agent_name}.json"


def read_state(project_root: str, agent_name: str) -> dict:
    """Read an agent's state file.  Returns empty dict if not found."""
    path = _state_path(project_root, agent_name)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def write_state(
    project_root: str,
    agent_name: str,
    *,
    last_commit: Optional[str] = None,
    status: Optional[str] = None,
    extra: Optional[dict] = None,
    touch_last_run: bool = True,
) -> Path:
    """Write (or update) an agent's state file.

    ``touch_last_run=False`` records a status WITHOUT claiming the agent ran.
    Use it when reporting that a role was skipped: stamping ``last_run`` there
    would redefine the field as "last time we considered running it", which
    silently destroys the staleness signal that tells a driver a role has not
    actually executed in days.
    """
    path = _state_path(project_root, agent_name)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = read_state(project_root, agent_name)

    if last_commit is None:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=project_root,
        )
        last_commit = result.stdout.strip() if result.returncode == 0 else existing.get("last_commit", "")

    state = {
        "last_commit": last_commit,
        "last_run": (datetime.now(timezone.utc).isoformat() if touch_last_run
                     else existing.get("last_run", "")),
        "status": status or existing.get("status", ""),
    }
    if extra:
        state.update(extra)

    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return path


def get_last_commit(project_root: str, agent_name: str) -> Optional[str]:
    """Get the last_commit SHA from an agent's state file, or None."""
    state = read_state(project_root, agent_name)
    return state.get("last_commit") or None


def get_review_range(project_root: str, agent_name: str) -> str:
    """Get a git log range string for scoped reviews."""
    last = get_last_commit(project_root, agent_name)
    if last:
        return f"{last}..HEAD"
    return "--no-walk HEAD"


def get_changed_files_since(project_root: str, agent_name: str) -> list[str]:
    """Get list of files changed since this agent's last run."""
    last = get_last_commit(project_root, agent_name)
    if last:
        cmd = ["git", "diff", "--name-only", last, "HEAD"]
    else:
        cmd = ["git", "diff", "--name-only", "HEAD~1", "HEAD"]

    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=project_root,
    )
    if result.returncode != 0:
        return []

    return [f for f in result.stdout.strip().splitlines() if f]


def collect_state_files(project_root: str, agent_names: list[str]) -> list[str]:
    """Return paths to state files that exist (for git staging)."""
    paths = []
    for name in agent_names:
        rel = str(STATE_DIR / f"{name}.json")
        full = Path(project_root) / rel
        if full.exists():
            paths.append(rel)
    return paths
