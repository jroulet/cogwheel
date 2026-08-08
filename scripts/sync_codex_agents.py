"""Sync .codex/agents/*.toml model fields from the env-selected provider.

Single source of truth: CODEX_ROLE_MODELS in runtime_codex.py (read from
OPENCODE_MODEL_PROVIDER via the shared selection in runtime_opencode.py
for opencode, or CODEX_MODEL overrides). This script writes the `model`
field into each Codex agent TOML so interactive subagents use the same
tiered models as SDK builds — no manual model edits per provider.

Run automatically by launch_build.sh when AGENT_PROVIDER=codex, or
manually after changing the model routing.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = REPO_ROOT / ".codex" / "agents"

_ROLE_TO_AGENT_FILE = {
    "architect":    "architect.toml",
    "coder":        "coder.toml",
    "inspector":    "inspector.toml",
    "professor":    "professor.toml",
    "prof_review":  "prof_review.toml",
    "foreman_lite": "foreman_lite.toml",
    "test_dev":     "test_dev.toml",
    "librarian":    "librarian.toml",
    "tidier":       "tidier.toml",
    "dreamer":      "dreamer.toml",
    "simplifier":   "simplifier.toml",
}


def main():
    sys.path.insert(0, str(REPO_ROOT / ".claude" / "sdk"))
    from runtime_codex import CODEX_ROLE_MODELS

    n = 0
    for role, model in CODEX_ROLE_MODELS.items():
        fname = _ROLE_TO_AGENT_FILE.get(role)
        if not fname:
            print(f"  skip: no agent file for role {role}", file=sys.stderr)
            continue
        path = AGENTS_DIR / fname
        text = path.read_text()

        lines = text.splitlines()
        has_model = any(line.startswith("model = ") for line in lines)
        if has_model:
            lines = [
                f'model = "{model}"' if line.startswith("model = ") else line
                for line in lines
            ]
        else:
            out = []
            for line in lines:
                out.append(line)
                if line.startswith("name = "):
                    out.append(f'model = "{model}"')
            lines = out
        path.write_text("\n".join(lines) + "\n")
        n += 1

    print(f"synced {n} codex agent(s) → model routing in runtime_codex.py",
          file=sys.stderr)


if __name__ == "__main__":
    main()
