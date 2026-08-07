"""Sync .opencode/agents/*.md frontmatter models from the env-selected provider.

Single source of truth: OPENCODE_MODEL_PROVIDER in .env (or shell env),
read by runtime_opencode.py.  This script reads the resolved role→model
maps and rewrites the frontmatter so interactive subagents use the right
tiered models without any manual edits.

Run automatically by launch_build.sh when AGENT_PROVIDER=opencode, or
manually after changing OPENCODE_MODEL_PROVIDER.
"""
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = REPO_ROOT / ".opencode" / "agents"

_ROLE_TO_AGENT_FILE = {
    "architect":    "architect.md",
    "coder":        "coder.md",
    "inspector":    "inspector.md",
    "professor":    "professor.md",
    "prof_review":  "prof_review.md",
    "foreman_lite": "foreman_lite.md",
    "test_dev":     "test_dev.md",
    "librarian":    "librarian.md",
    "tidier":       "tidier.md",
    "dreamer":      "dreamer.md",
    "simplifier":   "simplifier.md",
}


def main():
    sys.path.insert(0, str(REPO_ROOT / ".claude" / "sdk"))
    from runtime_opencode import OPENCODE_ROLE_MODELS, OPENCODE_ROLE_VARIANTS

    n = 0
    for role, model in OPENCODE_ROLE_MODELS.items():
        fname = _ROLE_TO_AGENT_FILE.get(role)
        if not fname:
            print(f"  skip: no agent file for role {role}", file=sys.stderr)
            continue
        path = AGENTS_DIR / fname
        text = path.read_text()

        if re.search(r'^model:', text, re.MULTILINE):
            text = re.sub(r'^model:.*$', f'model: {model}', text, flags=re.MULTILINE)
        else:
            text = re.sub(r'^mode: subagent$', f'mode: subagent\nmodel: {model}',
                          text, flags=re.MULTILINE)
        path.write_text(text)
        n += 1

    print(f"synced {n} agent(s) → provider '{os.environ.get('OPENCODE_MODEL_PROVIDER', 'ai-commons')}'",
          file=sys.stderr)


if __name__ == "__main__":
    main()
