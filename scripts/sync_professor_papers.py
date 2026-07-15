#!/usr/bin/env python3
"""
Professor paper read-status utilities.

Read tracking uses marker files in `.serena/memories/professor/read.d/`:
each `<arxiv_id>` file (zero-byte) means the paper has been deeply read.
This format is merge-safe (directory of files; git unions adds across branches).

Two mechanisms create markers (belt + suspenders):
  - `.claude/hooks/professor-auto-mark-read.sh` — PostToolUse hook on
    `mcp__serena__write_memory`. Auto-creates markers when arxiv IDs appear in
    Professor topic memory writes. The authoritative path.
  - `--mark-read <id> [<id> ...]` — explicit CLI for the "Professor read but
    decided nothing novel to add" case (no synthesis write to trigger the hook).

Usage:
  python scripts/sync_professor_papers.py
      List unread papers (REFERENCES.md ∖ read.d/) and update
      professor_knowledge.md Paper Coverage section.
  python scripts/sync_professor_papers.py --mark-read 2501.17939 [2603.05784 ...]
      Create marker file(s).
  python scripts/sync_professor_papers.py --list-unread
      Print unread paper IDs only (one per line, for scripting).
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REFS_MD = REPO_ROOT / "references" / "REFERENCES.md"
READ_DIR = REPO_ROOT / ".serena/memories/professor/read.d"
KNOWLEDGE_PATH = REPO_ROOT / ".serena/memories/professor_knowledge.md"


# ── Parse REFERENCES.md ──────────────────────────────────────────────────

def parse_all_paper_ids():
    """Return the set of arxiv IDs referenced in REFERENCES.md."""
    if not REFS_MD.exists():
        return set()
    text = REFS_MD.read_text(encoding="utf-8")
    # Match table rows that start with `| <arxiv_id> |`
    return set(re.findall(r"^\|\s*(\d{4}\.\d{4,5})\s*\|", text, re.MULTILINE))


# ── Read-status (marker files) ───────────────────────────────────────────

def load_read_ids():
    """Return the set of arxiv IDs with marker files in read.d/."""
    if not READ_DIR.is_dir():
        return set()
    return {f.name for f in READ_DIR.iterdir()
            if f.is_file() and re.match(r"^\d{4}\.\d{4,5}$", f.name)}


def mark_read(arxiv_ids):
    """Touch marker file(s) in read.d/. Idempotent."""
    READ_DIR.mkdir(parents=True, exist_ok=True)
    created = 0
    for aid in arxiv_ids:
        if not re.match(r"^\d{4}\.\d{4,5}$", aid):
            print(f"  skip: {aid!r} not a valid arxiv ID", file=sys.stderr)
            continue
        marker = READ_DIR / aid
        if not marker.exists():
            marker.touch()
            created += 1
    return created


# ── Update professor_knowledge.md ────────────────────────────────────────

def update_knowledge_index(all_ids, read_ids):
    """Update the Paper Coverage section in professor_knowledge.md."""
    if not KNOWLEDGE_PATH.exists():
        return False

    text = KNOWLEDGE_PATH.read_text(encoding="utf-8")
    deeply_read = sorted(read_ids & all_ids)
    unread = sorted(all_ids - read_ids)

    new_section = (
        "## Paper Coverage\n"
        f"- **Deeply read** ({len(deeply_read)}): "
        + (", ".join(deeply_read) if deeply_read else "(none)") + "\n"
        f"- **Unread** ({len(unread)}): "
        + (", ".join(unread) if unread else "(none)") + "\n"
        "- Markers live in `professor/read.d/`; created automatically by\n"
        "  `professor-auto-mark-read.sh` hook on topic-memory writes, or\n"
        "  manually via `python scripts/sync_professor_papers.py --mark-read`.\n"
    )

    pattern = re.compile(r"## Paper Coverage\n.*?(?=\n## |\Z)", re.DOTALL)
    if pattern.search(text):
        new_text = pattern.sub(new_section, text)
    else:
        new_text = text.rstrip() + "\n\n" + new_section

    if new_text != text:
        KNOWLEDGE_PATH.write_text(new_text, encoding="utf-8")
        return True
    return False


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    # --mark-read mode
    if args and args[0] == "--mark-read":
        ids = args[1:]
        if not ids:
            print("Usage: sync_professor_papers.py --mark-read ID [ID ...]",
                  file=sys.stderr)
            sys.exit(1)
        n = mark_read(ids)
        print(f"  Marked {n} new paper(s) as deeply read")
        return

    all_ids = parse_all_paper_ids()
    read_ids = load_read_ids()
    unread = sorted(all_ids - read_ids)

    # --list-unread mode (for scripting)
    if args and args[0] == "--list-unread":
        for aid in unread:
            print(aid)
        return

    # Default: report + update knowledge index
    if not all_ids:
        print("  No papers found in REFERENCES.md")
        return

    print(f"  {len(all_ids)} papers total, {len(read_ids & all_ids)} deeply "
          f"read, {len(unread)} unread")
    if update_knowledge_index(all_ids, read_ids):
        print("  professor_knowledge.md: Paper Coverage updated")
    else:
        print("  professor_knowledge.md: up to date")


if __name__ == "__main__":
    main()
