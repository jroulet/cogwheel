#!/usr/bin/env python3
"""
Render fragment directories into canonical monolithic files.

Fragment directories (under `.claude/spec/` by convention) let concurrent
agentic work land as separate files instead of conflicting edits to one
monolithic file. `git merge` handles file-level add/delete cleanly — no
content conflicts. This script re-assembles them deterministically.

Default fragment directories and targets (override SURFACES to customize):

  changelog.d/                            -> CHANGELOG.md
  .claude/spec/spec_changelog.d/          -> .claude/spec/SPEC_CHANGELOG.md
  .claude/spec/contracts_changelog.d/     -> .claude/spec/DATA_CONTRACTS_CHANGELOG.md
  .claude/spec/completed.d/               -> .claude/spec/COMPLETED.md
  .claude/spec/todo.d/                    -> .claude/spec/TODO.md

Each fragment is a Markdown file with YAML frontmatter (--- delimited).
Each directory may have a _seed.md providing base metadata (preamble,
base version for versioned changelogs, or the full template for todo).

Usage:
  render_fragments.py                  # render all surfaces
  render_fragments.py --check          # dry-run: report what would change (exit 1 if diff)
  render_fragments.py --surface NAME   # render only one surface (changelog, spec, contracts,
                                       #   completed, todo)

Ported from gw_detection_ias (e2aff8c) with these generalizations:
- .agent/spec/ -> .claude/spec/ (skill convention)
- GW-specific canonical file names (PIPELINE_SPEC, PIPELINE_COMPLETED,
  PIPELINE_TODO) -> generic (SPEC, COMPLETED, TODO)
- COMPLETED section order generalized (no pipeline-specific ordering)
- Header text generalized (no "GWIAS-HM Pipeline" branding)

A surface whose fragment directory doesn't exist is silently skipped,
so skill installers can enable just the subset they need.
"""

import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# If installed under assets/infrastructure/ (skill source tree), walk up one
# more to reach the project root. If installed under scripts/ (target project),
# REPO_ROOT is already correct.
if REPO_ROOT.name == "assets":
    REPO_ROOT = REPO_ROOT.parent

# ── Surface definitions ──────────────────────────────────────────────────

SURFACES = {
    "changelog": {
        "frag_dir": "changelog.d",
        "target": "CHANGELOG.md",
    },
    "spec": {
        "frag_dir": ".claude/spec/spec_changelog.d",
        "target": ".claude/spec/SPEC_CHANGELOG.md",
    },
    "contracts": {
        "frag_dir": ".claude/spec/contracts_changelog.d",
        "target": ".claude/spec/DATA_CONTRACTS_CHANGELOG.md",
    },
    "completed": {
        "frag_dir": ".claude/spec/completed.d",
        "target": ".claude/spec/COMPLETED.md",
    },
    "todo": {
        "frag_dir": ".claude/spec/todo.d",
        "target": ".claude/spec/TODO.md",
    },
}

# Canonical spec/contracts files whose version fields are derived from
# the fragment changelogs. If either file is absent, the writeback is a no-op.
SPEC_FILE = ".claude/spec/SPEC.md"
CONTRACTS_FILE = ".claude/spec/DATA_CONTRACTS.yaml"


# ── Frontmatter parsing ─────────────────────────────────────────────────

def parse_frontmatter(text):
    """Parse YAML-like frontmatter from markdown text.

    Returns (metadata_dict, body_str).  Handles simple ``key: value`` pairs.
    Values wrapped in quotes are unquoted.  Bracketed lists like
    ``["a", "b"]`` are returned as Python lists.
    """
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end < 0:
        return {}, text
    header = text[4:end]
    body = text[end + 4:]          # skip closing ---\n
    if body.startswith("\n"):
        body = body[1:]
    meta = {}
    for line in header.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([\w_]+)\s*:\s*(.*)", line)
        if m:
            key, val = m.group(1), m.group(2).strip()
            # Strip outer quotes
            if (val.startswith('"') and val.endswith('"')) or \
               (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]
            # Handle YAML list syntax: ["a", "b"]
            if val.startswith("[") and val.endswith("]"):
                val = [v.strip().strip('"').strip("'")
                       for v in val[1:-1].split(",") if v.strip()]
            meta[key] = val
    return meta, body


def load_fragments(frag_dir):
    """Load all .md fragments (excluding _seed.md) from *frag_dir*.

    Returns list of ``(meta, body, filename)`` sorted by filename.
    """
    d = REPO_ROOT / frag_dir
    if not d.is_dir():
        return []
    frags = []
    for f in sorted(d.iterdir()):
        if f.suffix == ".md" and f.name != "_seed.md":
            text = f.read_text(encoding="utf-8")
            meta, body = parse_frontmatter(text)
            frags.append((meta, body, f.name))
    return frags


def load_seed(frag_dir):
    """Load ``_seed.md`` from *frag_dir*.  Returns ``(meta, body)``."""
    seed = REPO_ROOT / frag_dir / "_seed.md"
    if not seed.exists():
        return {}, ""
    text = seed.read_text(encoding="utf-8")
    return parse_frontmatter(text)


# ── Version utilities ────────────────────────────────────────────────────

def parse_version(v):
    parts = str(v).strip('"').strip("'").split(".")
    return tuple(int(p) for p in parts)


def fmt_version(t):
    return ".".join(str(x) for x in t)


def bump_version(base, level):
    M, N, P = parse_version(base)
    if level == "major":
        return fmt_version((M + 1, 0, 0))
    if level == "minor":
        return fmt_version((M, N + 1, 0))
    return fmt_version((M, N, P + 1))


# ── CHANGELOG.md ─────────────────────────────────────────────────────────

CHANGELOG_ADVISORY = (
    "> **When something breaks after a `git pull`, look here first.**\n"
    "> Breaking changes are listed in reverse chronological order.\n"
    "> Search for the script or module name that failed to find the relevant entry.\n"
    "\n---\n"
)


def render_changelog(frag_dir):
    fragments = load_fragments(frag_dir)
    if not fragments:
        return None

    # Group by date
    by_date = {}
    for meta, body, fname in fragments:
        date = meta.get("date", "0000-00-00")
        by_date.setdefault(date, []).append((fname, body))

    dates = sorted(by_date, reverse=True)

    parts = [
        "<!-- Generated by scripts/render_fragments.py from changelog.d/. "
        "Do not edit directly. -->\n",
        "# Changelog\n\n",
        CHANGELOG_ADVISORY,
    ]

    for i, date in enumerate(dates):
        entries = sorted(by_date[date], key=lambda x: x[0])  # stable by filename
        bodies = [body.rstrip() for _, body in entries]
        combined = "\n\n---\n\n".join(bodies)
        parts.append(f"\n## {date}\n\n{combined}\n")
        if i < len(dates) - 1:
            parts.append("\n---\n")

    return "".join(parts) + "\n" if not "".join(parts).endswith("\n") else "".join(parts)


# ── Versioned changelogs (spec + contracts) ──────────────────────────────

def _render_versioned_changelog(frag_dir, header):
    """Shared logic for spec and contracts changelogs.

    Returns ``(rendered_text, latest_version, latest_date)`` or
    ``(None, None, None)`` if the fragment directory is empty.
    """
    seed_meta, _ = load_seed(frag_dir)
    fragments = load_fragments(frag_dir)
    if not fragments:
        return None, None, None

    # Separate migrated (explicit version) from new (bump-only)
    migrated = []
    new_frags = []
    for meta, body, fname in fragments:
        if "version" in meta:
            migrated.append((meta, body, fname))
        else:
            new_frags.append((meta, body, fname))

    # Sort new fragments by date ascending, then filename for stability
    new_frags.sort(key=lambda x: (x[0].get("date", ""), x[2]))

    # Derive versions for new fragments from the seed's from_version
    base = seed_meta.get("from_version", "0.0.0")
    derived = []
    current = base
    for meta, body, fname in new_frags:
        current = bump_version(current, meta.get("bump", "patch"))
        derived.append((current, meta.get("date", ""), body))

    # Combine all entries
    all_entries = []
    for meta, body, fname in migrated:
        all_entries.append((meta["version"], meta.get("date", ""), body))
    all_entries.extend(derived)

    # Sort by version descending
    all_entries.sort(key=lambda x: parse_version(x[0]), reverse=True)

    # Build output
    parts = [
        f"<!-- Generated by scripts/render_fragments.py. "
        f"Do not edit directly. -->\n",
        header,
    ]
    for version, date, body in all_entries:
        # Put the heading/body after the bullet separator, rather than after
        # a literal trailing space on the version line.
        parts.append(f"- `{version}` ({date}):\n{body.rstrip()}\n\n")

    result = "".join(parts).rstrip("\n") + "\n"
    latest_version = all_entries[0][0] if all_entries else base
    latest_date = all_entries[0][1] if all_entries else ""
    return result, latest_version, latest_date


SPEC_CHANGELOG_HEADER = (
    "# SPEC Changelog\n\n"
    f"Version history for `{SPEC_FILE}`.\n"
    "Add a new entry by creating a fragment in `spec_changelog.d/`.\n\n"
    "---\n\n"
)


def render_spec_changelog(frag_dir):
    return _render_versioned_changelog(frag_dir, SPEC_CHANGELOG_HEADER)


CONTRACTS_CHANGELOG_HEADER = (
    "# DATA_CONTRACTS Changelog\n\n"
    f"Version history for `{CONTRACTS_FILE}`.\n"
    "Add a new entry by creating a fragment in `contracts_changelog.d/`.\n\n"
    "---\n\n"
)


def render_contracts_changelog(frag_dir):
    return _render_versioned_changelog(frag_dir, CONTRACTS_CHANGELOG_HEADER)


# ── COMPLETED.md ─────────────────────────────────────────────────────────

COMPLETED_HEADER = (
    "# Completed Items\n\n"
    "Archived from `TODO.md`. Items retain their original section "
    "classification and tags.\n\n"
    "---\n"
)


def render_completed(frag_dir):
    fragments = load_fragments(frag_dir)
    if not fragments:
        return None

    # Group by section; subsection is optional
    sections = {}  # section -> [(subsection, date, body, fname)]
    for meta, body, fname in fragments:
        sec = meta.get("section", "Uncategorized")
        subsec = meta.get("subsection", "")
        date = meta.get("date", "0000-00-00")
        sections.setdefault(sec, []).append((subsec, date, body, fname))

    # Section order: read from _seed.md `section_order` list if present,
    # otherwise alphabetical with "Uncategorized" last.
    seed_meta, _ = load_seed(frag_dir)
    explicit_order = seed_meta.get("section_order")
    if isinstance(explicit_order, list):
        known = {s: i for i, s in enumerate(explicit_order)}
        all_secs = sorted(sections.keys(),
                          key=lambda s: (known.get(s, 999), s))
    else:
        all_secs = sorted(sections.keys(),
                          key=lambda s: (s == "Uncategorized", s))

    parts = [
        "<!-- Generated by scripts/render_fragments.py from completed.d/. "
        "Do not edit directly. -->\n",
        COMPLETED_HEADER,
    ]

    for sec in all_secs:
        items = sections[sec]
        parts.append(f"\n## {sec}\n\n")

        # Group by subsection
        by_subsec = {}
        for subsec, date, body, fname in items:
            by_subsec.setdefault(subsec, []).append((date, body, fname))

        # Order subsections: "" (no subsec) first, then alphabetical
        subsecs = sorted(by_subsec.keys(), key=lambda s: (s != "", s))
        for subsec in subsecs:
            if subsec:
                parts.append(f"### {subsec}\n\n")
            entries = sorted(by_subsec[subsec],
                             key=lambda x: x[0], reverse=True)  # newest first
            for _, body, _ in entries:
                parts.append(body.rstrip() + "\n\n")

    return "".join(parts).rstrip("\n") + "\n"


# ── TODO.md ──────────────────────────────────────────────────────────────
# Template-based renderer: the seed (_seed.md) is the full TODO file
# structure with items replaced by <!-- ITEMS --> markers.  The render
# script reads the template, finds markers, and inserts items for the
# current section/subsection.


def render_todo(frag_dir):
    _, template = load_seed(frag_dir)
    fragments = load_fragments(frag_dir)
    if not template:
        return None

    # Build item lookup: (section, subsection) -> [(fname, body)]
    items_by_loc = {}
    for meta, body, fname in fragments:
        key = (meta.get("section", ""), meta.get("subsection", ""))
        items_by_loc.setdefault(key, []).append((fname, body))
    for key in items_by_loc:
        items_by_loc[key].sort(key=lambda x: x[0])

    lines = template.splitlines(keepends=True)
    output = [
        "<!-- Generated by scripts/render_fragments.py from todo.d/. "
        "Do not edit directly. -->\n",
    ]
    current_section = ""
    current_subsection = ""

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("## ") and not stripped.startswith("## Workflow"):
            current_section = stripped[3:].strip()
            current_subsection = ""
        elif stripped.startswith("### "):
            current_subsection = stripped[4:].strip()

        if stripped == "<!-- ITEMS -->":
            key = (current_section, current_subsection)
            if key in items_by_loc:
                for _, body in items_by_loc.pop(key):
                    output.append(body.rstrip() + "\n\n")
            continue

        output.append(line)

    # A fragment whose `section:` matches no template header would
    # otherwise vanish silently from the rendered TODO (it bit three
    # fragments before this warning existed, 2026-07-18).
    for (section, subsection), items in sorted(items_by_loc.items()):
        names = ", ".join(fname for fname, _ in items)
        print(f"  WARNING: todo fragment(s) [{names}] declare "
              f"section={section!r} subsection={subsection!r}, which "
              "matches no header in todo.d/_seed.md — they were NOT "
              "rendered. Use one of the template's '## ' sections "
              "(e.g. 'Backlog', 'In progress').")

    return "".join(output).rstrip("\n") + "\n"


# ── Version writeback ────────────────────────────────────────────────────

def update_spec_version(version, last_date, check_only=False):
    """Write spec_version and last_updated into SPEC.md.

    *last_date* is the date string from the latest changelog fragment,
    so ``last_updated`` tracks content changes, not render runs.
    """
    spec = REPO_ROOT / SPEC_FILE
    if not spec.exists():
        return False
    text = spec.read_text(encoding="utf-8")

    new_text = re.sub(
        r"(spec_version:\s*).+",
        rf"\g<1>{version}",
        text,
    )
    if last_date:
        new_text = re.sub(
            r"(last_updated:\s*).+",
            rf"\g<1>{last_date}",
            new_text,
        )
    if new_text != text and not check_only:
        spec.write_text(new_text, encoding="utf-8")
    return new_text != text


def update_contracts_version(version, last_date, check_only=False):
    """Write schema_version and last_updated into DATA_CONTRACTS.yaml.

    *last_date* is the date string from the latest changelog fragment.
    """
    dc = REPO_ROOT / CONTRACTS_FILE
    if not dc.exists():
        return False
    text = dc.read_text(encoding="utf-8")

    new_text = re.sub(
        r'(schema_version:\s*)"[^"]+"',
        rf'\g<1>"{version}"',
        text,
    )
    if last_date:
        new_text = re.sub(
            r'(last_updated:\s*)"[^"]+"',
            rf'\g<1>"{last_date}"',
            new_text,
        )
    if new_text != text and not check_only:
        dc.write_text(new_text, encoding="utf-8")
    return new_text != text


# ── Main entry point ─────────────────────────────────────────────────────

def write_if_changed(target_path, content, check_only=False):
    """Write *content* to *target_path* only if it differs.

    Returns True if the file was (or would be) changed.
    """
    path = REPO_ROOT / target_path
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing == content:
            return False
    if not check_only:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return True


def check_wiki_links():
    """Return [(source_fragment, missing_target)] for unresolved [[links]].

    A fragment references another by its filename stem, `[[some_slug]]`.  Both
    the open todo.d set and the completed.d archive are valid targets: work
    that shipped keeps its record, and links to it stay meaningful.
    """
    known = set()
    sources = []
    for surface in ("todo", "completed"):
        frag_dir = SURFACES[surface]["frag_dir"]
        if not os.path.isdir(frag_dir):
            continue
        for name in sorted(os.listdir(frag_dir)):
            if not name.endswith(".md"):
                continue
            known.add(name[:-3])
            sources.append((surface, frag_dir, name))

    dangling = []
    for _surface, frag_dir, name in sources:
        with open(os.path.join(frag_dir, name), encoding="utf-8") as handle:
            text = handle.read()
        for target in re.findall(r"\[\[([^\]]+)\]\]", text):
            if target not in known:
                dangling.append((name, target))
    return dangling


def main():
    args = sys.argv[1:]
    check_only = "--check" in args
    if check_only:
        args.remove("--check")

    surface_filter = None
    if "--surface" in args:
        idx = args.index("--surface")
        if idx + 1 < len(args):
            surface_filter = args[idx + 1]
            args = args[:idx] + args[idx + 2:]
        else:
            print("Error: --surface requires a name", file=sys.stderr)
            sys.exit(1)

    changed = False

    # ── Changelog ──
    if surface_filter in (None, "changelog"):
        content = render_changelog(SURFACES["changelog"]["frag_dir"])
        if content is not None:
            if write_if_changed(SURFACES["changelog"]["target"],
                                content, check_only):
                changed = True
                label = "would change" if check_only else "updated"
                print(f"  {SURFACES['changelog']['target']}: {label}")

    # ── Spec changelog ──
    if surface_filter in (None, "spec"):
        content, version, last_date = render_spec_changelog(
            SURFACES["spec"]["frag_dir"])
        if content is not None:
            if write_if_changed(SURFACES["spec"]["target"],
                                content, check_only):
                changed = True
                label = "would change" if check_only else "updated"
                print(f"  {SURFACES['spec']['target']}: {label}")
            if version and update_spec_version(version, last_date,
                                               check_only):
                changed = True
                label = "would change" if check_only else "updated"
                print(f"  {SPEC_FILE} spec_version -> "
                      f"{version}: {label}")

    # ── Contracts changelog ──
    if surface_filter in (None, "contracts"):
        content, version, last_date = render_contracts_changelog(
            SURFACES["contracts"]["frag_dir"])
        if content is not None:
            if write_if_changed(SURFACES["contracts"]["target"],
                                content, check_only):
                changed = True
                label = "would change" if check_only else "updated"
                print(f"  {SURFACES['contracts']['target']}: {label}")
            if version and update_contracts_version(version, last_date,
                                                    check_only):
                changed = True
                label = "would change" if check_only else "updated"
                print(f"  {CONTRACTS_FILE} schema_version -> "
                      f"{version}: {label}")

    # ── Completed ──
    if surface_filter in (None, "completed"):
        content = render_completed(SURFACES["completed"]["frag_dir"])
        if content is not None:
            if write_if_changed(SURFACES["completed"]["target"],
                                content, check_only):
                changed = True
                label = "would change" if check_only else "updated"
                print(f"  {SURFACES['completed']['target']}: {label}")

    # ── Todo ──
    if surface_filter in (None, "todo"):
        content = render_todo(SURFACES["todo"]["frag_dir"])
        if content is not None:
            if write_if_changed(SURFACES["todo"]["target"],
                                content, check_only):
                changed = True
                label = "would change" if check_only else "updated"
                print(f"  {SURFACES['todo']['target']}: {label}")

    # ── Cross-reference integrity ──
    # Fragments link each other with [[stem]].  A link whose target has been
    # deleted still RENDERS FINE, so the graph rots silently: four such links
    # accumulated unnoticed across earlier sessions, every one pointing at a
    # fragment retired when its work completed.  Nothing was checking.
    dangling = check_wiki_links()
    if dangling:
        print("\n  DANGLING [[wiki-links]] "
              f"({len(dangling)}) — target fragment does not exist:",
              file=sys.stderr)
        for source, target in dangling:
            print(f"    {source} -> [[{target}]]", file=sys.stderr)
        print("  Repoint to the completed.d record if the work shipped, "
              "or drop the link.", file=sys.stderr)

    if check_only and (changed or dangling):
        if changed:
            print("\nFragment render check: files are stale.", file=sys.stderr)
        if dangling:
            print("Fragment render check: dangling cross-references.",
                  file=sys.stderr)
        sys.exit(1)
    elif not changed:
        print("  All surfaces up to date.")


if __name__ == "__main__":
    main()
