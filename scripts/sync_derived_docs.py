#!/usr/bin/env python3
"""Deterministic doc/spec consistency checks.

Checks are pure Python functions that verify docs match code.
Run without --check to auto-fix (where possible).
Run with --check to verify only (non-zero exit = issues found).

Add project-specific checks by defining new check_* functions.
The main() function auto-discovers and runs all of them.
"""

import argparse
import sys
from pathlib import Path


# ── Check framework ──────────────────────────────────────────────────────────


class CheckResult:
    def __init__(self, name: str):
        self.name = name
        self.issues: list[str] = []
        self.fixed: list[str] = []

    def issue(self, msg: str):
        self.issues.append(msg)

    def fix(self, msg: str):
        self.fixed.append(msg)

    @property
    def ok(self) -> bool:
        return not self.issues


def _get_project_root() -> Path:
    """Find the project root (directory containing .git)."""
    p = Path.cwd()
    while p != p.parent:
        if (p / ".git").exists():
            return p
        p = p.parent
    return Path.cwd()


# ── Built-in checks ─────────────────────────────────────────────────────────


def check_spec_version_format(root: Path, check_only: bool) -> CheckResult:
    """Verify SPEC.md has a valid spec_version in its frontmatter."""
    result = CheckResult("spec_version_format")
    spec = root / ".claude" / "spec" / "SPEC.md"
    if not spec.exists():
        result.issue("SPEC.md not found")
        return result
    content = spec.read_text(encoding="utf-8")
    if "spec_version:" not in content:
        result.issue("SPEC.md missing spec_version field")
    return result


def check_todo_no_completed(root: Path, check_only: bool) -> CheckResult:
    """Verify TODO.md has no [x] items (they should be in COMPLETED.md)."""
    result = CheckResult("todo_no_completed")
    todo = root / ".claude" / "spec" / "TODO.md"
    if not todo.exists():
        return result
    for i, line in enumerate(todo.read_text(encoding="utf-8").splitlines(), 1):
        if "[x] **" in line.lower() or "[X] **" in line:
            result.issue(f"TODO.md:{i}: completed item still in TODO — move to COMPLETED.md")
    return result


def check_changelog_advisory(root: Path, check_only: bool) -> CheckResult:
    """Verify CHANGELOG.md advisory block is near the top."""
    result = CheckResult("changelog_advisory")
    changelog = root / "CHANGELOG.md"
    if not changelog.exists():
        return result
    head = changelog.read_text(encoding="utf-8")[:500]
    if "When something breaks" not in head:
        result.issue("CHANGELOG.md advisory block missing or displaced from top")
    return result


# ── Add project-specific checks below ────────────────────────────────────────
# def check_my_custom_thing(root: Path, check_only: bool) -> CheckResult:
#     result = CheckResult("my_custom_thing")
#     ...
#     return result


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Deterministic doc/spec checks")
    parser.add_argument("--check", action="store_true", help="Check only (non-zero exit on issues)")
    args = parser.parse_args()

    root = _get_project_root()

    # Auto-discover all check_* functions in this module
    checks = [
        v for k, v in sorted(globals().items())
        if k.startswith("check_") and callable(v)
    ]

    all_ok = True
    for check_fn in checks:
        result = check_fn(root, check_only=args.check)
        if result.issues:
            all_ok = False
            for issue in result.issues:
                print(f"  [{result.name}] {issue}")
        if result.fixed:
            for fix in result.fixed:
                print(f"  [{result.name}] FIXED: {fix}")

    if not all_ok:
        if args.check:
            print(f"\n{len(checks)} checks run, issues found.")
            sys.exit(1)
        else:
            print(f"\n{len(checks)} checks run, some issues auto-fixed.")
    else:
        if not args.check:
            print(f"{len(checks)} checks run, all OK.")


if __name__ == "__main__":
    main()
