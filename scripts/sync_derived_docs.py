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


def check_data_contracts(root: Path, check_only: bool) -> CheckResult:
    """Keep DATA_CONTRACTS.yaml honest against the code (drift detection).

    For every registered artifact, verify each declared producer/consumer still
    resolves: the module file exists and the named function/method token is
    present in it. This catches the common drift — a producer/consumer renamed,
    moved, or deleted without updating the contract — as capabilities are added.
    Reliable and false-positive-free (only checks the DECLARED side).

    Not covered here: auto-discovering NEW, undeclared consumers from call sites
    (gw's jedi+ripgrep CONSUMER_GRAPH layer) — a documented follow-up; doing it
    without false positives needs heavier machinery than fits this check.
    """
    result = CheckResult("data_contracts")
    contracts = root / ".claude" / "spec" / "DATA_CONTRACTS.yaml"
    if not contracts.exists():
        return result
    try:
        import yaml
    except ImportError:
        return result
    data = yaml.safe_load(contracts.read_text(encoding="utf-8")) or {}
    artifacts = data.get("artifacts", {}) or {}

    def _check_ref(artifact: str, role: str, ref: dict):
        module = (ref or {}).get("module", "")
        func = (ref or {}).get("function", "")
        # Skip sentinel producers (prebuilt/shipped data, no in-repo module).
        if not module or not module.endswith(".py") or module.startswith("("):
            return
        mod_path = root / module
        if not mod_path.is_file():
            result.issue(
                f"{artifact}: declared {role} module '{module}' not found")
            return
        # Confirm the named function/method token appears in the module.
        if func and not func.startswith("("):
            token = func.split(".")[-1]  # Class.method -> method
            if token and token not in mod_path.read_text(encoding="utf-8"):
                result.issue(
                    f"{artifact}: declared {role} '{module}::{func}' — "
                    f"'{token}' not found in the module (renamed/removed?)")

    for name, info in artifacts.items():
        _check_ref(name, "producer", info.get("producer", {}))
        for c in info.get("consumers", []) or []:
            _check_ref(name, "consumer", c)
    return result


def check_consumer_graph(root: Path, check_only: bool) -> CheckResult:
    """Cross-check actual loader call-sites (CONSUMER_GRAPH.json, from
    regenerate_consumer_graph.py) against DATA_CONTRACTS.yaml's declared
    consumers, and flag ACTUAL-but-undeclared consumers — the drift that
    accumulates as capabilities are added. Advisory (informational); silently
    skips if the graph cache is absent (run scripts/regenerate_consumer_graph.py).
    """
    result = CheckResult("consumer_graph")
    contracts = root / ".claude" / "spec" / "DATA_CONTRACTS.yaml"
    graph = root / ".claude" / "spec" / "CONSUMER_GRAPH.json"
    if not contracts.exists() or not graph.exists():
        return result
    try:
        import yaml
        import json
    except ImportError:
        return result
    data = yaml.safe_load(contracts.read_text(encoding="utf-8")) or {}
    artifacts = data.get("artifacts", {}) or {}
    try:
        g = json.loads(graph.read_text(encoding="utf-8"))
    except ValueError:
        return result

    def _is_private(fn: str) -> bool:
        tail = fn.split(".")[-1]
        return tail.startswith("_") and not (tail.startswith("__") and tail.endswith("__"))

    for loader, entry in (g.get("loaders", {}) or {}).items():
        artifact = entry.get("artifact")
        info = artifacts.get(artifact)
        if not info:
            continue
        declared = {(c.get("module", ""), c.get("function", ""))
                    for c in info.get("consumers", []) or []}
        declared_fns = {f for _, f in declared}
        for c in entry.get("callers", []) or []:
            fn, mod = c.get("name", ""), c.get("file", "")
            # Skip module-level references (jedi returns the file stem as the
            # "caller" for a call not inside any function) — not a consumer.
            if fn == Path(mod).stem:
                continue
            if _is_private(fn) or fn == loader:
                continue
            if (mod, fn) in declared or fn in declared_fns:
                continue
            result.issue(
                f"{artifact}: actual consumer '{mod}::{fn}' (via {loader}) is "
                f"not in DATA_CONTRACTS.yaml — add it or confirm it's transient")
    return result


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
