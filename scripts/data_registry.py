#!/usr/bin/env python3
"""Data registry — resolve data file paths from data_registry.yaml.

Usage in code:
    from data_registry import get_path, get_root

    # Resolve a data entry to its full path
    path = get_path("training_data")

    # Get a storage root (respects env var overrides)
    root = get_root("cluster")

Usage as CLI:
    python scripts/data_registry.py list              — list all entries
    python scripts/data_registry.py resolve <entry>    — full path for an entry
    python scripts/data_registry.py roots              — list storage roots
    python scripts/data_registry.py validate           — check all paths exist

Reads from .claude/spec/data_registry.yaml by default.
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML required: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


_REGISTRY_PATH = ".claude/spec/data_registry.yaml"
_registry_cache: dict | None = None


def _find_project_root() -> Path:
    """Find the project root (directory containing .git)."""
    p = Path.cwd()
    while p != p.parent:
        if (p / ".git").exists():
            return p
        p = p.parent
    return Path.cwd()


def _load_registry(registry_path: str | None = None) -> dict:
    """Load and cache the data registry."""
    global _registry_cache
    if _registry_cache is not None:
        return _registry_cache

    root = _find_project_root()
    path = Path(registry_path) if registry_path else root / _REGISTRY_PATH
    if not path.exists():
        _registry_cache = {"storage_roots": {}, "entries": {}}
        return _registry_cache

    _registry_cache = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _registry_cache


def get_root(root_name: str, registry_path: str | None = None) -> Path:
    """Resolve a storage root, respecting env var overrides."""
    reg = _load_registry(registry_path)
    roots = reg.get("storage_roots", {})
    if root_name not in roots:
        raise KeyError(f"Unknown storage root: {root_name}. Known: {list(roots.keys())}")

    root_cfg = roots[root_name]
    env_var = root_cfg.get("env_override")
    if env_var and os.environ.get(env_var):
        return Path(os.environ[env_var])
    return Path(root_cfg["path"])


def get_path(entry_name: str, registry_path: str | None = None) -> Path:
    """Resolve a data entry to its full filesystem path."""
    reg = _load_registry(registry_path)
    entries = reg.get("entries", {})
    if entry_name not in entries:
        raise KeyError(f"Unknown data entry: {entry_name}. Known: {list(entries.keys())}")

    entry = entries[entry_name]
    root = get_root(entry["storage_root"], registry_path)
    return root / entry["relative_path"]


# ── CLI ──────────────────────────────────────────────────────────────────────


def cmd_list(reg: dict):
    entries = reg.get("entries", {})
    if not entries:
        print("No data entries registered.")
        return
    for name, info in entries.items():
        print(f"  {name}: {info.get('description', '(no description)')}")
        print(f"    root: {info.get('storage_root', '?')}, path: {info.get('relative_path', '?')}")
        print(f"    format: {info.get('format', '?')}, created_by: {info.get('created_by', '?')}")


def cmd_resolve(reg: dict, entry_name: str):
    try:
        path = get_path(entry_name)
        exists = path.exists()
        print(f"{entry_name}: {path}")
        print(f"  exists: {exists}")
    except KeyError as e:
        print(str(e))


def cmd_roots(reg: dict):
    roots = reg.get("storage_roots", {})
    if not roots:
        print("No storage roots configured.")
        return
    for name, cfg in roots.items():
        env_var = cfg.get("env_override", "")
        env_val = os.environ.get(env_var, "") if env_var else ""
        resolved = env_val or cfg.get("path", "?")
        override = f" (from ${env_var})" if env_val else ""
        print(f"  {name}: {resolved}{override}")
        print(f"    description: {cfg.get('description', '(none)')}")


def cmd_validate(reg: dict):
    entries = reg.get("entries", {})
    all_ok = True
    for name, info in entries.items():
        try:
            path = get_path(name)
            if path.exists():
                print(f"  OK: {name} -> {path}")
            else:
                print(f"  MISSING: {name} -> {path}")
                all_ok = False
        except KeyError as e:
            print(f"  ERROR: {name} -> {e}")
            all_ok = False
    if not all_ok:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Data registry path resolver")
    parser.add_argument("--registry", default=None, help="Path to registry YAML")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("list", help="List all data entries")
    sub = subparsers.add_parser("resolve", help="Resolve a data entry path")
    sub.add_argument("entry")
    subparsers.add_parser("roots", help="List storage roots")
    subparsers.add_parser("validate", help="Check all paths exist")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    reg = _load_registry(args.registry)

    if args.command == "list":
        cmd_list(reg)
    elif args.command == "resolve":
        cmd_resolve(reg, args.entry)
    elif args.command == "roots":
        cmd_roots(reg)
    elif args.command == "validate":
        cmd_validate(reg)


if __name__ == "__main__":
    main()
