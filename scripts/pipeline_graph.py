#!/usr/bin/env python3
"""Pipeline graph — trace data artifact dependencies.

Usage:
    pipeline_graph.py resolve <artifact>       — find what produces this artifact
    pipeline_graph.py trace <artifact>         — full producer/consumer chain
    pipeline_graph.py consumers_of <artifact>  — all code that reads this artifact
    pipeline_graph.py inputs_for <module>      — what artifacts a module consumes

Reads from .claude/spec/DATA_CONTRACTS.yaml (or the path set by
--contracts-file).
"""

import argparse
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML required: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


def load_contracts(path: str = ".claude/spec/DATA_CONTRACTS.yaml") -> dict:
    """Load and parse the data contracts file."""
    p = Path(path)
    if not p.exists():
        print(f"Contracts file not found: {path}", file=sys.stderr)
        sys.exit(1)
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _try_registry_path(artifact: str) -> str | None:
    """Look up the artifact's disk path from data_registry.yaml if available."""
    registry_path = Path(".claude/spec/data_registry.yaml")
    if not registry_path.exists():
        return None
    try:
        reg = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
        entries = reg.get("entries", {})
        # Match by name or by schema_ref
        for name, entry in entries.items():
            if name == artifact or entry.get("schema_ref") == artifact:
                import os
                root_name = entry.get("storage_root", "")
                roots = reg.get("storage_roots", {})
                root_cfg = roots.get(root_name, {})
                env_var = root_cfg.get("env_override", "")
                root_path = os.environ.get(env_var, root_cfg.get("path", "")) if env_var else root_cfg.get("path", "")
                return str(Path(root_path) / entry.get("relative_path", ""))
    except Exception:
        pass
    return None


def cmd_resolve(contracts: dict, artifact: str):
    """Find what produces an artifact and where it lives on disk."""
    artifacts = contracts.get("artifacts", {})
    if artifact not in artifacts:
        print(f"Unknown artifact: {artifact}")
        print(f"Known artifacts: {', '.join(artifacts.keys())}")
        return
    info = artifacts[artifact]
    producer = info.get("producer", {})
    print(f"Artifact: {artifact}")
    print(f"  Description: {info.get('description', '(none)')}")
    print(f"  Format: {info.get('format', '(unknown)')}")
    print(f"  Producer: {producer.get('module', '?')}::{producer.get('function', '?')}")

    # Enrich with registry path if available
    disk_path = _try_registry_path(artifact)
    if disk_path:
        print(f"  Disk path: {disk_path}")
        print(f"  Exists: {Path(disk_path).exists()}")


def cmd_trace(contracts: dict, artifact: str):
    """Full producer/consumer chain for an artifact."""
    artifacts = contracts.get("artifacts", {})
    if artifact not in artifacts:
        print(f"Unknown artifact: {artifact}")
        return
    info = artifacts[artifact]
    producer = info.get("producer", {})
    consumers = info.get("consumers", [])
    print(f"=== {artifact} ===")
    print(f"  {info.get('description', '')}")
    print(f"  Format: {info.get('format', '?')}")
    print(f"  Fields: {', '.join(info.get('fields', []))}")
    print(f"\n  Producer:")
    print(f"    {producer.get('module', '?')}::{producer.get('function', '?')}")
    print(f"\n  Consumers ({len(consumers)}):")
    for c in consumers:
        print(f"    {c.get('module', '?')}::{c.get('function', '?')}")


def cmd_consumers_of(contracts: dict, artifact: str):
    """All code that reads a given artifact."""
    artifacts = contracts.get("artifacts", {})
    if artifact not in artifacts:
        print(f"Unknown artifact: {artifact}")
        return
    consumers = artifacts[artifact].get("consumers", [])
    print(f"Consumers of {artifact}:")
    for c in consumers:
        print(f"  {c.get('module', '?')}::{c.get('function', '?')}")


def cmd_inputs_for(contracts: dict, module: str):
    """What data artifacts a module consumes."""
    artifacts = contracts.get("artifacts", {})
    found = []
    for name, info in artifacts.items():
        for c in info.get("consumers", []):
            if module in c.get("module", ""):
                found.append((name, c.get("function", "?")))
        producer = info.get("producer", {})
        if module in producer.get("module", ""):
            found.append((name, f"PRODUCES via {producer.get('function', '?')}"))
    if not found:
        print(f"No artifacts reference module: {module}")
        return
    print(f"Artifacts involving {module}:")
    for name, role in found:
        print(f"  {name} — {role}")


def main():
    parser = argparse.ArgumentParser(
        description="Trace data artifact dependencies from DATA_CONTRACTS.yaml",
    )
    parser.add_argument(
        "--contracts-file", default=".claude/spec/DATA_CONTRACTS.yaml",
        help="Path to contracts file",
    )
    subparsers = parser.add_subparsers(dest="command")

    sub = subparsers.add_parser("resolve", help="Find what produces an artifact")
    sub.add_argument("artifact")

    sub = subparsers.add_parser("trace", help="Full producer/consumer chain")
    sub.add_argument("artifact")

    sub = subparsers.add_parser("consumers_of", help="All code that reads an artifact")
    sub.add_argument("artifact")

    sub = subparsers.add_parser("inputs_for", help="What artifacts a module consumes")
    sub.add_argument("module")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    contracts = load_contracts(args.contracts_file)

    if args.command == "resolve":
        cmd_resolve(contracts, args.artifact)
    elif args.command == "trace":
        cmd_trace(contracts, args.artifact)
    elif args.command == "consumers_of":
        cmd_consumers_of(contracts, args.artifact)
    elif args.command == "inputs_for":
        cmd_inputs_for(contracts, args.module)


if __name__ == "__main__":
    main()
