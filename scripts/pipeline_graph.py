#!/usr/bin/env python3
"""Pipeline graph — trace data artifact dependencies.

Usage:
    pipeline_graph.py resolve <artifact>       — find what produces this artifact
    pipeline_graph.py trace <artifact>         — full producer/consumer chain
    pipeline_graph.py consumers_of <artifact>  — all code that reads this artifact
    pipeline_graph.py inputs_for <module>      — what artifacts a module consumes

Reads from .claude/spec/DATA_CONTRACTS.yaml (or the path set by
--contracts-file). Also usable in-process:

    from pipeline_graph import PipelineGraph
    pg = PipelineGraph()
    pg.trace("posterior_samples")      # -> dict | None
    pg.consumers_of("posterior_samples")  # -> list[dict]
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


DEFAULT_CONTRACTS = ".claude/spec/DATA_CONTRACTS.yaml"
DEFAULT_REGISTRY = ".claude/spec/data_registry.yaml"
DEFAULT_GRAPH = ".claude/spec/CONSUMER_GRAPH.json"


def load_contracts(path: str = DEFAULT_CONTRACTS) -> dict:
    """Load and parse the data contracts file (exits if missing — CLI use)."""
    p = Path(path)
    if not p.exists():
        print(f"Contracts file not found: {path}", file=sys.stderr)
        sys.exit(1)
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


class PipelineGraph:
    """Importable, declared-consumers-only view of DATA_CONTRACTS.yaml.

    Artifacts live under the ``artifacts:`` key; each entry uses
    ``producer.module``/``producer.function`` and a list of
    ``consumers[].module``/``consumers[].function``. Lazily loaded so that
    construction never raises on a missing/empty contracts file.
    """

    def __init__(self, contracts_path: str = DEFAULT_CONTRACTS,
                 registry_path: str = DEFAULT_REGISTRY,
                 graph_path: str = DEFAULT_GRAPH):
        self.contracts_path = Path(contracts_path)
        self.registry_path_file = Path(registry_path)
        self.graph_path = Path(graph_path)
        self._contracts = None
        self._graph = None

    @property
    def contracts(self) -> dict:
        if self._contracts is None:
            if self.contracts_path.exists():
                self._contracts = yaml.safe_load(
                    self.contracts_path.read_text(encoding="utf-8")) or {}
            else:
                self._contracts = {}
        return self._contracts

    @property
    def graph(self) -> dict:
        """Cached actual-caller graph from CONSUMER_GRAPH.json (regen tool)."""
        if self._graph is None:
            import json as _json
            if self.graph_path.exists():
                try:
                    self._graph = _json.loads(
                        self.graph_path.read_text(encoding="utf-8"))
                except ValueError:
                    self._graph = {}
            else:
                self._graph = {}
        return self._graph

    def _actual_callers(self, artifact: str) -> list:
        """Actual callers for an artifact, from CONSUMER_GRAPH.json (may be empty)."""
        out = []
        for loader, entry in (self.graph.get("loaders", {}) or {}).items():
            if entry.get("artifact") != artifact:
                continue
            for c in entry.get("callers", []) or []:
                out.append({"module": c.get("file", ""),
                            "function": c.get("name", ""), "via": loader})
        return out

    @property
    def artifacts(self) -> dict:
        return self.contracts.get("artifacts", {}) or {}

    def trace(self, artifact: str):
        """Full info dict for an artifact (producer, consumers, fields), or None."""
        return self.artifacts.get(artifact)

    def consumers_of(self, artifact: str) -> list:
        """Consumers of an artifact — declared (YAML) unioned with actual
        (CONSUMER_GRAPH.json). Each entry is {module, function, source} where
        source is 'contracts', 'graph', or 'both'.
        """
        info = self.artifacts.get(artifact)
        if info is None and not self._actual_callers(artifact):
            return []
        by_key = {}
        for c in (info or {}).get("consumers", []) or []:
            key = (c.get("module", ""), c.get("function", ""))
            by_key[key] = {"module": key[0], "function": key[1], "source": "contracts"}
        for c in self._actual_callers(artifact):
            key = (c["module"], c["function"])
            if key in by_key:
                by_key[key]["source"] = "both"
            else:
                # module may differ (graph gives the caller's file); also try to
                # merge on function name alone before adding as graph-only.
                merged = next((v for k, v in by_key.items()
                               if k[1] == c["function"]), None)
                if merged:
                    merged["source"] = "both"
                    if not merged["module"]:
                        merged["module"] = c["module"]
                else:
                    by_key[key] = {"module": c["module"], "function": c["function"],
                                   "source": "graph", "via": c.get("via", "")}
        return sorted(by_key.values(), key=lambda d: (d["function"], d["module"]))

    def resolve(self, artifact: str):
        """Producer + format + disk path for an artifact, or None if unknown."""
        info = self.artifacts.get(artifact)
        if not info:
            return None
        return {
            "producer": info.get("producer", {}),
            "format": info.get("format"),
            "description": info.get("description"),
            "disk_path": self.registry_path(artifact),
        }

    def inputs_for(self, module: str) -> list:
        """Artifacts a module produces/consumes: list of (name, role)."""
        found = []
        for name, info in self.artifacts.items():
            for c in info.get("consumers", []) or []:
                if module in c.get("module", ""):
                    found.append((name, c.get("function", "?")))
            producer = info.get("producer", {}) or {}
            if module in producer.get("module", ""):
                found.append((name, f"PRODUCES via {producer.get('function', '?')}"))
        return found

    def registry_path(self, artifact: str):
        """Look up the artifact's disk path from data_registry.yaml if available."""
        if not self.registry_path_file.exists():
            return None
        try:
            reg = yaml.safe_load(
                self.registry_path_file.read_text(encoding="utf-8")) or {}
            entries = reg.get("entries", {})
            for name, entry in entries.items():
                if name == artifact or entry.get("schema_ref") == artifact:
                    root_name = entry.get("storage_root", "")
                    roots = reg.get("storage_roots", {})
                    root_cfg = roots.get(root_name, {})
                    env_var = root_cfg.get("env_override", "")
                    root_path = (os.environ.get(env_var, root_cfg.get("path", ""))
                                 if env_var else root_cfg.get("path", ""))
                    return str(Path(root_path) / entry.get("relative_path", ""))
        except Exception:
            pass
        return None


def _try_registry_path(artifact: str) -> "str | None":
    """Back-compat module-level helper (delegates to PipelineGraph)."""
    return PipelineGraph().registry_path(artifact)


def cmd_resolve(pg: PipelineGraph, artifact: str):
    """Find what produces an artifact and where it lives on disk."""
    info = pg.trace(artifact)
    if info is None:
        print(f"Unknown artifact: {artifact}")
        print(f"Known artifacts: {', '.join(pg.artifacts.keys())}")
        return
    producer = info.get("producer", {})
    print(f"Artifact: {artifact}")
    print(f"  Description: {info.get('description', '(none)')}")
    print(f"  Format: {info.get('format', '(unknown)')}")
    print(f"  Producer: {producer.get('module', '?')}::{producer.get('function', '?')}")

    disk_path = pg.registry_path(artifact)
    if disk_path:
        print(f"  Disk path: {disk_path}")
        print(f"  Exists: {Path(disk_path).exists()}")


def cmd_trace(pg: PipelineGraph, artifact: str):
    """Full producer/consumer chain for an artifact."""
    info = pg.trace(artifact)
    if info is None:
        print(f"Unknown artifact: {artifact}")
        return
    producer = info.get("producer", {})
    consumers = pg.consumers_of(artifact)
    print(f"=== {artifact} ===")
    print(f"  {info.get('description', '')}")
    print(f"  Format: {info.get('format', '?')}")
    print(f"  Fields: {', '.join(info.get('fields', []))}")
    print(f"\n  Producer:")
    print(f"    {producer.get('module', '?')}::{producer.get('function', '?')}")
    print(f"\n  Consumers ({len(consumers)}):")
    for c in consumers:
        print(f"    {c.get('module', '?')}::{c.get('function', '?')}")


def cmd_consumers_of(pg: PipelineGraph, artifact: str):
    """All code that reads a given artifact."""
    if pg.trace(artifact) is None:
        print(f"Unknown artifact: {artifact}")
        return
    print(f"Consumers of {artifact}:")
    for c in pg.consumers_of(artifact):
        src = c.get("source", "contracts")
        tag = {"contracts": "declared", "graph": "ACTUAL-undeclared", "both": "declared+actual"}.get(src, src)
        print(f"  {c.get('module', '?')}::{c.get('function', '?')}  [{tag}]")


def cmd_inputs_for(pg: PipelineGraph, module: str):
    """What data artifacts a module consumes."""
    found = pg.inputs_for(module)
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
        "--contracts-file", default=DEFAULT_CONTRACTS,
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

    # CLI keeps the fail-loud behavior: error if the contracts file is missing.
    load_contracts(args.contracts_file)
    pg = PipelineGraph(contracts_path=args.contracts_file)

    if args.command == "resolve":
        cmd_resolve(pg, args.artifact)
    elif args.command == "trace":
        cmd_trace(pg, args.artifact)
    elif args.command == "consumers_of":
        cmd_consumers_of(pg, args.artifact)
    elif args.command == "inputs_for":
        cmd_inputs_for(pg, args.module)


if __name__ == "__main__":
    main()
