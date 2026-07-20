#!/usr/bin/env python3
"""Regenerate .claude/spec/CONSUMER_GRAPH.json from the live codebase.

Statically finds, for each tracked data-loader function/method, every call
site (the ACTUAL consumers) using ripgrep (fast recall) + jedi (precise
resolution). `sync_derived_docs.py::check_consumer_graph` then cross-checks
these against DATA_CONTRACTS.yaml's declared `consumers` and flags drift as
capabilities are added. Mechanism ported from gw_detection_ias.

Scope note: this only attributes cleanly for artifacts with a DEDICATED named
loader (e.g. `EventData.from_npz`). Artifacts read via generic loaders
(`pd.read_feather`, `np.load`, `pd.read_csv`) can't be call-site-attributed to
one artifact and are covered instead by the declared-side check
(`sync_derived_docs.py::check_data_contracts`). Add trackable loaders to
LOADERS below as the codebase grows.

Usage: python scripts/regenerate_consumer_graph.py
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    import jedi
except ImportError:
    print("jedi required: pip install jedi", file=sys.stderr)
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent
GRAPH_PATH = ROOT / ".claude" / "spec" / "CONSUMER_GRAPH.json"

# ── Config ───────────────────────────────────────────────────────────────
# Single source of truth: each tracked loader maps to the DATA_CONTRACTS.yaml
# artifact it loads. `method` is the bare name (ripgrep pattern); `class_name`
# is None for a free function / constructor; `definitions` pins the expected
# jedi full-name(s) so call sites of same-named functions elsewhere are not
# mis-attributed. Only include loaders that are SPECIFIC to one artifact.
LOADERS: dict[str, dict] = {
    "EventData.from_npz": {
        "artifact": "event_data_npz",
        "method": "from_npz",
        "class_name": "EventData",
        "definitions": [
            ("cogwheel/data.py", "cogwheel.data.EventData.from_npz"),
        ],
    },
    "PostProcessor": {
        "artifact": "posterior_samples",
        "method": "PostProcessor",   # constructor call: reads samples.feather
        "class_name": None,
        "definitions": [
            ("cogwheel/postprocessing.py",
             "cogwheel.postprocessing.PostProcessor"),
        ],
    },
    "LookupTable": {
        "artifact": "coherent_score_lookup_tables",
        "method": "LookupTable",     # constructor: loads/builds the cache
        "class_name": None,
        "definitions": [
            ("cogwheel/likelihood/marginalization/lookup_table.py",
             "cogwheel.likelihood.marginalization.lookup_table.LookupTable"),
        ],
    },
    "LensAmplificationSurrogate.load": {
        "artifact": "lens_amplification_surrogate",
        "method": "load",
        "class_name": "LensAmplificationSurrogate",
        "definitions": [
            ("cogwheel/lensing/surrogate.py",
             "cogwheel.lensing.surrogate.LensAmplificationSurrogate.load"),
        ],
    },
}

# ripgrep --glob exclusions. Keep in sync with EXCLUDED_PREFIXES in
# sync_derived_docs.py::check_consumer_graph_freshness.
EXCLUDE_GLOBS = [
    "!**/__pycache__/**", "!docs/**", "!.claude/**", "!.serena/**",
    "!references/**", "!build/**",
]


def _rg_path() -> str:
    """Resolve ripgrep robustly: anchor to the interpreter's bin, then PATH.

    Hook environments run with a minimal PATH that lacks the env's bin.
    """
    cand = os.path.join(os.path.dirname(sys.executable), "rg")
    if os.path.isfile(cand) and os.access(cand, os.X_OK):
        return cand
    found = shutil.which("rg")
    if found:
        return found
    raise FileNotFoundError(
        "ripgrep (rg) not found next to the interpreter "
        f"({cand}) or on PATH — install it in the project env")


def rg_candidates(method: str) -> list:
    """Fast recall pass: every `method(` call site as (relpath, line, col)."""
    rg = _rg_path()
    pattern = rf"\b{method}\("
    cmd = [rg, "-n", "--type=py",
           *[f"--glob={g}" for g in EXCLUDE_GLOBS], pattern, str(ROOT)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    out = []
    for line in result.stdout.splitlines():
        parts = line.split(":", 2)
        if len(parts) < 3:
            continue
        path, lineno_s, content = parts
        try:
            lineno = int(lineno_s)
        except ValueError:
            continue
        idx = content.find(f".{method}(")
        if idx >= 0:
            col = idx + 1
        else:
            idx = content.find(f"{method}(")
            if idx < 0:
                continue
            col = idx
        rel = path[len(str(ROOT)) + 1:] if path.startswith(str(ROOT)) else path
        out.append((rel, lineno, col))
    return out


def _files_importing(module_basenames: set) -> set:
    """Files that import any of the given module basenames (narrows jedi work)."""
    if not module_basenames:
        return set()
    alternation = "|".join(re.escape(m) for m in module_basenames)
    pattern = rf"^\s*(?:from|import)\b.*\b({alternation})\b"
    cmd = [_rg_path(), "-l", "--type=py",
           *[f"--glob={g}" for g in EXCLUDE_GLOBS], pattern, str(ROOT)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    files = set()
    for line in result.stdout.splitlines():
        files.add(line[len(str(ROOT)) + 1:] if line.startswith(str(ROOT)) else line)
    return files


def enclosing_caller(script, line: int):
    """Qualified name (Class.method) of the function enclosing `line`, or None."""
    names = script.get_names(all_scopes=True, definitions=True, references=False)
    containers = []
    for n in names:
        if n.type != "function":
            continue
        try:
            start = n.line
            end = (n.get_definition_end_position()[0]
                   if hasattr(n, "get_definition_end_position") else None)
        except Exception:
            continue
        if end is None or not (start <= line <= end):
            continue
        containers.append((end - start, n))
    if not containers:
        return None
    containers.sort(key=lambda x: -x[0])   # outermost = largest span
    outermost = containers[0][1]
    parts = [outermost.name]
    try:
        parent = outermost.parent()
    except Exception:
        parent = None
    while parent is not None and parent.type == "class":
        parts.append(parent.name)
        try:
            parent = parent.parent()
        except Exception:
            break
    return ".".join(reversed(parts))


def _module_name_from_path(rel_path: str) -> str:
    return rel_path[:-3].replace("/", ".") if rel_path.endswith(".py") else rel_path


def walk_loader(project, short_name: str, config: dict, script_cache: dict) -> dict:
    """Resolve the actual callers of one tracked loader via jedi."""
    method = config["method"]
    expected = {full for _, full in config["definitions"]}
    defined_in = [path for path, _ in config["definitions"]]

    basenames = {Path(p).stem for p, _ in config["definitions"]}
    basenames.update(config.get("extra_importer_modules", []))
    importer_files = _files_importing(basenames)
    importer_files.update(defined_in)

    candidates = [c for c in rg_candidates(method) if c[0] in importer_files]
    callers = {}
    for rel_path, lineno, col in candidates:
        try:
            if rel_path not in script_cache:
                script_cache[rel_path] = jedi.Script(path=rel_path, project=project)
            script = script_cache[rel_path]
            defs = script.goto(line=lineno, column=col, follow_imports=True)
        except Exception:
            continue
        if not any(d.full_name and d.full_name in expected for d in defs):
            continue
        caller = enclosing_caller(script, lineno)
        if caller is None:
            caller = _module_name_from_path(rel_path).rsplit(".", 1)[-1]
        if caller in (short_name, method) and rel_path in defined_in:
            continue
        callers[(caller, rel_path)] = True

    return {
        "artifact": config["artifact"],
        "defined_in": defined_in,
        "callers": [{"name": n, "file": f} for (n, f) in sorted(callers.keys())],
    }


def main() -> int:
    project = jedi.Project(str(ROOT))
    script_cache: dict = {}
    out = {
        "schema_version": "1.0",
        "generated_by": "scripts/regenerate_consumer_graph.py (jedi.goto + ripgrep)",
        "generated_at": time.strftime("%Y-%m-%d"),
        "loaders": {},
    }
    t0 = time.time()
    for short_name, config in LOADERS.items():
        t = time.time()
        entry = walk_loader(project, short_name, config, script_cache)
        out["loaders"][short_name] = entry
        print(f"  {short_name}: {len(entry['callers'])} callers ({time.time()-t:.1f}s)",
              file=sys.stderr)
    # Keep git clean when only the date would change.
    if GRAPH_PATH.exists():
        try:
            existing = json.loads(GRAPH_PATH.read_text())
            if existing.get("loaders") == out["loaders"]:
                out["generated_at"] = existing["generated_at"]
        except Exception:
            pass
    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = GRAPH_PATH.with_suffix(GRAPH_PATH.suffix + ".tmp")
    tmp.write_text(json.dumps(out, indent=2) + "\n")
    tmp.replace(GRAPH_PATH)
    print(f"Wrote {GRAPH_PATH.relative_to(ROOT)} ({time.time()-t0:.1f}s total)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
