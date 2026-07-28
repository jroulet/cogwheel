#!/usr/bin/env python3
"""Fail a commit that changes an API still referenced by a SKIPPED test.

THE FAILURE THIS CATCHES
------------------------
A build changes a signature, a constant's value, or a schema tag.  Some test
encoding the old contract breaks.  If that test is skipped -- behind
``skipUnless(COGWHEEL_TRAIN_TIER)``, ``@expectedFailure``, ``@skip``, a
pytest skip mark, or a module-level gate -- it does not run, does not fail,
and reports nothing.  The suite is green.  The breakage surfaces whenever
someone next runs the slow tier, with no link back to the commit that caused
it.

Observed 2026-07-27: a coordinate migration plus an admission change left 25
tests erroring at setup, undetected for a whole build cycle, because every one
of them sat in a file nobody ran.  Later the same day 14 engine-backed classes
were gated, and the next two builds changed ``reconstruct_farfield``'s
signature and bumped ``_FARFIELD_AXIS_SCHEMA`` -- both still referenced from
inside gated classes.

This is the mechanical half of the guard.  The persuasion half lives in
``.claude/crew/test_dev.md`` (step 7) and ``.claude/crew/inspector.md``
(check 5b), which ask agents to audit by READING.  Prompts can be skipped;
this cannot.

WHAT IT DOES *NOT* DO
---------------------
It does not decide whether the test is still correct -- that needs judgment.
It only reports that a symbol you changed is referenced from a test that will
not tell you if it broke.  Fix the test, or delete it, or bypass with
``--no-verify`` if you are certain.

Exit 0 = clean.  Exit 1 = drift found.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PKG = 'cogwheel'
TESTS = f'{PKG}/tests'

#: A decorator gates a test if its rendered name contains any of these.
#: Deliberately broad: `_TRAIN_TIER_SKIP` (a module-level
#: `unittest.skipUnless(...)` handle) must match as readily as `unittest.skip`,
#: and erring toward flagging is correct -- a false positive costs a glance, a
#: false negative costs a build cycle.
_GATE_MARKERS = ('skip', 'expectedfailure', 'xfail')


def _run(*args: str) -> str:
    return subprocess.run(args, cwd=REPO, capture_output=True,
                          text=True, check=False).stdout


def _staged_paths() -> list[str]:
    out = _run('git', 'diff', '--cached', '--name-only', '--diff-filter=ACM')
    return [p for p in out.splitlines()
            if p.startswith(f'{PKG}/') and p.endswith('.py')
            and not p.startswith(TESTS)]


def _head_source(path: str) -> str | None:
    proc = subprocess.run(['git', 'show', f'HEAD:{path}'], cwd=REPO,
                          capture_output=True, text=True, check=False)
    return proc.stdout if proc.returncode == 0 else None


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Argument NAMES and kinds -- not defaults, annotations or the body.

    A docstring edit, a retyped annotation or a changed default must NOT
    trip the guard; a new required argument or a rename must.
    """
    a = node.args
    parts = [
        ','.join(x.arg for x in a.posonlyargs),
        ','.join(x.arg for x in a.args),
        a.vararg.arg if a.vararg else '',
        ','.join(x.arg for x in a.kwonlyargs),
        a.kwarg.arg if a.kwarg else '',
        # Which kwonly args are REQUIRED (default None) matters: adding a
        # required keyword breaks callers, adding an optional one does not.
        ','.join('R' if d is None else 'O' for d in a.kw_defaults),
        str(len(a.defaults)),
    ]
    return '|'.join(parts)


def _api_surface(source: str) -> dict[str, str]:
    """{symbol: fingerprint} for defs and module-level constant literals."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    surface: dict[str, str] = {}

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                surface[f'{prefix}{child.name}'] = _signature(child)
            elif isinstance(child, ast.ClassDef):
                surface[f'{prefix}{child.name}'] = 'class'
                walk(child, f'{prefix}{child.name}.')

    walk(tree, '')
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    surface[target.id] = f'const:{ast.unparse(node.value)}'
                except Exception:                       # noqa: BLE001
                    pass
    return surface


def _changed_symbols() -> dict[str, str]:
    """Bare symbol names whose signature or constant value changed."""
    changed: dict[str, str] = {}
    for path in _staged_paths():
        head = _head_source(path)
        if head is None:
            continue                       # brand-new file: nothing to break
        try:
            now = (REPO / path).read_text()
        except OSError:
            continue
        before, after = _api_surface(head), _api_surface(now)
        for name, fingerprint in after.items():
            old = before.get(name)
            if old is None or old == fingerprint:
                continue
            bare = name.split('.')[-1]
            kind = ('constant value' if fingerprint.startswith('const:')
                    else 'signature')
            changed[bare] = f'{kind} changed in {path} ({name})'
        for name in before:
            if name not in after:
                bare = name.split('.')[-1]
                changed[bare] = f'removed from {path} ({name})'
    return changed


def _decorator_name(node: ast.AST) -> str:
    try:
        return ast.unparse(node).lower()
    except Exception:                                   # noqa: BLE001
        return ''


def _is_gated(node: ast.AST) -> bool:
    decorators = getattr(node, 'decorator_list', [])
    return any(marker in _decorator_name(d)
               for d in decorators for marker in _GATE_MARKERS)


def _referenced(node: ast.AST, names: set[str]) -> set[str]:
    """Which of ``names`` this subtree mentions, as a Name or an attribute."""
    found = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id in names:
            found.add(child.id)
        elif isinstance(child, ast.Attribute) and child.attr in names:
            found.add(child.attr)
    return found


def _module_helpers(tree: ast.Module) -> dict[str, ast.AST]:
    """Module-level functions, by name."""
    return {n.name: n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _helper_symbols(helpers: dict[str, ast.AST],
                    symbols: dict[str, str]) -> dict[str, set[str]]:
    """{helper: changed symbols it reaches}, following helper->helper calls.

    A gated test rarely touches the changed API directly -- it calls a
    module-level fixture builder that does.  Resolving only direct references
    misses exactly the case this hook exists for (measured 2026-07-27: every
    `reconstruct_farfield` reference in the gated far-field classes lived in
    shared helpers, so a body-only scan reported nothing).  Fixpoint because
    helpers call helpers.
    """
    direct = {name: _referenced(node, set(symbols))
              for name, node in helpers.items()}
    calls = {name: _referenced(node, set(helpers))
             for name, node in helpers.items()}
    reach = {name: set(found) for name, found in direct.items()}
    changed = True
    while changed:
        changed = False
        for name, callees in calls.items():
            for callee in callees:
                if callee != name and not reach[callee] <= reach[name]:
                    reach[name] |= reach[callee]
                    changed = True
    return reach


def _gated_references(symbols: dict[str, str], advisory: list
                      ) -> list[tuple[str, str, str, str]]:
    """(test_file, gated_test, symbol, how) for refs reachable from skips.

    Also appends ``(file, test, symbol)`` to ``advisory`` for RUNNING tests
    that reference the same symbols -- reported as blast radius, never a
    block, since a running test announces its own breakage.
    """
    findings: list[tuple[str, str, str, str]] = []
    if not symbols:
        return findings
    for path in sorted((REPO / TESTS).glob('test_*.py')):
        try:
            tree = ast.parse(path.read_text())
        except OSError:
            continue
        except SyntaxError as exc:                  # never fail silently
            findings.append((str(path.relative_to(REPO)), '<unparseable>',
                             '-', f'SyntaxError line {exc.lineno}'))
            continue
        helpers = _module_helpers(tree)
        reach = _helper_symbols(helpers, symbols)
        rel = str(path.relative_to(REPO))

        def scan(node: ast.AST, prefix: str, gated: bool) -> None:
            for child in ast.iter_child_nodes(node):
                if not isinstance(child, (ast.ClassDef, ast.FunctionDef,
                                          ast.AsyncFunctionDef)):
                    continue
                here = gated or _is_gated(child)
                name = f'{prefix}{child.name}'
                hits = set(_referenced(child, set(symbols)))
                for helper in _referenced(child, set(helpers)):
                    hits |= reach.get(helper, set())
                if not here:
                    # A RUNNING test that breaks reports itself, loudly, the
                    # moment the suite runs -- so this is advisory blast
                    # radius, never a block.  Blocking here would fire on
                    # every routine signature change and train everyone to
                    # --no-verify past the gate, which is how the gate that
                    # DOES matter stops being read.
                    if isinstance(child, ast.ClassDef):
                        for symbol in sorted(hits):
                            advisory.append((rel, name, symbol))
                    scan(child, f'{name}.', here)
                    continue
                for symbol in sorted(_referenced(child, set(symbols))):
                    findings.append((rel, name, symbol, 'directly'))
                for helper in sorted(_referenced(child, set(helpers))):
                    for symbol in sorted(reach.get(helper, ())):
                        findings.append(
                            (rel, name, symbol, f'via {helper}()'))

        scan(tree, '', False)
    # De-duplicate while keeping order.
    seen: set[tuple[str, str, str, str]] = set()
    unique = []
    for item in findings:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def main() -> int:
    symbols = _changed_symbols()
    advisory: list[tuple[str, str, str]] = []
    findings = _gated_references(symbols, advisory)
    if advisory:
        by_symbol: dict[str, set[str]] = {}
        for _file, test, symbol in advisory:
            by_symbol.setdefault(symbol, set()).add(test)
        print('  [advisory] changed API is referenced by running tests '
              '(they will report their own breakage):')
        for symbol in sorted(by_symbol):
            tests = sorted(by_symbol[symbol])
            shown = ', '.join(tests[:3])
            more = f' (+{len(tests) - 3} more)' if len(tests) > 3 else ''
            print(f'      {symbol}: {shown}{more}')
    if not findings:
        return 0

    # Targeted acknowledgement.  A gated test flagged here is not necessarily
    # BROKEN -- the honest resolution is often "I ran it under its tier and it
    # passes", which neither a fix nor a deletion expresses.  Without a way to
    # say that, the only exit is ``--no-verify``, which also skips the
    # correctness gates that run BEFORE this one and trains exactly the habit
    # this file's docstring warns about.
    #
    # ``GATED_DRIFT_ACK`` takes a comma-separated list of ``Class`` or
    # ``Class.test_method`` names the committer has actually RUN.  It is
    # deliberately per-test rather than a blanket switch: acknowledging one
    # test says nothing about the next one, so drift in a test you did not
    # check still blocks.  Substring-free exact matching keeps it honest.
    acked = {entry.strip()
             for entry in os.environ.get('GATED_DRIFT_ACK', '').split(',')
             if entry.strip()}
    if acked:
        kept = []
        confirmed = []
        for finding in findings:
            _file, test_name, _symbol, _how = finding
            head = test_name.split('.', 1)[0]
            if test_name in acked or head in acked:
                confirmed.append(test_name)
            else:
                kept.append(finding)
        if confirmed:
            print('  [acknowledged] gated tests the committer states they ran '
                  '(GATED_DRIFT_ACK):')
            for name in sorted(set(confirmed)):
                print(f'      {name}')
        findings = kept
        if not findings:
            return 0

    print('===== PRE-COMMIT: changed API referenced by SKIPPED tests =====')
    print('  These tests do NOT run, so they will NOT report their own')
    print('  breakage. The suite can be green while they are broken.')
    print()
    for test_file, test_name, symbol, how in findings:
        print(f'  {test_file}::{test_name}')
        reason = symbols.get(symbol, 'unparseable test file')
        print(f'      references {symbol!r} {how} -- {reason}')
    print()
    print('  Fix the test, delete it with a reason, or -- if you RAN it under')
    print('  its tier and it passes -- acknowledge that specific test:')
    print('    GATED_DRIFT_ACK="ClassName,Other.test_method" git commit ...')
    print('  Blanket bypass (also skips the correctness gates above):')
    print('    git commit --no-verify')
    print('==============================================================')
    return 1


if __name__ == '__main__':
    sys.exit(main())
