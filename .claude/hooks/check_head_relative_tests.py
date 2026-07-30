"""Fail a commit that adds a test which reconstructs code from ``git HEAD``.

THE FAILURE THIS CATCHES
------------------------
A test compares the worktree against the PRE-change implementation by pulling
it out of git: ``git show HEAD:<module>``, then AST-extracting a function or
constant and exec'ing it as an "independent oracle".  That is valid only while
HEAD is still the pre-change commit -- i.e. DURING the build that introduces
the change, before that build commits.

The instant the build commits, HEAD becomes the NEW version.  The oracle then
either compares a version against itself (silently vacuous) or fails outright
because the change deleted a symbol the reconstruction needs.

Why no existing gate catches it: the tree-wide gate runs BEFORE the commit, so
the test reads the OLD HEAD and PASSES.  It goes red in the NEXT build's gate,
attributed to a build that never touched it.  A hole that is green in its own
gate cannot be closed by reviewer vigilance -- hence this check.

Observed 2026-07-29/30 (FINDINGS F043): three tests broke this way in one day.
``test_lensing_caustic_cusps._head_find_cusps`` AST-extracted the pre-refactor
``_find_cusps`` from HEAD, needing ``_CUSP_SPEED_REL_FRAC``; build 1b deleted
that constant.  The tests passed in 1b's own tree gate and went red in 1c's,
which had not touched ``_find_cusps`` at all.  Two astroid byte-identity tests
failed the same way against a moved float-path baseline.

WHAT IT DOES *NOT* DO
---------------------
It does not judge whether the comparison is still meaningful -- that needs
judgment.  It reports that a test's oracle is pinned to a MOVING reference.
The fix is one of:

  * retire it -- a within-build transition check is done once its transition
    commits (this is usually the right answer);
  * freeze the expectation as a GOLDEN VALUE TABLE of literals in the test,
    readable without git;
  * pin an explicit historical commit SHA instead of ``HEAD`` (works, but is
    opaque and still breaks when the rule legitimately changes).

If the test is a deliberate within-build check that you are removing before it
merges, acknowledge it by name:

    GATED_HEAD_ORACLE_ACK="ClassName,Other.test_method" git commit ...

Exit 0 = clean.  Exit 1 = a HEAD-relative oracle was added or modified.
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PKG = 'cogwheel'
TESTS = f'{PKG}/tests'

#: Textual signatures of a HEAD-relative reconstruction.  Deliberately broad --
#: a false positive costs a glance and an ack, a false negative costs a build
#: cycle in a LATER build that did not cause it.
_HEAD_PATTERNS = (
    re.compile(r"""['"]HEAD:"""),                    # 'HEAD:path/to/mod.py'
    re.compile(r"""f?['"]HEAD:\{"""),                # f'HEAD:{_MODULE_REL}'
    re.compile(r"""git\s*['"],\s*['"]show"""),       # 'git', 'show', ...
    re.compile(r"""\bshow['"]\s*,\s*f?['"]HEAD"""),  # 'show', f'HEAD:...'
)


def _run(*args: str) -> str:
    return subprocess.run(args, cwd=REPO, capture_output=True,
                          text=True, check=False).stdout


def _staged_test_paths() -> list[str]:
    """Staged test files only -- this class of bug lives in tests."""
    out = _run('git', 'diff', '--cached', '--name-only', '--diff-filter=ACM')
    return [p for p in out.splitlines()
            if p.startswith(f'{TESTS}/') and p.endswith('.py')]


def _added_lines(path: str) -> list[str]:
    """Lines this commit ADDS to `path` (so a pre-existing, already-acked
    oracle does not re-fire on every unrelated edit to the same file)."""
    diff = _run('git', 'diff', '--cached', '-U0', '--', path)
    return [ln[1:] for ln in diff.splitlines()
            if ln.startswith('+') and not ln.startswith('+++')]


def _enclosing_defs(source: str) -> list[tuple[int, int, str]]:
    """(start_line, end_line, qualified_name) for every def/class."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    spans: list[tuple[int, int, str]] = []

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef)):
                name = f'{prefix}{child.name}'
                end = getattr(child, 'end_lineno', child.lineno)
                spans.append((child.lineno, end, name))
                walk(child, f'{name}.')

    walk(tree, '')
    return spans


def _owner(spans: list[tuple[int, int, str]], line: int) -> str:
    """Innermost def/class containing `line`, for the ack name and message."""
    best = ''
    for start, end, name in spans:
        if start <= line <= end and len(name) >= len(best):
            best = name
    return best or '<module level>'


def _head_oracle_helpers(source: str) -> set[str]:
    """Names of functions in this file whose BODY reconstructs code from HEAD.

    The antipattern spreads by REUSE, not only by re-introduction: build 1d
    (2026-07-30) added two tests whose only HEAD-relative content was the line
    ``head = _head_training_module()`` -- a call to a helper committed weeks
    earlier.  Scanning added lines for ``git show HEAD:`` saw nothing, the
    tests passed their own build's gate, and they went red in the very next
    run, which is precisely the failure F043 describes.  So resolve helpers
    first, then treat a CALL to one as the same finding as the literal.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    lines = source.splitlines()
    helpers: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        end = getattr(node, 'end_lineno', node.lineno)
        body = '\n'.join(lines[node.lineno - 1:end])
        if any(pat.search(body) for pat in _HEAD_PATTERNS):
            helpers.add(node.name)
    return helpers


def _acked() -> set[str]:
    raw = os.environ.get('GATED_HEAD_ORACLE_ACK', '')
    return {piece.strip() for piece in raw.split(',') if piece.strip()}


def main() -> int:
    findings: list[tuple[str, int, str, str]] = []
    for path in _staged_test_paths():
        added = _added_lines(path)
        try:
            source = (REPO / path).read_text()
        except OSError:
            continue
        # A line is HEAD-relative if it carries the literal pattern OR calls a
        # helper in this file whose body does.
        helpers = _head_oracle_helpers(source)
        call_re = (re.compile(r'\b(?:' + '|'.join(map(re.escape, helpers))
                              + r')\s*\(')
                   if helpers else None)

        def _is_head_relative(text: str) -> bool:
            if any(pat.search(text) for pat in _HEAD_PATTERNS):
                return True
            return bool(call_re and call_re.search(text))

        if not any(_is_head_relative(ln) for ln in added):
            continue                      # nothing HEAD-relative added here
        spans = _enclosing_defs(source)
        added_stripped = {ln.strip() for ln in added}
        for lineno, text in enumerate(source.splitlines(), 1):
            if not _is_head_relative(text):
                continue
            # Only report occurrences this commit actually introduced.
            if text.strip() not in added_stripped:
                continue
            findings.append((path, lineno, _owner(spans, lineno),
                             text.strip()))

    if not findings:
        return 0

    acked = _acked()
    live = [f for f in findings
            if not any(a == f[2] or f[2].startswith(f'{a}.') or a in f[2]
                       for a in acked)]

    if not live:
        print('  [acknowledged] HEAD-relative test oracles '
              '(GATED_HEAD_ORACLE_ACK):')
        for _p, _l, owner, _t in findings:
            print(f'      {owner}')
        return 0

    print('===== PRE-COMMIT: test oracle pinned to a MOVING git HEAD =====')
    print('  These reconstruct pre-change code from `git show HEAD` --')
    print('  directly, or by CALLING a helper in the same file that does.')
    print('  They PASS this gate (HEAD is still the old version) and BREAK')
    print('  the NEXT build once this commit lands. See FINDINGS F043.')
    print()
    for path, lineno, owner, text in live:
        print(f'  {path}:{lineno}  in {owner}')
        print(f'      {text[:96]}')
    print()
    print('  Fix: retire it (a within-build transition check is done once')
    print('  its transition commits), or freeze a golden-value table, or')
    print('  pin an explicit commit SHA instead of HEAD.')
    print('  If it is a deliberate within-build check, acknowledge it:')
    print('    GATED_HEAD_ORACLE_ACK="ClassName,Other.test_method" '
          'git commit ...')
    print('  Blanket bypass (also skips the correctness gates): --no-verify')
    print('===============================================================')
    return 1


if __name__ == '__main__':
    sys.exit(main())
