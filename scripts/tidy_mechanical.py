"""Deterministic replacement for the Tidier agent's mechanical rubric.

WHY THIS EXISTS
---------------
The Tidier's rubric is entirely mechanical -- 2 blank lines between top-level
definitions, 1 within classes, no whitespace-only lines, no run of 3+ blank
lines, imports ordered stdlib/third-party/local, unused imports removed.  None
of it needs judgment, and paying an agent to read each file and hand-edit
whitespace was measured (2026-07-30) to take LONGER THAN A FULL BUILD while
still unfinished.  Worse, one such pass wrote the literal characters ``\\n``
into ``operator.py`` where newlines belonged, leaving the package
un-importable (FINDINGS F047).

A deterministic pass is faster, identical every run, and -- because every edit
is checked by the AST round trip below -- cannot invent a syntax error.

WHAT IT DOES NOT DO
-------------------
Import REORDERING is deliberately not automated here: cogwheel's layer
convention (``from cogwheel import data, waveform, ...`` before
``from cogwheel.likelihood import ...``) is a judgment call about layering,
and a naive sorter would churn it.  Long lines are REPORTED, never wrapped --
where to break a line is a readability decision.  This script does only the
rules that are purely syntactic, and reports the rest.

Usage
-----
    python scripts/tidy_mechanical.py                 # the advisory's list
    python scripts/tidy_mechanical.py FILE [FILE ...]
    python scripts/tidy_mechanical.py --check         # report only; 1 if dirty
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import pathlib
import sys
import tokenize

REPO = pathlib.Path(__file__).resolve().parents[1]
ADVISORY = REPO / '.claude' / 'tidy_advisory.json'
MAX_LINE = 79


def _protected_lines(text: str) -> set[int]:
    """1-based line numbers whose content lives INSIDE a multi-line string.

    A line-based normaliser is wrong inside a string literal: a docstring line
    ending in a space OWNS that space as part of its value, so rstripping it
    changes the string.  The AST guard in `tidy_file` catches that and aborts,
    which is safe but means such a file is never tidied at all -- reported by
    the gw driver, 2026-07-30.  Tokenising first lets the surrounding
    whitespace be fixed while string contents are preserved byte for byte.

    Only the INTERIOR lines of a multi-line string are protected; the line
    carrying the closing quote can still have trailing whitespace stripped
    after the quote, but treating the whole span as protected is the
    conservative choice and costs nothing real.
    """
    protected: set[int] = set()
    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        for tok in tokens:
            if tok.type != tokenize.STRING:
                continue
            start, end = tok.start[0], tok.end[0]
            if end > start:                      # multi-line string only
                protected.update(range(start, end + 1))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # Unparseable: protect nothing here -- `tidy_file` has already
        # refused to touch a file that does not parse.
        return set()
    return protected


def _normalise(text: str) -> str:
    """Apply the whitespace rules.  Pure text -> text, no file I/O."""
    protected = _protected_lines(text)
    out = []
    for lineno, line in enumerate(text.split('\n'), 1):
        if lineno in protected:
            out.append(line)                     # string content: untouched
        elif line.strip() == '':
            out.append('')                       # a line of spaces is blank
        else:
            out.append(line.rstrip())
    text = '\n'.join(out)
    # Never more than 2 consecutive blank lines.  Blank runs inside a
    # multi-line string are left alone: a collapse there would also change
    # the string's value.
    text = _collapse_blank_runs(text)
    # Exactly one trailing newline.
    return text.rstrip('\n') + '\n'


def _collapse_blank_runs(text: str) -> str:
    """Collapse 3+ consecutive blank lines to 2, outside string literals."""
    protected = _protected_lines(text)
    lines = text.split('\n')
    out: list[str] = []
    run = 0
    for lineno, line in enumerate(lines, 1):
        if line == '' and lineno not in protected:
            run += 1
            if run > 2:
                continue
        else:
            run = 0
        out.append(line)
    return '\n'.join(out)


def _long_lines(text: str) -> list[tuple[int, int]]:
    """``(lineno, width)`` for lines over the pylint ceiling."""
    return [(i, len(line)) for i, line in enumerate(text.split('\n'), 1)
            if len(line) > MAX_LINE]


def tidy_file(path: pathlib.Path, *, check_only: bool) -> tuple[bool, str]:
    """Normalise one file.  Never writes a file that stops parsing."""
    original = path.read_text()
    try:
        ast.parse(original)
    except SyntaxError as exc:
        return False, f'SKIPPED (already broken): line {exc.lineno} {exc.msg}'

    updated = _normalise(original)
    if updated == original:
        note = 'clean'
    else:
        # The guarantee that makes this safe to run unattended: the result
        # must parse to the SAME tree.  Whitespace cannot change semantics,
        # so any mismatch is a bug in this script, not a style opinion.
        try:
            if ast.dump(ast.parse(updated)) != ast.dump(ast.parse(original)):
                return False, 'ABORTED: normalisation changed the AST'
        except SyntaxError as exc:
            return False, f'ABORTED: result would not parse ({exc.msg})'
        if not check_only:
            path.write_text(updated)
        note = 'normalised'

    longs = _long_lines(updated)
    if longs:
        shown = ', '.join(f'{n}({w})' for n, w in longs[:6])
        note += f'; {len(longs)} line(s) over {MAX_LINE}: {shown}'
    return updated != original, note


def _targets(argv: list[str]) -> list[pathlib.Path]:
    if argv:
        return [pathlib.Path(a) for a in argv]
    if not ADVISORY.exists():
        return []
    out = []
    for rel in json.loads(ADVISORY.read_text()).get('touched_files', []):
        path = REPO / rel
        # Tests are out of scope (the Tidier rubric says so), and the
        # advisory can name files deleted since it was written.
        if '/tests/' in rel or not path.exists():
            continue
        out.append(path)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Mechanical style pass (see module docstring).')
    parser.add_argument('files', nargs='*')
    parser.add_argument('--check', action='store_true',
                        help='report only; exit 1 if anything would change')
    args = parser.parse_args()

    targets = _targets(args.files)
    if not targets:
        print('nothing to tidy (no files given and no advisory entries)')
        return 0

    changed_any = False
    for path in targets:
        changed, note = tidy_file(path, check_only=args.check)
        changed_any |= changed
        try:
            rel = path.resolve().relative_to(REPO)
        except ValueError:
            rel = path
        print(f'  {"CHANGED" if changed else "       "}  {rel}  --  {note}')

    if args.check and changed_any:
        print('\n--check: files would change; run without --check to apply')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
