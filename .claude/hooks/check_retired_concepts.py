"""Fail a commit that reintroduces a RETIRED concept into production code.

THE FAILURE THIS CATCHES
------------------------
The caustic-relative sequence retires constants and coordinates as it goes,
and its standing rule is blunt: *default is DELETE, not re-point -- changing
`3.0` to `rho*` IS the scar, it preserves the shape of the wrong idea*. Step 8
of that sequence makes the rule mechanical. But step 8 is LAST, so on the
current ordering every ghost accumulates across eight steps before anything
checks for them.

This is step 8's cheap half, pulled forward. Each build that retires a name
adds it to `retired_concepts.json`; the name cannot then come back into
production code without a deliberate acknowledgement.

Written because the same session that RECORDED the ghost rule (F045 -- a
retired test left as a skipped shell kept its helper alive, and the next build
reused it) had already violated it twice. A rule the author breaks the same day
needs a mechanism, not another paragraph.

WHAT IT DOES *NOT* DO
---------------------
It checks PRODUCTION code only (`cogwheel/**`, excluding `cogwheel/tests/`).
Tests and specs may name a retired concept while recording its retirement --
that is history, and deleting history is how the reason for a decision gets
lost. In production code the name IS the concept, so there it is a defect.

It matches whole words, so `_WEDGE_EPS` does not fire on `_WEDGE_EPSILON_NEW`.

Acknowledge a deliberate reintroduction (a genuine un-retirement) with:

    GATED_RETIRED_ACK="_WEDGE_EPS,other_name" git commit ...

and REMOVE the entry from retired_concepts.json in the same commit -- an
acknowledged name that stays in the list will fire again next time.

Exit 0 = clean. Exit 1 = a retired concept is back.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REGISTRY = Path(__file__).resolve().parent / 'retired_concepts.json'
PKG = 'cogwheel'
TESTS = f'{PKG}/tests'


def _staged_production_paths() -> list[str]:
    out = subprocess.run(
        ['git', 'diff', '--cached', '--name-only', '--diff-filter=ACM'],
        cwd=REPO, capture_output=True, text=True, check=False).stdout
    return [p for p in out.splitlines()
            if p.startswith(f'{PKG}/') and p.endswith('.py')
            and not p.startswith(TESTS)]


def _added_lines(path: str) -> list[tuple[int, str]]:
    """(lineno, text) for lines this commit ADDS, so a pre-existing mention
    does not re-fire on every unrelated edit to the same file."""
    diff = subprocess.run(['git', 'diff', '--cached', '-U0', '--', path],
                          cwd=REPO, capture_output=True, text=True,
                          check=False).stdout
    out, lineno = [], 0
    for line in diff.splitlines():
        hunk = re.match(r'^@@ -\d+(?:,\d+)? \+(\d+)', line)
        if hunk:
            lineno = int(hunk.group(1))
            continue
        if line.startswith('+') and not line.startswith('+++'):
            out.append((lineno, line[1:]))
            lineno += 1
    return out


def _retired() -> list[dict]:
    try:
        return json.loads(REGISTRY.read_text()).get('retired', [])
    except (OSError, ValueError):
        return []


def main() -> int:
    entries = _retired()
    if not entries:
        return 0
    acked = {p.strip() for p in
             os.environ.get('GATED_RETIRED_ACK', '').split(',') if p.strip()}
    patterns = [(e, re.compile(rf'\b{re.escape(e["name"])}\b'))
                for e in entries if e.get('name') not in acked]
    if not patterns:
        return 0

    findings = []
    for path in _staged_production_paths():
        for lineno, text in _added_lines(path):
            # A line that merely RECORDS the retirement is fine, and is how a
            # future reader learns where the idea went.
            low = text.lower()
            if any(w in low for w in ('retired', 'deleted', 'removed',
                                      'no longer', 'used to')):
                continue
            for entry, pat in patterns:
                if pat.search(text):
                    findings.append((path, lineno, entry, text.strip()))

    if not findings:
        return 0

    print('===== PRE-COMMIT: a RETIRED concept is back in production code ====')
    print('  The sequence retiring these says: default is DELETE, not')
    print('  re-point -- re-pointing preserves the shape of the wrong idea.')
    print()
    for path, lineno, entry, text in findings:
        print(f'  {path}:{lineno}  reintroduces {entry["name"]!r}')
        print(f'      {text[:92]}')
        print(f'      retired by {entry.get("retired_by", "?")}: '
              f'{entry.get("note", "")[:96]}')
    print()
    print('  If this is a deliberate un-retirement, ack it AND remove the')
    print('  entry from .claude/hooks/retired_concepts.json in this commit:')
    print('    GATED_RETIRED_ACK="NAME" git commit ...')
    print('==================================================================')
    return 1


if __name__ == '__main__':
    sys.exit(main())
