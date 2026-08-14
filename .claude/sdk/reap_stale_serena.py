#!/usr/bin/env python
"""Reap stale serena MCP servers (and their pyright children) for ONE project.

Why this exists (measured 2026-08-14): 16 serena + 16 pyright instances had
accumulated on the box — every session reconnect and every build-crew agent
spawns a stdio pair, and when the client dies non-gracefully (watchdog kill,
crash, session exit) the pair persists and keeps re-indexing the repo on
every commit. The accumulation pinned swap and read as "serena suddenly
became unreasonably slow" / mystery 240s timeouts. A 21-hour-stale quintet
traced to one killed build's five crew agents.

Discrimination (ALL must hold to reap):
  * process cwd or --project arg matches THIS project root;
  * the process is ORPHANED (ppid == 1) or its parent is dead;
  * older than --min-age-hours (default 10);
  * not in the explicit --protect list.
Servers of OTHER projects are never touched (multi-project box: e.g. a gw
build's serena must survive a cogwheel reap).

Default is DRY-RUN: prints the verdict per process. Pass --apply to kill
(SIGTERM, 3 s grace, SIGKILL survivors), then reap pyright orphans whose
parent died in this sweep.

Usage:
  python .claude/sdk/reap_stale_serena.py                 # dry-run, this repo
  python .claude/sdk/reap_stale_serena.py --apply
  python .claude/sdk/reap_stale_serena.py --min-age-hours 2 --apply
  python .claude/sdk/reap_stale_serena.py --protect 1234 5678 --apply
"""
import argparse
import os
import re
import signal
import subprocess
import sys
import time


def _ps_rows(pattern):
    out = subprocess.run(['ps', '-eo', 'pid,ppid,etimes,args',
                          '--no-headers'],
                         capture_output=True, text=True).stdout
    rows = []
    for line in out.splitlines():
        parts = line.split(None, 3)
        if len(parts) < 4 or pattern not in parts[3]:
            continue
        rows.append((int(parts[0]), int(parts[1]), int(parts[2]), parts[3]))
    return rows


def _cwd(pid):
    try:
        return os.readlink(f'/proc/{pid}/cwd')
    except OSError:
        return None


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', default=None,
                        help='Project root to reap for (default: the repo '
                             'containing this script).')
    parser.add_argument('--min-age-hours', type=float, default=10.0)
    parser.add_argument('--protect', type=int, nargs='*', default=[])
    parser.add_argument('--apply', action='store_true',
                        help='Actually kill (default: dry-run).')
    args = parser.parse_args()

    project = args.project or os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..'))
    protect = set(args.protect)
    min_age = args.min_age_hours * 3600

    victims = []
    kept = []
    for pid, ppid, age, argv in _ps_rows('serena start-mcp-server'):
        cwd = _cwd(pid)
        m = re.search(r'--project (\S+)', argv)
        proc_project = m.group(1) if m else cwd
        reasons = []
        if proc_project != project:
            reasons.append(f'other project ({proc_project})')
        if pid in protect:
            reasons.append('protected')
        if age < min_age:
            reasons.append(f'young ({age // 3600}h)')
        if ppid != 1 and _alive(ppid):
            reasons.append(f'parent {ppid} alive')
        if reasons:
            kept.append((pid, age, '; '.join(reasons)))
        else:
            victims.append((pid, age))

    for pid, age, why in kept:
        print(f'KEEP {pid:>8}  age {age // 3600:>3}h  {why}')
    for pid, age in victims:
        print(f'{"REAP" if args.apply else "would-reap":>10} {pid:>8}  '
              f'age {age // 3600:>3}h')

    if not args.apply:
        print(f'\ndry-run: {len(victims)} reapable, {len(kept)} kept '
              f'(pass --apply to kill)')
        return 0

    for pid, _ in victims:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(3)
    for pid, _ in victims:
        if _alive(pid):
            os.kill(pid, signal.SIGKILL)
            print(f'SIGKILL {pid} (survived TERM)')

    orphans = 0
    for pid, ppid, age, _argv in _ps_rows('pyright-langserver'):
        if ppid == 1 and age >= min_age and pid not in protect:
            try:
                os.kill(pid, signal.SIGKILL)
                orphans += 1
                print(f'REAP orphan pyright {pid} (age {age // 3600}h)')
            except ProcessLookupError:
                pass

    n_serena = len(_ps_rows('serena start-mcp-server'))
    n_pyright = len(_ps_rows('pyright-langserver'))
    print(f'\nreaped {len(victims)} serena + {orphans} orphan pyright; '
          f'remaining on box: {n_serena} serena, {n_pyright} pyright')
    return 0


if __name__ == '__main__':
    sys.exit(main())
