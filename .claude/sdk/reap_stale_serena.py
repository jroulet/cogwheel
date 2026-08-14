#!/usr/bin/env python
"""Reap stale serena MCP servers (and their pyright children) for ONE project.

Why this exists (measured 2026-08-14): 16 serena + 16 pyright instances had
accumulated on the box — every session reconnect and every build-crew agent
spawns a stdio pair, and when the client dies non-gracefully (watchdog kill,
crash, session exit) the pair persists and keeps re-indexing the repo on
every commit. The accumulation pinned swap and read as "serena suddenly
became unreasonably slow" / mystery 240s timeouts. A 21-hour-stale quintet
traced to one killed build's five crew agents.

Discrimination (ALL must hold to reap; judged per wrapper->server CHAIN at
its root, and chains are reaped whole):
  * root cwd or --project arg matches THIS project root;
  * the root is ORPHANED (ppid == 1) or its parent is dead;
  * root older than --min-age-hours (default 10);
  * no chain member in the explicit --protect list;
  * no chain member holds an ESTABLISHED TCP connection — an SSE server a
    live client is attached to is serving, whatever its parentage.
Chain-level verdicts replace the 2026-08-14 live-child rule: judging
members separately made an orphaned pair immortal (wrapper kept for its
live child, child kept for its live parent), which was exactly the leak
shape this script was born to clear.
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


def _established(pid):
    """True if the process holds any ESTABLISHED TCP connection."""
    out = subprocess.run(['lsof', '-a', '-p', str(pid), '-iTCP',
                          '-sTCP:ESTABLISHED', '-t'],
                         capture_output=True, text=True).stdout
    return bool(out.strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', default=None,
                        help='Project root to reap for (default: the repo '
                             'containing this script).')
    parser.add_argument('--min-age-hours', type=float, default=10.0)
    parser.add_argument('--protect', type=int, nargs='*', default=[])
    parser.add_argument('--apply', action='store_true',
                        help='Actually kill (default: dry-run).')
    parser.add_argument('--count-live', action='store_true',
                        help='Print the number of live serena server '
                             'instances for THIS project and exit (one '
                             'instance = one wrapper/server chain, counted '
                             'by its root process).')
    args = parser.parse_args()

    project = args.project or os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..'))
    protect = set(args.protect)
    min_age = args.min_age_hours * 3600

    rows = _ps_rows('serena start-mcp-server')

    def _project_of(pid, argv):
        m = re.search(r'--project (\S+)', argv)
        return m.group(1) if m else _cwd(pid)

    if args.count_live:
        # One logical instance = one wrapper->server chain; count roots
        # (rows whose parent is not itself a serena row), so a healthy
        # uv-wrapper pair and an orphaned bare server each count once.
        pids = {pid for pid, _, _, _ in rows}
        print(sum(1 for pid, ppid, _age, argv in rows
                  if ppid not in pids and _project_of(pid, argv) == project))
        return 0

    children = {}
    for pid, ppid, _age, _argv in rows:
        children.setdefault(ppid, []).append(pid)

    # Judge per CHAIN at its root, and reap chains whole. Judging members
    # separately made an orphaned wrapper/server pair immortal (found
    # 2026-08-14): the wrapper was kept for its live child, the child for
    # its live parent. The wrapper/server pair is ONE unit.
    info = {pid: (ppid, age, argv) for pid, ppid, age, argv in rows}
    roots = [pid for pid, (ppid, _, _) in info.items() if ppid not in info]

    def _chain(root):
        out, stack = [], [root]
        while stack:
            pid = stack.pop()
            out.append(pid)
            stack.extend(children.get(pid, []))
        return out

    victims = []
    kept = []
    for root in roots:
        ppid, age, argv = info[root]
        members = _chain(root)
        reasons = []
        proc_project = _project_of(root, argv)
        if proc_project != project:
            reasons.append(f'other project ({proc_project})')
        hit = protect.intersection(members)
        if hit:
            reasons.append(f'protected ({sorted(hit)})')
        if age < min_age:
            reasons.append(f'young ({age // 3600}h)')
        if ppid != 1 and _alive(ppid):
            reasons.append(f'parent {ppid} alive')
        if not reasons:
            # Only now pay for lsof: a chain that survives the cheap checks
            # is kept anyway. An ESTABLISHED TCP peer means a live client is
            # attached (SSE) — serving, whatever the parentage. This is the
            # real signal the old live-child proxy stood in for.
            serving = [p for p in members if _established(p)]
            if serving:
                reasons.append(f'actively serving (pid {serving[0]} has an '
                               f'ESTABLISHED TCP peer)')
        if reasons:
            kept.append((root, age, '; '.join(reasons)))
        else:
            victims.extend((pid, info[pid][1]) for pid in members)

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
