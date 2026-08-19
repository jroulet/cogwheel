#!/usr/bin/env python
"""Thin CLI over `cogwheel.lensing.tiling_plan.run`.

Refreshes the engine-free serve-route demand census and builds the demand-sized
tiling plan + campaign cost estimate, writing ONE combined JSON advisory to
``.claude/handoff/`` and printing the escalation verdict.  All sizing logic
lives in the module; this wrapper only parses arguments, calls `run`, and
dumps JSON.  It performs NO amplitude-engine evaluation and no tiling
arithmetic of its own -- I/O only.
"""
import argparse
import json
from pathlib import Path

from cogwheel.lensing.tiling_plan import run

_DEFAULT_OUT = '.claude/handoff/tiling_plan_and_cost_7a2.json'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-samples', type=int, default=10_000,
                        help='Serve-route census draw count (default 10000).')
    parser.add_argument('--seed', type=int, default=0,
                        help='Census sampler seed (default 0).')
    parser.add_argument('--f-min-hz', type=float, default=20.0,
                        help='Census frequency-grid lower edge in Hz.')
    parser.add_argument('--f-max-hz', type=float, default=1024.0,
                        help='Census frequency-grid upper edge in Hz.')
    parser.add_argument('--out', default=_DEFAULT_OUT,
                        help='Combined plan+cost JSON destination.')
    args = parser.parse_args()

    report = run(n_samples=args.n_samples, seed=args.seed,
                 f_min_hz=args.f_min_hz, f_max_hz=args.f_max_hz)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as stream:
        json.dump(report, stream, indent=2)

    totals = report['totals']
    escalation = report['escalation']
    verdict = 'ESCALATE' if escalation['should_escalate'] else 'within budget'
    print(f'total_calls {totals["total_calls"]}; '
          f'wall_clock_hours {totals["wall_clock_hours"]:.2f}; '
          f'max_region_share {escalation["max_region_share"]:.3f}; '
          f'verdict {verdict} -> {out_path}')
    if escalation['reasons']:
        for reason in escalation['reasons']:
            print(f'  escalation reason: {reason}')


if __name__ == '__main__':
    main()
