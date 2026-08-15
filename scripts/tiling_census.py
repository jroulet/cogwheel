#!/usr/bin/env python
"""Thin CLI over `cogwheel.lensing.tiling_census.run`.

Runs the engine-free tiling census / node-budget predictor against the
production ``TrainingConfig`` (or a subset of regions) and writes the JSON
advisory report.  All logic lives in the census module; this wrapper only
parses arguments, calls `run`, and dumps JSON.  It performs NO amplitude-engine
evaluation and no tiling arithmetic of its own -- I/O only.
"""
import argparse
import json

from cogwheel.lensing.surrogate_training import TrainingConfig
from cogwheel.lensing.tiling_census import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--regions', nargs='+', default=None,
                        help='Restrict the census to these region names '
                        '(intersected with each parity\'s admissible set). '
                        'Default: census every region.')
    parser.add_argument('--out', default='tiling_census.json',
                        help='JSON report destination.')
    args = parser.parse_args()

    regions = tuple(args.regions) if args.regions is not None else None
    report = run(config=TrainingConfig(), regions=regions)

    with open(args.out, 'w') as stream:
        json.dump(report, stream, indent=2)

    print(f'aggregate_call_count {report["aggregate_call_count"]}; '
          f'census_seconds {report["census_seconds"]:.1f}; '
          f'self_estimate_seconds {report["self_estimate_seconds"]:.1f} -> '
          f'{args.out}')


if __name__ == '__main__':
    main()
