#!/usr/bin/env python
"""Thin CLI over `cogwheel.lensing.surrogate_census.run`.

Draws fixture-scale lens prior samples, characterizes each into a surrogate
serve / fall-through outcome, computes per-chart held-out envelope eps, and
writes the JSON census report. All logic lives in the census module; this
wrapper only parses arguments, calls `run`, and dumps JSON. The lnL-tier stage
(which needs a full likelihood) is not wired here -- it is exercised
programmatically via the injected ``lnlike_pair`` argument of `run`.
"""
import argparse
import json

from cogwheel.lensing.surrogate_census import (
    CensusConfig, dropped_slivers_from_training_report, run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--surrogate-path', default=None,
                        help='Surrogate artifact (default: package data).')
    parser.add_argument('--output', default='census_report.json',
                        help='JSON report destination.')
    parser.add_argument('--n-samples', type=int, default=CensusConfig.n_samples)
    parser.add_argument('--seed', type=int, default=CensusConfig.seed)
    parser.add_argument('--f-min-hz', type=float, default=CensusConfig.f_min_hz)
    parser.add_argument('--f-max-hz', type=float, default=CensusConfig.f_max_hz)
    parser.add_argument('--n-freq', type=int, default=CensusConfig.n_freq)
    parser.add_argument('--max-heldout-per-chart', type=int,
                        default=CensusConfig.max_heldout_per_chart)
    parser.add_argument('--dropped-slivers-report', default=None,
                        help='Training-report JSON to read dropped gamma '
                        'slivers from (default: surrogate provenance).')
    args = parser.parse_args()

    config = CensusConfig(
        n_samples=args.n_samples, seed=args.seed, f_min_hz=args.f_min_hz,
        f_max_hz=args.f_max_hz, n_freq=args.n_freq,
        max_heldout_per_chart=args.max_heldout_per_chart)

    dropped_slivers = None
    if args.dropped_slivers_report is not None:
        with open(args.dropped_slivers_report) as stream:
            dropped_slivers = dropped_slivers_from_training_report(
                json.load(stream))

    report = run(surrogate_path=args.surrogate_path, config=config,
                 dropped_slivers=dropped_slivers)

    with open(args.output, 'w') as stream:
        json.dump(report, stream, indent=2)

    print(f'served {report["served"]}/{report["n_samples"]} '
          f'({report["served_fraction"]:.3f}); '
          f'artifact {report["artifact"]["size_bytes"]} bytes -> '
          f'{args.output}')


if __name__ == '__main__':
    main()
