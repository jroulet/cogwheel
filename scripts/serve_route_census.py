#!/usr/bin/env python
"""Thin CLI over `cogwheel.lensing.serve_route_census.run`.

Runs the engine-free serve-route DEMAND census (no artifact attached) and
writes its JSON report; with ``--with-artifact PATH`` it loads a trained
``lens_amplification_surrogate`` via the production
`LensAmplificationSurrogate.load` path and threads it through so the
``surrogate`` route becomes reachable (the 7b ACCEPTANCE census).  All
computation lives in the census module; this wrapper only parses arguments,
loads the artifact, calls `run`, prints the 8-label route breakdown and dumps
the JSON.  It performs NO amplitude-engine evaluation of its own -- both the
surrogate load and its `serve` path are engine-free (spline lookup, no
wave-optics amplitude).

Demand mode (no artifact) asserts that the ``surrogate`` route is empty: it is
unreachable without an attached artifact, so a non-zero count would signal a
classifier regression.
"""
import argparse
import json

from cogwheel.lensing.serve_route_census import ServeRouteCensusConfig, run
from cogwheel.lensing.surrogate import LensAmplificationSurrogate


def _print_breakdown(report: dict) -> None:
    """Print the 8-label route breakdown and the residual 3-way split."""
    n_samples = report['n_samples']
    print(f'serve-route census ({report["header"]["mode"]} mode): '
          f'{n_samples} draws')
    for route, count in report['route_counts'].items():
        pct = (100.0 * count / n_samples) if n_samples else 0.0
        print(f'  {route:24s} {count:7d}  {pct:6.2f}%')

    residual = report['residual_demand']
    print(f'residual_demand (engine_residual = {residual["total"]} draws, '
          f'split on {residual["split_gauge"]}):')
    for bucket in ('born_chart_demand', 'near_caustic_tube', 'interior',
                   'undetermined'):
        info = residual[bucket]
        print(f'  {bucket:20s} {info["count"]:7d}  '
              f'{100.0 * info["prior_mass_fraction"]:6.2f}%')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-samples', type=int, default=10_000,
                        help='Number of prior draws to classify.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Sampler seed.')
    parser.add_argument('--f-min-hz', type=float, default=20.0,
                        help='Low edge of the analysis frequency grid (Hz).')
    parser.add_argument('--f-max-hz', type=float, default=1024.0,
                        help='High edge of the analysis frequency grid (Hz).')
    parser.add_argument('--n-freq', type=int, default=128,
                        help='Number of geometric frequency nodes.')
    parser.add_argument('--with-artifact', default=None, metavar='PATH',
                        help='Attach a trained lens_amplification_surrogate '
                        '(engine-free load) to enable the surrogate route '
                        '(the 7b acceptance census). Default: demand mode.')
    parser.add_argument('--out', default='serve_route_census.json',
                        help='JSON report destination.')
    args = parser.parse_args()

    config = ServeRouteCensusConfig(
        n_samples=args.n_samples, seed=args.seed, f_min_hz=args.f_min_hz,
        f_max_hz=args.f_max_hz, n_freq=args.n_freq)

    artifact = (LensAmplificationSurrogate.load(args.with_artifact)
                if args.with_artifact is not None else None)

    report = run(config=config, artifact=artifact)

    if artifact is None and report['route_counts']['surrogate'] != 0:
        raise AssertionError(
            'demand mode must emit zero surrogate-route draws; got '
            f'{report["route_counts"]["surrogate"]}')

    with open(args.out, 'w') as stream:
        json.dump(report, stream, indent=2)

    _print_breakdown(report)
    print(f'-> {args.out}')


if __name__ == '__main__':
    main()
