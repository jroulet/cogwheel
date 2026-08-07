#!/usr/bin/env python
"""Thin CLI over `cogwheel.lensing.surrogate_training.train`.

Builds the multi-chart lensing-amplification surrogate artifact from the prior
box and writes a JSON training report. All logic lives in the training module;
this wrapper only parses arguments. Defaults run at smoke scale.
"""
import argparse

from cogwheel.lensing.surrogate_training import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('outdir', help='Directory for per-chart files, the '
                        'packed artifact, and the JSON report.')
    parser.add_argument('--artifact-path', default=None,
                        help='Packed-artifact destination (default: '
                        'outdir/lens_amplification_surrogate.npz).')
    parser.add_argument('--report-path', default=None,
                        help='JSON report destination (default: '
                        'outdir/training_report.json).')
    parser.add_argument('--engine-budget', type=int, default=None,
                        help='Per-chart engine-call budget (raises the '
                        'smoke-config default when set).')
    parser.add_argument('--regions', nargs='*',
                        choices=['tube', 'exterior', 'wedge_interior',
                                 'lobe_interior'], default=None,
                        help='Regions to train (default: all). '
                        'Example: --regions wedge_interior')
    args = parser.parse_args()

    from cogwheel.lensing.surrogate_training import TrainingConfig
    config = (TrainingConfig(engine_budget=args.engine_budget)
              if args.engine_budget is not None else None)
    report_path = args.report_path or f'{args.outdir}/training_report.json'
    regions = (tuple(args.regions)
               if args.regions is not None else None)
    _, report = train(outdir=args.outdir, artifact_path=args.artifact_path,
                      config=config, report_path=report_path,
                      regions=regions)
    print(f'Wrote {report["artifact"]["n_charts"]} charts, '
          f'{report["artifact"]["size_bytes"]} bytes -> '
          f'{report["artifact"]["path"]}')


if __name__ == '__main__':
    main()
