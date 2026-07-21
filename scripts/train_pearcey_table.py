#!/usr/bin/env python
"""Regenerate the universal Pearcey ``P(x, y)`` spline-table artifact.

Derives the box edges at build time (marching the asymptotic-vs-quadrature
handoff on a ray fan), samples the certified quadrature on a graded grid,
demodulates the cusp Fresnel carrier, fits the bicubic splines, then
CERTIFIES the table against held-out quadrature points before writing it.
The artifact is a plain-array ``.npz`` with a JSON provenance scalar
carrying the box edges and a SHA1 content hash (see
`cogwheel.lensing.chang_refsdal._pearcey_table`).

All numerical logic lives in the table module; this wrapper only parses
arguments, runs the certification gate, and writes the artifact.  Defaults
run at a modest resolution; raise ``--n-x`` / ``--n-y`` for the shipped
artifact.
"""
import argparse

from cogwheel.lensing.chang_refsdal._pearcey_table import (
    build_table, derive_box, held_out_error, save_table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',
                        help='Destination .npz path for the table artifact.')
    parser.add_argument('--oracle-tol', type=float, default=1e-8,
                        help='Absolute asymptotic-vs-quadrature handoff and '
                        'held-out certification tolerance (default 1e-8).')
    parser.add_argument('--margin', type=float, default=0.15,
                        help='Overlap-margin inflation on the derived box '
                        '(default 0.15).')
    parser.add_argument('--n-x', type=int, default=161,
                        help='Grid points along the x (along-cusp) axis.')
    parser.add_argument('--n-y', type=int, default=161,
                        help='Grid points along the y (transverse) axis.')
    parser.add_argument('--grading-power', type=float, default=1.6,
                        help='Knot-grading exponent (>1 clusters near the '
                        'caustic; default 1.6).')
    parser.add_argument('--held-out-samples', type=int, default=4000,
                        help='Number of held-out certification points.')
    parser.add_argument('--held-out-seed', type=int, default=0,
                        help='RNG seed for the held-out points.')
    args = parser.parse_args()

    box = derive_box(oracle_tol=args.oracle_tol, margin=args.margin)
    print(f'Derived box: x_max={box["x_max"]:.4g}, y_max={box["y_max"]:.4g} '
          f'(handoff {box["x_handoff"]:.4g} x {box["y_handoff"]:.4g}, '
          f'margin {box["margin"]:.2f}).')

    table = build_table(box, n_x=args.n_x, n_y=args.n_y,
                        grading_power=args.grading_power)

    worst = held_out_error(table, n_samples=args.held_out_samples,
                           seed=args.held_out_seed)
    print(f'Held-out max abs error: {worst:.3e} '
          f'(tolerance {args.oracle_tol:.1e}).')
    if not worst < args.oracle_tol:
        raise SystemExit(
            f'Refusing to ship: held-out error {worst:.3e} does not clear '
            f'the {args.oracle_tol:.1e} tolerance. Increase --n-x/--n-y or '
            f'shrink the box.')

    save_table(table, args.output)
    print(f'Wrote {args.n_x} x {args.n_y} table '
          f'(hash {table.provenance["content_hash"][:12]}...) -> '
          f'{args.output}')


if __name__ == '__main__':
    main()
