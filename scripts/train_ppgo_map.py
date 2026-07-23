#!/usr/bin/env python
"""Regenerate the certified-ppGO frequency-floor map artifact.

Runs the offline validation sweep: for each
``(parity, gamma-band, caustic-frame annulus)`` cell it places a
representative source, measures the F-normalized error
``|F - ppGO_full| / max|F|`` of the bare point-mass geometric-optics
reconstruction (`geometric_amplification`) against the exact engine total
over a log-spaced ``w`` grid below the Schwinger wall, and records the
**sup-over-w certified floor** at the 1e-4 bar (the smallest ``w`` above
the last upward re-crossing -- the ppGO error is non-monotone).  Cells
whose error never clears the bar below the wall are marked beyond-wall
(UNKNOWN); parity-invalid gamma bands are marked invalid.  The result is a
plain-array ``.npz`` with a JSON provenance scalar carrying the grid axes,
the certification bar, the safety-margin rule, the walls and a SHA1
content hash (see `cogwheel.lensing.ppgo_map`).

All numerical logic lives in the map module; this wrapper only parses
arguments, runs the sweep, prints the cell-status summary, and writes the
artifact.  The defaults run a COARSE SYNTHETIC sweep (small grid, reduced
walls) suitable for the in-build acceptance; the shipped production map is
a post-build driver step run with the true walls (443.7 / 58) and the full
default band edges (pass ``--production``).
"""
import argparse

import numpy as np

from cogwheel.lensing.ppgo_map import (
    ASTROID_WALL, SADDLE_WALL, build_map, map_summary, save_map)

#: Coarse synthetic band edges for the fast in-build sweep (below the wall).
_COARSE_GAMMA_EDGES = (0.05, 0.3, 0.6, 1.0, 1.3)
_COARSE_RHO_EDGES = (0.0, 0.9, 1.0, 2.5, np.inf)

#: Reduced walls for the coarse sweep (well within the engine ceilings).
_COARSE_ASTROID_WALL = 40.0
_COARSE_SADDLE_WALL = 30.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',
                        help='Destination .npz path for the map artifact.')
    parser.add_argument('--kappa', type=float, default=0.0,
                        help='External convergence (shipped model pins '
                        'kappa=0; the driver may sweep it).')
    parser.add_argument('--production', action='store_true',
                        help='Use the full default band edges and the true '
                        'Schwinger walls instead of the coarse synthetic '
                        'grid (offline driver step).')
    parser.add_argument('--astroid-wall', type=float, default=None,
                        help='Positive-parity Schwinger wall (default: '
                        f'{_COARSE_ASTROID_WALL} coarse / {ASTROID_WALL} '
                        'production).')
    parser.add_argument('--saddle-wall', type=float, default=None,
                        help='Macro-saddle Schwinger wall (default: '
                        f'{_COARSE_SADDLE_WALL} coarse / {SADDLE_WALL} '
                        'production).')
    args = parser.parse_args()

    if args.production:
        gamma_edges = None
        rho_edges = None
        astroid_wall = (ASTROID_WALL if args.astroid_wall is None
                        else args.astroid_wall)
        saddle_wall = (SADDLE_WALL if args.saddle_wall is None
                       else args.saddle_wall)
    else:
        gamma_edges = _COARSE_GAMMA_EDGES
        rho_edges = _COARSE_RHO_EDGES
        astroid_wall = (_COARSE_ASTROID_WALL if args.astroid_wall is None
                        else args.astroid_wall)
        saddle_wall = (_COARSE_SADDLE_WALL if args.saddle_wall is None
                       else args.saddle_wall)

    print(f'Building certified-ppGO map (kappa={args.kappa}, walls: '
          f'astroid={astroid_wall}, saddle={saddle_wall}, '
          f'{"production" if args.production else "coarse synthetic"}).')

    ppgo_map = build_map(kappa=args.kappa, astroid_wall=astroid_wall,
                         saddle_wall=saddle_wall, gamma_edges=gamma_edges,
                         rho_edges=rho_edges)

    summary = map_summary(ppgo_map)
    print(f'Cells: {summary["n_cells"]} total -> '
          f'{summary["n_certified"]} certified, '
          f'{summary["n_beyond_wall"]} beyond-wall, '
          f'{summary["n_invalid"]} invalid '
          f'({summary["n_interpolable"]} interpolable).')

    save_map(ppgo_map, args.output)
    print(f'Wrote certified-ppGO map '
          f'(hash {ppgo_map.provenance["content_hash"][:12]}...) -> '
          f'{args.output}')


if __name__ == '__main__':
    main()
