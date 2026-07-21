#!/usr/bin/env python
"""Geometry-only Monte-Carlo census of two Build-8d homogenization corners.

WHAT
----
Draws fixture-scale samples from the reduced Chang-Refsdal lens prior box (in
SAMPLED coordinates, no importance weights) and REPORTS two prior-box
fractions the owner needs to scope Build 8e:

(a) The shear-free ``gamma' == 0`` hit fraction -- expected ~0 (measure-zero
    for ``gamma ~ U(0, 1.6)``) -- plus the near-boundary ``gamma' < 0.01``
    fraction as a bonus.  ``gamma' = gamma / (1 - kappa)`` is the reduced shear
    the wave evaluator consumes; the Schwinger path requires ``gamma' > 0`` and
    the legacy operator path is its only serving route at ``gamma' == 0``.

(b) The UNRESOLVED-high-w NAMED-refusal corner fraction: over the frequency
    grid the likelihood uses, each ``(config, w)`` is classified as
    served-by-Schwinger (``w <= 60``), served-by-geometric (resolved:
    ``w*delta_min >= 4.0`` and, on positive parity, ``L > 48``), or a NAMED
    refusal (``w > 60`` and not geometric-eligible).  A config lands in the
    refusal corner if ANY grid node is a named refusal.  Reported with a Wilson
    95% score interval.

WHY
---
This is a REPORTING deliverable, not a gate: it quantifies (a) how often the
documented ``gamma' == 0`` legacy-path exception is actually reached and (b)
how much prior-box volume the unresolved-high-w corner (owned by the Build-8e
cusp fast-serving build) covers, so 8e's value is measurable.  It never runs a
full amplification evaluation: parity, reduced shear, resolution, and the
branch gates are all cheap geometric predicates read from the ENGINE itself
(`operator.select_branch`, the mass-sheet maps, the real-image delay
separation), so the reported classification mirrors production dispatch exactly
without touching engine code.

Separation of concerns: every function here is pure computation over plain
arrays; only `main` reads argv, prints, and writes the JSON artifact.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from cogwheel.lensing.chang_refsdal import _schwinger, geometry, operator
from cogwheel.lensing.prior import (FixedLensGeometryPrior,
                                    UniformLensMassPrior,
                                    UniformReducedShearPrior,
                                    UniformSourcePositionPrior)
from cogwheel.lensing.waveform import dimensionless_frequency
from cogwheel.prior import CombinedPrior

# Production dispatch constants, imported (never hardcoded) so the census
# tracks the engine if a threshold ever moves.
W_CEILING = _schwinger.W_CEILING_SCHWINGER  # Schwinger hard ceiling (60.0).
RHO_END = operator.RHO_END                  # Resolution onset (4.0).
L_MAX = operator.L_MAX                       # Cancellation-exponent ceiling (48).
ENGINE_W_CEILING = 500.0                     # Certified frequency ceiling (SPEC).
GAMMA_PRIME_NEAR_ZERO = 0.01                 # Near-boundary bonus threshold.


class _LensPriorBox(CombinedPrior):
    """Lens-only combined prior for drawing sampled-coordinate points.

    Composes exactly the four reduced Chang-Refsdal lens subpriors (no CBC
    subpriors), so its ``.generate_random_samples`` draws ``gamma``,
    ``m_lens_msun`` and the shear-frame source ``(y1, y2)`` from the SAME
    sampled->standard machinery the production `LensedIASPrior` uses -- the
    prior box is READ from these classes, never hardcoded here.
    ``UniformLensMassPrior`` precedes ``UniformSourcePositionPrior`` because the
    latter is conditioned on ``m_lens_msun``.
    """
    prior_classes = [FixedLensGeometryPrior, UniformLensMassPrior,
                     UniformReducedShearPrior, UniformSourcePositionPrior]


@dataclass(frozen=True)
class CensusConfig:
    """Settings for a homogenization-corner census run."""
    n_samples: int = 200_000
    seed: int = 0
    f_min_hz: float = 20.0
    f_max_hz: float = 1024.0
    n_freq: int = 128


@dataclass(frozen=True)
class ConfigOutcome:
    """Per-config geometry-only classification outcome.

    ``gamma_prime_engine`` is ``nan`` when the geometry itself refuses the
    config (``engine_refused``), in which case no refusal-corner or served
    tallies are meaningful and all are zero.
    """
    engine_refused: bool
    refusal_corner: bool
    gamma_prime_engine: float
    n_schwinger: int
    n_geometric: int
    n_refusal: int
    max_w: float
    w_exceeds_engine: bool


def draw_samples(config: CensusConfig) -> pd.DataFrame:
    """Draw ``config.n_samples`` lens prior samples in sampled coordinates.

    Parameters
    ----------
    config : CensusConfig
        Census settings (``n_samples`` and ``seed`` are read).

    Returns
    -------
    pandas.DataFrame
        One row per sample; columns are the prior's sampled + standard params
        (``gamma``, ``m_lens_msun``, ``y1``, ``y2`` and the fixed geometry).
    """
    return _LensPriorBox().generate_random_samples(config.n_samples,
                                                    seed=config.seed)


def _reduce_shear(y: np.ndarray, gamma: float, kappa: float
                  ) -> tuple[float, np.ndarray, float]:
    """Dispatch to the engine mass-sheet map matching the config's parity.

    Positive parity (``|gamma| < 1 - kappa``) uses `operator._mass_sheet_map`;
    the macro saddle (``|gamma| > 1 - kappa``) uses
    `operator._saddle_mass_sheet_map`.  The exact parity boundary
    (``|gamma| == 1 - kappa``) is rejected upstream by `geometry.macro_matrix`,
    so it never reaches here.

    Returns
    -------
    lam, y_scaled, gamma_prime : tuple[float, np.ndarray, float]
        As returned by the selected engine map.
    """
    lam = 1.0 - float(kappa)
    if abs(float(gamma)) < lam:
        return operator._mass_sheet_map(y, gamma, kappa)
    return operator._saddle_mass_sheet_map(y, gamma, kappa)


def classify_config(gamma: float, m_lens_msun: float, y: np.ndarray,
                    kappa: float, beta: float, z_lens: float,
                    f_grid_hz: np.ndarray) -> ConfigOutcome:
    """Classify one lens config against the production branch gates.

    Builds the config's ``w`` grid, determines parity and reduced shear from
    the engine maps, and labels every ``w`` node as served-by-Schwinger
    (``w <= W_CEILING``), served-by-geometric (resolved -- positive parity also
    requires ``L > L_MAX``), or a named refusal (``w > W_CEILING`` and not
    geometric-eligible).  The expensive real-image delay separation is computed
    at most ONCE per config, and only when some node exceeds the ceiling (the
    only regime in which the geometric branch or a refusal can occur).

    Any `geometry.LensDomainError` raised while building the macro matrix,
    reducing the shear, or solving for images (the F012 near-axial census
    refusal) marks the config ``engine_refused`` -- a geometry refusal that
    precedes any wave/geometric dispatch, reported in its own bucket.

    Parameters
    ----------
    gamma, m_lens_msun : float
        Reduced shear magnitude and redshifted lens mass (solar masses).
    y : np.ndarray
        Shape ``(2,)`` physical-frame source position.
    kappa, beta, z_lens : float
        Fixed lens geometry (convergence, shear orientation, lens redshift).
    f_grid_hz : np.ndarray
        Frequency grid (Hz) the likelihood evaluates on.

    Returns
    -------
    ConfigOutcome
        Per-config classification tallies.
    """
    w_grid = np.asarray(
        dimensionless_frequency(f_grid_hz, m_lens_msun, z_lens), dtype=float)
    max_w = float(w_grid.max())
    w_exceeds_engine = bool(max_w > ENGINE_W_CEILING)

    try:
        matrix = geometry.macro_matrix(gamma, beta, kappa)
        lam, y_scaled, gamma_prime = _reduce_shear(y, gamma, kappa)
        abs_yprime = float(np.linalg.norm(y_scaled))
        is_saddle = abs(float(gamma)) > lam

        high = w_grid > W_CEILING
        delta_min = 0.0
        if np.any(high):
            # delta_min is frequency-independent; solve for images once.
            delta_min = operator._real_delay_min_separation(
                np.asarray(y, dtype=float), matrix)
    except geometry.LensDomainError:
        return ConfigOutcome(
            engine_refused=True, refusal_corner=False,
            gamma_prime_engine=float('nan'), n_schwinger=0, n_geometric=0,
            n_refusal=0, max_w=max_w, w_exceeds_engine=w_exceeds_engine)

    resolved = w_grid * delta_min >= RHO_END  # w-independent resolution mask
    if is_saddle:
        # Saddle geometric gate (operator._saddle_grid): resolved AND above the
        # Schwinger ceiling; no separate cancellation-exponent condition.
        geometric = high & resolved
    else:
        # Positive-parity gate (operator.select_branch): resolved AND the
        # cancellation exponent L = w*|y'| exceeds L_MAX. select_branch is
        # w-INDEPENDENT: a resolved, strongly-cancelling node below the
        # Schwinger ceiling is still served geometrically in production
        # (INS-1-003: the earlier `high &` masking here undercounted
        # served_by_geometric in the owner-facing split).
        geometric = resolved & (w_grid * abs_yprime > L_MAX)
    refusal = high & ~geometric
    schwinger = ~high & ~geometric  # Schwinger serves the non-geometric band.

    return ConfigOutcome(
        engine_refused=False,
        refusal_corner=bool(np.any(refusal)),
        gamma_prime_engine=float(gamma_prime),
        n_schwinger=int(np.count_nonzero(schwinger)),
        n_geometric=int(np.count_nonzero(geometric)),
        n_refusal=int(np.count_nonzero(refusal)),
        max_w=max_w, w_exceeds_engine=w_exceeds_engine)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion.

    Preferred over the normal approximation for small ``k`` (the near-zero and
    refusal-corner fractions), where it stays inside ``[0, 1]`` and does not
    collapse to a zero-width interval at ``k = 0``.

    Parameters
    ----------
    k : int
        Number of successes.
    n : int
        Number of trials.
    z : float, optional
        Standard-normal quantile (``1.96`` -> 95%).

    Returns
    -------
    tuple[float, float]
        ``(lower, upper)`` bounds, clipped to ``[0, 1]``.
    """
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def run(config: CensusConfig) -> dict:
    """Run the census and return a JSON-serializable report dict.

    Parameters
    ----------
    config : CensusConfig
        Census settings.

    Returns
    -------
    dict
        The full numeric report (see module docstring for the two headline
        fractions).
    """
    frame = draw_samples(config)
    f_grid = np.logspace(math.log10(config.f_min_hz),
                         math.log10(config.f_max_hz), config.n_freq)

    def _column(name: str, default: float) -> np.ndarray:
        if name in frame.columns:
            return frame[name].to_numpy(dtype=float)
        return np.full(len(frame), default, dtype=float)

    gamma = frame['gamma'].to_numpy(dtype=float)
    m_lens = frame['m_lens_msun'].to_numpy(dtype=float)
    y1 = frame['y1'].to_numpy(dtype=float)
    y2 = frame['y2'].to_numpy(dtype=float)
    kappa = _column('kappa', 0.0)
    beta = _column('beta', 0.0)
    z_lens = _column('z_lens', 0.0)

    # (a) gamma' = gamma / (1 - kappa) -- the analytic definition, cross-checked
    # per-config against the engine mass-sheet map below.
    gamma_prime = gamma / (1.0 - kappa)

    n = len(frame)
    n_refusal_corner = 0
    n_engine_refused = 0
    n_w_exceeds_engine = 0
    total_schwinger = 0
    total_geometric = 0
    total_refusal = 0
    max_w_seen = 0.0
    max_gamma_prime_diff = 0.0

    for i in range(n):
        outcome = classify_config(
            gamma[i], m_lens[i], np.array([y1[i], y2[i]]),
            kappa[i], beta[i], z_lens[i], f_grid)
        if outcome.engine_refused:
            n_engine_refused += 1
        else:
            n_refusal_corner += int(outcome.refusal_corner)
            total_schwinger += outcome.n_schwinger
            total_geometric += outcome.n_geometric
            total_refusal += outcome.n_refusal
            diff = abs(outcome.gamma_prime_engine - gamma_prime[i])
            max_gamma_prime_diff = max(max_gamma_prime_diff, diff)
        n_w_exceeds_engine += int(outcome.w_exceeds_engine)
        max_w_seen = max(max_w_seen, outcome.max_w)

    n_classifiable = n - n_engine_refused
    n_gamma_prime_zero = int(np.count_nonzero(gamma_prime == 0.0))
    n_gamma_prime_near_zero = int(
        np.count_nonzero(gamma_prime < GAMMA_PRIME_NEAR_ZERO))

    corner_lo, corner_hi = wilson_interval(n_refusal_corner, n)
    corner_lo_cls, corner_hi_cls = wilson_interval(
        n_refusal_corner, n_classifiable)

    return {
        'config': asdict(config),
        'n_samples': n,
        'gamma_prime_zero': {
            'count': n_gamma_prime_zero,
            'fraction': n_gamma_prime_zero / n if n else 0.0,
            'note': 'measure-zero for gamma ~ U(0, 1.6); expected ~0',
        },
        'gamma_prime_near_zero': {
            'threshold': GAMMA_PRIME_NEAR_ZERO,
            'count': n_gamma_prime_near_zero,
            'fraction': n_gamma_prime_near_zero / n if n else 0.0,
            'analytic_expectation': GAMMA_PRIME_NEAR_ZERO / 1.6,
        },
        'unresolved_high_w_refusal_corner': {
            'count': n_refusal_corner,
            'fraction_of_prior_box': n_refusal_corner / n if n else 0.0,
            'wilson95_of_prior_box': [corner_lo, corner_hi],
            'fraction_of_classifiable': (
                n_refusal_corner / n_classifiable if n_classifiable else 0.0),
            'wilson95_of_classifiable': [corner_lo_cls, corner_hi_cls],
            'definition': ('any frequency node with w > %.1f that is not '
                           'geometric-eligible (a NAMED SchwingerCertification'
                           'Error refusal under the Build-8d homogenization; '
                           'the Build-8e uniform-asymptotics build owns '
                           'serving this corner)' % W_CEILING),
        },
        'engine_refused': {
            'count': n_engine_refused,
            'fraction': n_engine_refused / n if n else 0.0,
            'note': ('geometry refusals (macro-matrix parity boundary, Type '
                     'III, or F012 near-axial census) preceding wave/geometric '
                     'dispatch'),
        },
        'node_classification_totals': {
            'served_by_schwinger': total_schwinger,
            'served_by_geometric': total_geometric,
            'named_refusal': total_refusal,
            'n_nodes_per_config': config.n_freq,
            'note': ('the schwinger/geometric split mirrors production '
                     'operator.select_branch exactly: the positive-parity '
                     'geometric gate (resolved AND L > L_MAX) is '
                     'w-independent, so strongly-cancelling resolved nodes '
                     'below w = %.1f count as served_by_geometric; the '
                     'saddle geometric gate additionally requires the node '
                     'above the Schwinger ceiling.' % W_CEILING),
        },
        'w_range': {
            'max_w_seen': max_w_seen,
            'engine_ceiling': ENGINE_W_CEILING,
            'n_configs_exceeding_engine_ceiling': n_w_exceeds_engine,
        },
        'gamma_prime_cross_check': {
            'max_abs_diff_engine_vs_analytic': max_gamma_prime_diff,
            'note': ('engine mass-sheet map gamma_prime vs analytic '
                     'gamma/(1-kappa); should agree to ~machine precision'),
        },
        'thresholds': {
            'w_ceiling_schwinger': W_CEILING,
            'rho_end': RHO_END,
            'l_max': L_MAX,
        },
    }


def _print_report(report: dict) -> None:
    """Print a concise human-readable summary of the census report."""
    n = report['n_samples']
    gz = report['gamma_prime_zero']
    gnz = report['gamma_prime_near_zero']
    corner = report['unresolved_high_w_refusal_corner']
    refused = report['engine_refused']
    print(f'geometry-only homogenization-corner census over {n} prior draws')
    print(f"  gamma'==0            : {gz['count']} "
          f"({gz['fraction']:.3e})  [expected ~0]")
    print(f"  gamma'<{gnz['threshold']:g}            : {gnz['count']} "
          f"({gnz['fraction']:.4f})  [analytic {gnz['analytic_expectation']:.4f}]")
    lo, hi = corner['wilson95_of_prior_box']
    print(f"  unresolved-high-w    : {corner['count']} "
          f"({corner['fraction_of_prior_box']:.4f})  "
          f"Wilson95 [{lo:.4f}, {hi:.4f}]")
    print(f"  engine-refused       : {refused['count']} "
          f"({refused['fraction']:.4e})")
    print(f"  max w seen           : {report['w_range']['max_w_seen']:.1f} "
          f"(engine ceiling {report['w_range']['engine_ceiling']:.0f})")
    print(f"  gamma' cross-check   : max |engine-analytic| = "
          f"{report['gamma_prime_cross_check']['max_abs_diff_engine_vs_analytic']:.2e}")


def main() -> None:
    """Parse arguments, run the census, print a summary, write the JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-samples', type=int, default=CensusConfig.n_samples)
    parser.add_argument('--seed', type=int, default=CensusConfig.seed)
    parser.add_argument('--f-min-hz', type=float, default=CensusConfig.f_min_hz)
    parser.add_argument('--f-max-hz', type=float, default=CensusConfig.f_max_hz)
    parser.add_argument('--n-freq', type=int, default=CensusConfig.n_freq)
    parser.add_argument('--output', default='homogenization_corners_census.json',
                        help='JSON report destination.')
    args = parser.parse_args()

    config = CensusConfig(
        n_samples=args.n_samples, seed=args.seed, f_min_hz=args.f_min_hz,
        f_max_hz=args.f_max_hz, n_freq=args.n_freq)

    report = run(config)

    with open(args.output, 'w') as stream:
        json.dump(report, stream, indent=2)

    _print_report(report)
    print(f'-> wrote {args.output}')


if __name__ == '__main__':
    main()
