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

# --- Build-8e corner-scoping constants (MEASURED only; never acted on) ------

#: Relaxed cancellation-exponent trust ceiling used ONLY to measure category
#: (b).  Production fires the geometric branch at ``L > L_MAX = 48``, a
#: conservative margin below the wave kernel's certified ceiling of 60 (the
#: operator module docstring: "just below the kernel ceiling of 60").  Category
#: (b) counts -- WITHOUT touching production -- how many refused high-``w`` nodes
#: are resolved fold configs that a relaxed gate (trusting the branch out to the
#: kernel ceiling) plus an image-census check would move onto the geometric
#: branch.  `operator.select_branch` and `operator.L_MAX` are NOT modified.
L_MAX_RELAXED = 60.0

#: Topology heuristic, NOT the resolvable/hard-core threshold: a third real
#: image whose Fermat delay lies within this multiple of the merging pair's
#: delay separation marks a three-image coalescence (cusp-type); otherwise the
#: refusal is a two-image fold.  Reads the local Morse structure (the delays are
#: the Morse critical values), not any diffraction scale.
CUSP_CLUSTER_DELAY_RATIO = 3.0

#: Source-plane cusp 2/3-law scalings (the semicubical caustic
#: ``27 y**2 = -8 x**3``): the along-cusp coordinate scales as ``w**(1/2)``, the
#: transverse coordinate as ``w**(3/4)``.
CUSP_PARALLEL_W_EXPONENT = 0.5
CUSP_PERP_W_EXPONENT = 0.75

#: Fixed candidate-threshold grids for the fraction-vs-threshold tables.  Fixed
#: (not data-adaptive) so the CDFs are reproducible and comparable across runs;
#: the resolvable/hard-core split is read off at the arms' certified thresholds
#: post-build (the thresholds xi*/R* are normalization-dependent, not guessable
#: here).  The fold grid is the raw ``w * Delta_tau`` argument; the cusp grid is
#: the radial ``R`` argument, log-spaced to span its wide dynamic range.
FOLD_WDTAU_THRESHOLD_GRID = np.round(np.linspace(0.0, 8.0, 33), 6)
CUSP_R_THRESHOLD_GRID = np.round(
    np.concatenate(([0.0], np.logspace(-1.0, 2.0, 31))), 6)


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

    The Build-8e corner-scoping fields partition the config's HIGH-``w`` nodes
    (``w > W_CEILING``): ``n_highw == n_geometric_highw + n_refusal``, and the
    refused high-``w`` nodes split into category (b) (``n_relaxed_rescue``) and
    the residual (c)/(d) population whose fold/cusp arguments are carried in the
    ``cd_*`` arrays (routed by ``topology``, exactly one bucket per config).
    """
    engine_refused: bool
    refusal_corner: bool
    gamma_prime_engine: float
    n_schwinger: int
    n_geometric: int
    n_refusal: int
    max_w: float
    w_exceeds_engine: bool
    # --- Build-8e corner-scoping fields ---
    n_highw: int
    n_geometric_highw: int
    n_relaxed_rescue: int
    topology: str
    cd_fold_wdtau: np.ndarray
    cd_fold_xi: np.ndarray
    cd_cusp_R: np.ndarray
    n_cd_degenerate: int


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


def _longest_true_run(mask: np.ndarray) -> int:
    """Length of the longest run of consecutive ``True`` values in ``mask``."""
    best = run = 0
    for flag in mask:
        run = run + 1 if bool(flag) else 0
        best = max(best, run)
    return best


def _merging_pair_indices(delays: np.ndarray) -> tuple[int, int]:
    """Indices of the two images with the smallest Fermat-delay separation.

    This is the pair that realizes ``delta_min`` -- the merging minimum/saddle
    pair whose half separation is the fold argument's ``Delta_tau``.
    """
    diffs = np.abs(delays[:, None] - delays[None, :])
    upper = np.triu_indices(delays.size, k=1)
    flat = int(np.argmin(diffs[upper]))
    return int(upper[0][flat]), int(upper[1][flat])


def _classify_fold_or_cusp(delays: np.ndarray) -> str:
    """Fold- vs cusp-type topology from the real-image delay spectrum.

    The Fermat delays are the Morse critical values.  Two images (or a tight
    pair well separated from the rest) is a two-image fold; a cluster of three
    or more images whose consecutive delay gaps are all within
    `CUSP_CLUSTER_DELAY_RATIO` of the smallest gap is a three-image
    coalescence (cusp / near-cusp).

    Parameters
    ----------
    delays : np.ndarray
        Fermat delays of the real images (any order).

    Returns
    -------
    str
        ``'fold'``, ``'cusp'`` or ``'degenerate'`` (fewer than two images).
    """
    if delays.size < 2:
        return 'degenerate'
    if delays.size == 2:
        return 'fold'
    ordered = np.sort(delays)
    gaps = np.diff(ordered)
    gmin = float(gaps.min())
    near = gaps <= 0.0 if gmin <= 0.0 else gaps <= CUSP_CLUSTER_DELAY_RATIO * gmin
    cluster_size = _longest_true_run(near) + 1
    return 'cusp' if cluster_size >= 3 else 'fold'


def _image_census_matches(mags: np.ndarray, pair: tuple[int, int]) -> bool:
    """Image-count-match guard for the category-(b) measurement.

    The resolved geometric (stationary-phase) branch is trustworthy only when
    the real-image census is the expected one for a fold: an even number of
    images (Chang-Refsdal admits 2 or 4) whose merging pair is one minimum and
    one saddle.  The image parity is read from the SIGN of the signed
    magnification (`geometry.magnification`): a minimum/maximum is positive, a
    saddle negative.

    Parameters
    ----------
    mags : np.ndarray
        Signed magnifications of the real images.
    pair : tuple[int, int]
        Indices of the merging (smallest-delay-separation) pair.

    Returns
    -------
    bool
        ``True`` when the census matches (even count, merging pair is one
        positive-parity and one negative-parity image).
    """
    if mags.size not in (2, 4):
        return False
    i, j = pair
    return (mags[i] > 0.0) != (mags[j] > 0.0)


def _fold_arguments(refused_w: np.ndarray, delta_min: float
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Fold-catastrophe argument for each refused node.

    ``Delta_tau = (tau_minus - tau_plus) / 2 = delta_min / 2`` is the half
    Fermat-delay separation of the merging minimum/saddle pair (the pair that
    realizes ``delta_min``); the fold argument is
    ``xi = (3 * w * Delta_tau / 4)**(2/3)``, reported alongside the raw
    ``w * Delta_tau``.

    Parameters
    ----------
    refused_w : np.ndarray
        Dimensionless frequencies of the refused nodes (``w >= 0``).
    delta_min : float
        Smallest pairwise real-image Fermat-delay separation.

    Returns
    -------
    w_delta_tau, xi : tuple[np.ndarray, np.ndarray]
        The raw ``w * Delta_tau`` argument and the fold argument ``xi``.
    """
    w_delta_tau = refused_w * (0.5 * float(delta_min))
    xi = np.cbrt(0.75 * w_delta_tau) ** 2  # (3/4 * w*Dtau)**(2/3); arg >= 0.
    return w_delta_tau, xi


def _cusp_arguments(refused_w: np.ndarray, delta_parallel: float,
                    delta_perp: float) -> np.ndarray:
    """Cusp (Pearcey) radial argument ``R`` for each refused node.

    ``R = sqrt(x**2 + y**2)`` with ``x = w**(1/2) * delta_parallel`` (along the
    cusp tangent / soft axis) and ``y = w**(3/4) * delta_perp`` (along the cusp
    normal / hard axis), the 2/3-law scalings of the semicubical caustic
    ``27 y**2 = -8 x**3``.  Proportionality constants are set to one: they are
    normalization-dependent and are read off at the arm's certified threshold.

    Parameters
    ----------
    refused_w : np.ndarray
        Dimensionless frequencies of the refused nodes (``w >= 0``).
    delta_parallel, delta_perp : float
        Source-plane offsets from the nearest caustic point along the cusp
        tangent (soft axis) and normal (hard axis).

    Returns
    -------
    np.ndarray
        The radial cusp argument ``R`` per node.
    """
    x = refused_w ** CUSP_PARALLEL_W_EXPONENT * abs(float(delta_parallel))
    y = refused_w ** CUSP_PERP_W_EXPONENT * abs(float(delta_perp))
    return np.sqrt(x * x + y * y)


def classify_config(gamma: float, m_lens_msun: float, y: np.ndarray,
                    kappa: float, beta: float, z_lens: float,
                    f_grid_hz: np.ndarray) -> ConfigOutcome:
    """Classify one lens config against the production branch gates.

    Builds the config's ``w`` grid, determines parity and reduced shear from
    the engine maps, and labels every ``w`` node as served-by-Schwinger
    (``w <= W_CEILING``), served-by-geometric (resolved -- positive parity also
    requires ``L > L_MAX``), or a named refusal (``w > W_CEILING`` and not
    geometric-eligible).  The refused HIGH-``w`` nodes are then scoped for
    Build 8e: category (a) is the geometric-served high-``w`` nodes; category
    (b) MEASURES how many refused nodes a relaxed cancellation ceiling
    (`L_MAX_RELAXED`) plus an image-census guard would move onto the geometric
    branch (production is untouched); the residual (c)/(d) refused nodes carry
    their fold (``w * Delta_tau``, ``xi``) or cusp (``R``) geometry-only
    arguments, routed by the config's `topology`.

    The single real-image solve (`geometry.find_images`) is shared by the
    resolution gate, the fold argument, the topology and the census guard; the
    nearest-caustic solve runs only for cusp-type refusal configs.  Any
    `geometry.LensDomainError` (macro-matrix parity boundary, Type III, or the
    F012 near-axial census) marks the config ``engine_refused`` -- a geometry
    refusal preceding any wave/geometric dispatch, reported in its own bucket.

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
        Per-config classification tallies and corner-scoping arguments.
    """
    w_grid = np.asarray(
        dimensionless_frequency(f_grid_hz, m_lens_msun, z_lens), dtype=float)
    max_w = float(w_grid.max())
    w_exceeds_engine = bool(max_w > ENGINE_W_CEILING)
    empty = np.empty(0)
    source = np.asarray(y, dtype=float)

    images: list[np.ndarray] = []
    delays_all: np.ndarray | None = None
    try:
        matrix = geometry.macro_matrix(gamma, beta, kappa)
        lam, y_scaled, gamma_prime = _reduce_shear(y, gamma, kappa)
        abs_yprime = float(np.linalg.norm(y_scaled))
        is_saddle = abs(float(gamma)) > lam

        high = w_grid > W_CEILING
        delta_min = 0.0
        if np.any(high):
            # One image solve, reused for the gate (delta_min), the fold
            # argument, the topology and the census guard.  This inline
            # delta_min mirrors operator._real_delay_min_separation exactly
            # (real images only; 0.0 when fewer than two real images).
            images = geometry.find_images(source, matrix)
            if len(images) >= 2:
                delays_all = np.array(
                    [geometry.delay(image, source, matrix) for image in images])
                diffs = np.abs(delays_all[:, None] - delays_all[None, :])
                upper = diffs[np.triu_indices(delays_all.size, k=1)]
                delta_min = float(np.min(upper))
    except geometry.LensDomainError:
        return ConfigOutcome(
            engine_refused=True, refusal_corner=False,
            gamma_prime_engine=float('nan'), n_schwinger=0, n_geometric=0,
            n_refusal=0, max_w=max_w, w_exceeds_engine=w_exceeds_engine,
            n_highw=0, n_geometric_highw=0, n_relaxed_rescue=0, topology='none',
            cd_fold_wdtau=empty, cd_fold_xi=empty, cd_cusp_R=empty,
            n_cd_degenerate=0)

    resolved = w_grid * delta_min >= RHO_END  # w-independent resolution mask
    l_arr = w_grid * abs_yprime               # cancellation exponent L = w*|y'|
    if is_saddle:
        # Saddle geometric gate (operator._saddle_grid): resolved AND above the
        # Schwinger ceiling; no separate cancellation-exponent condition.
        geometric = high & resolved
    else:
        # Positive-parity gate (operator.select_branch): resolved AND the
        # cancellation exponent L exceeds L_MAX.  select_branch is
        # w-INDEPENDENT: a resolved, strongly-cancelling node below the
        # Schwinger ceiling is still served geometrically in production.
        geometric = resolved & (l_arr > L_MAX)
    refusal = high & ~geometric
    schwinger = ~high & ~geometric  # Schwinger serves the non-geometric band.

    n_highw = int(np.count_nonzero(high))
    n_geometric_highw = int(np.count_nonzero(high & geometric))
    n_refusal = int(np.count_nonzero(refusal))

    # Corner-scoping defaults for configs with no refused high-w node.
    n_relaxed_rescue = 0
    topology = 'none'
    cd_fold_wdtau = empty
    cd_fold_xi = empty
    cd_cusp_R = empty
    n_cd_degenerate = 0

    if n_refusal > 0:
        if delays_all is not None:
            topology = _classify_fold_or_cusp(delays_all)
            pair = _merging_pair_indices(delays_all)
            mags = np.array(
                [geometry.magnification(image, matrix) for image in images])
            census_ok = _image_census_matches(mags, pair)
        else:  # fewer than two real images: no fold/cusp structure to exploit.
            topology = 'degenerate'
            census_ok = False

        # Category (b), MEASURED only: resolved refused positive-parity nodes
        # within the relaxed cancellation ceiling whose census matches.  Saddle
        # refusals have no L gate, so a relaxed L_MAX never rescues them.
        if (not is_saddle) and census_ok:
            relaxed = refusal & resolved & (l_arr <= L_MAX_RELAXED)
            n_relaxed_rescue = int(np.count_nonzero(relaxed))
            cd_mask = refusal & ~relaxed
        else:
            cd_mask = refusal

        cd_w = w_grid[cd_mask]
        if topology == 'cusp':
            try:
                caustic = geometry.nearest_caustic_point(
                    gamma, beta, source, kappa=kappa)
            except geometry.LensDomainError:
                # Cannot locate the caustic frame: treat as hard-core.
                n_cd_degenerate = int(cd_w.size)
            else:
                offset = source - np.asarray(caustic.source, dtype=float)
                delta_parallel = float(offset @ caustic.soft_axis)
                delta_perp = float(offset @ caustic.hard_axis)
                cd_cusp_R = _cusp_arguments(cd_w, delta_parallel, delta_perp)
        elif topology == 'fold':
            cd_fold_wdtau, cd_fold_xi = _fold_arguments(cd_w, delta_min)
        else:  # degenerate: no exploitable fold/cusp structure -> hard-core.
            n_cd_degenerate = int(cd_w.size)

    return ConfigOutcome(
        engine_refused=False,
        refusal_corner=bool(n_refusal > 0),
        gamma_prime_engine=float(gamma_prime),
        n_schwinger=int(np.count_nonzero(schwinger)),
        n_geometric=int(np.count_nonzero(geometric)),
        n_refusal=n_refusal, max_w=max_w, w_exceeds_engine=w_exceeds_engine,
        n_highw=n_highw, n_geometric_highw=n_geometric_highw,
        n_relaxed_rescue=n_relaxed_rescue, topology=topology,
        cd_fold_wdtau=cd_fold_wdtau, cd_fold_xi=cd_fold_xi,
        cd_cusp_R=cd_cusp_R, n_cd_degenerate=n_cd_degenerate)


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


def _distribution_summary(count: int, total: float, total_sq: float,
                          minimum: float, maximum: float) -> dict:
    """Summary statistics of an argument distribution from running moments.

    Parameters
    ----------
    count : int
        Number of samples.
    total, total_sq : float
        Running sum and sum of squares of the samples.
    minimum, maximum : float
        Running min and max of the samples.

    Returns
    -------
    dict
        ``count``/``min``/``max``/``mean``/``std`` (the last four ``None`` when
        ``count == 0``).
    """
    if count <= 0:
        return {'count': 0, 'min': None, 'max': None, 'mean': None, 'std': None}
    mean = total / count
    variance = max(total_sq / count - mean * mean, 0.0)
    return {'count': count, 'min': minimum, 'max': maximum,
            'mean': mean, 'std': math.sqrt(variance)}


def _threshold_table(counts_ge: np.ndarray, grid: np.ndarray, total: int,
                     threshold_key: str, extra: dict | None = None
                     ) -> list[dict]:
    """Fraction-vs-candidate-threshold table with Wilson intervals.

    Each row reports, at a candidate threshold ``t``, the fraction of the
    (c)/(d) population with argument ``>= t`` (uniform-resolvable) and ``< t``
    (genuinely hard-core), plus a Wilson 95% interval on the resolvable
    fraction.  This IS the empirical CDF sampled on the fixed grid.

    Parameters
    ----------
    counts_ge : np.ndarray
        Per-threshold counts of samples with argument ``>= grid[j]``.
    grid : np.ndarray
        The fixed candidate-threshold grid.
    total : int
        Size of the (c)/(d) population feeding the table.
    threshold_key : str
        JSON key for the threshold value in each row.
    extra : dict, optional
        Mapping ``row_key -> callable(threshold) -> value`` for derived columns
        (e.g. ``xi`` from ``w * Delta_tau``).

    Returns
    -------
    list of dict
        One row per grid threshold.
    """
    rows = []
    for threshold, count_ge in zip(grid, counts_ge):
        count_ge = int(count_ge)
        low, high = wilson_interval(count_ge, total)
        fraction = count_ge / total if total else 0.0
        row = {
            threshold_key: float(threshold),
            'fraction_resolvable_ge': fraction,
            'wilson95_resolvable': [low, high],
            'fraction_hardcore_lt': (1.0 - fraction) if total else 0.0,
        }
        for key, func in (extra or {}).items():
            row[key] = float(func(float(threshold)))
        rows.append(row)
    return rows


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
        fractions, plus the ``corner_scoping`` section carrying the Build-8e
        category (a)-(d) fractions and the fold/cusp argument tables).
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

    # --- Build-8e corner-scoping accumulators -------------------------------
    n_highw_total = 0
    n_a_total = 0            # (a) geometric-now high-w nodes
    n_b_total = 0            # (b) relaxed-L_MAX rescue high-w nodes
    fold_configs = 0
    cusp_configs = 0
    degenerate_configs = 0
    n_degenerate_cd = 0     # (d) hard-core nodes with no fold/cusp structure
    fold_wdtau_ge = np.zeros(FOLD_WDTAU_THRESHOLD_GRID.size, dtype=np.int64)
    cusp_R_ge = np.zeros(CUSP_R_THRESHOLD_GRID.size, dtype=np.int64)
    n_fold_cd = 0
    n_cusp_cd = 0
    fold_sum = fold_sumsq = 0.0
    fold_min, fold_max = math.inf, -math.inf
    cusp_sum = cusp_sumsq = 0.0
    cusp_min, cusp_max = math.inf, -math.inf

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

            n_highw_total += outcome.n_highw
            n_a_total += outcome.n_geometric_highw
            n_b_total += outcome.n_relaxed_rescue
            if outcome.topology == 'fold':
                fold_configs += 1
            elif outcome.topology == 'cusp':
                cusp_configs += 1
            elif outcome.topology == 'degenerate':
                degenerate_configs += 1
            n_degenerate_cd += outcome.n_cd_degenerate

            fold = outcome.cd_fold_wdtau
            if fold.size:
                fold_wdtau_ge += (
                    fold[:, None] >= FOLD_WDTAU_THRESHOLD_GRID[None, :]
                ).sum(axis=0).astype(np.int64)
                n_fold_cd += int(fold.size)
                fold_sum += float(fold.sum())
                fold_sumsq += float((fold * fold).sum())
                fold_min = min(fold_min, float(fold.min()))
                fold_max = max(fold_max, float(fold.max()))
            cusp = outcome.cd_cusp_R
            if cusp.size:
                cusp_R_ge += (
                    cusp[:, None] >= CUSP_R_THRESHOLD_GRID[None, :]
                ).sum(axis=0).astype(np.int64)
                n_cusp_cd += int(cusp.size)
                cusp_sum += float(cusp.sum())
                cusp_sumsq += float((cusp * cusp).sum())
                cusp_min = min(cusp_min, float(cusp.min()))
                cusp_max = max(cusp_max, float(cusp.max()))
        n_w_exceeds_engine += int(outcome.w_exceeds_engine)
        max_w_seen = max(max_w_seen, outcome.max_w)

    n_classifiable = n - n_engine_refused
    n_gamma_prime_zero = int(np.count_nonzero(gamma_prime == 0.0))
    n_gamma_prime_near_zero = int(
        np.count_nonzero(gamma_prime < GAMMA_PRIME_NEAR_ZERO))

    corner_lo, corner_hi = wilson_interval(n_refusal_corner, n)
    corner_lo_cls, corner_hi_cls = wilson_interval(
        n_refusal_corner, n_classifiable)

    # --- Corner-scoping report assembly -------------------------------------
    n_cd_total = n_highw_total - n_a_total - n_b_total
    a_lo, a_hi = wilson_interval(n_a_total, n_highw_total)
    b_lo, b_hi = wilson_interval(n_b_total, n_highw_total)
    cd_lo, cd_hi = wilson_interval(n_cd_total, n_highw_total)

    def _fraction(count: int, denom: int) -> float:
        return count / denom if denom else 0.0

    corner_scoping = {
        'high_w_node_categories': {
            'n_high_w_nodes': n_highw_total,
            'definition': ('categories (a)-(d) partition the high-w nodes '
                           '(w > %.1f): n_high_w_nodes == a + b + (c+d)'
                           % W_CEILING),
            'a_geometric_now': {
                'count': n_a_total,
                'fraction': _fraction(n_a_total, n_highw_total),
                'wilson95': [a_lo, a_hi],
                'note': 'high-w nodes already served by the geometric branch',
            },
            'b_geometric_under_relaxed_l_max': {
                'count': n_b_total,
                'fraction': _fraction(n_b_total, n_highw_total),
                'wilson95': [b_lo, b_hi],
                'relaxed_l_max': L_MAX_RELAXED,
                'production_l_max': L_MAX,
                'note': ('MEASURED ONLY: resolved refused positive-parity nodes '
                         'a relaxed cancellation ceiling plus the image-census '
                         'guard would move onto the geometric branch; '
                         'operator.L_MAX and operator.select_branch are '
                         'UNCHANGED'),
            },
            'cd_uniform_or_hardcore': {
                'count': n_cd_total,
                'fraction': _fraction(n_cd_total, n_highw_total),
                'wilson95': [cd_lo, cd_hi],
                'note': ('residual refused nodes; the (c) uniform-resolvable / '
                         '(d) genuinely-hard-core split is read off the fold '
                         'and cusp threshold tables below at the arms\' '
                         'certified thresholds'),
            },
            'correlation_caveat': ('nodes within a config share geometry; the '
                                   'Wilson intervals treat nodes as independent '
                                   'and are indicative, not rigorous'),
        },
        'fold_argument': {
            'definition': ('xi = (3 w Delta_tau / 4)**(2/3), Delta_tau = '
                           'delta_min / 2 (half the merging minimum/saddle '
                           'Fermat-delay separation)'),
            'population': 'refused fold-type nodes not rescued by category (b)',
            'w_delta_tau_distribution': _distribution_summary(
                n_fold_cd, fold_sum, fold_sumsq, fold_min, fold_max),
            'fraction_vs_threshold': _threshold_table(
                fold_wdtau_ge, FOLD_WDTAU_THRESHOLD_GRID, n_fold_cd,
                'w_delta_tau_threshold',
                {'xi_threshold': lambda t: float(np.cbrt(0.75 * t) ** 2)}),
        },
        'cusp_argument': {
            'definition': ('R = sqrt(x**2 + y**2), x = w**(1/2) delta_parallel, '
                           'y = w**(3/4) delta_perp (cusp 2/3 law; offsets from '
                           'nearest_caustic_point soft/hard axes)'),
            'population': 'refused cusp-type nodes not rescued by category (b)',
            'R_distribution': _distribution_summary(
                n_cusp_cd, cusp_sum, cusp_sumsq, cusp_min, cusp_max),
            'fraction_vs_threshold': _threshold_table(
                cusp_R_ge, CUSP_R_THRESHOLD_GRID, n_cusp_cd, 'R_threshold'),
        },
        'topology_config_counts': {
            'fold': fold_configs,
            'cusp': cusp_configs,
            'degenerate': degenerate_configs,
            'degenerate_hardcore_nodes': n_degenerate_cd,
            'note': ('per refusal-corner config topology from the real-image '
                     'delay spectrum; degenerate (< 2 real images) nodes have '
                     'no exploitable fold/cusp structure and count as (d)'),
        },
        'partition_check': {
            'cd_node_total': n_cd_total,
            'fold_plus_cusp_plus_degenerate': (
                n_fold_cd + n_cusp_cd + n_degenerate_cd),
            'consistent': (n_fold_cd + n_cusp_cd + n_degenerate_cd
                           == n_cd_total),
        },
        'production_l_max_unchanged': L_MAX,
    }

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
        'corner_scoping': corner_scoping,
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

    cats = report['corner_scoping']['high_w_node_categories']
    print(f"  high-w nodes         : {cats['n_high_w_nodes']} "
          f"(categories a+b+cd partition)")
    for key, label in (('a_geometric_now', 'a geometric-now'),
                       ('b_geometric_under_relaxed_l_max', 'b relaxed-L_MAX'),
                       ('cd_uniform_or_hardcore', 'cd uniform/hard')):
        cat = cats[key]
        clo, chi = cat['wilson95']
        print(f"    {label:<16}: {cat['count']} ({cat['fraction']:.4f})  "
              f"Wilson95 [{clo:.4f}, {chi:.4f}]")
    topo = report['corner_scoping']['topology_config_counts']
    print(f"  refusal topologies   : fold={topo['fold']} cusp={topo['cusp']} "
          f"degenerate={topo['degenerate']}")
    check = report['corner_scoping']['partition_check']
    print(f"  cd partition check   : {check['consistent']} "
          f"({check['fold_plus_cusp_plus_degenerate']} == "
          f"{check['cd_node_total']})")


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
