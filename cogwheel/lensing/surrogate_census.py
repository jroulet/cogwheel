"""Census / validation tool for the lens-amplification surrogate.

WHAT
----
Draws fixture-scale prior samples (in SAMPLED coordinates, no importance
weights) from the registered lens subpriors and, for each, decides whether
the multi-chart `LensAmplificationSurrogate` serves the point or falls
through to the exact engine.  It then reports:

- ``served_fraction`` and a five-way, MUTUALLY-EXCLUSIVE breakdown of the
  fall-through causes (``gamma-guard``, ``dropped-sliver``, ``cusp-window``,
  ``refusal-ball``, ``out-of-box``), plus a separate ``engine-refused`` bucket
  for domain refusals the geometry itself raises;
- per-chart held-out envelope error ``eps`` against a FRESH engine oracle;
- ``(gamma, image_count, eta)``-partitioned lnL error tiers versus the exact
  engine (dependency-injected; never partitioned by the gauge angle ``theta``,
  FINDINGS F017), each carrying its target accuracy bar (a target, not a
  silent gate);
- the measured on-disk artifact size.

WHY
---
The surrogate is a purely additive speed layer that must never answer where
the engine would refuse and must stay accurate where it does answer.  This
tool measures both properties without ever trusting the surrogate's own
labels: the fall-through causes are attributed by calling the surrogate's OWN
guard predicates (`surrogate._tube_serves` / `surrogate._farfield_serves`,
one source of truth), and the held-out envelope error uses a fresh
`ChangRefsdalChannels.evaluate` oracle (FINDINGS F002 -- never the surrogate's
own reconstruction).

Separation of concerns: every function here is pure computation returning
plain dicts/dataclasses; artifact loading and its size stat live in `run`, and
JSON writing lives in the thin CLI (`scripts/census_lens_surrogate.py`).  The
lnL-tier stage is dependency-injected (an optional ``lnlike_pair`` callable) so
the always-run stages need only the surrogate and the engine, never a full
`EventData`.
"""
from __future__ import annotations

import dataclasses
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from cogwheel.lensing import surrogate as _surrogate
from cogwheel.lensing.chang_refsdal import (ChangRefsdalChannels,
                                            farfield_envelope_from_partition)
from cogwheel.lensing.prior import (FixedLensGeometryPrior,
                                    UniformLensMassPrior,
                                    UniformReducedShearPrior,
                                    UniformSourcePositionPrior)
from cogwheel.lensing.waveform import dimensionless_frequency
from cogwheel.prior import CombinedPrior

# ---------------------------------------------------------------------------
# Tier thresholds (targets, not silent gates) with provenance.
#
# CROWN_LNL_TOL: Professor override of the brief's unreachable 0.01 nats --
#   dlnL ~ eps * SNR^2 floors near 0.04 for a dense 4D spline, so 0.05 is the
#   realistic crown target.
# STRONG_SADDLE_LNL_TOL: saddle-family gamma' ~ 1.25-1.3 measures ~0.04 nats
#   (FINDINGS F016); 0.1 sits a factor >= 10 below RB_ATOL.
# RESCUED_LNL_TOL: the rescued strong-shear RB gap is seed-swinging and
#   binning-limited, inseparable from surrogate error (F016), so it is gated
#   only at the standard relative-binning tolerance RB_ATOL = 1.5 nats.
CROWN_LNL_TOL = 0.05
STRONG_SADDLE_LNL_TOL = 0.1
RESCUED_LNL_TOL = 1.5  # == RB_ATOL

# gamma' onset of the strong-shear regime (positive parity).  Below it and
# away from the caustic, a config is a "crown" far-field point.
STRONG_SHEAR_ONSET = 0.5

# A source within this caustic distance is "near the caustic" (the tube
# region); such positive-parity points are held to the strong-saddle bar
# rather than the tight crown bar.  Reuses the surrogate's caustic floor.
CROWN_CAUSTIC_MARGIN = _surrogate._DEFAULT_CAUSTIC_FLOOR

# Fraction of a serving chart's log_w training range, at each end, that counts
# as a "band edge" (Professor Q2 sub-flag).
BAND_EDGE_FRACTION = 0.10

# Deliberate denominator floor for the max-normalized envelope error currency
# (Professor Q3): avoids over-weighting deep-cancellation troughs that carry
# negligible |E|^2-weighted lnL.
EPS_DENOM_FLOOR = 1e-6

# Named refusals the geometry / engine may raise; a sample that hits one is
# counted separately from the surrogate's own guard fall-throughs.
_ENGINE_REFUSALS = _surrogate._REFUSAL_ERRORS

# Empty refused-point set for the refusal-ball toggle probe.
_EMPTY_REFUSED = np.empty((0, 3), dtype=float)

# The five mutually-exclusive surrogate fall-through categories.
_FALLTHROUGH_CATEGORIES = ('gamma-guard', 'dropped-sliver', 'cusp-window',
                           'refusal-ball', 'out-of-box')


class CensusError(RuntimeError):
    """Raised when a defensive census invariant is violated."""


class _LensSampledPrior(CombinedPrior):
    """Lens-only combined prior for drawing sampled-coordinate points.

    Composes exactly the four reduced Chang-Refsdal lens subpriors (no CBC
    subpriors), so its ``.generate_random_samples`` draws ``gamma``,
    ``m_lens_msun`` and the shear-frame source ``(y1, y2)`` from the SAME
    sampled->standard machinery the production `LensedIASPrior` uses.
    ``UniformLensMassPrior`` precedes ``UniformSourcePositionPrior`` because the
    latter is conditioned on ``m_lens_msun``.
    """
    prior_classes = [FixedLensGeometryPrior, UniformLensMassPrior,
                     UniformReducedShearPrior, UniformSourcePositionPrior]


@dataclass(frozen=True)
class CensusConfig:
    """Fixture-scale census settings (NOT a full-box campaign).

    Attributes
    ----------
    n_samples : int
        Number of prior draws.
    seed : int
        Seed for the prior sampler (reproducibility).
    f_min_hz, f_max_hz : float
        Analysis frequency band (Hz) defining each sample's ``w`` band via
        ``w = dimensionless_frequency(f, m_lens_msun, 0)``.
    n_freq : int
        Number of log-spaced frequency nodes across the band.
    max_heldout_per_chart : int
        Cap on served samples per chart used for the held-out envelope eps
        (bounds the number of expensive fresh-oracle evaluations).
    max_binning_floor_configs : int
        Cap on configs measured by `binning_floor` (each costs two exact
        RB evaluations, one per tolerance).
    """
    n_samples: int = 256
    seed: int = 0
    f_min_hz: float = 20.0
    f_max_hz: float = 1024.0
    n_freq: int = 128
    max_heldout_per_chart: int = 10
    max_binning_floor_configs: int = 8


@dataclass
class SampleRecord:
    """Per-sample census outcome (plain, JSON-friendly scalars)."""
    gamma: float
    m_lens_msun: float
    y1: float
    y2: float
    log_w_min: float
    log_w_max: float
    served: bool
    engine_refused: bool = False
    category: str | None = None       # fall-through cause, if not served
    chart_index: int | None = None    # serving chart, if served
    eta: float | None = None
    theta: float | None = None
    image_count: int | None = None
    band_edge: bool = False


# ---------------------------------------------------------------------------
# Stage 1: sampling.

def draw_samples(config: CensusConfig) -> pd.DataFrame:
    """Draw ``config.n_samples`` fixture-scale lens prior samples.

    Uses the lens subpriors' own rejection sampler in SAMPLED coordinates
    (no importance weights).  The returned frame carries the standard columns
    ``gamma``, ``m_lens_msun``, ``y1`` and ``y2`` (plus the fixed geometry).

    Parameters
    ----------
    config : CensusConfig
        Census settings (``n_samples`` and ``seed`` are read).

    Returns
    -------
    pandas.DataFrame
        One row per sample; columns are the prior's sampled + standard params.
    """
    prior = _LensSampledPrior()
    return prior.generate_random_samples(config.n_samples, seed=config.seed)


def _frequency_grid(config: CensusConfig) -> np.ndarray:
    """Log-spaced analysis frequency grid (Hz)."""
    return np.geomspace(config.f_min_hz, config.f_max_hz, config.n_freq)


# ---------------------------------------------------------------------------
# Stage 2: serve decision and fall-through categorization.

def _normalize_slivers(
        dropped_slivers: Sequence[Sequence[float]] | None
) -> tuple[tuple[float, float], ...]:
    """Coerce a dropped-gamma-sliver list to a tuple of ``(lo, hi)`` pairs."""
    if not dropped_slivers:
        return ()
    return tuple((float(lo), float(hi)) for lo, hi in dropped_slivers)


def classify_fallthrough(
        surrogate: _surrogate.LensAmplificationSurrogate, *,
        gamma: float, log_w_min: float, log_w_max: float, eta: float,
        theta: float, image_count: int, y1_eig: float, y2_eig: float,
        dropped_slivers: tuple[tuple[float, float], ...]) -> str:
    """Attribute a single fall-through cause for a NON-served sample.

    Assumes the geometry partition succeeded (so ``eta``, ``theta``,
    ``image_count`` and the eigenframe source are available) and that
    `surrogate.select_chart` returned ``None``.  The cause is decided by
    calling the surrogate's OWN guard predicates -- never by re-deriving the
    guard math -- in the same priority order the guard stack uses, toggling a
    single guard off to detect whether it alone blocked service:

    1. ``gamma-guard``  -- ``|gamma - 1| < _GAMMA_GUARD_BAND`` (checked first).
    2. ``dropped-sliver`` -- ``gamma`` inside a training-dropped metamorphosis
       band (a subset of ``out-of-box`` on the gamma axis, so checked first).
    3. ``cusp-window``  -- some TUBE chart would serve but for its cusp
       exclusion (detected by relaxing ``cusp_windows`` to empty and
       re-calling `surrogate._tube_serves`).  Per Professor Q7 a near-cusp
       source projecting onto a neighbouring arc with out-of-range ``theta``
       still fails the theta-range gate with cusps relaxed, so it correctly
       falls to ``out-of-box``.
    4. ``refusal-ball`` -- some FAR-FIELD chart would serve but for its
       engine-refusal exclusion ball (detected by relaxing ``refused_points``
       to empty and re-calling `surrogate._farfield_serves`).
    5. ``out-of-box``   -- outside every chart's certified box otherwise.

    Returns
    -------
    str
        One of `_FALLTHROUGH_CATEGORIES`.
    """
    # (2) gamma guard band near the det-A = 0 parity boundary (checked first).
    if abs(gamma - 1.0) < _surrogate._GAMMA_GUARD_BAND:
        return 'gamma-guard'

    # Training-dropped metamorphosis sliver (gamma-only; before out-of-box).
    for lo, hi in dropped_slivers:
        if lo <= gamma <= hi:
            return 'dropped-sliver'

    # cusp-window: a tube chart blocked ONLY by its cusp exclusion.
    for chart in surrogate.charts:
        if isinstance(chart, _surrogate.TubeChart):
            relaxed = dataclasses.replace(chart, cusp_windows=())
            if _surrogate._tube_serves(relaxed, gamma, log_w_min, log_w_max,
                                       eta, theta, image_count):
                return 'cusp-window'

    # refusal-ball: a far-field chart blocked ONLY by its exclusion ball.
    # The far-field containment/exclusion test is in the chart's caustic-fixed
    # ``(rho, theta_c)`` axes (Build 8h-b3), so map the eigenframe source to
    # those axes via the shared scalar-reach normalisation first.
    rho, theta_c = _surrogate._to_caustic_fixed(gamma, y1_eig, y2_eig)
    for chart in surrogate.charts:
        if isinstance(chart, _surrogate.FarFieldChart):
            relaxed = dataclasses.replace(chart, refused_points=_EMPTY_REFUSED)
            if _surrogate._farfield_serves(relaxed, gamma, log_w_min,
                                           log_w_max, eta, image_count,
                                           rho, theta_c):
                return 'refusal-ball'

    return 'out-of-box'


def _chart_index(charts: Sequence, chart) -> int:
    """Index of ``chart`` in ``charts`` by identity (charts are ``eq=False``)."""
    for i, candidate in enumerate(charts):
        if candidate is chart:
            return i
    raise CensusError('Selected chart is not in the surrogate chart list.')


def _chart_log_w_range(chart) -> tuple[float, float]:
    """The chart's ``log_w`` training range ``(lo, hi)``."""
    return float(chart.log_w_grid[0]), float(chart.log_w_grid[-1])


def _is_band_edge(chart, log_w_min: float, log_w_max: float) -> bool:
    """Whether the query's ``log_w`` band touches the chart's outer band edge.

    True when ``[log_w_min, log_w_max]`` reaches into the outer
    `BAND_EDGE_FRACTION` of the serving chart's ``log_w`` training range at
    either end (Professor Q2).
    """
    lo, hi = _chart_log_w_range(chart)
    span = hi - lo
    if span <= 0.0:
        return False
    margin = BAND_EDGE_FRACTION * span
    return log_w_min < lo + margin or log_w_max > hi - margin


def characterize_sample(
        surrogate: _surrogate.LensAmplificationSurrogate,
        engine_factory: Callable[[np.ndarray], ChangRefsdalChannels], *,
        gamma: float, m_lens_msun: float, y1: float, y2: float,
        f_grid: np.ndarray,
        dropped_slivers: tuple[tuple[float, float], ...]) -> SampleRecord:
    """Decide serve / fall-through for one sample, mirroring production order.

    Mirrors `LensedRelativeBinningLikelihood._surrogate_coefficients`: the
    gamma guard fires (via `may_serve`) before any geometry is built; then a
    fresh geometry-only partition supplies the certified physical
    ``(eta, theta, image_count)``; then the full guard stack
    (`surrogate.select_chart`) decides.  A named engine refusal from the
    geometry is recorded in its own bucket -- it is NOT a surrogate guard
    fall-through.

    Parameters
    ----------
    surrogate : LensAmplificationSurrogate
        The surrogate under test.
    engine_factory : callable
        Maps a ``w`` grid to a FRESH `ChangRefsdalChannels` (fresh so labels
        start from the deterministic initial assignment).
    gamma, m_lens_msun, y1, y2 : float
        Sample lens parameters (``beta = kappa = z_lens = 0``).
    f_grid : np.ndarray
        Analysis frequency grid (Hz).
    dropped_slivers : tuple of (float, float)
        Training-dropped gamma bands.

    Returns
    -------
    SampleRecord
    """
    w_grid = dimensionless_frequency(f_grid, m_lens_msun, 0.0)
    log_w = np.log(w_grid)
    log_w_min, log_w_max = float(log_w.min()), float(log_w.max())
    y1_eig, y2_eig = _surrogate._rotate_to_eigenframe(y1, y2, 0.0)
    rho, theta_c = _surrogate._to_caustic_fixed(gamma, y1_eig, y2_eig)

    record = SampleRecord(
        gamma=float(gamma), m_lens_msun=float(m_lens_msun), y1=float(y1),
        y2=float(y2), log_w_min=log_w_min, log_w_max=log_w_max, served=False)

    # gamma guard band fires before geometry is built (mirrors may_serve).
    if abs(gamma - 1.0) < _surrogate._GAMMA_GUARD_BAND:
        record.category = 'gamma-guard'
        return record

    # Fresh geometry-only partition; a named refusal is its own bucket.
    try:
        geom = engine_factory(w_grid).geometry_partition(
            gamma=gamma, y=(y1, y2), beta=0.0, kappa=0.0)
    except _ENGINE_REFUSALS:
        record.engine_refused = True
        return record

    eta = float(geom.caustic_distance)
    theta = float(geom.caustic_theta)
    image_count = int(geom.real_mask.sum())
    record.eta, record.theta, record.image_count = eta, theta, image_count

    chart = _surrogate.select_chart(
        surrogate.charts, gamma=gamma, log_w_min=log_w_min,
        log_w_max=log_w_max, eta=eta, theta=theta, image_count=image_count,
        rho=rho, theta_c=theta_c)

    if chart is not None:
        record.served = True
        record.chart_index = _chart_index(surrogate.charts, chart)
        record.band_edge = _is_band_edge(chart, log_w_min, log_w_max)
        return record

    record.category = classify_fallthrough(
        surrogate, gamma=gamma, log_w_min=log_w_min, log_w_max=log_w_max,
        eta=eta, theta=theta, image_count=image_count, y1_eig=y1_eig,
        y2_eig=y2_eig, dropped_slivers=dropped_slivers)
    return record


def characterize(
        surrogate: _surrogate.LensAmplificationSurrogate,
        samples: pd.DataFrame, f_grid: np.ndarray,
        dropped_slivers: tuple[tuple[float, float], ...], *,
        engine_factory: Callable[[np.ndarray], ChangRefsdalChannels]
) -> list[SampleRecord]:
    """Characterize every sample row into a `SampleRecord`."""
    return [characterize_sample(
                surrogate, engine_factory, gamma=float(row.gamma),
                m_lens_msun=float(row.m_lens_msun), y1=float(row.y1),
                y2=float(row.y2), f_grid=f_grid, dropped_slivers=dropped_slivers)
            for row in samples.itertuples(index=False)]


def fallthrough_breakdown(records: Sequence[SampleRecord]) -> dict:
    """Aggregate serve counts + the five-way fall-through breakdown.

    Also verifies (defensively) that the buckets partition the sample set:
    ``served + engine_refused + sum(five categories) == n_samples``.  The five
    guard categories partition the surrogate's own fall-throughs; the
    ``engine_refused`` bucket (named geometry refusals) is reported separately
    because it is not a surrogate guard decision.

    Raises
    ------
    CensusError
        If the buckets do not partition the sample set, or an unknown category
        appears.
    """
    n = len(records)
    served = sum(r.served for r in records)
    engine_refused = sum(r.engine_refused for r in records)
    counts = {name: 0 for name in _FALLTHROUGH_CATEGORIES}
    for r in records:
        if r.served or r.engine_refused:
            continue
        if r.category not in counts:
            raise CensusError(
                f'Unknown fall-through category: {r.category!r}.')
        counts[r.category] += 1

    total = served + engine_refused + sum(counts.values())
    if total != n:
        raise CensusError(
            f'Fall-through buckets do not partition: {total} != {n}.')

    return {
        'n_samples': n,
        'served': int(served),
        'served_fraction': (served / n) if n else float('nan'),
        'engine_refused': int(engine_refused),
        'fallthrough': {k: int(v) for k, v in counts.items()},
        'fallthrough_total': int(n - served),
        'partition_ok': True}


# ---------------------------------------------------------------------------
# Stage 3: per-chart held-out envelope eps against a FRESH engine oracle.

def _eps_stats(values: Sequence[float]) -> dict:
    """Max / mean / count summary of a list of eps values."""
    if not values:
        return {'count': 0, 'max': None, 'mean': None}
    arr = np.asarray(values, dtype=float)
    return {'count': int(arr.size), 'max': float(arr.max()),
            'mean': float(arr.mean())}


def heldout_envelope_eps(
        surrogate: _surrogate.LensAmplificationSurrogate,
        records: Sequence[SampleRecord], f_grid: np.ndarray, *,
        max_per_chart: int,
        engine_factory: Callable[[np.ndarray], ChangRefsdalChannels]
) -> list[dict]:
    """Per-chart held-out envelope error against a fresh engine oracle.

    For each served sample (an off-node, in-box point of its serving chart),
    compares the surrogate envelope ``E_sur`` (via `surrogate.serve`) to a
    FRESH engine reference (F002 -- never the surrogate's own reconstruction)
    on the sample's own ``w`` grid.  The reference and its normalization mirror
    the label the SERVING chart is trained on (Build 8g-b):

    - a `FarFieldChart` is referenced against the far-field label
      ``E_ff = F - sum_a H_a e^{1j w tau_a}``
      (`farfield_envelope_from_partition`), F-normalized by ``max|exact_total|``
      (``max|E_ff| ~ 1e-4`` is too tiny a denominator);
    - a `TubeChart` keeps ``partition.envelope`` normalized by ``max|E|``
      (unchanged).

    In both cases the error is
    ``max|E_sur - E_eng| / max(denom, EPS_DENOM_FLOOR)`` (the deliberate 1e-6
    floor).

    Parameters
    ----------
    surrogate : LensAmplificationSurrogate
    records : sequence of SampleRecord
        All characterized samples (only served ones are used).
    f_grid : np.ndarray
        Analysis frequency grid (Hz).
    max_per_chart : int
        Cap on evaluated samples per chart (bounds oracle cost).
    engine_factory : callable
        Fresh-oracle factory.

    Returns
    -------
    list of dict
        One entry per chart the surrogate holds, with its type, image count,
        and the ``eps`` summary over its held-out served samples.
    """
    per_chart: dict[int, list[float]] = defaultdict(list)
    seen: dict[int, int] = defaultdict(int)

    for record in records:
        if not record.served:
            continue
        chart_index = record.chart_index
        if seen[chart_index] >= max_per_chart:
            continue
        w_grid = dimensionless_frequency(f_grid, record.m_lens_msun, 0.0)
        try:
            partition = engine_factory(w_grid).evaluate(
                gamma=record.gamma, y=(record.y1, record.y2), beta=0.0,
                kappa=0.0)
        except _ENGINE_REFUSALS:
            continue
        # Reference envelope + normalization mirror the label each chart is
        # trained on (Build 8g-b): a far-field chart is referenced against the
        # far-field label ``E_ff = F - sum_a H_a e^{1j w tau_a}`` and
        # F-normalized by ``max|exact_total|``; a tube chart keeps the
        # caustic-region ``partition.envelope`` reference and ``max|E|``
        # normalization (byte-identical to HEAD).
        if isinstance(surrogate.charts[chart_index], _surrogate.FarFieldChart):
            env_eng = farfield_envelope_from_partition(partition)
            denom_base = float(np.max(np.abs(partition.exact_total)))
        else:
            env_eng = np.asarray(partition.envelope)
            denom_base = float(np.max(np.abs(env_eng)))
        if not np.all(np.isfinite(env_eng)):
            continue
        env_sur, served, _definition = surrogate.serve(
            w_grid, gamma=record.gamma, y1=record.y1, y2=record.y2,
            beta=0.0, eta=partition.caustic_distance,
            theta=partition.critical_theta,
            image_count=int(partition.real_mask.sum()))
        if not served:
            continue
        denom = max(denom_base, EPS_DENOM_FLOOR)
        eps = float(np.max(np.abs(env_sur - env_eng)) / denom)
        per_chart[chart_index].append(eps)
        seen[chart_index] += 1

    report = []
    for index, chart in enumerate(surrogate.charts):
        report.append({
            'chart_index': index,
            'chart_type': type(chart).__name__,
            'image_count': (None if chart.image_count is None
                            else int(chart.image_count)),
            'eps': _eps_stats(per_chart.get(index, []))})
    return report


# ---------------------------------------------------------------------------
# Stage 4: (gamma, image_count, eta)-partitioned lnL error tiers.

def assign_tier(gamma: float, eta: float, *, kappa: float = 0.0) -> str:
    """Assign a served config to a lnL accuracy tier by CERTIFIED axes only.

    Partitions strictly by ``gamma`` (parity + shear strength) and ``eta``
    (caustic distance) -- never by the gauge angle ``theta`` (F017).  With the
    census fixed at ``kappa = 0`` the reduced shear equals ``gamma``.

    - ``strong_saddle`` -- macro-saddle parity (``gamma > 1``), OR strong shear
      (``gamma' >= STRONG_SHEAR_ONSET``), OR near the caustic
      (``eta <= CROWN_CAUSTIC_MARGIN``); target `STRONG_SADDLE_LNL_TOL`.
    - ``crown`` -- positive parity, weak shear, away from the caustic; target
      `CROWN_LNL_TOL`.

    The ``rescued`` tier (positive-parity configs the exact wave branch refuses
    that route through the Schwinger cross-parity fallback) is not assigned
    here -- it needs engine-path introspection and is applied best-effort via
    an injected predicate in `lnl_error_tiers`.
    """
    gamma_prime = gamma / (1.0 - kappa)
    if gamma > 1.0:
        return 'strong_saddle'
    if gamma_prime >= STRONG_SHEAR_ONSET:
        return 'strong_saddle'
    if eta <= CROWN_CAUSTIC_MARGIN:
        return 'strong_saddle'
    return 'crown'


def _lens_par_dic(base_par_dic: dict, record: SampleRecord) -> dict:
    """Merge a base CBC par_dic with a record's reduced lens parameters."""
    return {**base_par_dic,
            'm_lens_msun': record.m_lens_msun, 'z_lens': 0.0,
            'y1': record.y1, 'y2': record.y2, 'gamma': record.gamma,
            'beta': 0.0, 'kappa': 0.0}


def lnl_error_tiers(
        records: Sequence[SampleRecord],
        lnlike_pair: Callable[[dict], tuple[float, float]] | None, *,
        base_par_dic: dict | None = None,
        rescued_predicate: Callable[[dict], bool] | None = None) -> dict | None:
    """Partition served-config lnL errors into accuracy tiers.

    For each SERVED sample, evaluates ``lnlike_pair(par_dic) -> (lnl_served,
    lnl_exact)`` and records ``|lnl_served - lnl_exact|``, partitioned by
    `assign_tier` (certified ``gamma`` / ``eta`` axes only).  A band-edge
    sub-tier collects served samples whose ``log_w`` band touches the outer
    `BAND_EDGE_FRACTION` of their serving chart's range (Professor Q2).

    Dependency-injected: pass ``lnlike_pair=None`` (the default) to skip the
    stage entirely (returns ``None``) when no likelihood is available.

    Parameters
    ----------
    records : sequence of SampleRecord
    lnlike_pair : callable or None
        Maps a full par_dic to ``(lnl_served, lnl_exact)``; ``None`` skips.
    base_par_dic : dict or None
        CBC intrinsic/extrinsic parameters merged with each sample's lens
        parameters.  Required when ``lnlike_pair`` is given.
    rescued_predicate : callable or None
        Best-effort ``par_dic -> bool`` flag routing a positive-parity config
        into the ``rescued`` tier; ``None`` leaves the tier empty.

    Returns
    -------
    dict or None
        Per-tier ``{target_nats, max, mean, count}`` plus the band-edge
        sub-tier and a ``rescued_detection_enabled`` flag; ``None`` if skipped.
    """
    if lnlike_pair is None:
        return None
    if base_par_dic is None:
        raise CensusError('lnl_error_tiers requires base_par_dic when '
                          'lnlike_pair is provided.')

    tier_errors: dict[str, list[float]] = {
        'crown': [], 'strong_saddle': [], 'rescued': []}
    band_edge_errors: list[float] = []

    for record in records:
        if not record.served:
            continue
        par_dic = _lens_par_dic(base_par_dic, record)
        lnl_served, lnl_exact = lnlike_pair(par_dic)
        dlnl = abs(float(lnl_served) - float(lnl_exact))

        tier = assign_tier(record.gamma, record.eta)
        if (rescued_predicate is not None and record.gamma < 1.0
                and rescued_predicate(par_dic)):
            tier = 'rescued'
        tier_errors[tier].append(dlnl)
        if record.band_edge:
            band_edge_errors.append(dlnl)

    targets = {'crown': CROWN_LNL_TOL, 'strong_saddle': STRONG_SADDLE_LNL_TOL,
               'rescued': RESCUED_LNL_TOL}
    report = {name: {'target_nats': targets[name], **_eps_stats(errors)}
              for name, errors in tier_errors.items()}
    report['band_edge'] = _eps_stats(band_edge_errors)
    report['rescued_detection_enabled'] = rescued_predicate is not None
    return report


def binning_floor(records: Sequence[SampleRecord],
                  lnlike_exact_factory: Callable[
                      [float], Callable[[dict], float]] | None, *,
                  base_par_dic: dict | None = None,
                  pn_phase_tol: float = 0.05,
                  refine_factor: float = 4.0,
                  max_configs: int | None = None) -> dict | None:
    """Measured RB-binning lnL floor: exact-RB at ``delta`` vs ``delta/4``.

    Owner-approved census line (2026-07-20): the relative-binning
    tolerance ``pn_phase_tol`` (delta) sets its own lnL error floor,
    independent of the surrogate's spline-eps floor, and the
    enable-by-default decision should see BOTH floors side by side, each
    with its knob and cost slope.  This evaluates the EXACT engine's RB
    lnL at the working ``pn_phase_tol`` and at ``pn_phase_tol /
    refine_factor`` on the same served configs; the |difference| is a
    direct measurement of the binning floor at working resolution.  The
    surrogate artifact is delta-independent (bins only move the spline's
    ``w`` query abscissae), so this stage never retrains anything --
    only the likelihood's per-event summaries are rebuilt per tolerance.

    Dependency-injected like `lnl_error_tiers`:
    ``lnlike_exact_factory(pn_phase_tol)`` returns a callable mapping a
    full ``par_dic`` to the exact-engine RB lnL at that tolerance;
    ``None`` skips the stage (returns ``None``).

    Parameters
    ----------
    records : sequence of SampleRecord
        Census sample records; only SERVED samples are measured (the
        floor line accompanies the served-tier report).
    lnlike_exact_factory : callable or None
        ``pn_phase_tol -> (par_dic -> lnl_exact)``; ``None`` skips.
    base_par_dic : dict or None
        CBC parameters merged with each sample's lens parameters.
        Required when ``lnlike_exact_factory`` is given.
    pn_phase_tol : float
        The working binning tolerance delta [rad].
    refine_factor : float
        The refinement ratio for the fine reference (default 4).
    max_configs : int or None
        Cap on measured configs (each costs two exact evaluations).

    Returns
    -------
    dict or None
        ``{pn_phase_tol, refine_factor, max, mean, count}`` or ``None``
        if skipped.
    """
    if lnlike_exact_factory is None:
        return None
    if base_par_dic is None:
        raise CensusError('binning_floor requires base_par_dic when '
                          'lnlike_exact_factory is provided.')
    if refine_factor <= 1.0:
        raise CensusError('refine_factor must exceed 1.')

    lnl_coarse = lnlike_exact_factory(float(pn_phase_tol))
    lnl_fine = lnlike_exact_factory(float(pn_phase_tol) / refine_factor)

    errors: list[float] = []
    for record in records:
        if not record.served:
            continue
        if max_configs is not None and len(errors) >= max_configs:
            break
        par_dic = _lens_par_dic(base_par_dic, record)
        errors.append(abs(float(lnl_coarse(par_dic))
                          - float(lnl_fine(par_dic))))
    return {'pn_phase_tol': float(pn_phase_tol),
            'refine_factor': float(refine_factor),
            **_eps_stats(errors)}


# ---------------------------------------------------------------------------
# Artifact size + top-level entry.

def _resolve_artifact_path(
        surrogate_path: str | Path | None) -> Path:
    """Resolve the artifact path used for the size stat (package default)."""
    if surrogate_path is not None:
        return Path(surrogate_path)
    return _surrogate.LensAmplificationSurrogate._default_artifact_path()


def _dropped_slivers_from(
        surrogate: _surrogate.LensAmplificationSurrogate,
        override: Sequence[Sequence[float]] | None
) -> tuple[tuple[float, float], ...]:
    """Resolve dropped gamma slivers: explicit override, else provenance.

    The training driver persists ``dropped_gamma_slivers`` (a flat
    ``[[lo, hi], ...]`` list, collected across parities) into the surrogate's
    serialized provenance at save time, so this is populated by default.  A
    caller-supplied ``override`` -- e.g. loaded via
    `dropped_slivers_from_training_report` -- takes precedence when given.
    """
    if override is not None:
        return _normalize_slivers(override)
    return _normalize_slivers(surrogate.provenance.get('dropped_gamma_slivers'))


def dropped_slivers_from_training_report(
        report: dict) -> list[list[float]]:
    """Collect dropped gamma slivers from a training-report dict.

    The training driver records the metamorphosis sub-bands it dropped
    refusal-conservatively under ``report['parities'][label]
    ['dropped_gamma_slivers']`` as ``[[lo, hi], ...]``.  This gathers them
    across every parity into one flat list, suitable to pass as the census
    ``dropped_slivers`` override.

    Parameters
    ----------
    report : dict
        A parsed ``training_report.json``.

    Returns
    -------
    list of [float, float]
        All dropped ``(lo, hi)`` gamma bands, across parities.
    """
    parities = report.get('parities', {})
    slivers: list[list[float]] = []
    for entry in parities.values():
        for lo, hi in entry.get('dropped_gamma_slivers', []):
            slivers.append([float(lo), float(hi)])
    return slivers


def run(*, surrogate: _surrogate.LensAmplificationSurrogate | None = None,
        surrogate_path: str | Path | None = None,
        config: CensusConfig | None = None,
        lnlike_pair: Callable[[dict], tuple[float, float]] | None = None,
        base_par_dic: dict | None = None,
        rescued_predicate: Callable[[dict], bool] | None = None,
        dropped_slivers: Sequence[Sequence[float]] | None = None,
        engine_factory: Callable[[np.ndarray], ChangRefsdalChannels] | None
        = None,
        lnlike_exact_factory: Callable[
            [float], Callable[[dict], float]] | None = None,
        pn_phase_tol: float = 0.05) -> dict:
    """Run the full census and return the report dict (does not write it).

    Loads the surrogate (if not supplied), draws fixture-scale prior samples,
    characterizes each into a serve / fall-through outcome, computes per-chart
    held-out envelope eps and -- if a likelihood pair is injected -- the lnL
    error tiers, and measures the on-disk artifact size.

    Parameters
    ----------
    surrogate : LensAmplificationSurrogate or None
        A pre-loaded surrogate; if ``None`` it is loaded from
        ``surrogate_path`` (or the package default).
    surrogate_path : str or Path or None
        Artifact path; also the path whose size is measured.
    config : CensusConfig or None
        Census settings (fixture-scale defaults if ``None``).
    lnlike_pair : callable or None
        ``par_dic -> (lnl_served, lnl_exact)``; ``None`` skips the lnL tiers.
    base_par_dic : dict or None
        CBC parameters merged with each sample's lens parameters for the lnL
        tiers.
    rescued_predicate : callable or None
        Best-effort rescued-tier flag.
    dropped_slivers : sequence of (float, float) or None
        Training-dropped gamma bands; if ``None``, read from provenance.
    engine_factory : callable or None
        Fresh `ChangRefsdalChannels` factory; defaults to the real engine.
    lnlike_exact_factory : callable or None
        ``pn_phase_tol -> (par_dic -> lnl_exact)`` for the measured
        RB-binning floor line (`binning_floor`); ``None`` skips it.
    pn_phase_tol : float
        Working binning tolerance delta [rad] for the floor line.

    Returns
    -------
    dict
        The census report (JSON-serializable).
    """
    config = config or CensusConfig()
    if engine_factory is None:
        engine_factory = ChangRefsdalChannels
    artifact_path = _resolve_artifact_path(surrogate_path)
    if surrogate is None:
        surrogate = _surrogate.LensAmplificationSurrogate.load(surrogate_path)

    slivers = _dropped_slivers_from(surrogate, dropped_slivers)
    f_grid = _frequency_grid(config)

    records = characterize(surrogate, draw_samples(config), f_grid, slivers,
                           engine_factory=engine_factory)

    breakdown = fallthrough_breakdown(records)
    per_chart = heldout_envelope_eps(
        surrogate, records, f_grid,
        max_per_chart=config.max_heldout_per_chart,
        engine_factory=engine_factory)
    tiers = lnl_error_tiers(records, lnlike_pair, base_par_dic=base_par_dic,
                            rescued_predicate=rescued_predicate)
    floor = binning_floor(records, lnlike_exact_factory,
                          base_par_dic=base_par_dic,
                          pn_phase_tol=pn_phase_tol,
                          max_configs=config.max_binning_floor_configs)

    size_bytes = (os.path.getsize(artifact_path)
                  if artifact_path.exists() else None)

    return {
        'config': dataclasses.asdict(config),
        'artifact': {
            'path': str(artifact_path),
            'size_bytes': None if size_bytes is None else int(size_bytes),
            'n_charts': len(surrogate.charts),
            'chart_types': [type(c).__name__ for c in surrogate.charts]},
        'served_fraction': breakdown['served_fraction'],
        'served': breakdown['served'],
        'n_samples': breakdown['n_samples'],
        'engine_refused': breakdown['engine_refused'],
        'fallthrough': breakdown['fallthrough'],
        'fallthrough_total': breakdown['fallthrough_total'],
        'partition_ok': breakdown['partition_ok'],
        'per_chart_eps': per_chart,
        'lnl_tiers': tiers,
        'binning_floor': floor,
        'dropped_gamma_slivers': [list(s) for s in slivers]}
