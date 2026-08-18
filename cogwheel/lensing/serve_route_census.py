"""Engine-free serve-route DEMAND census for the lens-amplification stack.

WHAT
----
`run` draws full-reach lens prior samples and classifies each draw into
EXACTLY ONE of eight mutually-exclusive serve routes -- the analytic rung
that WOULD answer it (or the exact-wave engine that would have to) -- WITHOUT
ever evaluating a wave-optics amplitude.  It then aggregates the draw-level
routes into ``(region x gamma-band x w-band)`` cells and splits the
exact-wave-demand population three ways by caustic-relative reach, so a
downstream campaign planner (the "7b" acceptance tool) can size each analytic
family against the demand this census maps.

The draw-level routes mirror the BAND-SPLIT serving semantics of the
production likelihood rungs: ``ppgo_above_ceiling`` is the above-ceiling
split (exact engine at or below ``W_CEILING_SCHWINGER_QD``, fold-corrected
ppGO above, gated on ``W_CEILING_SCHWINGER_QD * min_delta_tau >= RHO_END``)
and ``saddle_c3`` covers BOTH the whole-band-analytic admit and the c3 band
split (exact engine at or below the per-draw certificate split ``w_split``,
analytic zero envelope above; ``w_split`` is recorded per draw).

Two classification granularities are produced per draw:

* a DRAW-LEVEL route (one of `SERVE_ROUTES`), decided by a first-admitting
  waterfall of whole-band analytic intercepts followed by a per-node pass;
* a per-node ROUTE-KIND vector (`ROUTE_KINDS`) recording, for each frequency
  node in the draw's w-band, which serving arm would answer it.  The KIND
  vector is the D2-reflection-invariant object: saddle lobes swap ``0 <-> pi``
  under reflection, flipping any lobe/internal index, but never the arm KIND.

WHY
---
A production training campaign spends its budget building analytic charts for
the sources the exact engine cannot cheaply answer.  Sizing that campaign
needs a MAP of demand: how many draws each analytic rung already covers, how
many fall through to the exact wave engine, and where (region, gamma, w) the
residual demand concentrates.  Measuring this by calling the engine on every
draw would itself cost the campaign; this census answers it engine-free.

ENGINE-FREE BY CONSTRUCTION (no-CALL, not no-import)
----------------------------------------------------
Importing anything under ``cogwheel.lensing`` necessarily runs the package
``__init__`` chain and loads the amplitude-engine module OBJECTS; the
achievable, load-bearing guarantee is that this census makes ZERO engine
CALLS.  The forbidden exact-wave DOORS -- ``ChangRefsdalChannels.evaluate``,
``_schwinger.f_schwinger`` / ``_f_schwinger_mpmath`` and the ``mpmath``
special functions -- are never touched: a draw (or node) that would reach them
is COUNTED as engine demand, not evaluated.  To keep the guarantee visible,
this module's top-level imports carry NO engine-bearing symbol; every
engine-adjacent import lives inside `_load_production_modules`, invoked from
`run`.  The geometry partition, the closed-form ``select_branch`` gate and the
uniform-asymptotic arms are all pure analytic paths and ARE used freely.

DEFINITIONS
-----------
'engine' (forbidden CALL): wave-optics amplitude EVALUATION -- the channel
``evaluate`` path, ``_schwinger`` / ``_f_schwinger`` and the ``mpmath``
special functions.  'geometry' + 'uniform arms' (allowed, pure): image finding
(Newton quartic), caustic curves, delays, the ``select_branch`` branch gate,
``cancellation_exponent`` and the fold / ghost+ppGO / Pearcey uniform arms
offered by ``operator._uniform_arm_value``.
"""

from __future__ import annotations

import dataclasses
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

# Engine-free layers only (no wave-optics evaluation door is exposed here):
# ``ppgo_map.caustic_rho`` is the single authoritative caustic-relative gauge
# and ``ppgo_map._gamma_band_edges`` supplies the production gamma bands;
# ``geometry`` supplies the macro matrix and the domain-refusal exception.
from cogwheel.lensing import ppgo_map
from cogwheel.lensing.chang_refsdal import geometry

# ---------------------------------------------------------------------------
# Route vocabulary
# ---------------------------------------------------------------------------

#: The eight MECE draw-level serve routes, listed in label-enumeration order.
#: The DECISION order (a first-admitting waterfall) is DIFFERENT and is
#: documented on `classify_draw`; it mirrors the PRODUCTION rung order of
#: ``likelihood._amplification_coefficients``: ``engine_refused`` is decided
#: first, then ``surrogate`` (artifact mode only), ``ppgo_above_ceiling``
#: (the above-ceiling band-split rung -- in production it fires BEFORE the
#: saddle rung), ``saddle_c3`` (whole-band-analytic OR c3 band-split),
#: ``born_analytic``, and finally the per-node pass resolves
#: ``wave_refused`` / ``engine_residual`` / ``analytics_engine_hosted``
#: (in that precedence order -- see `classify_draw`).
SERVE_ROUTES: tuple[str, ...] = (
    'surrogate',
    'ppgo_above_ceiling',
    'saddle_c3',
    'born_analytic',
    'analytics_engine_hosted',
    'engine_residual',
    'wave_refused',
    'engine_refused',
)

#: Per-node route KINDS (the reflection-invariant per-frequency arm labels).
#: ``geometric`` = stationary-phase asymptote; ``fold`` / ``ghost_ppgo`` /
#: ``pearcey`` = the three uniform-asymptotic arms of
#: ``operator._uniform_arm_value``, tried in that order; ``exact_wave`` = a
#: node that would fall through to the exact-wave engine (true engine
#: demand, only reachable at ``w <= W_CEILING_SCHWINGER_QD``); ``refused`` =
#: an above-QD-ceiling wave-branch node BOTH arms decline -- a deterministic
#: production refuser (``SchwingerCertificationError``, ``lnL = -inf``; no
#: exact engine exists above the ceiling), NOT engine demand.
ROUTE_KINDS: tuple[str, ...] = (
    'geometric', 'fold', 'ghost_ppgo', 'pearcey', 'exact_wave', 'refused',
)

#: The three uniform-arm KINDS a served wave node can carry (the subset of
#: `ROUTE_KINDS` produced by ``operator._uniform_arm_value``).  The engine
#: offers the arms to EVERY wave node above the DD ceiling (both the
#: ``(60, 150]`` mpmath band and the above-QD-ceiling regime).  These are
#: node-level labels only: the draw-level ``ppgo_above_ceiling`` route is
#: the whole-band above-ceiling INTERCEPT (the production
#: ``_ppgo_above_ceiling`` rung), never a node-pass outcome.
_UNIFORM_ARM_KINDS: frozenset[str] = frozenset(
    {'fold', 'ghost_ppgo', 'pearcey'})

#: The Born weak-deflection exterior floor: ``caustic_rho > _BORN_RHO_FLOOR``
#: is the geometric coverage predicate the production first-class Born
#: intercept (`likelihood.LensedRelativeBinningLikelihood._born_residual_`
#: ``analytic``) keys on, alongside ``kappa == 0`` and ``beta == 0``.
_BORN_RHO_FLOOR = 2.0

#: The three-way caustic-relative reach split of the exact-wave-demand
#: population, keyed on ``caustic_rho`` (NEVER ``rho_lobe`` -- the two gauges
#: differ by ~10x at the median).  ``> 2`` is Born-chart demand, ``(1, 2]`` is
#: the near-caustic / tube shell, ``<= 1`` is the caustic-relative interior.
_RESIDUAL_BORN_FLOOR = 2.0
_RESIDUAL_TUBE_FLOOR = 1.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ServeRouteCensusConfig:
    """Settings for a serve-route demand census.

    ``n_samples`` and ``seed`` are read by the reused
    `surrogate_census.draw_samples` (which samples the lens subpriors' own
    rejection sampler in SAMPLED coordinates, no importance weights).  The
    frequency-grid fields set the physical Hz grid mapped, per draw, to a
    dimensionless ``w`` band via ``waveform.dimensionless_frequency``; the
    census deliberately spans the FULL production Hz range so that the
    ``w > 150`` ppGO-above-ceiling regime is reachable (it does NOT inherit
    the ``scripts/census_dry_run.py`` training-wall ``w`` cap, which is a
    training budget, not the physical prior).
    """

    n_samples: int = 10_000
    seed: int = 0
    f_min_hz: float = 20.0
    f_max_hz: float = 1024.0
    n_freq: int = 128


def _frequency_grid(config: ServeRouteCensusConfig) -> np.ndarray:
    """Geometric analysis frequency grid in Hz."""
    return np.geomspace(config.f_min_hz, config.f_max_hz, config.n_freq)


# ---------------------------------------------------------------------------
# Lazily-loaded engine-adjacent production symbols
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ProductionModules:
    """Engine-adjacent callables bound once, lazily, from `run`.

    Bundling them keeps the pure classifier a thin caller of production
    predicates (one source of truth) while keeping this module's top-level
    imports engine-free.  None of these attributes is an exact-wave door;
    the doors (``evaluate`` / ``f_schwinger`` / ``mpmath``) are deliberately
    absent so the classifier CANNOT call them.
    """

    channels_cls: Any                     # ChangRefsdalChannels (constructor)
    draw_samples: Any                     # surrogate_census.draw_samples
    dimensionless_frequency: Any          # waveform.dimensionless_frequency
    saddle_farfield_serves: Any           # likelihood._saddle_farfield_...
    saddle_c3_split_point: Any            # likelihood._saddle_c3_split_point
    saddle_min_image_sep: Any             # likelihood._saddle_min_image_sep
    saddle_min_sep_floor: float           # likelihood._SADDLE_FARFIELD_MIN...
    macro_matrix: Any                     # geometry.macro_matrix
    select_branch: Any                    # operator.select_branch
    uniform_arm_value: Any                # operator._uniform_arm_value
    cancellation_exponent: Any            # operator.cancellation_exponent
    real_delay_min_separation: Any        # operator._real_delay_min_separation
    fold_amplification: Any               # operator._airy_fold.fold_amplif...
    ghost_ppgo_amplification: Any         # operator._ghost_ppgo_amplification
    w_ceiling_dd: float                   # _schwinger.W_CEILING_SCHWINGER
    w_ceiling_qd: float                   # _schwinger.W_CEILING_SCHWINGER_QD
    rho_end: float                        # operator.RHO_END (likelihood's own)
    refusal_errors: tuple[type[BaseException], ...]  # surrogate._REFUSAL_ERRORS


def _load_production_modules() -> _ProductionModules:
    """Import the engine-adjacent production modules lazily.

    Deferred so ``import cogwheel.lensing.serve_route_census`` binds no
    engine-bearing symbol at module top.  Importing a class is not calling
    it: the census still makes zero engine evaluations.
    """
    from cogwheel.lensing import surrogate as sg
    from cogwheel.lensing import surrogate_census as sc
    from cogwheel.lensing.chang_refsdal import ChangRefsdalChannels, _schwinger
    from cogwheel.lensing.chang_refsdal import operator as op
    from cogwheel.lensing.likelihood import (
        _SADDLE_FARFIELD_MIN_IMAGE_SEP, _saddle_c3_split_point,
        _saddle_farfield_analytic_serves, _saddle_min_image_sep)
    from cogwheel.lensing.waveform import dimensionless_frequency

    return _ProductionModules(
        channels_cls=ChangRefsdalChannels,
        draw_samples=sc.draw_samples,
        dimensionless_frequency=dimensionless_frequency,
        saddle_farfield_serves=_saddle_farfield_analytic_serves,
        saddle_c3_split_point=_saddle_c3_split_point,
        saddle_min_image_sep=_saddle_min_image_sep,
        saddle_min_sep_floor=float(_SADDLE_FARFIELD_MIN_IMAGE_SEP),
        macro_matrix=geometry.macro_matrix,
        select_branch=op.select_branch,
        uniform_arm_value=op._uniform_arm_value,
        cancellation_exponent=op.cancellation_exponent,
        real_delay_min_separation=op._real_delay_min_separation,
        fold_amplification=op._airy_fold.fold_amplification,
        ghost_ppgo_amplification=op._ghost_ppgo_amplification,
        w_ceiling_dd=float(_schwinger.W_CEILING_SCHWINGER),
        w_ceiling_qd=float(_schwinger.W_CEILING_SCHWINGER_QD),
        rho_end=float(op.RHO_END),
        refusal_errors=tuple(sg._REFUSAL_ERRORS),
    )


# ---------------------------------------------------------------------------
# Per-draw classification result
# ---------------------------------------------------------------------------

@dataclass
class DrawResult:
    """Serve-route classification of a single lens prior draw.

    Carries the draw parameters (``kappa = beta = 0`` are fixed by the
    census contract and omitted), the draw-level `SERVE_ROUTES` label, the
    per-node `ROUTE_KINDS` vector, the aggregation coordinates, the
    caustic-relative reach and the draw's dimensionless ``w`` band.  No
    per-draw weight is carried: the draws are equal-weight by construction.

    ``w_split`` is the saddle-c3 certificate split frequency recorded on a
    ``saddle_c3`` draw (``None`` on every other route): the exact engine
    serves the band nodes at or below ``w_split`` and the analytic zero
    envelope serves above it, so the tiling design can size the below-split
    chart demand from this detail.  A whole-band-analytic admit records its
    (``<= w_lo``) split point too.  The ``ppgo_above_ceiling`` engine-below
    band needs no per-draw detail: its split is the constant
    ``W_CEILING_SCHWINGER_QD``.
    """

    gamma: float
    m_lens_msun: float
    y1: float
    y2: float
    route: str
    node_route_kinds: tuple[str, ...]
    region: str
    gamma_band: str
    w_band: str
    caustic_rho: float | None
    log_w_min: float
    log_w_max: float
    w_split: float | None = None

    def as_record(self) -> dict[str, Any]:
        """JSON-serializable per-draw record (no weight column)."""
        return {
            'gamma': float(self.gamma),
            'm_lens_msun': float(self.m_lens_msun),
            'y1': float(self.y1),
            'y2': float(self.y2),
            'route': self.route,
            'node_route_kinds': list(self.node_route_kinds),
            'region': self.region,
            'gamma_band': self.gamma_band,
            'w_band': self.w_band,
            'caustic_rho': (None if self.caustic_rho is None
                            else float(self.caustic_rho)),
            'log_w_min': float(self.log_w_min),
            'log_w_max': float(self.log_w_max),
            'w_split': (None if self.w_split is None
                        else float(self.w_split)),
        }


# ---------------------------------------------------------------------------
# Aggregation-coordinate helpers (pure)
# ---------------------------------------------------------------------------

#: Region admissibility by parity -- the SAME per-parity tuples the production
#: trainer honours (mirrors ``tiling_census._REGIONS_BY_PARITY``): index 0 the
#: tube shell, index 1 the interior family, index 2 the exterior family.
_REGIONS_BY_PARITY: dict[int, tuple[str, str, str]] = {
    1: ('tube', 'wedge_interior', 'exterior'),
    -1: ('tube', 'lobe_interior', 'lobe_exterior'),
}


def _parity_of(gamma: float) -> int:
    """Macro parity sign: ``+1`` positive-parity astroid, ``-1`` macro saddle.

    Mirrors the production discriminator (``'positive' if gamma < 1 else
    'saddle'``): ``gamma == 1`` is the parity boundary and is assigned to the
    saddle side, consistent with ``surrogate_census``.
    """
    return 1 if gamma < 1.0 else -1


def _region_of(gamma: float, caustic_rho: float | None) -> str:
    """Map a draw to a parity-specific region by caustic-relative reach.

    ``rho > 2`` is the exterior family, ``1 < rho <= 2`` the tube shell and
    ``rho <= 1`` the caustic-relative interior family; an unresolved ``rho``
    (the ``gamma`` parity boundary or ``gamma == 0``) is ``'undetermined'``.
    ``rho`` is a scalar reach GAUGE, not a domain predicate (F073) -- these
    bands are aggregation coordinates, not exact interior/exterior tests.
    """
    if caustic_rho is None:
        return 'undetermined'
    tube, interior, exterior = _REGIONS_BY_PARITY[_parity_of(gamma)]
    if caustic_rho > 2.0:
        return exterior
    if caustic_rho > 1.0:
        return tube
    return interior


def _gamma_band_of(gamma: float, edges: np.ndarray) -> str:
    """Label the production gamma band containing ``gamma`` (or out-of-grid)."""
    if gamma < edges[0] or gamma >= edges[-1]:
        return 'out_of_grid'
    index = int(np.searchsorted(edges, gamma, side='right')) - 1
    return f'{edges[index]:.6g}-{edges[index + 1]:.6g}'


def _w_band_of(log_w_max: float, w_dd: float, w_qd: float) -> str:
    """Label a draw's w band by its CEILING ``w_hi = exp(log_w_max)``.

    The edges are the production Schwinger serving-regime ceilings
    (``W_CEILING_SCHWINGER`` and ``W_CEILING_SCHWINGER_QD``), single-sourced
    from ``_schwinger``.  The band ceiling (not the floor) is the demand-
    relevant key: a draw reaches the ppGO-above-ceiling regime only when its
    ``w_hi`` exceeds the QD ceiling.
    """
    w_hi = math.exp(log_w_max)
    if w_hi <= w_dd:
        return f'w_hi<={w_dd:g}'
    if w_hi <= w_qd:
        return f'{w_dd:g}<w_hi<={w_qd:g}'
    return f'w_hi>{w_qd:g}'


# ---------------------------------------------------------------------------
# Per-node arm pass (second granularity)
# ---------------------------------------------------------------------------

def _resolve_arm_kind(mods: _ProductionModules, w: float, source: np.ndarray,
                      gamma: float) -> str:
    """Which uniform arm answers a served wave node (label only).

    ``operator._uniform_arm_value`` returns the first certified value but not
    which arm produced it; this re-probes the SAME arms in the SAME
    documented order (fold, then ghost+ppGO, then Pearcey) purely to LABEL
    the node.  The served decision itself is made by ``_uniform_arm_value`` in
    `_classify_nodes` -- this is a labelling helper, not a second gate.  It is
    called only after ``_uniform_arm_value`` returned non-``None``, so the
    winning arm (and every arm before it) is known not to raise.
    """
    if mods.fold_amplification(w, source, gamma) is not None:
        return 'fold'
    if mods.ghost_ppgo_amplification(w, source, gamma) is not None:
        return 'ghost_ppgo'
    return 'pearcey'


def _classify_nodes(mods: _ProductionModules, *, gamma: float,
                    source: np.ndarray, w_grid: np.ndarray,
                    delta_min: float, eta: float) -> tuple[str, ...]:
    """Route each frequency node through the PRODUCTION band ladder.

    Mirrors the operator's per-node serving ladder exactly (the
    ``operator.py`` node loop, ~:923 / :1231):

    * ``w <= W_CEILING_SCHWINGER`` (DD ceiling, 60): production sends the
      node straight to the DD exact engine -- no arms, no geometric
      asymptote -- so the node is unconditionally exact-wave demand.
    * ``60 < w <= W_CEILING_SCHWINGER_QD`` (QD ceiling, 150): production
      offers ``operator._uniform_arm_value`` (fold -> ghost+ppGO ->
      Pearcey) FIRST; decliners fall to the exact mpmath engine, so an
      arm-declined node is exact-wave demand.
    * ``w > 150``: the authoritative `operator.select_branch` gate decides
      geometric vs wave (positive parity supplies the measured cancellation
      exponent ``L = w|y'|``; the macro saddle passes ``inf``, which leaves
      that leg vacuous -- exactly as the production per-node routers do).
      A geometric node is served by the stationary-phase asymptote; a wave
      node is offered the arms; a wave node BOTH arms decline is a
      DETERMINISTIC production refuser (``SchwingerCertificationError``,
      ``lnL = -inf`` -- no exact engine exists above the QD ceiling) and is
      labelled ``'refused'``, never exact-wave demand.

    ``select_branch`` is consulted ONLY above the QD ceiling: at or below
    it production runs the band ladder unconditionally, so a geometric (or
    refused) label there would be unfaithful.

    Each node is wrapped in the refusal-plus-degeneracy except tuple
    (``ZeroDivisionError`` included: ``caustic_rho`` raises it at ``gamma ==
    0``, not a domain error), and a degenerate node routes to exact-wave
    demand -- the conservative (demand-overcounting) direction, kept even
    above the QD ceiling where the ladder outcome is indeterminate.

    Returns the per-node `ROUTE_KINDS` vector (one entry per frequency node).
    """
    # The saddle (macro-saddle, gamma >= 1) has no cancellation-exponent
    # analogue; ``cancellation_exponent`` would raise there (1 - kappa <=
    # |gamma|).  Mirror the production routers: pass ``inf`` for the saddle so
    # only the resolution and caustic-distance legs of the gate are live.
    saddle_host = (1.0 <= abs(gamma))  # kappa == 0 by the census contract
    node_except = mods.refusal_errors + (ValueError, ZeroDivisionError)

    kinds: list[str] = []
    for w_value in w_grid:
        w = float(w_value)
        try:
            if w > mods.w_ceiling_qd:
                exponent = (math.inf if saddle_host
                            else mods.cancellation_exponent(
                                w, source, gamma, 0.0))
                if mods.select_branch(
                        w, delta_min, exponent, eta) == 'geometric':
                    kinds.append('geometric')
                elif mods.uniform_arm_value(w, source, gamma) is not None:
                    kinds.append(_resolve_arm_kind(mods, w, source, gamma))
                else:
                    kinds.append('refused')
            elif w > mods.w_ceiling_dd:
                if mods.uniform_arm_value(w, source, gamma) is not None:
                    kinds.append(_resolve_arm_kind(mods, w, source, gamma))
                else:
                    kinds.append('exact_wave')
            else:
                kinds.append('exact_wave')
        except node_except:
            kinds.append('exact_wave')
    return tuple(kinds)


# ---------------------------------------------------------------------------
# Draw-level classifier (pure)
# ---------------------------------------------------------------------------

def _caustic_rho_or_none(gamma: float, abs_y: float) -> float | None:
    """Caustic-relative reach ``rho``, or ``None`` where the gauge is undefined.

    ``caustic_rho`` raises ``ValueError`` / ``LensDomainError`` at the parity
    boundary and a raw ``ZeroDivisionError`` at ``gamma == 0`` (no caustic);
    all three collapse to ``None`` (the ``'undetermined'`` bucket).
    """
    try:
        return float(ppgo_map.caustic_rho(gamma, abs_y, 0.0))
    except (ValueError, geometry.LensDomainError, ZeroDivisionError):
        return None


def _ppgo_ceiling_gate_passes(mods: _ProductionModules, geom: Any) -> bool:
    """Mirror of the production ``_ppgo_above_ceiling`` admission gate.

    Leg-for-leg mirror of the gate in
    ``likelihood.LensedRelativeBinningLikelihood._ppgo_above_ceiling`` up to
    (and excluding) the serve itself: at least two real-image delays, a
    positive minimum pairwise delay separation, and the ceiling-keyed
    resolution gate ``W_CEILING_SCHWINGER_QD * min_delta_tau >= RHO_END``
    (both constants bound from the production modules, never re-typed).
    The caller has already established ``w_hi > W_CEILING_SCHWINGER_QD``
    (the rung's entry guard).  A ``True`` here is the production serve:
    fold-corrected ppGO above the ceiling, exact engine below -- the
    engine-below band is the constant split at the ceiling, so no per-draw
    split detail is recorded.  All quantities are geometry-only
    (``geom.delays`` / ``geom.real_mask``): zero engine calls.
    """
    real = np.asarray(geom.real_mask, dtype=bool)
    real_delays = np.asarray(geom.delays)[real]
    if len(real_delays) < 2:
        return False
    delta_taus = np.diff(np.sort(real_delays))
    positive_deltas = delta_taus[delta_taus > 0]
    if len(positive_deltas) == 0:
        return False
    min_delta_tau = float(np.min(positive_deltas))
    return mods.w_ceiling_qd * min_delta_tau >= mods.rho_end


def _saddle_c3_route(mods: _ProductionModules, real_images: np.ndarray,
                     source: np.ndarray, matrix: np.ndarray, w_lo: float,
                     w_hi: float) -> tuple[bool, float | None]:
    """Whether the saddle c3 rung serves this draw END-TO-END, plus w_split.

    Thin mirror of the serve decision in
    ``likelihood.LensedRelativeBinningLikelihood._saddle_farfield_analytic``,
    calling the production helpers (``_saddle_farfield_analytic_serves``,
    ``_saddle_min_image_sep``, ``_saddle_c3_split_point``) so the served set
    and the counted set can never skew:

    * whole-band admit -- the gate passes at the band floor ``w_lo``
      (equivalently ``w_split <= w_lo``): analytic zero envelope over the
      whole band, no engine demand;
    * band split -- the gate fails at ``w_lo`` but the separation backstop
      holds and the certificate split point satisfies
      ``w_lo < w_split < w_hi`` with ``w_split <= W_CEILING_SCHWINGER_QD``:
      analytic zero envelope above ``w_split``, exact engine below;
    * refuse -- a merging pair (``est`` and hence ``w_split`` is ``None``;
      the whole-draw refusal), an under-separated pair (separation
      backstop), a certificate failing across the whole band
      (``w_split >= w_hi``), or a split beyond the exact engine's ceiling
      (``w_split > W_CEILING_SCHWINGER_QD``).

    Returns ``(serves, w_split)``; ``w_split`` is recorded on every serve
    (for the whole-band admit it is the same certificate inversion,
    ``<= w_lo``) so the tiling design can size the below-split chart demand.
    """
    if mods.saddle_farfield_serves(real_images, source, matrix, w_lo):
        w_split = mods.saddle_c3_split_point(real_images, source, matrix)
        return True, None if w_split is None else float(w_split)
    min_sep = mods.saddle_min_image_sep(real_images)
    if min_sep is None or min_sep < mods.saddle_min_sep_floor:
        return False, None
    w_split = mods.saddle_c3_split_point(real_images, source, matrix)
    if (w_split is None or w_split >= w_hi
            or w_split > mods.w_ceiling_qd):
        return False, None
    return True, float(w_split)


def classify_draw(mods: _ProductionModules, *, gamma: float,
                  m_lens_msun: float, y1: float, y2: float,
                  f_grid: np.ndarray, gamma_edges: np.ndarray,
                  artifact: Any | None = None) -> DrawResult:
    """Classify one lens prior draw into exactly one `SERVE_ROUTES` label.

    Decision waterfall (first-admitting), mirroring the PRODUCTION rung
    order of ``likelihood._amplification_coefficients`` (surrogate ->
    above-ceiling ppGO -> saddle c3 -> Born -> exact engine):

    1. ``engine_refused`` -- the real geometry partition (or its construction)
       raises a named domain refusal; the source produces ``lnL = -inf`` and
       is not a training target.  Decided FIRST via the production geometry
       path, never a heuristic; this is the largest single population, so
       folding it into ``engine_residual`` would grossly over-size the
       campaign.
    2. ``surrogate`` -- artifact mode only (WP2 threads a loaded artifact in).
       In demand mode (``artifact is None``) this label is never emitted; the
       invariant is asserted below.
    3. ``ppgo_above_ceiling`` -- the band ceiling exceeds the Schwinger QD
       ceiling (``w_hi > W_CEILING_SCHWINGER_QD``) AND the production
       ceiling-keyed resolution gate admits (`_ppgo_ceiling_gate_passes`:
       ``W_CEILING_SCHWINGER_QD * min_delta_tau >= RHO_END``).  Production
       serves the fold-corrected ppGO carrier above the ceiling and the
       exact engine below (the constant 150 split).  This rung fires BEFORE
       the saddle rung, so a draw both rungs would serve counts here --
       production precedence is the law.
    4. ``saddle_c3`` -- a macro saddle (``gamma > 1``) the c3 rung serves
       END-TO-END (`_saddle_c3_route`, a thin caller of the production
       helpers; the certificate is never reimplemented): EITHER the
       whole-band admit at the band floor ``w_lo`` (``w_split <= w_lo``:
       zero envelope, no engine demand) OR the c3 band split
       (``w_lo < w_split < w_hi`` and ``w_split <= W_CEILING_SCHWINGER_QD``:
       analytic zero envelope above ``w_split``, exact engine below).  A
       merging pair (``est`` is ``None``) refuses the whole draw and falls
       onward.  ``w_split`` is recorded in the result detail.
    5. ``born_analytic`` -- the Born weak-deflection exterior by geometric
       predicate alone: ``kappa == beta == 0`` (fixed here), ``gamma != 0``
       and ``caustic_rho > 2`` (the production Born intercept's coverage
       predicate, minus the chart-box test that demand mode has no chart
       for).
    6. per-node pass, resolved in this precedence: ``wave_refused`` (any
       node is an above-QD-ceiling deterministic refuser -- production
       raises ``SchwingerCertificationError`` and the draw's ``lnL`` is
       ``-inf``, so it is a production REFUSAL, not engine demand), then
       ``engine_residual`` (any node is exact-wave demand), then
       ``analytics_engine_hosted`` (every node served by the engine-hosted
       analytics -- geometric asymptote and/or uniform arms).

    The per-node pass is reached ONLY after intercepts 1-5 decline, so
    ``wave_refused`` is now a DERIVED, named set: intercept 3 absorbs every
    above-ceiling draw whose ceiling gate passes and intercept 4 absorbs
    the c3-served saddles (whole-band AND band-split -- a ``w_hi > 150``
    saddle the c3 rung serves classifies ``saddle_c3``, never
    ``wave_refused``), so only draws with above-ceiling nodes whose ceiling
    gate FAILED (``W_CEILING_SCHWINGER_QD * min_delta_tau < RHO_END`` --
    the deferred-2b unresolved near-caustic corner) can still carry a
    refuser node.  Within the node pass ``wave_refused`` precedes
    ``engine_residual`` because production's any-refuser ->
    whole-grid-refusal is DETERMINISTIC (no engine exists above the
    ceiling), while an exact-wave node is merely potential engine demand.
    The labels stay MECE: exactly one route per draw (asserted below).
    """
    w_grid = mods.dimensionless_frequency(f_grid, m_lens_msun, 0.0)
    log_w = np.log(w_grid)
    log_w_min, log_w_max = float(log_w.min()), float(log_w.max())
    w_lo = float(w_grid.min())
    w_hi = float(w_grid.max())
    source = np.array([y1, y2], dtype=float)
    abs_y = math.hypot(y1, y2)
    rho = _caustic_rho_or_none(gamma, abs_y)

    region = _region_of(gamma, rho)
    gamma_band = _gamma_band_of(gamma, gamma_edges)
    w_band = _w_band_of(log_w_max, mods.w_ceiling_dd, mods.w_ceiling_qd)

    def _result(route: str, kinds: tuple[str, ...],
                w_split: float | None = None) -> DrawResult:
        return DrawResult(
            gamma=gamma, m_lens_msun=m_lens_msun, y1=y1, y2=y2, route=route,
            node_route_kinds=kinds, region=region, gamma_band=gamma_band,
            w_band=w_band, caustic_rho=rho, log_w_min=log_w_min,
            log_w_max=log_w_max, w_split=w_split)

    # --- Intercept 1: engine_refused (decided first, via the real path) ---
    try:
        geom = mods.channels_cls(w_grid).geometry_partition(
            gamma=gamma, y=(y1, y2), beta=0.0, kappa=0.0)
    except mods.refusal_errors:
        return _result('engine_refused', ())

    # --- Intercept 2: surrogate (artifact mode only) ---
    if _artifact_surrogate_serves(
            artifact, geom=geom, w_grid=w_grid, gamma=gamma, y1=y1, y2=y2,
            log_w_min=log_w_min, log_w_max=log_w_max):
        return _result('surrogate', ())

    # --- Intercept 3: ppgo_above_ceiling (above-ceiling band split) ---
    # Production rung order: `_ppgo_above_ceiling` fires BEFORE the saddle
    # rung, so a draw both rungs would serve counts here.  Entry guard and
    # gate mirror the production method exactly (constants bound from
    # production).
    if w_hi > mods.w_ceiling_qd and _ppgo_ceiling_gate_passes(mods, geom):
        return _result('ppgo_above_ceiling', ())

    matrix = mods.macro_matrix(gamma, 0.0, 0.0)
    # ``geom.images`` is already the real-only array (``geometry.find_images``);
    # it must NOT be re-masked with the length-4 channel ``real_mask`` (the
    # double-mask hazard on 2-image saddle draws).
    real_images = np.asarray(geom.images)
    eta = float(geom.caustic_distance)

    # --- Intercept 4: saddle_c3 (macro saddle far-field, c3 band split) ---
    # Gated on the saddle parity exactly as the production serve rung;
    # `_saddle_c3_route` is a thin caller of the production helpers, so the
    # census served set is a byte-faithful mirror (served == counted) of
    # BOTH the whole-band-analytic admit and the c3 band split.
    if gamma > 1.0:
        c3_serves, w_split = _saddle_c3_route(
            mods, real_images, source, matrix, w_lo, w_hi)
        if c3_serves:
            return _result('saddle_c3', (), w_split=w_split)

    # --- Intercept 5: born_analytic (Born exterior, geometric predicate) ---
    if gamma != 0.0 and rho is not None and rho > _BORN_RHO_FLOOR:
        return _result('born_analytic', ())

    # --- Per-node pass (reached ONLY after intercepts 1-5 decline) ---
    delta_min = mods.real_delay_min_separation(source, matrix)
    kinds = _classify_nodes(mods, gamma=gamma, source=source, w_grid=w_grid,
                            delta_min=delta_min, eta=eta)

    if 'refused' in kinds:
        # Only reachable with above-ceiling nodes AND a failed ceiling gate
        # (intercept 3 absorbed every gate-passer): the deferred-2b
        # unresolved near-caustic corner, a production refusal.
        route = 'wave_refused'
    elif 'exact_wave' in kinds:
        route = 'engine_residual'
    else:
        route = 'analytics_engine_hosted'

    # MECE invariant: demand mode never emits 'surrogate'; every intercept
    # returns before the node pass, so an intercept route and a node-pass
    # route can never coexist for one draw (a band-split serve's
    # engine-below demand is per-draw DETAIL -- w_split / the constant
    # ceiling split -- not a second label).
    assert not (artifact is None and route == 'surrogate')
    return _result(route, kinds)


def _artifact_surrogate_serves(artifact: Any | None, *, geom: Any,
                               w_grid: np.ndarray, gamma: float, y1: float,
                               y2: float, log_w_min: float,
                               log_w_max: float) -> bool:
    """Whether a loaded surrogate artifact would serve this draw (engine-free).

    Demand mode passes ``artifact is None`` and this is always ``False`` -- no
    ``surrogate`` label is emitted.  Artifact mode (the 7b acceptance census)
    threads a loaded `LensAmplificationSurrogate` in and mirrors the production
    surrogate intercept (`likelihood._amplification_coefficients`): the cheap
    `may_serve` ``(gamma, log w)``-box pre-check, then the full multi-chart
    `serve` guard stack keyed on the geometry-only partition's caustic distance
    (``eta``), caustic arc angle (``theta`` gauge) and real-image count.

    The whole-band ``may_serve`` / ``serve`` are checked over the draw's full
    ``w`` band; the production Born-residual band split (which serves a chart
    only up to a per-cell ``w_trust``) is deliberately NOT reproduced -- this
    census records the surrogate's whole-draw INTERCEPT route, not the per-bin
    envelope, so a sub-band serve is treated as a whole-draw serve.

    ``serve`` recomputes NO geometry and evaluates only the chart splines, so
    consulting the surrogate makes ZERO exact-wave engine calls -- the census's
    engine-free guarantee is preserved.  Returns the surrogate's own ``served``
    verdict.
    """
    if artifact is None:
        return False
    if not artifact.may_serve(gamma, log_w_min, log_w_max):
        return False
    _, served, _ = artifact.serve(
        w_grid, gamma=gamma, y1=y1, y2=y2, beta=0.0,
        eta=float(geom.caustic_distance), theta=float(geom.caustic_theta),
        image_count=int(geom.real_mask.sum()))
    return bool(served)


# ---------------------------------------------------------------------------
# Aggregation (pure)
# ---------------------------------------------------------------------------

def _cell_key(result: DrawResult) -> str:
    """Aggregation cell key ``region|gamma_band|w_band``."""
    return f'{result.region}|{result.gamma_band}|{result.w_band}'


def aggregate_cells(results: list[DrawResult]) -> dict[str, dict[str, Any]]:
    """Bucket draws into ``(region x gamma-band x w-band)`` route tallies."""
    cells: dict[str, dict[str, Any]] = {}
    for result in results:
        cell = cells.setdefault(
            _cell_key(result),
            {'region': result.region, 'gamma_band': result.gamma_band,
             'w_band': result.w_band, 'total': 0,
             'routes': {route: 0 for route in SERVE_ROUTES}})
        cell['total'] += 1
        cell['routes'][result.route] += 1
    return cells


def residual_demand(results: list[DrawResult]) -> dict[str, Any]:
    """Split the ``engine_residual`` population by caustic-relative reach.

    Only true engine demand is split: ``wave_refused`` draws are production
    refusals (``lnL = -inf``), not demand, and are EXCLUDED here by the
    ``route == 'engine_residual'`` filter.

    Three-way split keyed on ``caustic_rho`` (NEVER ``rho_lobe``): Born-chart
    demand (``rho > 2``), near-caustic / tube (``1 < rho <= 2``) and interior
    (``rho <= 1``); an unresolved ``rho`` is a fourth ``undetermined`` bucket.
    Count and prior-mass fraction are EQUAL by construction (equal-weight
    draws), so a single fraction (count / total) is reported and this equality
    is stated in the census header.
    """
    counts: dict[str, int] = defaultdict(int)
    for result in results:
        if result.route != 'engine_residual':
            continue
        rho = result.caustic_rho
        if rho is None:
            counts['undetermined'] += 1
        elif rho > _RESIDUAL_BORN_FLOOR:
            counts['born_chart_demand'] += 1
        elif rho > _RESIDUAL_TUBE_FLOOR:
            counts['near_caustic_tube'] += 1
        else:
            counts['interior'] += 1

    total = sum(counts.values())

    def _bucket(name: str) -> dict[str, float | int]:
        count = int(counts.get(name, 0))
        fraction = (count / total) if total else 0.0
        return {'count': count, 'prior_mass_fraction': float(fraction)}

    return {
        'total': int(total),
        'split_gauge': 'caustic_rho',
        'note': ('count and prior_mass_fraction are equal by construction '
                 '(equal-weight draws); the split is on caustic-relative rho, '
                 'never rho_lobe.'),
        'born_chart_demand': _bucket('born_chart_demand'),
        'near_caustic_tube': _bucket('near_caustic_tube'),
        'interior': _bucket('interior'),
        'undetermined': _bucket('undetermined'),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(config: ServeRouteCensusConfig | None = None,
        artifact: Any | None = None) -> dict[str, Any]:
    """Run the serve-route demand census and return its report dict.

    Parameters
    ----------
    config : ServeRouteCensusConfig, optional
        Census settings; defaults to fixture-scale
        `ServeRouteCensusConfig` (``n_samples = 10_000``).
    artifact : object, optional
        A loaded surrogate artifact enabling the ``surrogate`` route (WP2).
        ``None`` (default) is DEMAND mode: the ``surrogate`` label is never
        emitted.

    Returns
    -------
    dict
        JSON-serializable report under ``schema = 'serve_route_census_v1'``.
        The function neither prints nor writes files.
    """
    config = config or ServeRouteCensusConfig()
    mods = _load_production_modules()
    f_grid = _frequency_grid(config)
    gamma_edges = ppgo_map._gamma_band_edges()
    samples = mods.draw_samples(config)

    results: list[DrawResult] = []
    for row in samples.itertuples(index=False):
        results.append(classify_draw(
            mods, gamma=float(row.gamma),
            m_lens_msun=float(row.m_lens_msun), y1=float(row.y1),
            y2=float(row.y2), f_grid=f_grid, gamma_edges=gamma_edges,
            artifact=artifact))

    route_counts = {route: 0 for route in SERVE_ROUTES}
    for result in results:
        route_counts[result.route] += 1

    return {
        'schema': 'serve_route_census_v1',
        'header': {
            'mode': 'artifact' if artifact is not None else 'demand',
            'engine_free_guarantee': (
                'zero exact-wave engine calls: ChangRefsdalChannels.evaluate, '
                '_schwinger.f_schwinger / _f_schwinger_mpmath and the mpmath '
                'special functions are never invoked; a draw or node that '
                'would reach them is COUNTED as engine demand.'),
            'equal_weight_note': (
                'draws are equal-weight by construction (sampled coordinates, '
                'no importance weights), so every count equals its '
                'prior-mass fraction times n_samples.'),
            'serve_routes_decision_order': [
                'engine_refused', 'surrogate', 'ppgo_above_ceiling',
                'saddle_c3', 'born_analytic',
                'wave_refused|engine_residual|analytics_engine_hosted'],
            'serve_routes': list(SERVE_ROUTES),
            'route_kinds': list(ROUTE_KINDS),
            'w_band_edges': {
                'w_ceiling_dd': mods.w_ceiling_dd,
                'w_ceiling_qd': mods.w_ceiling_qd,
                'keyed_on': 'band ceiling w_hi = exp(log_w_max)'},
            'gamma_band_edges': [float(edge) for edge in gamma_edges],
        },
        'config': dataclasses.asdict(config),
        'n_samples': len(results),
        'route_counts': route_counts,
        'records': [result.as_record() for result in results],
        'cells': aggregate_cells(results),
        'residual_demand': residual_demand(results),
    }
