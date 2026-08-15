"""Engine-free tiling census + node-budget predictor for the lens surrogate.

WHAT
----
`run` walks the SAME ``parity x gamma-band x region`` tiling the production
trainer (`cogwheel.lensing.surrogate_training._train_band_charts`) walks, but
calls ONLY the engine-free geometry + tiler helpers to COUNT fold arcs, tiles
and interpolation nodes.  It never evaluates a wave-optics amplitude, so it is
a cheap pre-campaign advisory: it predicts the campaign engine call-count,
flags silent-empty or exploding tile counts against two-sided expected bands,
cross-checks against the production ``_self_estimate``, and answers four
standing design questions (Q1-Q4) about the tiling.

WHY
---
A production training campaign costs hours of engine calls.  A silent-empty
region (a band whose tiler returns ``[]`` and vanishes with no record) or an
exploding tile count (a mis-sized grid) is far cheaper to catch here than
after the campaign.  The census is REPORT EVIDENCE, never an assertion gate:
it computes counts + verdicts + question answers and RETURNS them for a human
or a downstream gate to read; it neither prints nor writes files (the CLI in
``scripts/tiling_census.py`` does all I/O).

CONSERVATIVE UPPER BOUND (no ppGO trim modeled)
-----------------------------------------------
``per_region`` counts and ``aggregate_call_count`` are computed WITHOUT
modeling the certified ppGO map's stratum/window trim
(``surrogate_training._apply_ppgo_trim``, which every real ``train()`` run
installs via ``get_certified_ppgo_map()``); with a map, production drops whole
ppGO-certified strata/windows, which only REDUCES counts.  The census
therefore reports a conservative UPPER BOUND on the real campaign's node and
call count -- never an underestimate -- and a real campaign's intentional
ppGO-served empties are not mis-flagged as ``SILENT_EMPTY`` coverage holes.
The returned dict carries ``ppgo_trim_modeled: False`` so a downstream cost
estimate can treat the number programmatically as an upper bound.

ENGINE-FREE BY CONSTRUCTION
---------------------------
The module top-level imports only ``numpy`` and the engine-free
``chang_refsdal.geometry`` + ``ppgo_map`` layers, so
``import cogwheel.lensing.tiling_census`` pulls NO amplitude engine.  The
production tilers live in ``surrogate_training`` (which imports the
``ChangRefsdalChannels`` engine class at module load); the census imports that
module LAZILY inside `run`, and calls only its pure geometry + tiler helpers --
it makes ZERO engine calls (``ChangRefsdalChannels.evaluate``, ``_schwinger``
and the ``mpmath`` special-function paths are never touched).

DEFINITIONS
-----------
'engine' (forbidden): wave-optics amplitude EVALUATION only -- the channel
evaluate path, ``_schwinger`` / ``_f_schwinger`` and the ``mpmath`` special
functions.  'geometry' (allowed, pure): image finding (Newton quartic),
caustic curves, magnifications, delays and the closed-form
``ppgo_error_estimate`` ``w**-3`` series term.  The census calls the latter
freely and none of the former.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cogwheel.lensing import ppgo_map
from cogwheel.lensing.chang_refsdal import geometry

# ---------------------------------------------------------------------------
# Report-evidence constants (NEVER asserted -- returned for a human gate)
# ---------------------------------------------------------------------------

#: Verdict labels for a single (region x parity) count against its band.
SILENT_EMPTY = 'SILENT_EMPTY'
IN_BAND = 'IN_BAND'
EXPLOSION = 'EXPLOSION'

#: Region admissibility by parity (the SAME per-parity tuples the production
#: ``_train_band_charts`` / ``_self_estimate`` honour): the positive-parity
#: astroid charts a tube shell, an origin-centred exterior remainder and a
#: wedge interior; the macro-saddle deltoid charts a tube shell and two
#: lobe-local (interior + exterior) families.
_REGIONS_BY_PARITY: dict[int, tuple[str, ...]] = {
    1: ('tube', 'exterior', 'wedge_interior'),
    -1: ('tube', 'lobe_interior', 'lobe_exterior'),
}

#: Two-sided expected ``(low, high)`` bands per ``(region, parity)`` for the
#: ``(n_arcs, n_tiles, n_nodes)`` triple.  These are COARSE REPORT EVIDENCE --
#: sanity envelopes a human reads, NOT invariants.  ``'arcs'`` is ``None``
#: where fold-arc counting does not apply (only the tube region charts arcs).
#: A count of 0 (or below the low edge) is flagged ``SILENT_EMPTY`` so a
#: region that vanished silently is surfaced; above the high edge is
#: ``EXPLOSION``.
_EXPECTED_BANDS: dict[tuple[str, int], dict[str, tuple[int, int] | None]] = {
    ('tube', 1): {'arcs': (1, 4), 'tiles': (1, 200), 'nodes': (1, 10 ** 9)},
    ('tube', -1): {'arcs': (1, 6), 'tiles': (1, 400), 'nodes': (1, 10 ** 9)},
    ('exterior', 1): {'arcs': None, 'tiles': (1, 10000),
                      'nodes': (1, 10 ** 10)},
    ('wedge_interior', 1): {'arcs': None, 'tiles': (1, 10000),
                            'nodes': (1, 10 ** 10)},
    ('lobe_interior', -1): {'arcs': None, 'tiles': (1, 10000),
                            'nodes': (1, 10 ** 10)},
    ('lobe_exterior', -1): {'arcs': None, 'tiles': (1, 10000),
                            'nodes': (1, 10 ** 10)},
}

#: Engine evaluations charged per interpolation node, mirroring the ``8`` in
#: ``surrogate_training._self_estimate`` (a node's held-out probe + envelope
#: labels).  Kept as a named constant so the cross-check reads the same model.
_LABELS_PER_NODE = 8

#: Seconds per engine evaluation, mirroring the ``0.09`` in ``_self_estimate``.
_SECONDS_PER_LABEL = 0.09

#: The census aggregate (tile-count aware) and the production ``_self_estimate``
#: (one tile per region) may diverge by up to this factor before the divergence
#: is flagged.  A REPORT SIGNAL, never raised.
_CROSS_CHECK_FACTOR = 5000.0

#: Q2 deltoid mis-allocation ratio above which a far-field-tiling redesign is
#: advised (Professor: 2-3 is the danger zone).
_Q2_MISALLOC_THRESHOLD = 2.5

#: Q4 saddle far-field serve-floor coefficient: ``w_floor = (2e4 * K)**(1/3)``.
_Q4_SADDLE_FLOOR_COEFF = 2.0e4

#: Q4 astroid double-double product ceiling: ``w * |y| <= _DD_PRODUCT_MARGIN``
#: sets the ceiling ``st._DD_PRODUCT_MARGIN / sqrt(s)``.  Read at the use
#: site from the production module (whose import is deferred, see below)
#: rather than mirrored here -- a third re-typed copy is mirror-drift
#: waiting to happen (two allowlisted mirrors already exist), and the part0
#: absorber guard rightly flagged a new one.


class TilingCensusError(Exception):
    """Base class for tiling-census failures."""


class MalformedConfigError(TilingCensusError):
    """The supplied config lacks a field the census multiplies against."""


#: TrainingConfig fields the node budget reads; a missing one is malformed.
_REQUIRED_CONFIG_FIELDS = (
    'n_gamma', 'n_u', 'n_theta', 'n_rho', 'n_theta_c', 'w_nodes_per_decade',
    'f_max', 'gamma_band_halfwidth', 'min_gamma_band', 'max_tube_arcs',
    'n_farfield_tiles_per_side', 'n_caustic_samples',
    'gamma_refine_near_one_window', 'gamma_refine_near_one_width',
)


@dataclass
class _BandCtx:
    """Engine-free derived geometry of one topology-stable gamma band.

    Mirrors the setup chain at the head of
    ``surrogate_training._train_band_charts`` (every quantity is computed by a
    pure geometry / tiler helper), so the region counters below are thin
    callers rather than re-derivations.
    """

    parity: int
    band: tuple[float, float]
    structure: Any
    gamma_mid: float
    tube_arcs: list
    max_eta_max: float
    reach_scalar: float
    coordinate_radius_min: float
    reach_max: float
    strata: list
    m_lo_region: float
    m_hi_region: float
    y_outer_region: float
    rho_outer_region: float
    exclusion_rho: float
    degenerate: bool
    inradius: float
    encloses: bool
    saddle_lobe_admissions: Any = None
    notes: dict[str, str] = field(default_factory=dict)


def _load_production_modules() -> tuple[Any, Any]:
    """Lazily import the engine-bearing production modules.

    Deferred so ``import cogwheel.lensing.tiling_census`` stays engine-free:
    ``surrogate_training`` imports ``ChangRefsdalChannels`` at module load.
    Importing the class is not calling it -- the census still makes zero
    engine evaluations.
    """
    from cogwheel.lensing import surrogate_training as st
    from cogwheel.lensing import surrogate as sg
    return st, sg


def _validate_config(config: Any) -> None:
    """Raise `MalformedConfigError` if a required node-budget field is absent."""
    missing = [f for f in _REQUIRED_CONFIG_FIELDS if not hasattr(config, f)]
    if missing:
        raise MalformedConfigError(
            'TrainingConfig is missing field(s) the tiling census multiplies '
            f'against: {missing}. Pass a cogwheel.lensing.surrogate_training.'
            'TrainingConfig (or a compatible object exposing these fields).')


def _spatial_nodes_per_tile(region: str, config: Any) -> int:
    """Spatial interpolation nodes per tile, exactly as ``_self_estimate``.

    tube -> ``n_theta * n_u``; exterior -> ``n_rho * n_theta_c``; the
    interior families store one caustic-relative chart per tile (unit spatial
    factor -- the gamma and w axes carry their density).
    """
    return {
        'tube': config.n_theta * config.n_u,
        'exterior': config.n_rho * config.n_theta_c,
        'wedge_interior': 1,
        'lobe_interior': 1,
        'lobe_exterior': 1,
    }[region]


def _w_nodes(config: Any) -> int:
    """Effective w-axis node count, mirroring ``_self_estimate``.

    ``int(w_nodes_per_decade * 2.0)`` -- roughly two decades of w per chart.
    """
    return int(config.w_nodes_per_decade * 2.0)


def _build_band_ctx(st: Any, box: Any, parity: int,
                    band: tuple[float, float], structure: Any,
                    config: Any) -> _BandCtx:
    """Replicate the engine-free per-band setup of ``_train_band_charts``.

    Raises whatever the underlying geometry helpers raise (e.g.
    ``geometry.LensDomainError`` on a degenerate band); the caller records the
    band as dropped and continues.
    """
    gamma_mid = 0.5 * (band[0] + band[1])
    tube_arcs = st._tube_training_arcs(structure, parity, config.max_tube_arcs)
    arc_r_min = [st._min_curvature_radius(band, arc, config.n_caustic_samples)
                 for arc in tube_arcs]
    max_eta_max = (config.f_max * max(arc_r_min)
                   if arc_r_min else config.f_max * 0.05)
    reach_scalar = st._scalar_caustic_reach(gamma_mid)
    coordinate_radius_min, reach_max = st._coordinate_radius_bounds(band, parity)
    strata, _beyond = st._mass_strata(box, parity)
    if strata:
        m_lo_region, m_hi_region = strata[0][0], strata[-1][1]
    else:
        m_lo_region, m_hi_region = box.m_lens_range
    y_outer_region = float(st._lens_prior._source_scale(m_lo_region))
    rho_outer_region = 1.0 + y_outer_region - coordinate_radius_min
    physical_exclusion_radius = reach_max + max_eta_max
    exclusion_rho = 1.0 + physical_exclusion_radius - coordinate_radius_min
    inradius, encloses = st._caustic_inradius(
        gamma_mid, parity, config.n_caustic_samples)
    saddle_lobe_admissions = None
    notes: dict[str, str] = {}
    if parity != 1:
        try:
            saddle_lobe_admissions = st._saddle_lobe_admissions(
                band, config, eta_max=max_eta_max)
        except geometry.LensDomainError as exc:
            notes['saddle_lobe_admissions'] = f'refused: {exc}'
    return _BandCtx(
        parity=parity, band=band, structure=structure, gamma_mid=gamma_mid,
        tube_arcs=tube_arcs, max_eta_max=max_eta_max, reach_scalar=reach_scalar,
        coordinate_radius_min=coordinate_radius_min, reach_max=reach_max,
        strata=strata, m_lo_region=m_lo_region, m_hi_region=m_hi_region,
        y_outer_region=y_outer_region, rho_outer_region=rho_outer_region,
        exclusion_rho=exclusion_rho,
        degenerate=rho_outer_region <= 1.0,
        inradius=inradius, encloses=encloses,
        saddle_lobe_admissions=saddle_lobe_admissions, notes=notes)


# ---------------------------------------------------------------------------
# Per-region tile counters (thin callers: each len()s a production tiler)
# ---------------------------------------------------------------------------

def _count_tube(ctx: _BandCtx) -> tuple[int, int, str | None]:
    """Return ``(n_arcs, n_tiles, skip_reason)`` for the tube shell.

    One tube chart per selected fold arc; ``n_tiles == n_arcs``.
    """
    n_arcs = len(ctx.tube_arcs)
    if n_arcs == 0:
        return 0, 0, 'no_fold_arc_selected'
    return n_arcs, n_arcs, None


def _count_exterior(ctx: _BandCtx, config: Any, st: Any
                    ) -> tuple[int, str | None]:
    """Positive-parity origin-centred exterior remainder tile count."""
    if ctx.parity != 1:
        return 0, 'wrong_parity'
    if ctx.degenerate:
        return 0, 'exterior_band_degenerate'
    admission = st._interior_admission(
        ctx.band, 1, ctx.reach_scalar, config, eta_max=ctx.max_eta_max)
    cusp_angles = st._cusp_source_angles(ctx.gamma_mid, config.n_caustic_samples)
    tiles = st._farfield_exterior_tiles(
        ctx.rho_outer_region, config.n_farfield_tiles_per_side,
        admission=admission, source_magnitude_max=ctx.y_outer_region,
        cusp_angles=cusp_angles, gamma=ctx.gamma_mid, gamma_band=ctx.band,
        ghost_drop_count=[0])
    if not tiles:
        return 0, 'zero_column_admission'
    return len(tiles), None


def _count_wedge_interior(ctx: _BandCtx, config: Any, st: Any
                          ) -> tuple[int, str | None]:
    """Positive-parity wedge caustic-relative interior tile count."""
    if ctx.parity != 1:
        return 0, 'wrong_parity'
    if not ctx.encloses:
        return 0, 'caustic_not_origin_enclosing'
    if ctx.reach_scalar <= ctx.max_eta_max:
        return 0, 'tube_shell_fills_interior'
    if not ctx.strata:
        return 0, 'no_mass_strata'
    n_tiles = 0
    gamma_rep = float(np.median(
        st._log_reach_gamma_axis(ctx.band, config.n_gamma, 'gamma')))
    for m_lo, _m_hi in ctx.strata:
        y_extent = float(st._lens_prior._source_scale(m_lo))
        grid_rho_extent = min(1.0, y_extent / ctx.coordinate_radius_min)
        r_extent = min(grid_rho_extent,
                       1.0 - ctx.max_eta_max / ctx.coordinate_radius_min)
        tiles = st._wedge_interior_tiles(
            gamma_rep, r_extent, config.n_farfield_tiles_per_side)
        n_tiles += len(tiles)
    return n_tiles, (None if n_tiles else 'zero_wedge_admission')


def _count_lobe(ctx: _BandCtx, config: Any, st: Any, *, exterior: bool
                ) -> tuple[int, str | None]:
    """Macro-saddle lobe interior/exterior tile count (canonical +y1 lobe)."""
    if ctx.parity == 1:
        return 0, 'wrong_parity'
    admissions = ctx.saddle_lobe_admissions
    if admissions is None:
        return 0, 'saddle_lobe_admissions_unavailable'
    n_tiles = 0
    for lens_center, lobe in zip(st._SADDLE_LOBE_CENTERS[1:], admissions[1:]):
        lobe_cusps = st._lobe_cusp_source_angles(
            ctx.gamma_mid, lens_center, lobe.centroid, config.n_caustic_samples)
        if exterior:
            tiles = st._lobe_exterior_tiles(
                lobe, lobe_cusps, config.n_farfield_tiles_per_side,
                ctx.rho_outer_region)
        else:
            tiles = st._lobe_interior_tiles(
                lobe, lobe_cusps, config.n_farfield_tiles_per_side)
        n_tiles += len(tiles)
    reason = None if n_tiles else 'zero_lobe_admission'
    return n_tiles, reason


def _count_region(ctx: _BandCtx, region: str, config: Any, st: Any
                  ) -> tuple[int | None, int, str | None]:
    """Dispatch to the region counter, returning ``(n_arcs, n_tiles, skip)``.

    ``n_arcs`` is ``None`` for the far-field / interior regions (fold arcs are
    a tube-only concept).
    """
    if region == 'tube':
        n_arcs, n_tiles, skip = _count_tube(ctx)
        return n_arcs, n_tiles, skip
    if region == 'exterior':
        n_tiles, skip = _count_exterior(ctx, config, st)
    elif region == 'wedge_interior':
        n_tiles, skip = _count_wedge_interior(ctx, config, st)
    elif region == 'lobe_exterior':
        n_tiles, skip = _count_lobe(ctx, config, st, exterior=True)
    elif region == 'lobe_interior':
        n_tiles, skip = _count_lobe(ctx, config, st, exterior=False)
    else:  # pragma: no cover - guarded by _REGIONS_BY_PARITY
        raise TilingCensusError(f'unknown region {region!r}')
    return None, n_tiles, skip


def _verdict(count: int, band: tuple[int, int] | None) -> str:
    """Two-sided verdict for one count against its report band."""
    if band is None:
        return IN_BAND
    low, high = band
    if count == 0 or count < low:
        return SILENT_EMPTY
    if count > high:
        return EXPLOSION
    return IN_BAND


# ---------------------------------------------------------------------------
# Standing design questions Q1-Q4 (engine-free diagnostics)
# ---------------------------------------------------------------------------

def _q1_arc_census(ctx_by_parity: dict[int, list[_BandCtx]], st: Any
                   ) -> dict[str, Any]:
    """Q1: detected vs trained fold-arc counts (post-F079 wrap fix)."""
    out: dict[str, Any] = {}
    for parity, name in ((1, 'astroid'), (-1, 'saddle')):
        rep = next((c for c in ctx_by_parity[parity]), None)
        if rep is None:
            out[name] = {'detected_arcs': None, 'trained_arcs': None,
                         'deferred': 'no topology-stable band'}
            continue
        detected = len(rep.structure.arcs)
        trained = len(rep.tube_arcs)
        out[name] = {
            'parity': parity,
            'detected_arcs': detected,
            'trained_arcs': trained,
            'detected_cusps': rep.structure.detected_cusps,
            'representative_band': [float(rep.band[0]), float(rep.band[1])],
        }
    return out


def _representative_saddle_ctx(ctx_by_parity: dict[int, list[_BandCtx]]
                               ) -> _BandCtx | None:
    """First non-degenerate saddle band with usable lobe admissions."""
    for ctx in ctx_by_parity.get(-1, []):
        if not ctx.degenerate and ctx.saddle_lobe_admissions is not None:
            return ctx
    for ctx in ctx_by_parity.get(-1, []):
        if not ctx.degenerate:
            return ctx
    return None


def _q2_deltoid_redesign(ctx_by_parity: dict[int, list[_BandCtx]], config: Any,
                         st: Any) -> dict[str, Any]:
    """Q2: deltoid far-field mis-allocation ratio + cusp-in-tile verdict.

    Tiles the representative saddle band with the LEGACY origin-polar additive
    scalar-reach deltoid tiler (`_farfield_tiles`) -- the gauge the lobe-local
    redesign replaced -- and measures the radial dynamic range spanned by
    equal per-tile node budgets: ``mis_alloc_ratio = max(outer rho edge) /
    min(outer rho edge)`` over the emitted tiles.  A ratio above
    ``_Q2_MISALLOC_THRESHOLD``, OR any tile whose angular span strictly
    contains a deltoid cusp ray (where the ``2/3``-power reach is non-monotone
    across the tile), flags a redesign.
    """
    ctx = _representative_saddle_ctx(ctx_by_parity)
    if ctx is None:
        return {'mis_alloc_ratio': None, 'redesign_needed': None,
                'reason': 'deferred: no non-degenerate saddle band'}
    cusp_angles = st._deltoid_cusp_source_angles(
        ctx.gamma_mid, config.n_caustic_samples)
    try:
        tiles = st._farfield_tiles(
            ctx.exclusion_rho, ctx.rho_outer_region,
            config.n_farfield_tiles_per_side, cusp_angles=cusp_angles,
            gamma=ctx.gamma_mid, gamma_band=ctx.band)
    except geometry.LensDomainError as exc:
        return {'mis_alloc_ratio': None, 'redesign_needed': None,
                'reason': f'deferred: legacy tiler refused ({exc})'}
    if not tiles:
        return {'mis_alloc_ratio': None, 'redesign_needed': None,
                'reason': 'deferred: legacy deltoid tiler emitted no tile'}
    outer_edges = [center[0] + half[0] for center, half, _i, _j in tiles]
    lo_edge = min(outer_edges)
    mis_alloc_ratio = float(max(outer_edges) / lo_edge) if lo_edge > 0 else \
        float('inf')
    tol = 1e-9
    cusp_in_tile = any(
        (center[1] - half[1] + tol) < cusp < (center[1] + half[1] - tol)
        for center, half, _i, _j in tiles for cusp in cusp_angles)
    redesign = mis_alloc_ratio > _Q2_MISALLOC_THRESHOLD or cusp_in_tile
    if cusp_in_tile:
        reason = 'cusp ray strictly inside a tile angular span'
    elif mis_alloc_ratio > _Q2_MISALLOC_THRESHOLD:
        reason = (f'radial dynamic range {mis_alloc_ratio:.2f} exceeds '
                  f'{_Q2_MISALLOC_THRESHOLD}')
    else:
        reason = 'additive scalar-reach tiling within tolerance'
    return {'mis_alloc_ratio': mis_alloc_ratio, 'redesign_needed': bool(redesign),
            'reason': reason, 'n_tiles': len(tiles),
            'representative_band': [float(ctx.band[0]), float(ctx.band[1])]}


def _q3_near_cusp_kink(ctx_by_parity: dict[int, list[_BandCtx]], config: Any,
                       st: Any, sg: Any) -> dict[str, Any]:
    """Q3: near-cusp ``u = d**(2/3)`` coordinate-map kink check (engine-free).

    Over each non-straddling near-cusp deltoid far-field tile, build the shipped
    cusp-adapted angular map (`surrogate._deltoid_cusp_axis_map`) and verify the
    tabulated ``u`` is (a) strictly monotone in ``theta`` (single-sign
    differences) and (b) finite with a strictly positive minimum step (so the
    inverse ``theta(u)`` slope is bounded -- no ``d**(-1/3)`` blow-up).
    """
    ctx = _representative_saddle_ctx(ctx_by_parity)
    if ctx is None:
        return {'kink_free': None, 'worst_tile': None,
                'deferred': 'no non-degenerate saddle band'}
    cusp_angles = st._deltoid_cusp_source_angles(
        ctx.gamma_mid, config.n_caustic_samples)
    if not cusp_angles:
        return {'kink_free': None, 'worst_tile': None,
                'deferred': 'no deltoid cusp ray resolved for the band'}
    try:
        tiles = st._farfield_tiles(
            ctx.exclusion_rho, ctx.rho_outer_region,
            config.n_farfield_tiles_per_side, cusp_angles=cusp_angles,
            gamma=ctx.gamma_mid, gamma_band=ctx.band)
    except geometry.LensDomainError as exc:
        return {'kink_free': None, 'worst_tile': None,
                'deferred': f'legacy tiler refused ({exc})'}
    checked = 0
    worst_tile: dict[str, Any] | None = None
    kink_free = True
    for center, half, i, j in tiles:
        theta_lo = float(center[1] - half[1])
        theta_hi = float(center[1] + half[1])
        if not 0.0 <= theta_lo < theta_hi <= math.pi / 2.0:
            continue
        cusp = min(cusp_angles, key=lambda a: abs(a - center[1]))
        if theta_lo < cusp < theta_hi:  # straddle -> map is None by design
            continue
        try:
            result = sg._deltoid_cusp_axis_map(theta_lo, theta_hi, cusp)
        except ValueError:
            continue
        if result is None:
            continue
        _theta_fine, u_fine = result
        checked += 1
        du = np.diff(u_fine)
        monotone = bool(np.all(du > 0.0) or np.all(du < 0.0))
        finite = bool(np.all(np.isfinite(u_fine)))
        min_step = float(np.min(np.abs(du))) if du.size else 0.0
        tile_ok = monotone and finite and min_step > 0.0
        if not tile_ok:
            kink_free = False
            worst_tile = {
                'tile_ij': [int(i), int(j)],
                'theta_range': [theta_lo, theta_hi],
                'cusp_angle': float(cusp),
                'monotone': monotone, 'finite': finite,
                'min_step': min_step}
            break
    if checked == 0:
        return {'kink_free': None, 'worst_tile': None,
                'deferred': ('no non-straddling near-cusp far-field tile in '
                             'the representative saddle band')}
    return {'kink_free': bool(kink_free), 'worst_tile': worst_tile,
            'n_tiles_checked': checked}


def _saddle_farfield_floor(ctx: _BandCtx, box: Any) -> tuple[float | None,
                                                             str | None]:
    """Saddle far-field effective serve floor ``(2e4 * K)**(1/3)``.

    ``K = sum_a sqrt|mu_a| |c3_a|`` over the real images of a representative
    exterior source, recovered from ``geometry.ppgo_error_estimate`` (which
    returns ``K / w_min**3``).  Returns ``(None, reason)`` when the estimate is
    uncertifiable (empty / non-finite ingredients) -- never a divide.
    """
    admissions = ctx.saddle_lobe_admissions
    if admissions is None:
        return None, 'saddle_lobe_admissions_unavailable'
    lobe = admissions[1]
    centroid = np.asarray(lobe.centroid, dtype=float)
    cmag = float(np.hypot(centroid[0], centroid[1]))
    r_max = float(np.max(lobe.boundary_r))
    direction = centroid / cmag if cmag > 0 else np.array([1.0, 0.0])
    source = centroid + direction * (1.2 * r_max)
    w_min = float(box.w_range(-1)[0])
    try:
        matrix = geometry.macro_matrix(ctx.gamma_mid)
        real_images = np.asarray(geometry.find_images(source, matrix))
        est = geometry.ppgo_error_estimate(real_images, source, matrix, w_min)
    except geometry.LensDomainError as exc:
        return None, f'geometry refused ({exc})'
    if est is None or w_min <= 0.0:
        return None, 'uncertifiable: ppgo_error_estimate returned None'
    k_amplitude = est * w_min ** 3
    floor = float((_Q4_SADDLE_FLOOR_COEFF * k_amplitude) ** (1.0 / 3.0))
    return floor, None


def _q4_w_band_containment(ctx_by_parity: dict[int, list[_BandCtx]], box: Any,
                           st: Any, regions: tuple[str, ...] | None
                           ) -> dict[str, Any]:
    """Q4: trained w-band vs effective serve floor/ceiling containment."""
    out: dict[str, Any] = {}
    astroid_rep = next((c for c in ctx_by_parity.get(1, [])
                        if not c.degenerate), None)
    saddle_rep = _representative_saddle_ctx(ctx_by_parity)
    for parity in (1, -1):
        admissible = _admissible_regions(parity, regions)
        w_lo, w_hi = (float(v) for v in box.w_range(parity))
        for region in admissible:
            key = f'{region}:{parity:+d}'
            entry: dict[str, Any] = {'parity': parity, 'region': region,
                                     'w_band': [w_lo, w_hi]}
            if parity == 1:
                s = astroid_rep.y_outer_region if astroid_rep else None
                if s and s > 0.0:
                    ceiling = min(st._POSITIVE_W_CEILING,
                                  st._DD_PRODUCT_MARGIN / math.sqrt(s))
                else:
                    ceiling = float(st._POSITIVE_W_CEILING)
                entry.update({
                    'effective_floor': w_lo, 'effective_ceiling': float(ceiling),
                    'contained': bool(w_hi <= ceiling),
                    'source_magnitude': (float(s) if s else None)})
            elif region == 'tube':
                floor = float(ppgo_map.SADDLE_WALL)
                ceiling = float(st._SADDLE_W_CEILING)
                entry.update({
                    'effective_floor': floor, 'effective_ceiling': ceiling,
                    'contained': bool(w_lo >= floor and w_hi <= ceiling)})
            else:  # saddle far-field lobe regions
                if saddle_rep is None:
                    entry.update({'effective_floor': None, 'contained': None,
                                  'reason': 'no non-degenerate saddle band'})
                else:
                    floor, reason = _saddle_farfield_floor(saddle_rep, box)
                    if floor is None:
                        entry.update({'effective_floor': None,
                                      'contained': None, 'reason': reason})
                    else:
                        entry.update({'effective_floor': floor,
                                      'contained': bool(w_lo >= floor)})
            out[key] = entry
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _admissible_regions(parity: int, regions: tuple[str, ...] | None
                        ) -> tuple[str, ...]:
    """Per-parity admissible regions, intersected with a requested filter."""
    parity_regions = _REGIONS_BY_PARITY[parity]
    if regions is None:
        return parity_regions
    return tuple(r for r in parity_regions if r in regions)


def _collect_band_contexts(st: Any, box: Any, parity: int, config: Any
                           ) -> tuple[list[_BandCtx], list[dict[str, Any]]]:
    """Build the topology-stable band contexts for one parity.

    Returns ``(contexts, dropped)`` where ``dropped`` records slivers dropped
    by ``stable_gamma_bands`` and bands whose engine-free setup refused.
    """
    band = st._gamma_band(box, parity, config.gamma_band_halfwidth)
    sub_bands, sliver_dropped = st.stable_gamma_bands(
        band, parity, n_samples=config.n_caustic_samples,
        min_width=config.min_gamma_band,
        refine_near_one_window=config.gamma_refine_near_one_window,
        refine_near_one_width=config.gamma_refine_near_one_width)
    contexts: list[_BandCtx] = []
    dropped: list[dict[str, Any]] = [
        {'band': [float(lo), float(hi)], 'reason': 'topology_sliver_dropped'}
        for lo, hi in sliver_dropped]
    for sub, structure in sub_bands:
        gamma_mid = 0.5 * (sub[0] + sub[1])
        if gamma_mid == 0.0:  # degenerate caustic_rho(gamma=0); skip loudly
            dropped.append({'band': [float(sub[0]), float(sub[1])],
                            'reason': 'gamma_zero_degenerate'})
            continue
        try:
            contexts.append(
                _build_band_ctx(st, box, parity, sub, structure, config))
        except (geometry.LensDomainError, ZeroDivisionError, ValueError) as exc:
            dropped.append({'band': [float(sub[0]), float(sub[1])],
                            'reason': f'setup_refused: {exc}'})
    return contexts, dropped


def _census_region(contexts: list[_BandCtx], region: str, parity: int,
                   config: Any, st: Any) -> dict[str, Any]:
    """Aggregate one (region x parity) over its band contexts into a record."""
    total_arcs = 0
    total_tiles = 0
    arcs_apply = False
    skip_reasons: dict[str, int] = {}
    for ctx in contexts:
        n_arcs, n_tiles, skip = _count_region(ctx, region, config, st)
        if n_arcs is not None:
            arcs_apply = True
            total_arcs += n_arcs
        total_tiles += n_tiles
        if skip is not None:
            skip_reasons[skip] = skip_reasons.get(skip, 0) + 1
    spatial = _spatial_nodes_per_tile(region, config)
    n_nodes = total_tiles * spatial * config.n_gamma * _w_nodes(config)
    bands = _EXPECTED_BANDS.get((region, parity), {})
    verdict_tiles = _verdict(total_tiles, bands.get('tiles'))
    verdict_nodes = _verdict(n_nodes, bands.get('nodes'))
    verdict_arcs = (_verdict(total_arcs, bands.get('arcs'))
                    if arcs_apply else IN_BAND)
    overall = EXPLOSION if EXPLOSION in (verdict_tiles, verdict_nodes,
                                         verdict_arcs) else (
        SILENT_EMPTY if SILENT_EMPTY in (verdict_tiles, verdict_nodes,
                                         verdict_arcs) else IN_BAND)
    record: dict[str, Any] = {
        'parity': parity, 'region': region, 'n_bands': len(contexts),
        'n_arcs': (total_arcs if arcs_apply else None),
        'n_tiles': total_tiles, 'n_nodes': int(n_nodes),
        'spatial_nodes_per_tile': spatial, 'w_nodes': _w_nodes(config),
        'verdict_arcs': (verdict_arcs if arcs_apply else None),
        'verdict_tiles': verdict_tiles, 'verdict_nodes': verdict_nodes,
        'verdict': overall,
        'expected_bands': {k: (list(v) if v else None)
                           for k, v in bands.items()},
        'skip_reasons': skip_reasons}
    return record


def run(config: Any, regions: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Census the engine-free tiling and predict the campaign node budget.

    Parameters
    ----------
    config : TrainingConfig
        A ``cogwheel.lensing.surrogate_training.TrainingConfig`` (or a
        compatible object exposing the node-budget fields).
    regions : tuple of str, optional
        Restrict the census to these region names (intersected with each
        parity's admissible set).  ``None`` (default) censuses every region.

    Returns
    -------
    dict
        Per-(region x parity) counts + verdicts, the aggregate campaign engine
        call-count, a ``self_estimate_seconds`` cross-check equal to the
        production ``_self_estimate``, and the Q1-Q4 answer fields.  The
        function neither prints nor writes files.  ``per_region`` counts and
        ``aggregate_call_count`` are computed WITHOUT modeling the certified
        ppGO map's stratum/window trim (``surrogate_training._apply_ppgo_trim``,
        installed via ``get_certified_ppgo_map()`` in every real ``train()``
        run), so they are a conservative UPPER BOUND on the real campaign's
        node/call count, never an underestimate; the returned
        ``ppgo_trim_modeled`` key (``False``) flags this for downstream
        consumers.

    Raises
    ------
    MalformedConfigError
        If ``config`` lacks a field the node budget multiplies against.
    """
    _validate_config(config)
    st, sg = _load_production_modules()
    box = st.PriorBox.from_prior_classes()

    ctx_by_parity: dict[int, list[_BandCtx]] = {}
    dropped_by_parity: dict[str, list[dict[str, Any]]] = {}
    for parity in (1, -1):
        contexts, dropped = _collect_band_contexts(st, box, parity, config)
        ctx_by_parity[parity] = contexts
        dropped_by_parity[f'{parity:+d}'] = dropped

    per_region: dict[str, dict[str, Any]] = {}
    aggregate_call_count = 0
    for parity in (1, -1):
        for region in _admissible_regions(parity, regions):
            record = _census_region(
                ctx_by_parity[parity], region, parity, config, st)
            per_region[f'{region}:{parity:+d}'] = record
            aggregate_call_count += record['n_nodes'] * _LABELS_PER_NODE

    census_seconds = aggregate_call_count * _SECONDS_PER_LABEL
    self_estimate_seconds = float(st._self_estimate(config, regions))
    ratio = (census_seconds / self_estimate_seconds
             if self_estimate_seconds > 0 else float('inf'))

    return {
        'schema': 'tiling_census_v1',
        'config': {f: getattr(config, f) for f in _REQUIRED_CONFIG_FIELDS},
        'regions_requested': (list(regions) if regions is not None else None),
        'aggregate_call_count': int(aggregate_call_count),
        'ppgo_trim_modeled': False,
        'labels_per_node': _LABELS_PER_NODE,
        'census_seconds': float(census_seconds),
        'self_estimate_seconds': self_estimate_seconds,
        'cross_check': {
            'ratio_census_over_self_estimate': float(ratio),
            'documented_factor': _CROSS_CHECK_FACTOR,
            'within_documented_factor': bool(ratio <= _CROSS_CHECK_FACTOR),
            'note': ('_self_estimate charges one tile per region; the census '
                     'is tile-count aware, so a ratio above 1 is expected and '
                     'is a report signal, never an assertion.')},
        'per_region': per_region,
        'dropped_bands': dropped_by_parity,
        'q1_arc_census': _q1_arc_census(ctx_by_parity, st),
        'q2_deltoid_redesign': _q2_deltoid_redesign(ctx_by_parity, config, st),
        'q3_near_cusp_kink': _q3_near_cusp_kink(ctx_by_parity, config, st, sg),
        'q4_w_band_containment': _q4_w_band_containment(
            ctx_by_parity, box, st, regions),
    }
