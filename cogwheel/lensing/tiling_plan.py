"""Demand-sized tiling plan + campaign cost estimate (engine-free).

WHAT
    Predicts the lens-amplification training campaign's per-region tile plan
    and total engine-call cost by

    (a) refreshing the engine-free serve-route demand census at build HEAD,
    (b) enumerating demand-sized tiles per ``region x parity x gamma_band`` by
        delegating to the production tilers (a thin consumer, exactly the
        `tiling_census` pattern -- never a hand count),
    (c) sizing every axis by ``n = ceil(span / resolution)`` (Professor's
        laws: gamma resolution from the caustic-reach derivative, theta from
        the F083 density constant, w from the measured demand-band edges and
        the per-decade carrier density, the far-field annulus in a declared
        gauge),
    (d) gating each chart tile on POSITIVE ``engine_residual`` demand for its
        census cell, and
    (e) reconciling the cost with three independent cross-checks and emitting
        an escalation verdict.

WHY
    A blanket-count tiling (``config.n_gamma x _w_nodes`` over every admitted
    tile) explodes to ~2e6 calls, ~82% of them on astroid-exterior cells that
    Born / c3 / certified-map analytics already serve.  Sizing each axis to
    its measured resolution and gating on residual demand collapses that
    budget; this module makes the prediction auditable BEFORE any wave
    evaluation is spent.

ENGINE-FREE CONTRACT
    This module performs ZERO wave evaluations.  It imports only the two
    engine-free predictor siblings (`serve_route_census`, `tiling_census`),
    which defer every engine-adjacent import to their own
    ``_load_production_modules``; the exact-wave entry points
    (``ChangRefsdalChannels.evaluate``, ``_schwinger.f_schwinger`` /
    ``_f_schwinger_mpmath``) are never called and ``mpmath`` must never enter
    ``sys.modules`` during a run.

    BOOBY-TRAP NOTE (for the Test Developer): assert engine-free by
    ``mock.patch``-ing those evaluate entry points to raise and asserting the
    full ``run`` completes with zero calls, and by asserting
    ``'mpmath' not in sys.modules`` after a run -- a namespace-absence check
    alone is insufficient (importing the package loads engine module OBJECTS;
    the guarantee is NO CALL, not no-import).

This module is pure prediction: it neither prints nor writes files.  The CLI
(`scripts/tiling_plan.py`) parses arguments, calls `run`, and writes the JSON.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from cogwheel.lensing import serve_route_census, tiling_census

SCHEMA = 'tiling_plan_v1'

# --- cost model -------------------------------------------------------------
# One engine "call" is a single node-label evaluation; a node carries
# ``_LABELS_PER_NODE`` labels.  ``SECONDS_PER_CALL`` (0.0903 s) is the measured
# per-CALL DD-band smoke rate (campaign_tiling_design Fact 6).  ``tiling_census``
# charges the same granularity as ``_SECONDS_PER_LABEL = 0.09`` s per label
# (== per call); the ~0.3% gap is smoke-run jitter, reconciled explicitly in
# the emitted ``cost_model`` note rather than silently reconciled to one value.
_LABELS_PER_NODE = tiling_census._LABELS_PER_NODE
SECONDS_PER_CALL = 0.0903

# Engine-residual honest ledger (campaign_tiling_design Fact 1): the census
# refresh's measured engine_residual share is reconciled against this.
_CENSUS_ENGINE_RESIDUAL_LEDGER = 0.4119

# --- axis-sizing law constants ---------------------------------------------
# GAMMA (law 1): gamma_res(gamma) = _C_GAMMA * r_caustic / |d r_caustic/dgamma|.
_C_GAMMA = 0.4
_GAMMA_DIFF_STEP = 1.0e-3

# THETA (law 2): n_theta = ceil(kappa_theta * trimmed_arc_span).  kappa_theta is
# the F083 tube density constant in nodes-per-radian: ~10 nodes over the ~0.28-
# rad resolvable core sub-arc gave eps 4.3e-3 (~12x margin under the 5e-2 bar)
# at w~52, i.e. ~36 nodes/rad.  ONE constant per parity; saddle provisionally
# mirrors astroid pending its own calibration (recorded in the plan provenance).
_KAPPA_THETA_NODES_PER_RAD: dict[int, float] = {1: 36.0, -1: 36.0}

# W (law 3): interior charts carry the dense carrier phase (per-decade node
# density ``interior_w_nodes_per_decade``); tube / far-field charts use the
# coarse ``w_nodes_per_decade``.
_INTERIOR_REGIONS = frozenset({'wedge_interior', 'lobe_interior'})
_FARFIELD_REGIONS = frozenset({'exterior', 'lobe_exterior'})

# FAR-FIELD ANNULUS (law 4): the saddle lobe-exterior prior demand reaches
# rho_lobe ~ 20.2 (campaign_tiling_design Fact 3), served by the analytic
# ladder / deltoid redesign (OUT of scope here); recorded for the annulus gauge.
_SADDLE_LOBE_DEMAND_RHO_OUTER = 20.2

# ESCALATION tripwire: the module never raises on escalation -- it records the
# verdict for the owner to act on.
_ESCALATION_CALL_LIMIT = 5.0e5
_ESCALATION_REGION_SHARE = 0.40


class TilingPlanError(Exception):
    """Raised on a malformed sizing input the plan cannot size against."""


# ===========================================================================
# Axis-sizing laws (pure; no I/O, no engine call)
# ===========================================================================
def _wall_nearest_edge(band: tuple[float, float], parity: int) -> float:
    """Return the band edge nearest the ``gamma = 1`` parity wall.

    Bands butt the wall and never straddle it: the astroid (``parity == 1``,
    ``gamma < 1``) wall edge is the HIGH edge; the saddle (``parity == -1``,
    ``gamma > 1``) wall edge is the LOW edge.  Resolution tightens toward the
    wall because ``d r_caustic / d gamma -> inf`` there.
    """
    lo, hi = band
    return hi if parity == 1 else lo


def _gamma_resolution(st: Any, gamma: float, parity: int) -> float:
    """Law 1: ``_C_GAMMA * r_caustic / |d r_caustic/d gamma|`` at ``gamma``.

    The derivative is a central finite difference on the engine-free scalar
    caustic reach ``st._scalar_caustic_reach``; the step is clamped so both
    evaluation points stay strictly on the band's side of the parity wall
    (and away from ``gamma = 0``), where the reach is finite.  A vanishing or
    non-finite derivative yields ``inf`` (the band needs a single gamma node).
    """
    reach = float(st._scalar_caustic_reach(gamma))
    dist_wall = abs(gamma - 1.0)
    dist_zero = abs(gamma)
    step = min(_GAMMA_DIFF_STEP, 0.5 * dist_wall, 0.5 * dist_zero)
    if step <= 0.0:
        raise TilingPlanError(
            f'cannot size gamma resolution at gamma={gamma!r}: the band edge '
            'coincides with the parity wall or the origin.')
    r_plus = float(st._scalar_caustic_reach(gamma + step))
    r_minus = float(st._scalar_caustic_reach(gamma - step))
    dreach = abs(r_plus - r_minus) / (2.0 * step)
    if not math.isfinite(dreach) or dreach <= 0.0:
        return math.inf
    return _C_GAMMA * reach / dreach


def _n_gamma_in_band(st: Any, band: tuple[float, float], parity: int) -> int:
    """Law 1 node count: ``ceil(band_span / gamma_res(wall-nearest edge))``."""
    gamma_edge = _wall_nearest_edge(band, parity)
    resolution = _gamma_resolution(st, gamma_edge, parity)
    span = band[1] - band[0]
    if not math.isfinite(resolution) or resolution <= 0.0 or span <= 0.0:
        return 1
    return max(1, math.ceil(span / resolution))


def _n_theta_for_span(arc_span_rad: float, parity: int) -> int:
    """Law 2 node count for one fold arc: ``ceil(kappa_theta * span)``."""
    kappa = _KAPPA_THETA_NODES_PER_RAD[parity]
    if arc_span_rad <= 0.0:
        return 1
    return max(1, math.ceil(kappa * arc_span_rad))


def _trimmed_arc_span(st: Any, band: tuple[float, float], arc: Any,
                      parity: int, config: Any) -> tuple[float, bool]:
    """Resolvable theta span of one tube fold arc (F083 trim).

    Astroid arcs are narrowed to their resolvable sub-arc by
    ``st._trim_tube_arc``; saddle arcs are returned full (the trim is the
    identity for ``parity != 1``).  Returns ``(span_rad, trimmed)`` where
    ``trimmed`` is ``False`` when the astroid trim refused and the raw
    cusp-to-cusp span was used as a conservative fallback.
    """
    eta_max = config.f_max * st._min_curvature_radius(
        band, arc, config.n_caustic_samples)
    try:
        resolved = st._trim_tube_arc(
            band=band, arc=arc, eta_max=eta_max, parity=parity)
    except ValueError:
        return float(arc.theta_hi - arc.theta_lo), False
    return float(resolved.theta_hi - resolved.theta_lo), (parity == 1)


def _tube_spatial_total(st: Any, ctx: Any, parity: int, config: Any
                        ) -> tuple[int, int, list[float]]:
    """Heterogeneous tube spatial nodes: ``sum_arc(n_theta(arc) * n_u)``.

    Returns ``(n_arcs, spatial_total, theta_node_counts)``.  Each fold arc is a
    tile; its theta axis is sized independently from its trimmed span (law 2)
    while the depth axis keeps ``config.n_u``.
    """
    spatial_total = 0
    theta_counts: list[float] = []
    for arc in ctx.tube_arcs:
        span_rad, _trimmed = _trimmed_arc_span(st, ctx.band, arc, parity, config)
        n_theta = _n_theta_for_span(span_rad, parity)
        theta_counts.append(float(n_theta))
        spatial_total += n_theta * config.n_u
    return len(ctx.tube_arcs), spatial_total, theta_counts


def _resolve_dd_ceiling(w_ceiling_dd: float | None) -> float:
    """Resolve the DD-band ceiling, defaulting to the canonical constant.

    Production supplies ``w_ceiling_dd`` from the census header (the single
    source of truth at run time).  A ``None`` (direct helper callers, e.g.
    unit tests) resolves to ``chang_refsdal._schwinger.W_CEILING_SCHWINGER``
    -- the same constant the census header is built from -- via a lazy import
    that stays engine-free (reading a module float triggers no wave
    evaluation and never imports ``mpmath``).
    """
    if w_ceiling_dd is not None:
        return float(w_ceiling_dd)
    from cogwheel.lensing.chang_refsdal import _schwinger
    return float(_schwinger.W_CEILING_SCHWINGER)


def _measured_w_range(records: list[dict[str, Any]], region: str,
                      gamma_band_label: str, box: Any, parity: int,
                      w_ceiling_dd: float | None = None
                      ) -> tuple[float, float, str]:
    """Law 3 w-band edges from the demand cells' ``engine_residual`` draws.

    ``w = exp(log_w)`` (the records carry natural logs).  The upper edge is
    clipped at the DD-band ceiling ``w_ceiling_dd`` (single-sourced from the
    census header; ``None`` resolves to the canonical
    ``_schwinger.W_CEILING_SCHWINGER``).  ``engine_residual`` fires for any
    draw whose node kinds include ``exact_wave``, and that route also tallies
    draws straddling the ``(60, 150]`` QD/mpmath band with ``log_w_max`` left
    unclipped -- those are above-ceiling rungs, NOT DD-band chart demand, and
    must neither size nor inflate a chart tile that only serves
    ``w <= w_ceiling_dd``.  The lower edge stays measured, so this is a
    targeted clip of the above-ceiling leak, never a blanket
    ``[w_floor, ceiling]`` band (true DD demand edges are already below the
    ceiling).  Falls back to the prior box's ``w_range(parity)`` (also
    clipped) when the demand cell holds no residual draw (a tile admitted by
    geometry but empty of measured demand).
    """
    ceiling = _resolve_dd_ceiling(w_ceiling_dd)
    log_lo: list[float] = []
    log_hi: list[float] = []
    for rec in records:
        if (rec['route'] == 'engine_residual'
                and rec['region'] == region
                and rec['gamma_band'] == gamma_band_label):
            log_lo.append(rec['log_w_min'])
            log_hi.append(rec['log_w_max'])
    if not log_hi:
        w_lo, w_hi_raw = box.w_range(parity)
        w_hi = min(float(w_hi_raw), ceiling)
        source = ('prior_box_fallback_clipped_dd'
                  if w_hi < float(w_hi_raw) else 'prior_box_fallback')
        return float(w_lo), w_hi, source
    w_lo = math.exp(min(log_lo))
    w_hi_raw = math.exp(max(log_hi))
    w_hi = min(w_hi_raw, ceiling)
    source = 'measured_clipped_dd' if w_hi < w_hi_raw else 'measured'
    return w_lo, w_hi, source


def _n_w_nodes(w_lo: float, w_hi: float, region: str, config: Any) -> int:
    """Law 3 node count: ``ceil(per_decade * log10(w_hi / w_lo))``."""
    per_decade = (config.interior_w_nodes_per_decade
                  if region in _INTERIOR_REGIONS else config.w_nodes_per_decade)
    if w_hi <= w_lo or w_lo <= 0.0:
        return 1
    decades = math.log10(w_hi / w_lo)
    return max(1, math.ceil(per_decade * decades))


def _annulus_record(st: Any, ctx: Any, region: str) -> dict[str, Any] | None:
    """Law 4: the far-field annulus with an EXPLICITLY declared gauge.

    Astroid exterior tiles live in the caustic-relative gauge; saddle
    lobe-exterior tiles live in the lobe-local ``rho_lobe`` gauge.  The prior
    demand outer edge (``rho_lobe ~ 20``, served by the analytic ladder /
    deltoid redesign, OUT of scope here) is recorded with the band's caustic
    reach so the two gauges can be related downstream.
    """
    if region not in _FARFIELD_REGIONS:
        return None
    caustic_reach = float(st._scalar_caustic_reach(ctx.gamma_mid))
    if region == 'exterior':
        return {
            'gauge': 'caustic_rho',
            'rho_inner': float(ctx.exclusion_rho),
            'rho_outer': float(ctx.rho_outer_region),
            'caustic_reach': caustic_reach,
            'note': ('astroid origin-centred exterior remainder; inner edge is '
                     'the tube-shell exclusion rho, outer the prior source '
                     'reach, both in caustic_rho.')}
    return {
        'gauge': 'rho_lobe',
        'rho_inner': 1.0,
        'rho_outer': float(ctx.rho_outer_region),
        'prior_demand_rho_outer_lobe': _SADDLE_LOBE_DEMAND_RHO_OUTER,
        'caustic_reach': caustic_reach,
        'note': ('macro-saddle lobe-exterior in lobe-local rho_lobe; the full '
                 f'prior annulus (rho_lobe ~ {_SADDLE_LOBE_DEMAND_RHO_OUTER}) '
                 'is served by the analytic ladder / deltoid redesign, out of '
                 'scope for chart tiling.')}


def _saddle_bottom_anchor(ctx: Any, w_lo: float) -> dict[str, Any]:
    """Law 5 (record only): the ``w -> 0`` saddle-bottom anchor context.

    ``F(w->0) = -1j * sqrt(mu_macro)`` with the Morse ``n=1`` macro
    magnification ``mu_macro = 1 / |gamma^2 - 1|``.  Engine-free and purely
    informational: no serving ladder or ceiling is touched.
    """
    gamma = ctx.gamma_mid
    mu_macro = 1.0 / abs(gamma * gamma - 1.0)
    return {
        'anchor': 'F(w->0) = -1j * sqrt(mu_macro)  [Morse n=1]',
        'sqrt_mu_macro': float(math.sqrt(mu_macro)),
        'w_axis_lower_edge': float(w_lo),
        'note': 'record only; no serving ladder / ceiling is modified.'}


# ===========================================================================
# Demand gating + per-band / per-region planning
# ===========================================================================
def _residual_by_region_band(cells: dict[str, dict[str, Any]]
                             ) -> dict[tuple[str, str], int]:
    """Sum ``engine_residual`` demand per ``(region, gamma_band)`` census cell.

    A chart tile is admitted ONLY where this count is positive, which is what
    zeroes the astroid-exterior explosion: cells that Born / c3 / certified-map
    already serve carry no ``engine_residual`` demand.
    """
    lookup: dict[tuple[str, str], int] = {}
    for cell in cells.values():
        key = (cell['region'], cell['gamma_band'])
        lookup[key] = lookup.get(key, 0) + int(
            cell['routes'].get('engine_residual', 0))
    return lookup


def _band_tile_geometry(st: Any, ctx: Any, region: str, parity: int,
                        config: Any) -> tuple[int, int, list[float]]:
    """Return ``(n_tiles, spatial_total, theta_node_counts)`` for one band.

    ``spatial_total`` folds the per-tile spatial factor (heterogeneous for the
    tube, where each arc's theta axis is sized independently).  A tiler that
    admits nothing returns ``(0, 0, [])``.
    """
    if region == 'tube':
        return _tube_spatial_total(st, ctx, parity, config)
    _n_arcs, n_tiles, _skip = tiling_census._count_region(
        ctx, region, config, st)
    spatial = tiling_census._spatial_nodes_per_tile(region, config)
    return n_tiles, n_tiles * spatial, []


def _plan_band(st: Any, box: Any, ctx: Any, region: str, parity: int,
               config: Any, records: list[dict[str, Any]],
               residual_lookup: dict[tuple[str, str], int],
               gamma_edges: np.ndarray,
               w_ceiling_dd: float | None = None
               ) -> tuple[dict[str, Any] | None, str]:
    """Size one ``(region x band)`` demand-gated tile plan.

    Returns ``(entry, status)`` where ``status`` is ``'planned'``,
    ``'gated_no_demand'`` (positive geometry but zero census demand) or
    ``'empty'`` (the tiler admitted nothing).
    """
    label = serve_route_census._gamma_band_of(ctx.gamma_mid, gamma_edges)
    n_tiles, spatial_total, theta_counts = _band_tile_geometry(
        st, ctx, region, parity, config)
    if n_tiles == 0:
        return None, 'empty'
    if residual_lookup.get((region, label), 0) <= 0:
        return None, 'gated_no_demand'

    n_gamma = _n_gamma_in_band(st, ctx.band, parity)
    w_lo, w_hi, w_source = _measured_w_range(
        records, region, label, box, parity, w_ceiling_dd)
    n_w = _n_w_nodes(w_lo, w_hi, region, config)
    band_nodes = spatial_total * n_gamma * n_w
    entry: dict[str, Any] = {
        'gamma_band': label,
        'band': [float(ctx.band[0]), float(ctx.band[1])],
        'gamma_mid': float(ctx.gamma_mid),
        'n_tiles': int(n_tiles),
        'spatial_nodes_total': int(spatial_total),
        'theta_node_counts': ([int(c) for c in theta_counts]
                              if theta_counts else None),
        'n_gamma_in_band': int(n_gamma),
        'n_w': int(n_w),
        'w_lo': float(w_lo),
        'w_hi': float(w_hi),
        'w_range_source': w_source,
        'annulus': _annulus_record(st, ctx, region),
        'band_nodes': int(band_nodes),
    }
    if parity != 1:
        entry['saddle_bottom_anchor'] = _saddle_bottom_anchor(ctx, w_lo)
    return entry, 'planned'


def _plan_region(st: Any, box: Any, contexts: list[Any], region: str,
                 parity: int, config: Any, records: list[dict[str, Any]],
                 residual_lookup: dict[tuple[str, str], int],
                 gamma_edges: np.ndarray,
                 w_ceiling_dd: float | None = None) -> dict[str, Any]:
    """Aggregate one ``(region x parity)`` over its demand-gated bands."""
    status_counts = {'planned': 0, 'gated_no_demand': 0, 'empty': 0}
    bands: list[dict[str, Any]] = []
    region_tiles = 0
    region_nodes = 0
    for ctx in contexts:
        entry, status = _plan_band(
            st, box, ctx, region, parity, config, records,
            residual_lookup, gamma_edges, w_ceiling_dd)
        status_counts[status] += 1
        if entry is not None:
            bands.append(entry)
            region_tiles += entry['n_tiles']
            region_nodes += entry['band_nodes']
    return {
        'parity': parity,
        'region': region,
        'n_bands': len(contexts),
        'n_bands_planned': status_counts['planned'],
        'n_bands_gated_no_demand': status_counts['gated_no_demand'],
        'n_bands_empty': status_counts['empty'],
        'region_tiles': int(region_tiles),
        'region_nodes': int(region_nodes),
        'region_calls': int(region_nodes * _LABELS_PER_NODE),
        'bands': bands,
    }


# ===========================================================================
# Cost, cross-checks, escalation
# ===========================================================================
def _cross_checks(st: Any, config: Any, total_nodes: int,
                  census: dict[str, Any],
                  tiling_census_report: dict[str, Any]) -> dict[str, Any]:
    """Three reported ratios; a large divergence is a number, never a silent pass.

    (i)  plan nodes vs ``surrogate_training._self_estimate`` (blanket-count
         upper bound; ratio < 1 confirms the demand gate + axis sizing bit).
    (ii) plan nodes vs the ``tiling_census`` aggregate (also a no-demand-gate
         upper bound; ratio << 1 expected).
    (iii) measured ``engine_residual`` share vs the ~41% honest ledger.
    """
    self_estimate_seconds = float(st._self_estimate(config, None))
    seconds_per_node = _LABELS_PER_NODE * tiling_census._SECONDS_PER_LABEL
    self_estimate_nodes = self_estimate_seconds / seconds_per_node
    census_nodes = (tiling_census_report['aggregate_call_count']
                    / _LABELS_PER_NODE)
    residual = census['residual_demand']
    n_samples = census['n_samples']
    residual_share = residual['total'] / n_samples if n_samples else 0.0
    return {
        'i_vs_self_estimate': {
            'plan_nodes': int(total_nodes),
            'self_estimate_nodes': float(self_estimate_nodes),
            'ratio': (total_nodes / self_estimate_nodes
                      if self_estimate_nodes > 0 else float('inf')),
            'note': '_self_estimate is a blanket-count upper bound; ratio < 1 '
                    'reflects the demand gate and per-axis sizing.'},
        'ii_vs_tiling_census': {
            'plan_nodes': int(total_nodes),
            'tiling_census_nodes': float(census_nodes),
            'ratio': (total_nodes / census_nodes
                      if census_nodes > 0 else float('inf')),
            'note': 'tiling_census applies no demand gate; ratio << 1 expected.'},
        'iii_residual_vs_ledger': {
            'measured_engine_residual_share': float(residual_share),
            'ledger': _CENSUS_ENGINE_RESIDUAL_LEDGER,
            'ratio': (residual_share / _CENSUS_ENGINE_RESIDUAL_LEDGER
                      if _CENSUS_ENGINE_RESIDUAL_LEDGER else float('inf'))},
    }


def _escalation_verdict(total_calls: float, per_region: dict[str, Any],
                        total_nodes: int) -> dict[str, Any]:
    """Record (never raise) the escalation verdict + reason strings."""
    shares = {key: (rec['region_nodes'] / total_nodes if total_nodes else 0.0)
              for key, rec in per_region.items()}
    max_region_share = max(shares.values()) if shares else 0.0
    reasons: list[str] = []
    if total_calls > _ESCALATION_CALL_LIMIT:
        reasons.append(
            f'total_calls {total_calls:.0f} exceeds limit '
            f'{_ESCALATION_CALL_LIMIT:.0f}')
    if max_region_share > _ESCALATION_REGION_SHARE:
        reasons.append(
            f'max region node share {max_region_share:.3f} exceeds limit '
            f'{_ESCALATION_REGION_SHARE:.2f}')
    return {
        'should_escalate': bool(reasons),
        'reasons': reasons,
        'total_calls': float(total_calls),
        'max_region_share': float(max_region_share),
        'call_limit': _ESCALATION_CALL_LIMIT,
        'region_share_limit': _ESCALATION_REGION_SHARE,
        'per_region_node_share': {k: float(v) for k, v in shares.items()},
    }


def _census_summary(census: dict[str, Any]) -> dict[str, Any]:
    """Summarize the census refresh + reconcile its residual share to ledger."""
    residual = census['residual_demand']
    n_samples = census['n_samples']
    share = residual['total'] / n_samples if n_samples else 0.0
    return {
        'schema_source': census['schema'],
        'mode': census['header']['mode'],
        'n_samples': n_samples,
        'route_counts': census['route_counts'],
        'engine_residual_count': int(residual['total']),
        'engine_residual_share': float(share),
        'ledger_reference': _CENSUS_ENGINE_RESIDUAL_LEDGER,
        'share_over_ledger': (float(share / _CENSUS_ENGINE_RESIDUAL_LEDGER)
                              if _CENSUS_ENGINE_RESIDUAL_LEDGER else float('inf')),
        'residual_split': residual,
        'gamma_band_edges': census['header']['gamma_band_edges'],
    }


def _per_parity_totals(per_region: dict[str, Any]) -> dict[str, Any]:
    """Fold the per-region records into per-parity node / call totals."""
    totals: dict[str, dict[str, int]] = {
        '+1': {'nodes': 0, 'calls': 0}, '-1': {'nodes': 0, 'calls': 0}}
    for rec in per_region.values():
        key = f'{rec["parity"]:+d}'
        totals[key]['nodes'] += rec['region_nodes']
        totals[key]['calls'] += rec['region_calls']
    return totals


# ===========================================================================
# Public entry points
# ===========================================================================
def build_plan(census: dict[str, Any], config: Any,
               tiling_census_report: dict[str, Any]) -> dict[str, Any]:
    """Build the demand-sized tiling plan + cost estimate from a census dict.

    Parameters
    ----------
    census : dict
        A `serve_route_census.run` report (schema ``serve_route_census_v1``)
        holding ``cells``, ``records``, ``residual_demand`` and the header's
        ``gamma_band_edges``.
    config : TrainingConfig
        The production ``surrogate_training.TrainingConfig`` sizing the tilers.
    tiling_census_report : dict
        A `tiling_census.run` report for the same ``config`` (cross-check ii).

    Returns
    -------
    dict
        The combined plan + cost report under schema ``tiling_plan_v1``; this
        function neither prints nor writes files.
    """
    st, _sg = tiling_census._load_production_modules()
    box = st.PriorBox.from_prior_classes()
    gamma_edges = np.asarray(census['header']['gamma_band_edges'], dtype=float)
    w_ceiling_dd = float(census['header']['w_band_edges']['w_ceiling_dd'])
    records = census['records']
    residual_lookup = _residual_by_region_band(census['cells'])

    per_region: dict[str, Any] = {}
    total_nodes = 0
    for parity in (1, -1):
        contexts, _dropped = tiling_census._collect_band_contexts(
            st, box, parity, config)
        for region in tiling_census._admissible_regions(parity, None):
            rec = _plan_region(
                st, box, contexts, region, parity, config, records,
                residual_lookup, gamma_edges, w_ceiling_dd)
            per_region[f'{region}:{parity:+d}'] = rec
            total_nodes += rec['region_nodes']

    total_calls = total_nodes * _LABELS_PER_NODE
    wall_clock_s = total_calls * SECONDS_PER_CALL
    return {
        'schema': SCHEMA,
        'census_refresh': _census_summary(census),
        'config': {f: getattr(config, f)
                   for f in tiling_census._REQUIRED_CONFIG_FIELDS},
        'cost_model': {
            'labels_per_node': _LABELS_PER_NODE,
            'seconds_per_call': SECONDS_PER_CALL,
            'w_ceiling_dd': w_ceiling_dd,
            'note': ('a call is one node-label; SECONDS_PER_CALL=0.0903 is the '
                     'DD-band per-call smoke rate, tiling_census uses '
                     f'_SECONDS_PER_LABEL={tiling_census._SECONDS_PER_LABEL} '
                     's/label (== per call); the ~0.3% gap is smoke jitter.')},
        'per_region': per_region,
        'per_parity': _per_parity_totals(per_region),
        'totals': {
            'total_nodes': int(total_nodes),
            'total_calls': int(total_calls),
            'wall_clock_s': float(wall_clock_s),
            'wall_clock_hours': float(wall_clock_s / 3600.0)},
        'cross_checks': _cross_checks(
            st, config, total_nodes, census, tiling_census_report),
        'escalation': _escalation_verdict(total_calls, per_region, total_nodes),
    }


def run(*, n_samples: int = 10_000, seed: int = 0, f_min_hz: float = 20.0,
        f_max_hz: float = 1024.0, training_config: Any | None = None
        ) -> dict[str, Any]:
    """Refresh the demand census and build the demand-sized tiling plan.

    Parameters
    ----------
    n_samples, seed, f_min_hz, f_max_hz : int / float
        Serve-route census settings (defaults: the 10k / seed-0 / 20-1024 Hz
        build-HEAD refresh).
    training_config : TrainingConfig, optional
        The tiler sizing config; defaults to the production
        ``surrogate_training.TrainingConfig()``.

    Returns
    -------
    dict
        The combined plan + cost report (schema ``tiling_plan_v1``).  Pure
        prediction: no wave evaluation, no file write, no print.
    """
    census_config = serve_route_census.ServeRouteCensusConfig(
        n_samples=n_samples, seed=seed, f_min_hz=f_min_hz, f_max_hz=f_max_hz)
    census = serve_route_census.run(census_config, artifact=None)

    st, _sg = tiling_census._load_production_modules()
    config = training_config if training_config is not None else st.TrainingConfig()
    tiling_census_report = tiling_census.run(config, None)
    return build_plan(census, config, tiling_census_report)
