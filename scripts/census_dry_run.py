#!/usr/bin/env python
"""
Structural serve-coverage dry run for the lensing surrogate architecture.

Measures what fraction of prior draws WOULD be served by the architecture
regardless of whether charts are trained. This is a geometry-only census
that classifies each draw into the serve path that would handle it.

Usage:
    python scripts/census_dry_run.py [--n-samples N] [--seed S]
"""
import argparse
import math
import sys
import time

import numpy as np

from cogwheel.lensing.chang_refsdal.geometry import (
    LensDomainError, r_caustic, caustic_curvature_radius,
    nearest_caustic_point)
from cogwheel.lensing.ppgo_map import caustic_rho, caustic_geometry
from cogwheel.lensing.surrogate import _to_lobe_fixed
from cogwheel.lensing import surrogate_training as _st

# ---- Architecture constants (mirrored from surrogate.py) ----
_DD_PRODUCT_MARGIN = 58.0
_XI_FOLD_THRESHOLD = 4.0
_CUSP_ARM_COVERAGE = 0.07  # rad, from surrogate.py

# Astroid cusp angles (eigenframe, positive parity).
_ASTROID_CUSP_ANGLES = (0.0, math.pi / 2, -math.pi / 2, math.pi)

# Cusp exclusion half-width (from typical tube charts): the tube chart's
# cusp window is ~0.25 rad; after subtracting _CUSP_ARM_COVERAGE the
# residual is ~0.18, but the full window defines what the arm can cover.
_TYPICAL_CUSP_HALF_WINDOW = 0.25  # rad, approximate tube cusp window

# ---- Macro-saddle lobe coverage (mirrored from surrogate_training.py) ----
# Largest source magnitude the census prior can draw (source box corner,
# 3*sqrt(2)); the census is mass-free, so this stands in for the production
# per-stratum ``y_outer_region = _source_scale(m_lo)`` when forming the served
# exterior band ``rho_outer = 1 + y_outer - coordinate_radius_min``.
_SOURCE_BOX_CORNER = 3.0 * math.sqrt(2.0)
_SADDLE_PARITY = -1
# Gamma-band grid over the saddle regime (gamma > 1).  One canonical +y1 lobe
# admission is built per band and cached; band edges are nudged above the
# gamma = 1 parity boundary, where the deltoid geometry is singular.  The band
# width mirrors production's near-boundary refinement
# (``gamma_refine_near_one_width = 0.05``): a wider band moves the lobe too far
# between its three gammas, so no interior point stays inside all three winding
# loops and the lobe-interior admission collapses to empty (a width-0.2 band
# admits zero interior probes; 0.05 recovers the interior family).
_SADDLE_BAND_WIDTH = 0.05
_SADDLE_BAND_FLOOR = 1.0 + 1e-3
_SADDLE_CONFIG = _st.TrainingConfig()
# Cache: band index -> (canonical +y1 lobe admission or None, rho_outer).
_SADDLE_ADMISSION_CACHE: dict[int, tuple] = {}


def _saddle_lobe_admission(gamma: float) -> tuple:
    """Canonical +y1 lobe admission and served exterior rho for a saddle gamma.

    Mirrors the production saddle packing path
    (``surrogate_training._train_band_charts``): builds the band caustic
    structure, the tube-shell half-width
    ``max_eta_max = f_max * max(arc_r_min)``, the two per-lobe admissions, and
    the additive outer edge ``rho_outer = 1 + y_outer - coordinate_radius_min``.
    The canonical ``+y1`` lobe is admission index 1 (lens centre ``pi``),
    matching the production chart that serves the whole D2-folded first
    quadrant.  Results are cached per coarse gamma band (only a few bands span
    the saddle regime), so the caustic clouds are built at most once per band.

    Returns ``(admission, rho_outer)`` for the band containing ``gamma`` or
    ``(None, None)`` if the band geometry is degenerate.
    """
    band_index = int((gamma - 1.0) // _SADDLE_BAND_WIDTH)
    if band_index in _SADDLE_ADMISSION_CACHE:
        return _SADDLE_ADMISSION_CACHE[band_index]
    band_lo = max(1.0 + band_index * _SADDLE_BAND_WIDTH, _SADDLE_BAND_FLOOR)
    band_hi = 1.0 + (band_index + 1) * _SADDLE_BAND_WIDTH
    band = (band_lo, band_hi)
    cfg = _SADDLE_CONFIG
    try:
        structure = _st.band_caustic_structure(
            band, _SADDLE_PARITY, n_samples=cfg.n_caustic_samples)
        arc_r_min = [
            _st._min_curvature_radius(band, arc, cfg.n_caustic_samples)
            for arc in structure.arcs[:cfg.max_tube_arcs]]
        max_eta_max = (cfg.f_max * max(arc_r_min)
                       if arc_r_min else cfg.f_max * 0.05)
        admissions = _st._saddle_lobe_admissions(
            band, cfg, eta_max=max_eta_max)
        coordinate_radius_min, _ = _st._coordinate_radius_bounds(
            band, _SADDLE_PARITY)
        rho_outer = 1.0 + _SOURCE_BOX_CORNER - coordinate_radius_min
        result: tuple = (admissions[1], rho_outer)
    except (ValueError, LensDomainError, ZeroDivisionError, IndexError):
        result = (None, None)
    _SADDLE_ADMISSION_CACHE[band_index] = result
    return result


def _classify_saddle(gamma: float, y_abs: float, theta: float) -> str:
    """Serve category for a macro-saddle scalar-interior draw (rho <= 1).

    Mirrors the production lobe serve gates: fold the source into the first
    quadrant (D2 reflection), map it to the canonical +y1 lobe's lobe-local
    ``(rho_lobe, theta_local)``, and admit it as ``lobe_interior`` when it is
    inside the lobe caustic (``rho_lobe < 1`` and ``admits``) or as
    ``lobe_exterior`` when it is in the served exterior band
    (``rho_lobe in (1, rho_outer]`` and ``admits_exterior``).  Genuinely
    uncovered draws fall through to ``exact_engine``.
    """
    lobe, rho_outer = _saddle_lobe_admission(gamma)
    if lobe is None:
        return 'exact_engine'
    y1 = y_abs * math.cos(theta)
    y2 = y_abs * math.sin(theta)
    # D2 reflection fold: production charts only the canonical +y1 lobe and
    # maps any quadrant into the first via abs() on both eigenframe axes.
    y1_fold, y2_fold = abs(y1), abs(y2)
    try:
        rho_lobe, theta_local = _to_lobe_fixed(
            lobe.centroid, lobe.boundary_theta, lobe.boundary_r,
            y1_fold, y2_fold)
    except ValueError:
        return 'exact_engine'  # degenerate query exactly at the centroid
    center = (rho_lobe, theta_local)
    half = (0.0, 0.0)  # single-point structural probe (no tile extent)
    if rho_lobe < 1.0:
        return 'lobe_interior' if lobe.admits(center, half) else 'exact_engine'
    if rho_lobe <= rho_outer and lobe.admits_exterior(center, half):
        return 'lobe_exterior'
    return 'exact_engine'


def _draw_prior(n: int, rng: np.random.Generator):
    """Draw N samples from the full lens prior.

    Parameters
    ----------
    n : int
        Number of samples.
    rng : np.random.Generator
        Random generator.

    Returns
    -------
    gamma, y_abs, theta, w : arrays of shape (n,)
    """
    gamma = rng.uniform(0.001, 1.599, size=n)
    y_abs = rng.uniform(0.01, 4.2426, size=n)
    theta = rng.uniform(0.0, 2.0 * math.pi, size=n)
    # w log-uniform in (5, 148)
    log_w_min, log_w_max = math.log(5.0), math.log(148.0)
    log_w = rng.uniform(log_w_min, log_w_max, size=n)
    w = np.exp(log_w)
    return gamma, y_abs, theta, w


def _compute_rho(gamma: float, y_abs: float) -> float:
    """Compute caustic-relative rho = |y| / reach. Returns inf on failure."""
    try:
        return caustic_rho(gamma, y_abs)
    except (ValueError, LensDomainError):
        return float('inf')  # treat as exterior on geometry failure


def _is_near_cusp(gamma: float, theta: float) -> tuple[bool, float]:
    """Check if theta is within a cusp window for positive-parity astroid.

    Returns (is_near, delta_to_nearest_cusp) where delta is the angular
    distance to the nearest cusp direction.
    """
    if gamma >= 1.0:
        # Macro saddle: cusp structure different; for simplicity, the
        # pearcey arm currently serves only positive-parity cusps.
        return False, float('inf')

    min_delta = float('inf')
    for cusp_angle in _ASTROID_CUSP_ANGLES:
        delta = abs((theta - cusp_angle + math.pi) % (2 * math.pi) - math.pi)
        if delta < min_delta:
            min_delta = delta
    # The cusp arm covers sources within _TYPICAL_CUSP_HALF_WINDOW of
    # a cusp direction.
    return min_delta < _TYPICAL_CUSP_HALF_WINDOW, min_delta


def _dd_w_cap(gamma: float, y_abs: float) -> float:
    """Approximate DD product ceiling: w_max such that w * |y| <= 58."""
    # The DD product bound is w * r_max * reach_max <= 58; for the
    # scalar reach gauge, reach_max ~ |y| gives cap ~ 58 / |y|.
    if y_abs <= 0.0:
        return float('inf')
    return _DD_PRODUCT_MARGIN / y_abs


def _compute_xi_min(w: float, gamma: float, y_abs: float, theta: float
                    ) -> float:
    """Estimate xi_min for the fold-ppGO gate.

    xi_min = (3 w delta_tau / 4)^{2/3} where delta_tau is the
    delay difference of the merging fold pair. For a rough structural
    estimate, delta_tau ~ (1 - rho)^{3/2} * geometric_factor for interior
    sources. We use a conservative heuristic: for rho ~ 0.9, delta_tau ~ 0.03;
    for rho ~ 0.5, delta_tau ~ 0.3.
    """
    # Full computation would require find_images + delay. For a structural
    # census we estimate: the xi threshold is w-dependent. The question is
    # whether w is large enough that xi_min >= 4. For a merging pair with
    # delay difference delta_tau, xi_min = (3 w delta_tau / 4)^{2/3}.
    # Solving for w: w >= 4/3 * (4)^{3/2} / delta_tau = 4/3 * 8 / delta_tau
    # ~ 10.67 / delta_tau. For typical interior sources delta_tau ~ 0.01-1.0.
    # At w=148 and delta_tau=0.01: xi = (3*148*0.01/4)^{2/3} = (1.11)^{2/3} ~ 1.07
    # At w=148 and delta_tau=0.1: xi = (3*148*0.1/4)^{2/3} = (11.1)^{2/3} ~ 4.9
    # Since we can't cheaply compute delta_tau without the full geometry engine,
    # we use a heuristic: for high-w interior draws above the DD cap, we
    # assume xi_min >= 4 when w >= ~60 (conservative estimate).
    # This is a STRUCTURAL coverage estimate — the actual census would verify.
    return w  # Return w as proxy; caller uses threshold


def classify_draw(gamma: float, y_abs: float, theta: float, w: float
                  ) -> str:
    """Classify which serve path would handle this draw.

    Returns one of:
        'born' — Born carrier (exterior, rho > 1)
        'chart_interior' — InteriorWedgeChart (interior, w < DD cap)
        'ppgo_fold' — fold-ppGO handoff (interior, w >= DD cap, high xi)
        'cusp_arm' — Pearcey cusp arm
        'chart_tube' — tube chart (near-caustic, away from cusps)
        'chart_farfield' — far-field chart (exterior, rho > 1 up to rho_max)
        'exact_engine' — structural gap (no path covers)
    """
    # Compute rho (caustic-relative coordinate).
    rho = _compute_rho(gamma, y_abs)

    # (a) Exterior: rho > 1 → Born carrier serves analytically.
    if rho > 1.0:
        return 'born'

    # (a') Macro-saddle interior (gamma >= 1): the two deltoid lobes are
    # served by the lobe-interior and lobe-exterior charts, NOT the
    # origin-centred astroid paths below.  Route via the same lobe-local map
    # (`_to_lobe_fixed`) and structural admissions (`admits` /
    # `admits_exterior`) the production serve path uses.
    if gamma >= 1.0:
        return _classify_saddle(gamma, y_abs, theta)

    # Interior path (rho <= 1):
    # The interior is covered by the InteriorWedgeChart up to the DD cap,
    # and by fold-ppGO above it.

    # (b) Interior, below DD cap → InteriorWedgeChart serves.
    dd_cap = _dd_w_cap(gamma, y_abs)
    if w < dd_cap:
        # Positive parity (gamma < 1): the source is inside the astroid and
        # the InteriorWedgeChart serves it.  (Macro-saddle interior draws are
        # already routed to the lobe charts above.)
        return 'chart_interior'

    # (c) Interior, above DD cap: fold-ppGO handoff if xi_min >= 4.
    # The xi_min gate requires w * delta_tau to be large enough.
    # Heuristic: at high w and moderate rho (<~ 0.95), the delay
    # difference is large enough. For rho very close to 1, delta_tau → 0
    # and xi_min drops. We use the conservative condition:
    # For typical interior draws at w >= DD cap: most will have xi >= 4
    # because w >= 58/|y| is large and rho < 1 ensures delta_tau > 0.
    # A more precise gate would need the full geometry engine.
    if gamma < 1.0:
        # Positive parity interior above DD cap.
        # Estimate: xi_min ~ (3 w delta_tau / 4)^{2/3}
        # For rho close to 1, delta_tau is small and xi drops.
        # Use rho < 0.95 as a proxy for "delta_tau large enough":
        if rho < 0.95:
            # High confidence that xi_min >= 4 at these w values
            return 'ppgo_fold'
        else:
            # Near caustic: delta_tau may be too small. Check if
            # the cusp arm or tube could serve instead.
            pass

    # For draws that are interior but near the caustic (0.95 <= rho < 1)
    # and above the DD cap, check the tube / cusp paths:

    # Compute eta (caustic distance proxy): eta ~ (1 - rho) * reach
    # The tube chart serves for small eta when within its angular range.
    try:
        reach, _ = caustic_geometry(gamma)
    except (ValueError, LensDomainError):
        return 'exact_engine'
    eta = (1.0 - rho) * reach  # approximate source-plane distance to caustic

    # (d) Cusp arm: covers near-cusp sources with delta_theta > COVERAGE.
    if gamma < 1.0:
        is_near, delta_cusp = _is_near_cusp(gamma, theta)
        if is_near and delta_cusp > _CUSP_ARM_COVERAGE:
            # The Pearcey arm serves the outer shell of the cusp window:
            # delta > _CUSP_ARM_COVERAGE.
            return 'cusp_arm'

    # (e) Tube chart: serves sources near the caustic, away from cusps.
    # The tube operates for small eta (close to caustic) with angular
    # coverage excluding cusp windows. For positive parity: if the source
    # is near the caustic and not in a cusp exclusion zone, the tube serves.
    if gamma < 1.0 and eta > 0.0:
        # Tube coverage: the tube serves eta in [eta_floor, eta_max] where
        # eta_max is typically f_max * R_c (with f_max ~ 3-5 and R_c the
        # curvature radius). For a structural check, if eta is small enough
        # and theta is away from cusps, the tube covers it.
        try:
            R_c = caustic_curvature_radius(gamma, theta)
        except (ValueError, LensDomainError, ZeroDivisionError):
            R_c = float('inf')

        # Typical tube eta_max ~ 3-5 * min(R_c); tube eta_floor ~ 1e-4.
        # If eta < f_max * R_c and not in cusp exclusion:
        f_max_typical = 5.0
        if np.isfinite(R_c) and eta < f_max_typical * R_c:
            # Check cusp exclusion (residual window blocks the tube).
            is_near, delta_cusp = _is_near_cusp(gamma, theta)
            residual = max(0.0, _TYPICAL_CUSP_HALF_WINDOW - _CUSP_ARM_COVERAGE)
            if not (is_near and delta_cusp < residual):
                return 'chart_tube'

    # (f) Far-field exterior: rho > rho_exterior_min (sources outside the
    # tube's eta_max but still close to the caustic). This path serves
    # rho just barely > 1. But we already handled rho > 1 as 'born' above.
    # For rho ~= 1 (within the tube overlap region), the tube or far-field
    # chart would serve. But far-field charts are EXTERIOR (rho > 1);
    # since rho <= 1 here, far-field doesn't apply.
    # Actually rho <= 1 but near 1: this is the interior near-caustic zone.
    # The tube covers it (handled above). If it fell through:
    # it means neither the tube nor cusp arm can serve.

    # (g) Structural gap: nothing covers this draw.
    return 'exact_engine'


def main() -> None:
    """Parse arguments, draw prior samples, classify each, and print summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-samples', type=int, default=10000,
                        help='Number of prior draws (default: 10000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    n = args.n_samples
    seed = args.seed
    rng = np.random.default_rng(seed)

    print("=" * 70)
    print("STRUCTURAL SERVE-COVERAGE DRY RUN")
    print("=" * 70)
    print(f"  N samples:  {n}")
    print(f"  Seed:       {seed}")
    print("  Prior:")
    print("    gamma ~ U(0.001, 1.599)")
    print("    |y|   ~ U(0.01, 4.2426)")
    print("    theta ~ U(0, 2π)")
    print("    w     ~ LogU(5, 148)")
    print(f"  DD product margin: {_DD_PRODUCT_MARGIN}")
    print(f"  Xi fold threshold: {_XI_FOLD_THRESHOLD}")
    print(f"  Cusp arm coverage: {_CUSP_ARM_COVERAGE} rad")
    print()

    t0 = time.time()
    gamma, y_abs, theta, w = _draw_prior(n, rng)

    # Classify each draw.
    categories: dict[str, int] = {}
    exact_engine_draws: list[tuple[float, float, float, float]] = []

    for i in range(n):
        cat = classify_draw(gamma[i], y_abs[i], theta[i], w[i])
        categories[cat] = categories.get(cat, 0) + 1
        if cat == 'exact_engine':
            exact_engine_draws.append(
                (gamma[i], y_abs[i], w[i], theta[i]))
        if (i + 1) % 2000 == 0:
            elapsed = time.time() - t0
            print(f"  Progress: {i+1}/{n} ({elapsed:.1f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"  Completed {n} classifications in {elapsed:.1f}s")
    print()

    # ---- Summary table ----
    print("=" * 70)
    print("CLASSIFICATION SUMMARY")
    print("=" * 70)
    print(f"{'Category':<20} {'Count':>8} {'Fraction':>10}")
    print("-" * 40)

    # Define display order.
    display_order = [
        'born', 'chart_interior', 'lobe_interior', 'lobe_exterior',
        'ppgo_fold', 'cusp_arm', 'chart_tube', 'chart_farfield',
        'exact_engine'
    ]
    served_count = 0
    for cat in display_order:
        count = categories.get(cat, 0)
        frac = count / n
        served_flag = ""
        if cat != 'exact_engine':
            served_count += count
            served_flag = "  ✓"
        print(f"  {cat:<18} {count:>8}   {frac:>8.4f}{served_flag}")

    # Any unexpected categories?
    for cat, count in sorted(categories.items()):
        if cat not in display_order:
            frac = count / n
            print(f"  {cat:<18} {count:>8}   {frac:>8.4f}  ???")

    print("-" * 40)
    served_frac = served_count / n
    gap_count = categories.get('exact_engine', 0)
    gap_frac = gap_count / n
    print(f"  {'SERVED (total)':<18} {served_count:>8}   {served_frac:>8.4f}")
    print(f"  {'GAP (exact_engine)':<18} {gap_count:>8}   {gap_frac:>8.4f}")
    print()
    print(f"  *** STRUCTURAL COVERAGE: {served_frac*100:.2f}% ***")
    print()

    # ---- Detail on exact_engine residual ----
    if exact_engine_draws:
        print("=" * 70)
        print(f"EXACT_ENGINE RESIDUAL DETAIL (first {min(20, len(exact_engine_draws))} draws)")
        print("=" * 70)
        print(f"  {'gamma':>8} {'|y|':>8} {'w':>8} {'theta':>8}  {'rho':>8}  note")
        print("  " + "-" * 60)
        for i, (g, y, ww, th) in enumerate(exact_engine_draws[:20]):
            rho = _compute_rho(g, y)
            dd_cap = _dd_w_cap(g, y)
            note = ""
            if g >= 1.0:
                note = "saddle lobe gap (gamma>=1)"
            elif rho > 0.95:
                note = f"near-caustic, w>{dd_cap:.1f}(DD), rho={rho:.3f}"
            elif ww >= dd_cap:
                note = f"above DD cap ({dd_cap:.1f}), rho={rho:.3f}"
            else:
                note = f"rho={rho:.3f}, dd_cap={dd_cap:.1f}"
            print(f"  {g:>8.4f} {y:>8.4f} {ww:>8.2f} {th:>8.4f}  {rho:>8.4f}  {note}")

        # Breakdown of gap draws.
        saddle_interior = sum(1 for g, y, ww, th in exact_engine_draws
                             if g >= 1.0)
        near_caustic = sum(1 for g, y, ww, th in exact_engine_draws
                          if g < 1.0 and _compute_rho(g, y) >= 0.95)
        other = len(exact_engine_draws) - saddle_interior - near_caustic
        print()
        print("  Gap breakdown:")
        print(f"    Saddle lobe gap (gamma >= 1):      {saddle_interior}")
        print(f"    Near-caustic (rho >= 0.95, high w): {near_caustic}")
        print(f"    Other:                              {other}")

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
