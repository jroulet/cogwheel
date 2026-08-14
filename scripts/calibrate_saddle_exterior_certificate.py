#!/usr/bin/env python
"""Calibrate the c3+ghost certificate for 2-image macro-saddle EXTERIOR admission.

The next build replaces the scalar eta-floor admission leg of the tier-1
saddle-analytic serve with certificate admission::

    admit  iff  estimate(config, w_min) <= bar

    estimate = geometry.ppgo_error_estimate(real_images, source, matrix, w_min)
               + |ghost_kernel(w_min).kernel| * exp(-w_min * Im tau_c)

This script measures, per witness, the certificate ``estimate`` at
``w_min = W_FLOOR`` against the ACTUAL pointwise relative serve error of the
zero-envelope ``FARFIELD_KERNEL_SUM`` reconstruction over the production band
``w in [W_FLOOR, w_ceil]`` (w_ceil <= W_CEILING_SCHWINGER = 60, so every
oracle eval stays in the cheap double-double band), and reports the ratio
distribution ``ratio = estimate / max(err)`` -- mirroring the INTERIOR
c3-certificate calibration methodology (build ppgo_interior_certificate:
ratio distribution, median/p99/max, 0%-optimistic bar; the measured interior
true/certificate ratio was median 0.587 / p99 0.953 / max 0.980, 0%
optimistic).  Here ``ratio < 1`` means the certificate is OPTIMISTIC
(underestimates the true error); the safety factor must push the optimistic
fraction to 0%.

ORACLE / SERVE PLUMBING (repo rule: oracles must call shipping code).
The serve, oracle, and geometry helpers are REUSED by import from
``cogwheel/tests/test_lensing_saddle_tier1_accuracy.py`` (``_polar_source`` /
``_tier1_serve`` / ``_exact_total_w`` / ``_min_delta_tau``) -- the
pairing-validated production-shaped plumbing.  The eta-floor scan script
``scripts/measure_saddle_eta_floor.py`` mirrors the same helpers but is NOT
importable at HEAD: it imports ``_SADDLE_TIE_EPS`` from
``cogwheel.lensing.likelihood``, which does not exist there (the eta build
was reverted); HEAD's gate is ``_saddle_farfield_analytic_serves(real_delays,
w_lo, rho)`` with ``_SADDLE_FARFIELD_RHO_FLOOR = 2.0``.

PAIRING GATE (mandatory before any oracle claim).  Before any error is
scored, one known-resolved, production-gate-admitted config is checked:
serve and oracle must agree to better than the certified far-field bound
(max err < 1e-3; the certified population measured p90 ~5e-5, max ~7e-4).
A frame mismatch (e.g. a missing ``t_min`` demodulation) would read O(1).

WITNESS POPULATION.  Per gamma: connecting region (origin side, rho <= 0.4,
including the exact axis, where the mirror pair is delay-tied by symmetry),
transverse cone, generic off-axis directions, and -- decisively -- the
MEASURED WORST-CASE RIDGE of the eta scan (gamma=1.2, angle ~1.40-1.42 rad,
isotropic caustic-relative scale ~1.5-3.0, worst near scale 2.6 / eta ~2.9):
the certificate must not be optimistic where the serve is actually worst.

GHOST AVAILABILITY (measured while building this script).  On the macro
saddle the ghost is NOT merely unavailable on the principal axes: over much
of the connecting region, the transverse cone, and the ridge the image
quartic has NO complex-conjugate pair at all (``_ghost_candidates`` returns
0 candidates -- the two non-image quartic roots are real), so
``geometry.ghost_kernel`` raises ``GhostDomainError`` structurally, not just
on the measure-zero axis.  Such witnesses are recorded as
``ghost_unavailable`` together with their c3-only ratio, so the build can
decide whether they must refuse or need a separate admission story.

Usage::

    python scripts/calibrate_saddle_exterior_certificate.py --pilot
    python scripts/calibrate_saddle_exterior_certificate.py \
        --n-witnesses 60 --gammas 1.2 1.5 2.0 --n-w 24 --w-ceil 60 --seed 42
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np

from cogwheel.lensing.chang_refsdal import geometry
from cogwheel.lensing.chang_refsdal.operator import RHO_END
from cogwheel.lensing.chang_refsdal._schwinger import (
    SchwingerCertificationError)
from cogwheel.lensing.likelihood import _saddle_farfield_analytic_serves
from cogwheel.lensing.ppgo_map import caustic_rho
# Production-shaped serve/oracle helpers (see module docstring for why the
# eta-floor scan script cannot be the import source at HEAD).
from cogwheel.tests.test_lensing_saddle_tier1_accuracy import (
    W_FLOOR, _exact_total_w, _min_delta_tau, _polar_source, _tier1_serve)

#: Delay gaps at or below this are symmetry ties (mirror pair, delta_tau == 0
#: exactly up to roundoff), not resolvable separations.  Defined locally: the
#: reverted eta build's ``_SADDLE_TIE_EPS`` does not exist at HEAD.
_TIE_EPS = 1e-12

#: Default production-band ceiling: the double-double Schwinger domain
#: (W_CEILING_SCHWINGER = 60).  The mpmath band (60 < w <= 148,
#: ~85-120 s/eval) is priced separately via --time-mpmath, never swept.
W_CEIL_DEFAULT = 60.0

DEFAULT_GAMMAS = (1.2, 1.5, 2.0)


def _real_images(geom):
    """Real image positions of a partition.

    ``geom.images`` carries either all channel slots (length matching
    ``real_mask``; mask applies) or only the real images (2-image saddle
    exterior partitions at HEAD return shape (2, 2) against a length-4
    ``real_mask``).  Handle both.
    """
    images = np.asarray(geom.images)
    real = np.asarray(geom.real_mask, dtype=bool)
    if len(images) == len(real):
        return images[real]
    if len(images) != int(real.sum()):
        raise ValueError(
            f'geom.images has {len(images)} rows but real_mask sums to '
            f'{int(real.sum())} (mask length {len(real)}).')
    return images


def _min_image_separation(images) -> float:
    """Smallest pairwise Euclidean distance among the real images."""
    n = len(images)
    if n < 2:
        return float('nan')
    return min(float(np.linalg.norm(images[i] - images[j]))
               for i in range(n) for j in range(i + 1, n))


def _certificate(y, matrix, real_images, w_min):
    """Evaluate the c3+ghost certificate at ``w_min``.

    Returns a dict with keys ``est_c3``, ``ghost_term``, ``estimate``,
    ``ghost_status`` ('ok' | 'ghost_unavailable' | 'ghost_absent_unexpected'),
    ``im_tau_c``, and ``seconds``.  ``estimate`` is None unless both terms
    are available (production reads that as REFUSE).
    """
    t_start = time.perf_counter()
    est_c3 = geometry.ppgo_error_estimate(real_images, y, matrix, w_min)
    ghost_term = None
    im_tau_c = float('nan')
    try:
        ghost = geometry.ghost_kernel(np.array([w_min]), y, matrix)
        im_tau_c = float(ghost.delay.imag)
        ghost_term = float(abs(ghost.kernel[0]) * math.exp(-w_min * im_tau_c))
        ghost_status = 'ok'
    except geometry.GhostAbsentError:
        # Impossible on a 2-image census (4 real roots would be required);
        # record loudly rather than crash or silently zero the term.
        ghost_status = 'ghost_absent_unexpected'
    except geometry.GhostDomainError:
        ghost_status = 'ghost_unavailable'
    estimate = (est_c3 + ghost_term
                if est_c3 is not None and ghost_term is not None else None)
    return {'est_c3': est_c3, 'ghost_term': ghost_term, 'estimate': estimate,
            'ghost_status': ghost_status, 'im_tau_c': im_tau_c,
            'seconds': time.perf_counter() - t_start}


def _measure_witness(gamma, rho, angle, region, w_grid):
    """Measure one witness; returns a result dict, or None if not a 2-image,
    resolvable-or-symmetry-tied exterior config."""
    try:
        y = _polar_source(rho, angle, gamma)
    except geometry.LensDomainError:
        return None
    t_serve0 = time.perf_counter()
    try:
        geom, f_serve = _tier1_serve(w_grid, gamma, y)
    except geometry.LensDomainError:
        return None
    t_serve = time.perf_counter() - t_serve0
    if int(np.asarray(geom.real_mask).sum()) != 2:
        return None
    images = _real_images(geom)
    mdt = _min_delta_tau(geom)
    w_lo = float(w_grid.min())
    tied = mdt <= _TIE_EPS
    resolvable = w_lo * mdt >= RHO_END
    if not (tied or resolvable):
        return None

    matrix = geometry.macro_matrix(gamma, 0.0, 0.0)
    cert = _certificate(y, matrix, images, w_lo)

    t_oracle0 = time.perf_counter()
    try:
        f_exact = _exact_total_w(w_grid, gamma, y)
    except (geometry.LensDomainError, SchwingerCertificationError):
        # The exact engine can refuse marginal configs (e.g. near the
        # parity wall at the band ceiling); record rather than crash.
        return 'oracle_refused'
    t_oracle = time.perf_counter() - t_oracle0
    err = np.abs(f_serve - f_exact) / np.abs(f_exact)
    err_p90 = float(np.percentile(err, 90))
    err_max = float(err.max())

    ratio = (cert['estimate'] / err_max
             if cert['estimate'] is not None and err_max > 0 else None)
    ratio_c3 = (cert['est_c3'] / err_max
                if cert['est_c3'] is not None and err_max > 0 else None)
    return {'gamma': gamma, 'region': region, 'rho': rho, 'angle': angle,
            'y': y, 'eta': float(geom.caustic_distance),
            'sep': _min_image_separation(images), 'mdt': mdt, 'tied': tied,
            'err_p90': err_p90, 'err_max': err_max, 'ratio': ratio,
            'ratio_c3': ratio_c3, 't_serve': t_serve, 't_oracle': t_oracle,
            'n_w': len(w_grid), **cert}


def _anchor_witnesses(gammas):
    """Deterministic anchors as (gamma, region, rho, angle), ORDERED so a
    truncated (pilot) prefix keeps the decisive coverage: the gamma=1.2
    measured worst-case ridge (several scales), one exact-axis symmetry-tied
    mirror pair, one near-axis tied pair, one connecting-region witness, a
    ghost-available generic direction, and the transverse cone."""
    anchors = [
        # --- decisive gamma=1.2 block first ---
        (1.2, 'connecting', 0.3, 0.0),      # origin side, ON axis (tied)
        (1.2, 'connecting', 0.3, 0.03),     # near-axis mirror pair
        (1.2, 'ridge', 2.6, 1.415),         # measured worst (eta ~2.9)
        (1.2, 'ridge', 2.0, 1.41),
        (1.2, 'ridge', 1.5, 1.41),
        (1.2, 'ridge', 3.0, 1.40),
        (1.2, 'generic', 2.6, 0.7),         # ghost-available direction
        (1.2, 'transverse', 1.5, 0.5 * math.pi),
        # --- other gammas ---
        (1.5, 'connecting', 0.3, 0.0),
        (1.5, 'generic', 2.0, 1.0),         # ghost-available direction
        (2.0, 'connecting', 0.3, 0.0),
        (2.0, 'generic', 2.0, 1.0),
        (1.5, 'transverse', 1.5, 0.5 * math.pi),
        (2.0, 'transverse', 1.5, 0.5 * math.pi),
        (1.5, 'connecting', 0.3, 0.03),
        (2.0, 'connecting', 0.3, 0.03),
        (1.5, 'ridge', 2.2, 1.41),
        (2.0, 'ridge', 1.5, 1.41),
        (1.2, 'connecting', 0.15, 0.0),
        (1.2, 'generic', 1.0, 0.7),
    ]
    return [a for a in anchors if a[0] in gammas]


_REGION_RANGES = {
    # region: (rho_lo, rho_hi, angle_lo, angle_hi)
    'connecting': (0.05, 0.4, 0.0, 0.7),
    'transverse': (0.6, 3.0, 0.5 * math.pi - 0.35, 0.5 * math.pi),
    'generic': (0.8, 3.2, 0.5, 1.45),
    'ridge': (1.5, 3.0, 1.40, 1.42),
}


def _build_witness_list(gammas, n_witnesses, rng):
    """Ordered anchors first, then random draws across regions; truncate to
    ``n_witnesses``."""
    witnesses = _anchor_witnesses(gammas)
    regions = list(_REGION_RANGES)
    while len(witnesses) < n_witnesses:
        g = gammas[rng.integers(len(gammas))]
        region = regions[rng.integers(len(regions))]
        rho_lo, rho_hi, a_lo, a_hi = _REGION_RANGES[region]
        witnesses.append((g, region, float(rng.uniform(rho_lo, rho_hi)),
                          float(rng.uniform(a_lo, a_hi))))
    return witnesses[:n_witnesses]


def _pairing_gate(w_grid):
    """Verify serve/oracle frame pairing on ONE known-resolved config.

    Config: gamma=1.2, rho=3.0, angle=0.7 -- inside the certified
    far-from-caustic domain of the tier-1 accuracy suite; the production
    admission predicate is consulted, and the serve must match the exact
    engine to < 1e-3 (certified population measured p90 ~5e-5, max ~7e-4).
    Raises AssertionError on failure -- nothing downstream may be trusted.
    """
    gamma, rho, angle = 1.2, 3.0, 0.7
    y = _polar_source(rho, angle, gamma)
    geom, f_serve = _tier1_serve(w_grid, gamma, y)
    real = np.asarray(geom.real_mask, dtype=bool)
    real_delays = np.asarray(geom.delays)[real]
    rho_prod = caustic_rho(gamma, float(np.hypot(y[0], y[1])), kappa=0.0)
    admitted = _saddle_farfield_analytic_serves(
        real_delays, float(w_grid.min()), rho_prod)
    f_exact = _exact_total_w(w_grid, gamma, y)
    err_max = float((np.abs(f_serve - f_exact) / np.abs(f_exact)).max())
    assert admitted, (
        f'pairing-gate config unexpectedly refused by the production gate '
        f'(rho_prod={rho_prod:.3f})')
    assert err_max < 1e-3, (
        f'PAIRING GATE FAILED: max err {err_max:.3e} >= 1e-3 on a '
        f'known-resolved config -- serve and oracle frames are NOT paired.')
    print(f'PAIRING GATE PASS: gamma={gamma}, rho={rho}, angle={angle}, '
          f'production-gate admitted={admitted}, max err = {err_max:.3e} '
          f'< 1e-3')


def _print_table(rows):
    header = (f'{"region":<11s} {"gamma":>5s} {"rho":>5s} {"angle":>6s} '
              f'{"eta":>6s} {"sep":>6s} {"mdt":>9s} {"c3":>9s} '
              f'{"ghost":>9s} {"estimate":>9s} {"err_p90":>9s} '
              f'{"err_max":>9s} {"ratio":>8s} {"ghost_status":<18s} '
              f'{"t_orc":>6s} {"t_cert":>7s}')
    print(header)
    print('-' * len(header))
    for r in rows:
        def fmt(x, spec):
            return format(x, spec) if x is not None else '--'
        print(f'{r["region"]:<11s} {r["gamma"]:>5.2f} {r["rho"]:>5.2f} '
              f'{r["angle"]:>6.3f} {r["eta"]:>6.3f} {r["sep"]:>6.3f} '
              f'{r["mdt"]:>9.2e} {fmt(r["est_c3"], ".3e"):>9s} '
              f'{fmt(r["ghost_term"], ".3e"):>9s} '
              f'{fmt(r["estimate"], ".3e"):>9s} {r["err_p90"]:>9.2e} '
              f'{r["err_max"]:>9.2e} {fmt(r["ratio"], ".3f"):>8s} '
              f'{r["ghost_status"]:<18s} {r["t_oracle"]:>6.2f} '
              f'{r["seconds"]:>7.3f}')


def _summarize(rows):
    full = [r for r in rows if r['ratio'] is not None]
    unavailable = [r for r in rows if r['ghost_status'] != 'ok']
    print()
    print(f'witnesses measured: {len(rows)}; full certificate (c3+ghost) '
          f'available: {len(full)}; ghost unavailable: {len(unavailable)}')
    if full:
        ratios = np.array([r['ratio'] for r in full])
        optimistic = ratios < 1.0
        print(f'ratio = estimate / actual_max over the {len(full)} '
              f'full-certificate witnesses:')
        print(f'  min    = {ratios.min():.4f}')
        print(f'  median = {np.median(ratios):.4f}')
        print(f'  p99    = {np.percentile(ratios, 99):.4f}')
        print(f'  max    = {ratios.max():.4f}')
        print(f'  optimistic (ratio < 1): {int(optimistic.sum())}/'
              f'{len(full)} = {100.0 * optimistic.mean():.1f}%'
              f'  (interior bar: 0%)')
        for r in full:
            if r['ratio'] < 1.0:
                print(f'  OPTIMISTIC witness: {r["region"]} gamma='
                      f'{r["gamma"]:.2f} rho={r["rho"]:.2f} angle='
                      f'{r["angle"]:.3f} eta={r["eta"]:.3f} ratio='
                      f'{r["ratio"]:.4f}')
    if unavailable:
        print()
        print('ghost-unavailable witnesses (production reads: REFUSE) and '
              'their c3-only ratio (est_c3 / actual_max):')
        for r in unavailable:
            ratio_c3 = (f'{r["ratio_c3"]:.4f}' if r['ratio_c3'] is not None
                        else '--')
            print(f'  {r["region"]:<11s} gamma={r["gamma"]:.2f} '
                  f'rho={r["rho"]:.2f} angle={r["angle"]:.3f} '
                  f'eta={r["eta"]:.3f} sep={r["sep"]:.3f} '
                  f'mdt={r["mdt"]:.2e} c3_only_ratio={ratio_c3} '
                  f'[{r["ghost_status"]}]')


# ======================================================================
# Follow-up measurements (driver request 2026-08-14): root identity in the
# ghost-unavailable region, err(w) decay curves there, and a grown
# ghost-available ratio sample with per-(config, w) pointwise ratios.
# ======================================================================

def _jsonify(obj):
    """Recursively convert numpy / complex values to JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonify(obj.tolist())
    if isinstance(obj, complex):
        return {'re': float(obj.real), 'im': float(obj.imag)}
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    return obj


def _quartic_root_census(y, matrix):
    """All four image-quartic roots reconstructed to lens-plane positions.

    Mirrors ``geometry._ghost_candidates`` WITHOUT its ghost filters, so the
    roots it rejects can be identified: for each root the reconstructed
    position, the |Fermat-gradient| residual of the ORIGINAL lens equation
    (``A x - y - x / (x.x)``, bilinear so complex positions evaluate too),
    and -- for genuine real stationary points -- delay / Morse index.
    """
    y = np.asarray(y, dtype=float)
    source_radius, basis = geometry._source_frame(y)
    rotated = basis.T @ matrix @ basis
    a11 = float(rotated[0, 0])
    a12 = float(rotated[0, 1])
    a22 = float(rotated[1, 1])
    real_images = geometry.find_images(y, matrix)
    real_delays = [geometry.delay(x, y, matrix) for x in real_images]
    rows = []
    for raw_root in geometry._companion_roots(
            geometry.image_quartic_coefficients(source_radius, rotated)):
        u = complex(raw_root)
        row = {'root': u}
        denom = (a11 - u) * (a22 - u) - a12 * a12
        if abs(denom) <= 1e-12 * (1.0 + abs(a11 * a22) + abs(u) ** 2):
            row['reconstruction'] = 'degenerate_denominator'
            rows.append(row)
            continue
        x = basis @ np.array([source_radius * (a22 - u) / denom,
                              -source_radius * a12 / denom], dtype=complex)
        z = x[0] * x[0] + x[1] * x[1]
        grad = matrix @ x - y - x / z
        row['x'] = x
        row['lens_eq_residual'] = float(np.abs(grad).max())
        is_real = float(np.abs(x.imag).max()) <= 1e-9 * (
            1.0 + float(np.abs(x.real).max()))
        row['is_real_position'] = bool(is_real)
        matched = None
        for i, img in enumerate(real_images):
            if float(np.linalg.norm(x - img)) < 1e-6:
                matched = i
        row['matches_real_image'] = matched
        if is_real and row['lens_eq_residual'] < 1e-8:
            x_real = x.real
            tau = geometry.delay(x_real, y, matrix)
            hess = geometry.hessian(x_real, matrix)
            eigs = np.linalg.eigvalsh(hess)
            row['delay'] = float(tau)
            row['delta_tau_to_nearest_image'] = float(
                min(abs(tau - t) for t in real_delays))
            row['morse_index'] = int(geometry.morse_index(x_real, matrix))
            row['hessian_eigs'] = [float(e) for e in eigs]
            row['magnification'] = float(
                geometry.magnification(x_real, matrix))
        rows.append(row)
    return {'real_images': [list(map(float, x)) for x in real_images],
            'real_delays': [float(t) for t in real_delays],
            'roots': rows}


_ROOT_IDENTITY_WITNESSES = [
    # coordinator's four, plus on-axis/near-axis pairs where the exact axis
    # is a coordinate collapse of the source-aligned reduction
    (1.2, 'connecting', 0.3, 0.0),
    (1.2, 'connecting', 0.3, 0.03),
    (1.2, 'ridge', 2.0, 1.41),
    (1.2, 'transverse', 1.5, 0.5 * math.pi),
    (1.2, 'transverse', 1.5, 0.5 * math.pi - 0.03),
    (2.0, 'generic', 2.0, 1.0),
]


def _followup_root_identity():
    """Task 1: what ARE the two non-image quartic roots where the ghost is
    unavailable?"""
    results = []
    print('=== Task 1: root identity in the ghost-unavailable region ===')
    for gamma, region, rho, angle in _ROOT_IDENTITY_WITNESSES:
        y = _polar_source(rho, angle, gamma)
        matrix = geometry.macro_matrix(gamma, 0.0, 0.0)
        census = _quartic_root_census(y, matrix)
        results.append({'gamma': gamma, 'region': region, 'rho': rho,
                        'angle': angle, 'y': list(map(float, y)), **census})
        print(f'-- {region} gamma={gamma} rho={rho} angle={angle:.3f} '
              f'y=({y[0]:.4f},{y[1]:.4f}); real delays='
              f'{[f"{t:.4f}" for t in census["real_delays"]]}')
        for row in census['roots']:
            u = row['root']
            if row.get('reconstruction') == 'degenerate_denominator':
                print(f'   root u=({u.real:+.4f},{u.imag:+.4f}i): '
                      f'DEGENERATE reconstruction (axis collapse)')
                continue
            x = row['x']
            desc = (f'   root u=({u.real:+.4f},{u.imag:+.4f}i) -> '
                    f'x=({x[0].real:+.4f}{x[0].imag:+.3f}i,'
                    f'{x[1].real:+.4f}{x[1].imag:+.3f}i) '
                    f'|lens_eq|={row["lens_eq_residual"]:.2e} '
                    f'real={row["is_real_position"]} '
                    f'img_match={row["matches_real_image"]}')
            if 'delay' in row:
                desc += (f' tau={row["delay"]:+.4f} '
                         f'dtau_min={row["delta_tau_to_nearest_image"]:.4f} '
                         f'morse={row["morse_index"]} '
                         f'eigs=({row["hessian_eigs"][0]:+.3f},'
                         f'{row["hessian_eigs"][1]:+.3f}) '
                         f'mu={row["magnification"]:+.3f}')
            print(desc)
    print()
    return results


_ERROR_CURVE_WITNESSES = [
    (1.2, 'connecting', 0.3, 0.0),      # exactly on-axis (tied)
    (1.2, 'connecting', 0.3, 0.03),     # slightly off-axis
    (1.5, 'connecting', 0.3, 0.0),
    (2.0, 'connecting', 0.3, 0.0),
    (1.2, 'ridge', 2.0, 1.41),
    (1.2, 'ridge', 2.6, 1.415),
    (1.2, 'transverse', 1.5, 0.5 * math.pi),
    (2.0, 'generic', 2.0, 1.0),
]


def _fit_decay(w, err):
    """Fit err(w) as exp(-a*w) vs w**-k; return both fits with R^2 in the
    fitted (log) currency, and a verdict."""
    log_err = np.log(err)

    def r_squared(x_axis):
        slope, intercept = np.polyfit(x_axis, log_err, 1)
        fitted = slope * x_axis + intercept
        ss_res = float(np.sum((log_err - fitted) ** 2))
        ss_tot = float(np.sum((log_err - log_err.mean()) ** 2))
        return slope, 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    slope_w, r2_exp = r_squared(np.asarray(w))
    slope_logw, r2_pow = r_squared(np.log(w))
    verdict = 'exponential' if r2_exp > r2_pow else 'power_law'
    return {'exp_a': -slope_w, 'exp_r2': r2_exp,
            'pow_k': -slope_logw, 'pow_r2': r2_pow, 'verdict': verdict}


def _followup_error_curves(n_w=24, w_ceil=60.0):
    """Task 2: err(w) decay curves for ghost-unavailable witnesses."""
    w_grid = np.geomspace(W_FLOOR, w_ceil, n_w)
    results = []
    print(f'=== Task 2: err(w) curves, {n_w} nodes in [{W_FLOOR}, {w_ceil}] '
          f'===')
    for gamma, region, rho, angle in _ERROR_CURVE_WITNESSES:
        y = _polar_source(rho, angle, gamma)
        geom, f_serve = _tier1_serve(w_grid, gamma, y)
        f_exact = _exact_total_w(w_grid, gamma, y)
        err = np.abs(f_serve - f_exact) / np.abs(f_exact)
        # smallest w from which the tail [w, w_ceil] meets the production
        # contract (p90 <= 1e-3 AND max <= 1e-2 over the tail nodes)
        w_pass = None
        for i in range(len(w_grid)):
            tail = err[i:]
            if (np.percentile(tail, 90) <= 1e-3
                    and tail.max() <= 1e-2):
                w_pass = float(w_grid[i])
                break
        fit = _fit_decay(w_grid, err)
        results.append({'gamma': gamma, 'region': region, 'rho': rho,
                        'angle': angle, 'eta': float(geom.caustic_distance),
                        'mdt': _min_delta_tau(geom),
                        'w': list(map(float, w_grid)),
                        'err': list(map(float, err)),
                        'w_contract_pass': w_pass, **fit})
        curve = ' '.join(f'({w:.1f},{e:.1e})' for w, e in zip(w_grid, err))
        print(f'-- {region} gamma={gamma} rho={rho} angle={angle:.3f} '
              f'eta={geom.caustic_distance:.3f} '
              f'mdt={_min_delta_tau(geom):.2e}')
        print(f'   err(w): {curve}')
        print(f'   w_contract_pass={w_pass}  fit: exp a={fit["exp_a"]:.4f} '
              f'(R2={fit["exp_r2"]:.3f}) vs pow k={fit["pow_k"]:.2f} '
              f'(R2={fit["pow_r2"]:.3f}) -> {fit["verdict"]}')
    print()
    return results


def _pointwise_certificate(y, matrix, images, w_grid):
    """Per-w certificate estimate(w) over the grid, or None per point where
    unavailable.  ``ppgo_error_estimate`` accepts any positive scalar w (it
    is the c3 sum scaled by w**-3), so the interior per-(config, w)
    methodology applies directly."""
    try:
        ghost = geometry.ghost_kernel(w_grid, y, matrix)
        im_tau = float(ghost.delay.imag)
        ghost_terms = np.abs(ghost.kernel) * np.exp(-w_grid * im_tau)
    except geometry.GhostDomainError:
        return None
    estimates = []
    for w_i, g_i in zip(w_grid, ghost_terms):
        c3_i = geometry.ppgo_error_estimate(images, y, matrix, float(w_i))
        estimates.append(None if c3_i is None else c3_i + float(g_i))
    return estimates


def _followup_ratio_sample(n_sample=40, n_w=24, w_ceil=60.0, seed=42):
    """Task 3: grow the ghost-available ratio sample + coverage map."""
    gammas = (1.1, 1.2, 1.35, 1.5, 2.0, 2.5)
    rng = np.random.default_rng(seed + 1)
    w_grid = np.geomspace(W_FLOOR, w_ceil, n_w)
    results = []
    print(f'=== Task 3: ratio sample, {n_sample} draws, gamma in {gammas}, '
          f'angle in [0, pi/2], rho in [1.2, 4] ===')
    for _ in range(n_sample):
        gamma = float(gammas[rng.integers(len(gammas))])
        angle = float(rng.uniform(0.0, 0.5 * math.pi))
        rho = float(rng.uniform(1.2, 4.0))
        rec = {'gamma': gamma, 'angle': angle, 'rho': rho}
        row = _measure_witness(gamma, rho, angle, 'sampled', w_grid)
        if row is None:
            rec['status'] = 'filtered'  # not 2-image / unresolvable-untied
            results.append(rec)
            continue
        if row == 'oracle_refused':
            rec['status'] = 'oracle_refused'
            results.append(rec)
            continue
        rec.update({k: row[k] for k in
                    ('eta', 'sep', 'mdt', 'tied', 'est_c3', 'ghost_term',
                     'estimate', 'ghost_status', 'im_tau_c', 'err_p90',
                     'err_max', 'ratio', 'ratio_c3')})
        rec['y'] = list(map(float, row['y']))
        rec['status'] = 'measured'
        if row['ghost_status'] == 'ok':
            y = np.asarray(row['y'])
            matrix = geometry.macro_matrix(gamma, 0.0, 0.0)
            geom, f_serve = _tier1_serve(w_grid, gamma, y)
            f_exact = _exact_total_w(w_grid, gamma, y)
            err = np.abs(f_serve - f_exact) / np.abs(f_exact)
            estimates = _pointwise_certificate(
                y, matrix, _real_images(geom), w_grid)
            if estimates is not None:
                pw = [(float(w), float(e), est)
                      for w, e, est in zip(w_grid, err, estimates)
                      if est is not None and e > 0]
                rec['pointwise'] = [
                    {'w': w, 'err': e, 'estimate': est, 'ratio': est / e}
                    for w, e, est in pw]
        results.append(rec)

    measured = [r for r in results if r['status'] == 'measured']
    full = [r for r in measured if r.get('ratio') is not None]
    print(f'sampled {len(results)}; measured (2-image, resolvable-or-tied) '
          f'{len(measured)}; full certificate {len(full)}')
    if full:
        ratios = np.array([r['ratio'] for r in full])
        print(f'band-min ratio (estimate(w_lo=8)/actual_max), {len(full)} '
              f'witnesses: min={ratios.min():.4f} '
              f'median={np.median(ratios):.4f} '
              f'p99={np.percentile(ratios, 99):.4f} max={ratios.max():.4f} '
              f'optimistic={100.0 * np.mean(ratios < 1):.1f}%')
        pw_ratios = np.array([p['ratio'] for r in full
                              for p in r.get('pointwise', [])])
        if len(pw_ratios):
            print(f'pointwise per-(config,w) ratio, {len(pw_ratios)} points: '
                  f'min={pw_ratios.min():.4f} '
                  f'median={np.median(pw_ratios):.4f} '
                  f'p99={np.percentile(pw_ratios, 99):.4f} '
                  f'max={pw_ratios.max():.4f} '
                  f'optimistic={100.0 * np.mean(pw_ratios < 1):.1f}%')
        for r in full:
            worst = min((p['ratio'] for p in r.get('pointwise', [])),
                        default=r['ratio'])
            if r['ratio'] < 1.0 or worst < 1.0:
                print(f'  OPTIMISTIC: gamma={r["gamma"]:.2f} '
                      f'rho={r["rho"]:.2f} angle={r["angle"]:.3f} '
                      f'eta={r["eta"]:.3f} band ratio={r["ratio"]:.4f} '
                      f'worst pointwise={worst:.4f}')
    print('coverage per gamma (of measured): ghost-available / measured, '
          'angle range of availability:')
    for gamma in gammas:
        rows = [r for r in measured if r['gamma'] == gamma]
        ok = [r for r in rows if r['ghost_status'] == 'ok']
        if rows:
            ok_angles = sorted(r['angle'] for r in ok)
            span = (f'{ok_angles[0]:.2f}-{ok_angles[-1]:.2f} rad'
                    if ok_angles else '--')
            print(f'  gamma={gamma}: {len(ok)}/{len(rows)} available '
                  f'(angles {span})')
        else:
            print(f'  gamma={gamma}: 0 measured')
    print()
    return results


def _run_followup(args):
    """Run the three follow-up measurements and dump raw JSON."""
    import json
    import subprocess
    w_grid = np.geomspace(W_FLOOR, args.w_ceil, args.n_w)
    _pairing_gate(w_grid)
    print()
    payload = {
        'meta': {
            'sha': subprocess.run(
                ['git', 'rev-parse', '--short', 'HEAD'],
                capture_output=True, text=True).stdout.strip(),
            'w_floor': W_FLOOR, 'w_ceil': args.w_ceil, 'n_w': args.n_w,
            'seed': args.seed,
        },
        'root_identity': _followup_root_identity(),
        'error_curves': _followup_error_curves(
            n_w=args.n_w, w_ceil=args.w_ceil),
        'ratio_sample': _followup_ratio_sample(
            n_sample=40, n_w=args.n_w, w_ceil=args.w_ceil, seed=args.seed),
    }
    out_path = 'scripts/calibration_pilot_followup.json'
    with open(out_path, 'w') as stream:
        json.dump(_jsonify(payload), stream, indent=1)
    print(f'raw results written to {out_path}')


def _time_mpmath_band(gamma=1.2, rho=3.0, angle=0.7):
    """Time ONE single-point oracle eval at w=100 and one at w=148 (mpmath
    band) so a full-band calibration can be priced.  ~85-120 s EACH."""
    y = _polar_source(rho, angle, gamma)
    for w_point in (100.0, 148.0):
        t0 = time.perf_counter()
        _exact_total_w(np.array([w_point]), gamma, y)
        seconds = time.perf_counter() - t0
        print(f'mpmath-band oracle timing: single eval at w={w_point:.0f} '
              f'took {seconds:.1f} s  (gamma={gamma}, rho={rho}, '
              f'angle={angle})')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n-witnesses', type=int, default=30)
    parser.add_argument('--gammas', type=float, nargs='+',
                        default=list(DEFAULT_GAMMAS))
    parser.add_argument('--n-w', type=int, default=24)
    parser.add_argument('--w-ceil', type=float, default=W_CEIL_DEFAULT)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pilot', action='store_true',
                        help='small pilot: 12 witnesses, n_w=8, w<=60')
    parser.add_argument('--time-mpmath', action='store_true',
                        help='also time one oracle eval at w=100 and w=148 '
                             '(~85-120 s EACH; prices a full-band '
                             'calibration; off by default)')
    parser.add_argument('--followup', action='store_true',
                        help='run the follow-up measurements (root identity, '
                             'err(w) curves, grown ratio sample) and write '
                             'scripts/calibration_pilot_followup.json')
    args = parser.parse_args()
    if args.followup:
        _run_followup(args)
        return
    if args.pilot:
        args.n_witnesses = min(args.n_witnesses, 12)
        args.n_w = min(args.n_w, 8)
        args.w_ceil = min(args.w_ceil, 60.0)

    w_grid = np.geomspace(W_FLOOR, args.w_ceil, args.n_w)
    rng = np.random.default_rng(args.seed)

    print(f'certificate: ppgo_error_estimate + |ghost_kernel|*exp(-w*Im tau) '
          f'at w_min = {W_FLOOR}')
    print(f'actual: pointwise |F_serve - F_exact|/|F_exact| on {args.n_w} '
          f'geomspace nodes over [{W_FLOOR}, {args.w_ceil}]')
    print(f'gammas = {args.gammas}, n_witnesses = {args.n_witnesses}, '
          f'seed = {args.seed}')
    print()
    _pairing_gate(w_grid)
    print()

    t_total0 = time.perf_counter()
    rows = []
    n_skipped = 0
    for gamma, region, rho, angle in _build_witness_list(
            tuple(args.gammas), args.n_witnesses, rng):
        row = _measure_witness(gamma, rho, angle, region, w_grid)
        if not isinstance(row, dict):
            n_skipped += 1
            continue
        rows.append(row)
    t_total = time.perf_counter() - t_total0

    _print_table(rows)
    _summarize(rows)
    if n_skipped:
        print(f'\nskipped {n_skipped} witness placements (not 2-image, '
              f'unresolvable-and-untied, or domain refusal)')
    if rows:
        t_oracle = sum(r['t_oracle'] for r in rows)
        t_cert = sum(r['seconds'] for r in rows)
        t_serve = sum(r['t_serve'] for r in rows)
        print(f'\ntiming: {len(rows)} witnesses in {t_total:.1f} s '
              f'({t_total / len(rows):.2f} s/witness); oracle '
              f'{t_oracle:.1f} s total ({t_oracle / len(rows):.2f} '
              f's/witness, {t_oracle / len(rows) / args.n_w * 1e3:.0f} '
              f'ms/eval); serve+partition {t_serve:.1f} s; certificate '
              f'{t_cert:.2f} s total ({t_cert / len(rows) * 1e3:.1f} '
              f'ms/witness)')
    if args.time_mpmath:
        print()
        _time_mpmath_band()


if __name__ == '__main__':
    main()
