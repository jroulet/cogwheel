import sys
sys.path.insert(0, '.')
import math
import numpy as np
from cogwheel.lensing.chang_refsdal import geometry
from cogwheel.lensing.chang_refsdal.channels import (
    ChangRefsdalChannels, FARFIELD_KERNEL_SUM, reconstruct_farfield)
from cogwheel.lensing.likelihood import (
    _SADDLE_TIE_EPS, _saddle_farfield_analytic_serves)
from cogwheel.lensing.ppgo_map import caustic_geometry

W_FLOOR, W_CEIL, N_W = 8.0, 60.0, 24
P90_TOL, OUTLIER_TOL = 1e-3, 1e-2
ETA_PROBE = 1e6

def _polar_source(rho, angle, gamma, kappa=0.0):
    reach, _d = caustic_geometry(gamma, kappa=kappa)
    radius = rho * reach
    return radius * np.array([math.cos(angle), math.sin(angle)])

w_grid = np.linspace(W_FLOOR, W_CEIL, N_W)
w_lo = float(w_grid.min())

worst_overall = None
for gamma in [1.1, 1.15, 1.2, 1.25, 1.3]:
    angles = np.linspace(1.3, 2.3, 30)
    scales = np.linspace(1.4, 2.6, 30)
    worst = None
    n_fail = 0
    n_admit = 0
    for angle in angles:
        for scale in scales:
            y = _polar_source(scale, angle, gamma)
            try:
                geom = ChangRefsdalChannels(w_grid).geometry_partition(
                    gamma=gamma, y=(float(y[0]), float(y[1])), beta=0.0, kappa=0.0)
            except geometry.LensDomainError:
                continue
            if int(np.asarray(geom.real_mask).sum()) != 2:
                continue
            real = np.asarray(geom.delays)[np.asarray(geom.real_mask, dtype=bool)]
            if not _saddle_farfield_analytic_serves(real, w_lo, ETA_PROBE):
                continue
            n_admit += 1
            eta = float(geom.caustic_distance)
            envelope = np.zeros(w_grid.shape, dtype=complex)
            _k, f_serve = reconstruct_farfield(
                w_grid, envelope, geom.delays, geom.saddle_kernels, geom.real_mask,
                FARFIELD_KERNEL_SUM, geom.t_min)
            ch = ChangRefsdalChannels(w_grid)
            ch.reset()
            part = ch.evaluate(gamma=gamma, y=(float(y[0]), float(y[1])), beta=0.0, kappa=0.0)
            f_exact = part.exact_total
            err = np.abs(f_serve - f_exact) / np.abs(f_exact)
            p90 = float(np.percentile(err, 90))
            emax = float(err.max())
            failing = p90 > P90_TOL or emax > OUTLIER_TOL
            if failing:
                n_fail += 1
                if worst is None or eta > worst[0]:
                    worst = (eta, angle, scale, p90, emax)
    print(f'gamma={gamma}: n_admit={n_admit} n_fail={n_fail} worst={worst}')
    if worst is not None and (worst_overall is None or worst[0] > worst_overall[0]):
        worst_overall = (gamma,) + worst

print('WORST OVERALL:', worst_overall)
