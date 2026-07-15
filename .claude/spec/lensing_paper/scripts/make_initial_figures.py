from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / 'code'))

from chang_refsdal_operator import chang_refsdal_amplification
from chang_refsdal_geometry import (
    critical_point, delay, find_images, hessian, image_kernel, macro_matrix,
)

GAMMA = 0.2
KAPPA = 0.2
BETA = 0.0
A = macro_matrix(GAMMA, BETA, KAPPA)

# Representative source positions.  The fold and cusp points lie just inside
# the astroid caustic, so that all four images are present while the relevant
# image cluster is close to degeneracy.
_, y_fold_c, *_ = critical_point(GAMMA, np.pi / 4, BETA, KAPPA)
CASES = {
    'two-image region': np.array([0.55, 0.0]),
    'four-image region': np.array([0.10, 0.10]),
    'near a fold': 0.99 * y_fold_c,
    'near a cusp': np.array([-0.395, 0.0]),
}

W_LOW = np.geomspace(0.1, 8.0, 64)
W_HIGH = np.linspace(8.0, 40.0, 241)
W_SP = np.linspace(12.0, 40.0, 211)


def exact_grid(w, y):
    vals = []
    for wi in np.asarray(w):
        val, diag = chang_refsdal_amplification(
            float(wi), y, GAMMA, beta=BETA, kappa=KAPPA,
            max_order=70, dps=90, tolerance=2e-12,
            return_diagnostics=True,
        )
        if (not diag.converged) and diag.estimated_relative_tail > 1e-10:
            raise RuntimeError(f'operator series failed at w={wi}, y={y}: {diag}')
        vals.append(val)
    return np.asarray(vals)


def stationary_phase_grid(w, y, order=0):
    w = np.asarray(w, float)
    total = np.zeros_like(w, dtype=complex)
    for x in find_images(y, A):
        tau = delay(x, y, A)
        if order == 0:
            H = hessian(x, A)
            eig = np.linalg.eigvalsh(H)
            morse = int(np.sum(eig < 0.0))
            mu = 1.0 / np.linalg.det(H)
            kernel = np.full_like(w, np.sqrt(abs(mu)) * np.exp(-0.5j*np.pi*morse), dtype=complex)
        elif order == 2:
            kernel = image_kernel(w, x, A)
        else:
            raise ValueError(order)
        total += np.exp(1j*w*tau) * kernel
    return total


# Cache exact data so future revisions do not repeat the special-function sums.
cache = ROOT / 'data' / 'initial_figure_data.npz'
if cache.exists():
    data = dict(np.load(cache, allow_pickle=False))
else:
    data = {}
    for i, (name, y) in enumerate(CASES.items()):
        data[f'low_{i}'] = exact_grid(W_LOW, y)
        data[f'high_{i}'] = exact_grid(W_HIGH, y)
    np.savez(cache, **data)

# Figure 1: exact amplification with logarithmic low-frequency and linear
# high-frequency panels.  The two attached axes are visually joined.
fig = plt.figure(figsize=(7.2, 6.0), constrained_layout=True)
outer = fig.add_gridspec(2, 2)
for idx, (name, y) in enumerate(CASES.items()):
    r, c = divmod(idx, 2)
    inner = outer[r, c].subgridspec(1, 2, width_ratios=[1.0, 1.65], wspace=0.045)
    axl = fig.add_subplot(inner[0, 0])
    axh = fig.add_subplot(inner[0, 1], sharey=axl)
    Fl = data[f'low_{idx}']
    Fh = data[f'high_{idx}']
    axl.plot(W_LOW, np.abs(Fl), lw=1.2)
    axh.plot(W_HIGH, np.abs(Fh), lw=1.2)
    axl.set_xscale('log')
    axl.set_xlim(W_LOW[0], W_LOW[-1])
    axh.set_xlim(W_HIGH[0], W_HIGH[-1])
    axl.spines['right'].set_visible(False)
    axh.spines['left'].set_visible(False)
    axh.tick_params(labelleft=False, left=False)
    axl.tick_params(right=False)
    # Small diagonal marks show the change in coordinate sampling.
    d = .015
    kwargs = dict(transform=axl.transAxes, clip_on=False, linewidth=0.8, color='black')
    axl.plot((1-d, 1+d), (-d, +d), **kwargs)
    axl.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    kwargs.update(transform=axh.transAxes)
    axh.plot((-d, +d), (-d, +d), **kwargs)
    axh.plot((-d, +d), (1-d, 1+d), **kwargs)
    axl.set_title(name, fontsize=10)
    axl.set_ylabel(r'$|F(w)|$')
    if r == 1:
        axl.set_xlabel(r'$w$ (logarithmic)')
        axh.set_xlabel(r'$w$ (linear)')
    axl.grid(alpha=0.2)
    axh.grid(alpha=0.2)
fig.savefig(FIG / 'amplification_mixed.pdf', bbox_inches='tight')
fig.savefig(FIG / 'amplification_mixed.png', dpi=220, bbox_inches='tight')
plt.close(fig)

# Figure 2: stationary phase in regular two- and four-image regions.  The
# high-frequency range is sampled linearly, preventing apparent irregularity
# of the fringes.
fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.4), sharex=True, constrained_layout=True)
sp_cases = [('two-image region', np.array([0.90, 0.0])), ('four-image region', np.array([0.03, 0.10]))]
for ax, (name, y) in zip(axes, sp_cases):
    F = exact_grid(W_SP, y)
    G0 = stationary_phase_grid(W_SP, y, order=0)
    G2 = stationary_phase_grid(W_SP, y, order=2)
    ax.plot(W_SP, np.abs(F), lw=1.4, label='operator total')
    ax.plot(W_SP, np.abs(G0), lw=1.0, ls='--', label='leading stationary phase')
    ax.plot(W_SP, np.abs(G2), lw=1.0, ls=':', label=r'through $O(w^{-2})$')
    ax.set_ylabel(r'$|F(w)|$')
    ax.set_title(name, fontsize=10)
    ax.grid(alpha=0.2)
axes[-1].set_xlabel(r'$w$')
axes[1].legend(frameon=False, ncol=1, fontsize=8, loc='upper right')
fig.savefig(FIG / 'stationary_phase_comparison.pdf', bbox_inches='tight')
fig.savefig(FIG / 'stationary_phase_comparison.png', dpi=220, bbox_inches='tight')
plt.close(fig)


print('Wrote figures to', FIG)
