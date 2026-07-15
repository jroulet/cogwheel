from pathlib import Path
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / 'code'))

from chang_refsdal_global_tracking import GlobalChannelTracker
from chang_refsdal_topology_stable import build_fold_crossing_partition
from chang_refsdal_geometry import delay, find_images, image_kernel, macro_matrix
from chang_refsdal_exact_partition import interpolate_complex_linear
from exact_gauge_partition import reconstructed_total

GAMMA = 0.2
KAPPA = 0.2
BETA = 0.0
A = macro_matrix(GAMMA, BETA, KAPPA)


def stationary_phase_shifted(w, y):
    w = np.asarray(w, float)
    y = np.asarray(y, float)
    images = find_images(y, A)
    times = np.array([delay(x, y, A) for x in images])
    tmin = float(times.min())
    out = np.zeros_like(w, dtype=complex)
    for x, t in zip(images, times):
        out += np.exp(1j * w * (t - tmin)) * image_kernel(w, x, A)
    return out


def profile_error(approx, truth):
    floor = 0.15 * np.max(np.abs(truth))
    return np.abs(approx - truth) / np.maximum(np.abs(truth), floor)


def carrier_channel_ratios(fid, cand):
    return (
        np.exp(1j * np.multiply.outer(fid.w, cand.delays - fid.delays))
        * cand.kernels / fid.kernels
    )


def reconstruct_channels(fid, cand, node_indices):
    node_indices = np.asarray(node_indices, int)
    ratios = carrier_channel_ratios(fid, cand)
    interp = interpolate_complex_linear(
        fid.w[node_indices], ratios[node_indices], fid.w
    )
    return reconstructed_total(fid.w, fid.delays, fid.kernels * interp)


def reconstruct_direct(fid, cand, node_indices):
    node_indices = np.asarray(node_indices, int)
    ratio = cand.exact_total / fid.exact_total
    interp = interpolate_complex_linear(
        fid.w[node_indices], ratio[node_indices], fid.w
    )
    return fid.exact_total * interp


def greedy_curve(fid, cand, method, max_nodes=None):
    n = len(fid.w)
    if max_nodes is None:
        max_nodes = n
    nodes = {0, n - 1}
    counts, errors = [], []
    while len(nodes) <= min(max_nodes, n):
        idx = np.array(sorted(nodes), int)
        approx = (
            reconstruct_channels(fid, cand, idx)
            if method == 'channels'
            else reconstruct_direct(fid, cand, idx)
        )
        err = profile_error(approx, cand.exact_total)
        counts.append(len(idx))
        errors.append(float(np.max(err)))
        if len(nodes) == n or len(nodes) == max_nodes:
            break
        search = err.copy()
        search[idx] = -1.0
        nodes.add(int(np.argmax(search)))
    return np.asarray(counts), np.asarray(errors)


# Figure: isolated-image asymptotics versus the exact projected decomposition.
w = np.linspace(5.0, 40.0, 141)
y_hard = np.array([0.10, 0.10])
tracker = GlobalChannelTracker(w, operator_dps=75, operator_max_order=52)
part = tracker.evaluate(gamma=GAMMA, kappa=KAPPA, beta=BETA, y=y_hard)
sp = stationary_phase_shifted(w, y_hard)
proj = part.reconstructed
err_sp = profile_error(sp, part.exact_total)
err_proj = profile_error(proj, part.exact_total)

fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.2), sharex=True,
                         gridspec_kw={'height_ratios': [2.1, 1.0]},
                         constrained_layout=True)
axes[0].plot(w, np.abs(part.exact_total), lw=1.5, label='operator total')
axes[0].plot(w, np.abs(sp), lw=1.0, ls='--', label=r'isolated-image expansion through $w^{-2}$')
axes[0].plot(w, np.abs(proj), lw=0.9, ls=':', label='projected channel sum')
axes[0].set_ylabel(r'$|F(w)|$')
axes[0].legend(frameon=False, fontsize=8, ncol=3)
axes[0].grid(alpha=0.2)
axes[1].semilogy(w, np.maximum(err_sp, 1e-16), lw=1.1, ls='--', label='isolated-image expansion')
axes[1].semilogy(w, np.maximum(err_proj, 1e-16), lw=1.0, ls=':', label='projected channel sum')
axes[1].set_ylabel('normalized error')
axes[1].set_xlabel(r'$w$')
axes[1].grid(alpha=0.2)
axes[1].legend(frameon=False, fontsize=8)
fig.savefig(FIG / 'projected_residual_comparison.pdf', bbox_inches='tight')
fig.savefig(FIG / 'projected_residual_comparison.png', dpi=220, bbox_inches='tight')
plt.close(fig)

# Figure: a fixed-topology example where direct relative binning fails.
# Both sources lie outside the same generic fold and therefore have the same
# two-image topology.  The two additional labels form a virtual fold cluster.
w = np.linspace(5.0, 40.0, 141)
fold_fid = build_fold_crossing_partition(
    w, gamma=GAMMA, theta_c=4.0, eta_s=-0.040,
    operator_max_order=42, operator_dps=70,
)
fold_cand = build_fold_crossing_partition(
    w, gamma=GAMMA, theta_c=4.0, eta_s=-0.010,
    operator_max_order=42, operator_dps=70,
)
direct_ratio = fold_cand.exact_total / fold_fid.exact_total
ratios = []
labels = []
for j in range(fold_fid.persistent_kernels.shape[1]):
    ratios.append(fold_cand.persistent_kernels[:, j] /
                  fold_fid.persistent_kernels[:, j])
    labels.append(f'persistent {j+1}')
for j in range(fold_fid.cluster_kernels.shape[1]):
    ratios.append(fold_cand.cluster_kernels[:, j] /
                  fold_fid.cluster_kernels[:, j])
    labels.append(f'fold component {j+1}')

direct_phase = np.unwrap(np.angle(direct_ratio))
direct_phase -= direct_phase[0]
ratio_phase = [np.unwrap(np.angle(r)) - np.angle(r[0]) for r in ratios]

fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.7), sharex=True,
                         constrained_layout=True)
axes[0, 0].plot(w, np.abs(direct_ratio), lw=1.25)
axes[0, 1].plot(w, direct_phase, lw=1.25)
for r, ph, label in zip(ratios, ratio_phase, labels):
    axes[1, 0].plot(w, np.abs(r), lw=1.05, label=label)
    axes[1, 1].plot(w, ph, lw=1.05)
axes[0, 0].set_ylabel(r'$|F/F_0|$')
axes[0, 1].set_ylabel(r'$\Delta\arg(F/F_0)$')
axes[1, 0].set_ylabel(r'$|K_a/K_{a0}|$')
axes[1, 1].set_ylabel(r'$\Delta\arg(K_a/K_{a0})$')
axes[1, 0].set_xlabel(r'$w$')
axes[1, 1].set_xlabel(r'$w$')
axes[0, 0].set_title('complete amplification ratio', fontsize=10)
axes[0, 1].set_title('complete amplification ratio', fontsize=10)
axes[1, 0].set_title('separated ratios', fontsize=10)
axes[1, 1].set_title('separated ratios', fontsize=10)
axes[1, 0].legend(frameon=False, fontsize=8, ncol=2)
for ax in axes.flat:
    ax.grid(alpha=0.2)
fig.savefig(FIG / 'projected_kernel_ratios.pdf', bbox_inches='tight')
fig.savefig(FIG / 'projected_kernel_ratios.png', dpi=220, bbox_inches='tight')
plt.close(fig)


def reconstruct_fold_channels(fid, cand, node_indices):
    node_indices = np.asarray(node_indices, int)
    wn = fid.w[node_indices]
    total = np.zeros_like(cand.exact_total)
    if fid.persistent_kernels.shape[1]:
        q = cand.persistent_kernels / fid.persistent_kernels
        qi = interpolate_complex_linear(wn, q[node_indices], fid.w)
        # The candidate delay phases are retained exactly rather than
        # interpolated as part of q.
        total += reconstructed_total(
            fid.w, cand.persistent_delays, fid.persistent_kernels * qi
        )
    q = cand.cluster_kernels / fid.cluster_kernels
    qi = interpolate_complex_linear(wn, q[node_indices], fid.w)
    total += reconstructed_total(
        fid.w, cand.cluster_delays, fid.cluster_kernels * qi
    )
    return total


def reconstruct_fold_direct(fid, cand, node_indices):
    node_indices = np.asarray(node_indices, int)
    ratio = cand.exact_total / fid.exact_total
    interp = interpolate_complex_linear(
        fid.w[node_indices], ratio[node_indices], fid.w
    )
    return fid.exact_total * interp


def greedy_fold(fid, cand, method, max_nodes=None):
    n = len(fid.w)
    if max_nodes is None:
        max_nodes = n
    nodes = {0, n - 1}
    counts, errors = [], []
    while len(nodes) <= min(max_nodes, n):
        idx = np.array(sorted(nodes), int)
        approx = (
            reconstruct_fold_channels(fid, cand, idx)
            if method == 'channels'
            else reconstruct_fold_direct(fid, cand, idx)
        )
        err = profile_error(approx, cand.exact_total)
        counts.append(len(idx))
        errors.append(float(np.max(err)))
        if len(nodes) == n or len(nodes) == max_nodes:
            break
        search = err.copy()
        search[idx] = -1.0
        nodes.add(int(np.argmax(search)))
    return np.asarray(counts), np.asarray(errors)


# Figure and CSV: interpolation error with exact candidate delay phases.
nc, ec = greedy_fold(fold_fid, fold_cand, 'channels')
nd, ed = greedy_fold(fold_fid, fold_cand, 'direct')
target = 1e-3
n_channel_target = int(nc[np.flatnonzero(ec <= target)[0]]) if np.any(ec <= target) else -1
n_direct_target = int(nd[np.flatnonzero(ed <= target)[0]]) if np.any(ed <= target) else -1

fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
ax.semilogy(nc, ec, marker='o', ms=3, lw=1.0, label='separated ratios; delays exact')
ax.semilogy(nd, ed, marker='s', ms=3, lw=1.0, label='complete amplification ratio')
ax.axhline(target, ls='--', lw=0.9, label=r'$10^{-3}$ target')
ax.set_xlabel('number of frequency nodes')
ax.set_ylabel('maximum normalized complex error')
ax.grid(alpha=0.2)
ax.legend(frameon=False, fontsize=8)
fig.savefig(FIG / 'relative_binning_nodes.pdf', bbox_inches='tight')
fig.savefig(FIG / 'relative_binning_nodes.png', dpi=220, bbox_inches='tight')
plt.close(fig)

with (ROOT / 'data' / 'channel_benchmark.csv').open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['method', 'nodes', 'max_normalized_error'])
    for nval, eval_ in zip(nc, ec):
        writer.writerow(['separated_ratios_exact_delays', int(nval), float(eval_)])
    for nval, eval_ in zip(nd, ed):
        writer.writerow(['direct_total', int(nval), float(eval_)])

(ROOT / 'data' / 'channel_benchmark_summary.txt').write_text(
    'fixed topology: both points outside the same generic fold\n'
    'fiducial eta_s=-0.040\n'
    'candidate eta_s=-0.010\n'
    f'frequency range={w[0]}..{w[-1]} with {len(w)} validation points\n'
    f'separated-ratio nodes at 1e-3={n_channel_target}\n'
    f'direct nodes at 1e-3={n_direct_target}\n'
    f'max exact reconstruction error={max(fold_fid.reconstruction_error, fold_cand.reconstruction_error):.6e}\n'
)
print('channel nodes at target', n_channel_target)
print('direct nodes at target', n_direct_target)
print('projected reconstruction error', part.reconstruction_error)
