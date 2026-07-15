# Likelihood and Inference: Relative Binning, Non-Gaussian Noise, Factorized PE

The CBC likelihood in cogwheel is the stationary-Gaussian-noise Whittle form:
`log L = -1/2 sum_k 4 Re integral |d_k(f) - h_k(f;theta)|^2 / S_k(f) df` over detectors k.

## Relative binning / heterodyning (1806.08792)
A posterior sampler only visits waveforms close to the best-fit, so the ratio
r(f) = h(f)/h0(f) of a trial to a fiducial waveform is smooth in frequency even
though h(f) oscillates through thousands of cycles. Physically: PN phase is a sum
of a few power laws (chirp mass gamma=-5/3, mass ratio -1, spin -2/3, tidal 5/3,
time 1); a small parameter shift gives a slowly-varying differential phase. Within
a coarse bin r(f) ~ r0(b) + r1(b)(f - f_m(b)) (linear). The likelihood is
reconstructed from precomputed summary data A0, A1 (data.h0* moments) and B0, B1
(|h0|^2 moments) accumulated once at full resolution; per-waveform cost drops to
O(N_bins). ~60 bins suffice for GW170817 (T=2048 s, 4096 Hz) -> ~1e4x over full
matched filtering, ~10x over ROQ. cogwheel/likelihood/relative_binning.py implements
the A0/A1/B0/B1 scheme.
**Failure modes:** linear-in-bin approximation breaks for (a) too few in-band cycles
(short/high-mass BBH, ringdown-dominated), (b) too-coarse bins (phase error >1 rad —
raise polynomial order rather than add bins), (c) fiducial far from posterior bulk,
(d) higher modes need SEPARATE bin sets (modes are mutually oscillatory).

## Non-Gaussian / non-stationary noise (1908.05644)
Two effects break stationary-Gaussian S_n(f). (1) PSD DRIFT on few-to-tens-of-seconds
timescales -> a globally-estimated PSD mis-whitens locally; SNR loss is quadratic in
the fractional error. Fix: divide matched-filter overlaps by a locally-estimated
std sigma_z(t) tracked over ~1 s windows ("PSD drift correction"). (2) Loud GLITCHES:
remedy is INPAINTING (hole-filling) — solve for masked-window values so the
inverse-PSD-filtered data is zero there, making overlaps independent of template
values inside the hole. PE relevance: whitening is only locally valid; long signals
overlapping a glitch (as GW170817) need inpainting to avoid biased parameters.

## Factorized PE (2210.16278)
cogwheel's real-time framework: split the ~11-D CBC space into intrinsic
{m1,m2,chi1,chi2} and extrinsic {distance, RA, dec, psi, iota, phi_c, t_c}, then
semi-analytically MARGINALIZE over all seven extrinsic to get L_marg(d|intrinsic),
sampled with relative binning. Distance and phase integrate analytically; sky/time
via importance-sampled MC over detector time-delay dictionaries; inclination/
polarization numerically. Extrinsic restored in post-processing from conditional
posteriors. ~20x speedup, JS-divergence <0.1 nats vs full dynesty/PyMultiNest.
**Limitation:** the extrinsic-marginalization identities assume quadrupolar
aligned-spin ((l,m)=(2,+-2)) — precession or higher modes break the factorization
(see `mem:professor/marginalization` for the higher-mode generalization).

Sources: 1806.08792, 1908.05644, 2210.16278.
