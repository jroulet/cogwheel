# Bayesian foundations of GW parameter estimation

## Framework (1809.02293, Thrane & Talbot)
Given strain d and parameters theta (~15 CBC parameters), the posterior is
p(theta|d) = L(d|theta) pi(theta) / Z. Z = integral L pi dtheta is the evidence
(normalization for PE; compared via Bayes factors for model selection).

**The GW likelihood.** Frequency-domain noise assumed stationary Gaussian with one-sided
PSD S_n(f) -> the Whittle likelihood, a complex Gaussian with no sqrt in the normalization.
In inner-product form log L = log Z_N + kappa^2 - 1/2 rho_opt^2, where
<a,b> = 4 df Re sum a* b / P_j, rho_opt^2 = <mu,mu> is optimal SNR^2, and kappa^2 = <d,mu>
is the matched-filter overlap — PE is the Bayesian generalization of matched filtering.
Product over frequency bins and detectors (independent noise). Analytic marginalization
over phase (Bessel I_0; valid only for dominant l=|m|=2), time (FFT over t_c grid), and
distance (lookup table) — the same tricks LALInference/Bilby expose as flags, and the
foundation cogwheel extends with the coherent score.

**Validation via P-P plots.** Draw many theta_true from the prior, simulate data, run PE,
record the credible level at which each true value falls. If well-calibrated, the fraction
recovered within the X% credible interval equals X for all X — the cumulative percentile
plot lies on the diagonal within the binomial band. One does NOT expect the posterior to
peak at theta_true, only correct interval coverage. This is exactly what cogwheel's
validation/ injection-recovery module implements (see `mem:professor/validation` when written).

## Community standards
LALInference (1409.7215): the LSC C+Python toolkit — LAL data/PSD/waveforms, coherent-network
Whittle likelihood (optional PSD marginalization -> Student-t), three samplers (parallel-
tempered MCMC, nested sampling, MultiNest/BAMBI). Samples in (M_chirp, q) to decorrelate
masses; system-frame spin angles; CBC-specific jump proposals.
Bilby (1811.02042): modern modular Python reimplementation — core/gw/hyper packages,
Prior/Likelihood/Sampler abstractions, default GravitationalWaveTransient Whittle likelihood,
standard priors (uniform masses, isotropic sky, comoving-volume distance, sin-iota), common
interface to emcee/dynesty/PyMultiNest/etc., optional phase/time/distance marginalization.

## Where cogwheel diverges
Both standards sample near-physical coordinates with generic samplers. cogwheel instead:
(i) bespoke SAMPLED-vs-STANDARD coordinates to reduce correlations/multimodality;
(ii) FOLDING to collapse degenerate/symmetric modes and shrink the sampled volume;
(iii) RELATIVE BINNING for order-of-magnitude speedup (must agree with exact L within
tolerance); (iv) COHERENT-SCORE marginalization over extrinsic parameters.

## Pitfalls
Prior choice carries a Jacobian (mass parametrization); distance/inclination and tidal
parameters are strongly covariant; prior-boundary pileup (prefer HPDI near edges);
label switching / mode degeneracy (folding targets this); MCMC autocorrelation and burn-in;
under-coverage is the primary failure mode P-P tests surface.

Sources: 1809.02293, 1409.7215, 1811.02042.
