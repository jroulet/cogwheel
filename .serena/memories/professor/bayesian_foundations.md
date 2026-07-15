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
validation/ injection-recovery module implements.

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
(i) bespoke SAMPLED-vs-STANDARD coordinates to reduce correlations/multimodality
(see `mem:professor/priors_and_coordinates`);
(ii) FOLDING to collapse degenerate/symmetric modes and shrink the sampled volume;
(iii) RELATIVE BINNING for order-of-magnitude speedup (must agree with exact L within
tolerance); (iv) COHERENT-SCORE marginalization over extrinsic parameters.

## The authors' own frame (2402.11439, Roulet & Venumadhav review)
The canonical statement of cogwheel's design philosophy:
- **What is actually measured**: the data constrain OBSERVABLE COMBINATIONS, not the
  physical parametrization. Inspiral phase measures leading PN coefficients — first
  Mchirp^(-5/3) (exquisite at low mass), then corrections mixing q and aligned spins
  with similar slowly-varying powers of v across the band — hence the q–chi_eff
  degeneracy (the 1.5PN coefficient itself mixes both). High-mass events instead
  constrain a total-mass/aligned-spin combination through the merger frequency.
  Even for favorable sources only ~4-5 combinations of the 8 intrinsic parameters
  are meaningfully constrained; extrinsics contribute <=3 numbers per detector
  (amplitude, phase, arrival time), fewer in practice (near-coaligned LIGOs).
- **In-plane spins** appear only through the amplitude/phase of the subleading
  precession harmonic — precession is a power series in tan(beta/2), usually well
  approximated by the first two precession harmonics; natural frame z = J_hat,
  x toward observer. **Distance–inclination** degeneracy broken by higher modes
  (relative amplitude depends on q and iota; GW190412 the showcase).
- **Fisher-based principal components work only for the top ~2 combinations** —
  beyond that the prior dominates and Fisher fails. (Caution for lens-parameter
  coordinate design: use Fisher for the leading combos only.)
- **Discrete degeneracies**: phi -> phi+pi (quadrupole); simultaneous (phi, psi)
  pi/2 shifts; cos-iota flips and sky reflections for the near-aligned network —
  handled by folding (exact, general-sampler-compatible, postprocessing unfold).
- **Pipeline economics**: cost = (per-likelihood cost) x (number of evaluations);
  attack both. Relative binning by co-precessing m (same-m modes share phase
  evolution Phi_m ~ m Phi_orb(f/m)); ~few hundred coarse frequencies; ~1 ms per
  IMRPhenomXPHM call; PE needs ~1e5-1e7 waveform evaluations per event.
  Alternatives ranked: ROQ (no reference waveform, lower efficiency), multibanding
  (cost grows with duration), likelihood interpolation (RIFT).
- **Marginalization wisdom**: phase analytically (I_0, quadrupolar; undo by sampling
  2phi from von Mises); distance via 2D interpolation table in <d|h1>, <h1|h1>
  (tabulate log-ratio to an analytic approximation for dynamic range); time by
  quadrature/pruned FFT; draw marginalized params back from conditionals.
  Importance sampling: proposal must have HEAVIER tails than the target; monitor
  n_eff = (sum w)^2/sum w^2; reweight cheap-model posteriors (22-only -> HM,
  quasicircular -> eccentric).
- Odds ratios are NOT goodness-of-fit tests.

## Pitfalls
Prior choice carries a Jacobian (mass parametrization); distance/inclination and tidal
parameters are strongly covariant; prior-boundary pileup (prefer HPDI near edges);
label switching / mode degeneracy (folding targets this); MCMC autocorrelation and burn-in;
under-coverage is the primary failure mode P-P tests surface.

Sources: 1809.02293, 1409.7215, 1811.02042, 2402.11439.
