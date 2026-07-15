# Samplers and convergence

cogwheel wraps four samplers in cogwheel/sampling.py. Two families: nested sampling
(dynesty, nautilus, PyMultiNest) and ensemble MCMC (zeus).

## dynesty — dynamic nested sampling (1904.02180)
Nested sampling refactors Z = integral L pi dTheta into a 1-D integral over prior
volume X. Maintains K live points drawn from the prior under a hard constraint
L >= L_min; each step retires the lowest-L point and replaces it above threshold, so
the live set climbs through nested iso-likelihood shells whose enclosed volume shrinks
~exp(-1/K) per step. Evidence AND posterior fall out: dead points get weights
p_i ~ L_i dX_i (equal-weight resampling available). Stop on evidence: remaining
delta ln Z < ~0.01. "Dynamic" varies K to allocate live points where posterior mass is
(default 80/20 posterior/evidence). Failure: too few live points miss a narrow peak and
terminate early (L_max underestimated).

## nautilus — importance nested sampling + deep learning (2306.16923)
INS defines a pseudo-importance density g from bounding regions and importance-weights
ALL evaluated points (w = L pi / g), not just live-set entrants -> Z = sum w. Bounds are
LEARNED: a neural-net regressor predicts a likelihood score to carve proposals tracing
the iso-likelihood surface; multi-ellipsoidal for multimodal. Phases: exploration
(stop f_live < 0.01) then optional sampling to grow N_eff (stop N_eff > 10000).
Convergence: N_eff = (sum w)^2 / sum w^2, delta log Z ~ N_eff^-1/2. Often >10x more
efficient than dynesty/emcee. Failure specific to nautilus: importance-weight variance
— a poor g inflates weight variance, lowers N_eff, can slightly bias posterior
(resampling helps).

## zeus — ensemble slice sampling (2105.03468)
MCMC (NOT nested): samples the posterior directly, does NOT compute evidence. Replaces
Metropolis with 1-D slice sampling along directions from an ensemble of walkers. Slice
sampling is rejection-free (walkers always move), auto-tunes one length scale, adapts
locally -> handles strong correlations and multimodality far better than emcee/AIES.
Convergence: integrated autocorrelation time tau (chain length >> tau),
N_eff = N_steps/tau; discard burn-in. tau scales ~linearly with dimension (vs emcee's
exponential). Failure: long tau, unconverged/multimodal chains missing modes,
insufficient burn-in; needs >~2*D walkers.

## PyMultiNest / MultiNest (0809.3437)
Original multimodal NS engine: same machinery, but proposals use MULTI-ELLIPSOIDAL
decomposition (k-means/X-means EM bounding ellipsoids, enlargement factor) so separated
modes and curving degeneracies each get their own ellipsoid -> per-mode local evidence.
Stops on delta ln Z (~0.5). Failure: ellipsoidal bounds can under-cover the true
iso-likelihood surface (biased Z, under-sampled tails) or become inefficient for many
modes.

## Practical guidance
Need evidence / Bayes factors -> nested (nautilus for expensive likelihoods,
PyMultiNest for strong multimodality, dynesty a robust default). Cheap likelihood in
high-D, posterior-only -> zeus (but check convergence yourself). "Converged" means:
nested — delta ln Z below tolerance, ln Z stable, adequate live points; MCMC — chain
>> tau, adequate N_eff, burn-in removed. Universal ESS: N_eff = (sum w)^2/sum w^2
(weighted/nested), N_eff = N/tau (MCMC).

Sources: 1904.02180, 2306.16923, 2105.03468, 0809.3437.
