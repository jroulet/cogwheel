# Extrinsic marginalization (the coherent score)

Source: 2404.02435 (Roulet, Mushkin, Wadekar, Venumadhav, Zackay, Zaldarriaga —
"Fast marginalization algorithm for GW detection, PE and sky localization"). This is
the reference behind cogwheel/likelihood/marginalization/.

## What is marginalized and why
Computes the Gaussian likelihood ratio marginalized over the EXTRINSIC parameters —
distance D, reference phase phi_ref, geocenter time t, sky location, polarization psi
(and inclination analytically for aligned-spin). Intrinsic parameters are left for an
outer sampler. Because the waveform's analytic extrinsic dependence is known, the
marginalized likelihood is obtainable from a SINGLE waveform evaluation per intrinsic
point — the "coherent score."

## Core decomposition (cogwheel modules)
Two inputs carry all intrinsic dependence: the matched-filter timeseries z_mpd(t)
(mode m, polarization p, detector d) and the mode-mode covariances c (time-independent
template overlaps). These map to coherent_score_hm.py (higher modes) and
coherent_score_qas.py (quadrupole/aligned-spin). Both use relative binning generalized
MODE-BY-MODE — each (l,m) harmonic gets its own summary data, so relative binning stays
valid though higher modes break the single-phase quadrupole assumption.

## Integration strategy (three tiers)
1. Distance: 2D lookup-table interpolation (lookup_table.py) of <d|h1> and <h1|h1> at
   unit distance (Singer-Price). Valid — neither higher modes nor precession alter the
   D-dependence.
2. Orbital phase: trapezoid quadrature on a grid (~128 points) — NOT analytic, because
   higher modes couple phase non-trivially.
3. Time/sky/polarization: adaptive multiple importance sampling. Proposal over
   discretized detector arrival times (factorizable, inverse-transform sampled), mapped
   to sky location via a precomputed SKY DICTIONARY (skydict.py) — the partition of the
   sky by inter-detector time delays, which also supplies the time-delay prior.
   Quasi-Monte Carlo (scrambled Halton) reduces variance.

## Numerical gotchas
- Effective sample size N_eff is the reliable error tracer (ln L error ~ 3.4 N_eff^-0.76);
  target N_eff >~ 100 for ~10% precision; iterate adaptive proposals up to j_max.
- Importance-sampling variance DIVERGES when the proposal has tighter support than the
  posterior — err toward HEAVIER-tailed proposals (Cauchy KDE, uniform psi). Low N_eff
  signals proposal misspecification, not just noise.
- Phase and polarization are largely degenerate -> marginalize only one (phase) at high res.
- Lookup-table accuracy must stay within tolerance vs brute force — a natural cogwheel
  numerical-accuracy test target.
- Marginalization is provably >= maximization for detection (Neyman-Pearson): correctly
  penalizes fine-tuned high-inclination configs that maximization over-rewards.
- Higher modes break the distance-inclination degeneracy -> better distance/sky loc.
Validated against full 15-D sampling and P-P plots; ~50 ms per marginalized-L call.

Source: 2404.02435. See also `mem:professor/likelihood_and_inference`.
