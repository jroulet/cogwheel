# Priors and coordinates: the cogwheel sampled-coordinate system

Source: 2207.03508 (Roulet, Olsen, Mushkin, Islam, Venumadhav, Zackay, Zaldarriaga —
"Removing degeneracy and multimodality in gravitational wave source parameters").
THE foundational paper for cogwheel/prior.py and cogwheel/gw_prior/.

## Design principle
A quasicircular BBH has 15 parameters but data constrain only ~10 combinations.
Sample in coordinates that SEPARATELY control the observables the data measure —
amplitude a_k, phase phi_k, arrival time t_k at each detector — so well- and
poorly-measured directions decouple. Expected widths from SNR: da_k ~ a_k/rho_k,
dphi_k ~ 1/rho_k, dt_k ~ 1/(2 pi sigma_f rho_k). Transformations are chains of
triangular maps (x1,x2) -> (a(x2)x1 + b(x2), x2) with |J| = |a(x2)|, so Jacobians
stay tractable and compose multiplicatively.

## The coordinates and what each fixes
- **Mchirp, ln q**: chirp mass controls leading PN amplitude/phase; the mass backbone.
- **chi_eff, C_diff**: aligned spins enter phase mainly via chi_eff; the poorly
  measured orthogonal direction is sampled as the CUMULATIVE of the conditional
  prior (CDF coordinate) — uniform on (0,1), no Jacobian. Breaks q–spin degeneracy
  into well/poorly measured pieces.
- **Chirp distance d_hat = d_L/(M^{5/6}|R_k0|)**: inverse amplitude at the reference
  (loudest) detector; kills d_L–iota–sky correlations; mass-independent range.
  (Alternatively marginalize distance semianalytically.)
- **t_k0**: arrival time at the REFERENCE DETECTOR, not geocenter (geocenter t_c
  correlates with sky through the ~40 ms Earth-crossing time; t_k0 measured to <~1 ms).
- **cos theta_net, phi_hat_net**: sky in a polar system with z-axis through the two
  loudest detectors — theta_net encodes the well-measured time-delay ring.
- **phi_hat_ref**: (phi_k0 − phi_k0^ML)/2, deviation of arrival phase at reference
  detector from its ML value; width ~1/(2 rho_k0).
- **Precession**: theta_JN, phi_hat_JL, phi_12 + CDF coords for in-plane spin
  magnitudes. Azimuths defined about J relative to N_hat (Farr convention), NOT the
  orbital separation (separation-based azimuths couple spuriously to phi_ref).
  f_ref must be IN-BAND: f_ref = SNR^2-weighted mean frequency. phi_hat_JL is a
  remarkably well measured precession observable.

## Folding
Four approximate discrete symmetries give up to 2^4 posterior modes:
phi_hat_ref -> +pi (exact for even m); psi -> +pi/2 with compensating phi shift
(exact for |m|=2); phi_hat_net -> −phi_hat_net and cos theta_JN -> −cos theta_JN
(near-coaligned HL geometry). Folding samples P_folded = sum over the 2^N images,
then unfolds each sample probabilistically. It is EXACT even when the symmetry is
broken (Virgo, HM, precession) — only efficiency degrades. All 2^N evaluations
reuse the waveform (only extrinsic responses change) ~ one likelihood call.
Unfolding probabilities near 1/2^N verify the symmetries hold in full dimension.

## Recipe for NEW parameters (e.g. lensing)
1. Identify the leading dependence of detector-frame amplitude/phase/time on the
   new parameter; form the combination that controls ONE observable (expected
   width ~1/rho).
2. Put poorly measured orthogonal directions in conditional-CDF coordinates
   (uniform prior, unit Jacobian).
3. Keep every map triangular so Jacobians compose.
4. Hunt for discrete near-symmetries; add them to the folding set as
   one-parameter reflections. Place any coordinate branch cuts ON fold boundaries
   so they vanish in the folded posterior.

## Pitfalls
Apply P(y) = P(x)|dx/dy| consistently — prefer specifying the prior directly in
sampled coordinates (that's why chi_eff gets a uniform prior rather than a
transformed isotropic one). Watch branch cuts (phi_hat_ref is discontinuous at the
cut of arg R_k0 and bimodal before folding). Folding presumes the symmetry maps are
correct in the FULL space — verify with unfolding-probability histograms.

Source: 2207.03508.
