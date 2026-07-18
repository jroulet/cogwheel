---
spec_version: 0.7.0
last_updated: 2026-07-18
---

# cogwheel — Project Specification

## Mission

cogwheel (PyPI/conda package `cogwheel-pe`, import name `cogwheel`) is a scientific
Python library for **Bayesian parameter estimation of gravitational-wave sources**
from compact binary coalescences (black-hole / neutron-star mergers). Given
conditioned detector strain data, it infers a posterior over source parameters
(component masses, spins, tidal deformabilities, sky location, distance,
orientation, coalescence time/phase).

Three signature contributions:
1. **Sampled vs standard coordinates** — a custom sampling coordinate system that
   separates reparameterized "sampled" parameters from physical "standard"
   parameters to reduce correlations.
2. **Folding** — an algorithm that reduces posterior multimodality by sampling
   folded dimensions over half their range and summing reflected images.
3. **Relative binning (heterodyning) + marginalization** — a fast likelihood that
   interpolates the ratio of a trial waveform to a reference over coarse frequency
   bins (generalized to higher modes), plus analytic/semi-analytic marginalization
   over distance and over all extrinsic parameters (the "coherent score").

## Architecture

cogwheel is a layered scientific package. `cogwheel/__init__.py` sets version
metadata only; functionality lives in submodules.

### Data flow

```
EventData (strain + ASD)  ->  WaveformGenerator  ->  Likelihood
                                                        |
                  Prior (sampled<->standard) ----------> Posterior (folding) -> Sampler -> samples.feather -> postprocessing
```

### Layers / modules

| Layer | Purpose | Key modules |
|-------|---------|-------------|
| Data acquisition & conditioning | Download public GW strain (GWOSC/GWpy), condition it, store ASDs and event metadata, build `EventData` (incl. Gaussian-noise and injected-signal events). | `cogwheel/data.py`, `cogwheel/data/events_metadata.csv`, `cogwheel/data/example_asds/` |
| Waveform generation | Frequency-domain waveforms via LALSimulation; project onto detectors; register approximants. | `cogwheel/waveform.py`, `cogwheel/waveform_models/__init__.py`, `cogwheel/waveform_models/xode.py` |
| Likelihood | CBC likelihood, fast relative-binning likelihood (with higher modes), marginalized variants. | `cogwheel/likelihood/likelihood.py`, `relative_binning.py`, `marginalized_distance.py`, `marginalized_distance_phase.py`, `marginalized_extrinsic.py`, `marginalized_extrinsic_qas.py`, `reference_waveform_finder.py` |
| Extrinsic marginalization (coherent score) | Marginalize over extrinsic params from matched-filter timeseries; sky dictionary + lookup tables; numba-accelerated. | `cogwheel/likelihood/marginalization/base.py`, `coherent_score_hm.py`, `coherent_score_qas.py`, `skydict.py`, `lookup_table.py` |
| Priors & coordinates | Abstract `Prior` base with sampled<->standard transforms; composable subpriors; concrete GW priors; PN-inspired QMC proposals. | `cogwheel/prior.py`, `cogwheel/prior_ratio.py`, `cogwheel/pn_coordinates.py`, `cogwheel/gw_prior/combined.py`, `mass.py`, `spin.py`, `extrinsic.py`, `tides.py`, `pn.py`, `twosquircle.py`, `miscellaneous.py` |
| Posterior & sampling | Combine prior + likelihood into `Posterior` (with folding); find reference solution; sample via dynesty/nautilus/zeus/PyMultiNest. | `cogwheel/posterior.py`, `cogwheel/sampling.py`, `cogwheel/postprocessing.py` |
| Utilities & physics | JSON (de)serialization mixin, caching, detector geometry/response, sky-loc angles, cosmology. | `cogwheel/utils.py`, `cogwheel/gw_utils.py`, `cogwheel/skyloc_angles.py`, `cogwheel/cosmology.py` |
| Plotting | Corner plots with GW-specific LaTeX labels. | `cogwheel/plotting.py`, `cogwheel/gw_plotting.py` |
| Validation | End-to-end injection-recovery pipeline (PP-plots / coverage). | `cogwheel/validation/generate_injections.py`, `inference.py`, `analyze.py`, `injection_prior.py`, `example/config.py` |
| Microlensing engine (Chang–Refsdal wave optics) | Complete wave-optics amplification engine for microlensed-PE: double-double arithmetic substrate (`_dd.py`), exact gauge/cluster-split channel algebra (`_gauge.py`), image geometry (`geometry.py` — quartic solver, delays, magnifications, stationary-phase kernels), dd-accumulated complex-1F1 kernel (`_hyp1f1.py`), contour-free operator `F_op` (`operator.py`), and topology-stable `ChangRefsdalChannels` (`channels.py`, the public entry point). Limitations: positive-parity macro images only (`1-kappa > \|gamma\|`; macro saddles out of scope); certified frequency ceiling `w <= 500`; double-double product ceiling `w*sqrt(s) <= 60`; geometric branch taken only above `w*delta_min >= 4.0` and `L > 48`; wave-branch contraction oracle-certified at 1e-10 only to `L ~ 25-30`, and above it the overflow-safe contraction raises a named `CancellationError` (never a silent `nan` or finite-but-wrong number) so the band `L in [~30, 48]` is certified-or-refused rather than silently wrong — the accuracy extension remains open (FINDINGS F005, NARROWED). Batched fast path (Build 3c): `operator.F_op_grid(w_array, ...)` evaluates the whole wave-branch node grid in one call via the per-order weight-vector contraction — the w-independent monomial/table weights are scatter-added once per evaluation and each node reduces to a length-`dim` dot product, replacing the per-node 85x85 bilinear form at byte-unchanged refusal thresholds; scalar `F_op` delegates to it (single contraction path, single certification), and `channels._exact_total` calls it once per wave-branch node subset. Certified by `cogwheel/tests/test_lensing_batched_operator.py`: 70-dps-mpmath-oracle accuracy at 1e-10 across the F005 band, single-vs-batch refusal-decision identity with zero flips, and F010-style py_func-chain self-falsification. | `cogwheel/lensing/chang_refsdal/channels.py` (public), `operator.py`, `_hyp1f1.py`, `geometry.py`, `_gauge.py`, `_dd.py` |
| Microlensed waveform & likelihood (Chang–Refsdal PE) | Apply the engine's amplification to CBC waveforms and evaluate a fast relative-binning likelihood for microlensed events. `LensedWaveformGenerator` composes an ordinary `WaveformGenerator` and multiplies every harmonic mode by the shared factor `F(w(f))`, with `w = 8*pi*G*M_L*(1+z_L)*f/c^3` dimensionless and linear in `f`; it exposes the per-image `(tau_a, K_a)` decomposition the likelihood consumes, not only the collapsed total. `LensedRelativeBinningLikelihood` (subclass of `BaseLinearFree`) heterodynes against an *unlensed* reference `par_dic_0` and reconstructs the lensed `(d\|h)`/`(h\|h)` from delay-continuous frequency-moment summaries contracted mode-then-image (additive `M^2 + n_img^2` cost, no FFTs on the hot path); image-delay phases stay analytic, smooth kernels are interpolated, and a lens-aware bin guard raises `LensedBinningError` if `pi*Delta_f_bin*delta_t_max` breaches tolerance. Positive-parity macro images only: `1-kappa > \|gamma\|` is enforced by raising `geometry.LensDomainError` at the API boundary (constructor and every strain/likelihood path), never a warning or `nan`; the wave branch is certified-or-named-refusal everywhere (WP1, F005 NARROWED). Fast path (Builds 3-3f, SACR-C): the channel construction is the switched-analytic + single-envelope decomposition — persistent resolved images carried by the ANALYTIC saddle kernel `geometry.image_kernel` under their own carriers with smootherstep weights `S_a(w) = smootherstep(w*\|tau_a - tau_c\|, 0.5, 4)` keyed on the CRITICALITY separation (`tau_c` = the parked critical-carrier delay from `nearest_caustic_point`; supersedes the F008 full-cluster switch keying — see the F008 addendum), plus ONE beat-free interpolated transition envelope `E(w) = e^{-i w tau_c}(F - sum_a S_a H_a e^{i w tau_a})` whose demodulated phase is bounded at 4 rad by construction. The envelope is engine-evaluated (`F_op_grid`) on a leave-one-out-adaptive coarse grid (`_LOO_SEED_NODES = 8`, stop `4e-3`, ceiling `_LOO_MAX_NODES = 48`, node count config-independent) and candidates are rebuilt by closed-form dense reconstruction (`reconstruct_from_envelope`); the dd/1F1 ladder and operator contraction are numba-njit (`fastmath=False`, refusal logic and thresholds untouched in Python), and the frequency-independent nearest-caustic search is njit-accelerated (value-preserving to rel `1e-10`, branch-invariant). Ratio layer (Build 3g): per-proposal cost is cut by heterodyning the candidate envelope against a memoized FIDUCIAL envelope — a pure function of the candidate (all five lens-geometry params snapped to a fixed lattice, `m_lens`/`z_lens` shared exactly so the `w` grid never regrids) — interpolating only the ultra-smooth ratio `rho = e^{i w dtau_c} E_cand/E_fid` (~8 LOO nodes, config-independent) with the critical-delay difference pulled out analytically; image-count and envelope-health guards (and any fiducial-side refusal) fall back to the certified direct path, while candidate-side refusals propagate unswallowed on ratio, direct, and brute paths alike. Certified by `cogwheel/tests/test_lensing_ratio_layer.py` (lattice-point identity, perturbed ratio-vs-direct, ratio-vs-brute at inherited tolerances, bit-identical cache determinism, refusal symmetry, deep-band macro limit; measured warm single-thread lnlike ~9.8 ms, ~143x over brute force). Certified by `cogwheel/tests/test_lensing_fast_path.py`: numba-vs-mpmath preservation, a null-safe production-grid interpolation gate (`max\|dF\|/max\|F\| < 1e-3`, worst regime two-image `4.2e-4`), RB-vs-brute agreement on every lens regime, and single-thread timing guards (warm lnlike ~0.3 s/eval, ~50x over brute force; the few-ms target is deferred to a 2D surrogate-table decision). | `cogwheel/lensing/waveform.py` (`LensedWaveformGenerator`), `cogwheel/lensing/likelihood.py` (`LensedRelativeBinningLikelihood`) |
| Microlensed sampling layer (priors, folding, posterior) | Sampled lens coordinates per the locked reduced parametrization: kappa/beta/z_lens ELIMINATED (`FixedLensGeometryPrior` — kappa is exactly mass-sheet degenerate and never sampled; z_lens folds into the sampled REDSHIFTED lens mass since only `w ∝ M_L(1+z_L) f` enters), `ln m_lens` log-uniform with certified-domain provenance, reduced shear `gamma` uniform on [0, 0.45], and the source position as a unit box `(u1, u2)` in the shear frame scaled by a mass-conditioned factor keeping `w*sqrt(s) <= 58` by construction; astroid quadrant folding via `folded_reflected_params = ['u1','u2']`; NO phase-fold (the constant-lens-phase ~ orbital-phase degeneracy is 22-only and must not be assumed for XPHM). `LensedIASPrior` (registered, `default_likelihood_class = LensedRelativeBinningLikelihood`) composes these with the IAS CBC subpriors; `LensedPosterior` maps `geometry.LensDomainError`/`operator.CancellationError` to `lnL = -inf` at the POSTERIOR BOUNDARY ONLY (the engine/likelihood named-refusal contract is untouched); the in-memory fiducial cache is dropped on pickle so sampler workers rebuild deterministically. Certified by `cogwheel/tests/test_lensing_prior.py` (round-trips 1e-12, Jacobians, 1e4-draw domain safety, reflection/fold consistency, mass-sheet lnL invariance, refusal-net mutation check, XPHM no-phase-fold). Known sampler-efficiency note: ~41% of blind prior draws are finite (the gamma box overlaps the operator cancellation band near 0.45+curvature); an efficiency-motivated gamma re-bound is an open owner decision, not a correctness issue. | `cogwheel/lensing/prior.py` (`LensedIASPrior`), `cogwheel/lensing/posterior.py` (`LensedPosterior`) |

### Key abstractions

- **`EventData`** (`data.py`) — strain + ASD for an event.
- **`WaveformGenerator`** (`waveform.py`) — wraps LALSimulation approximants
  (IMRPhenomXPHM / XAS / XODE). Note: uses IMRPhenomXP, *not* Pv2, due to
  phase-convention differences (LIGO-T1500602).
- **`Likelihood`** classes (`likelihood/`) — `CBCLikelihood`,
  `RelativeBinningLikelihood`, `Marginalized*`.
- **`Prior`** classes (`prior.py`, `gw_prior/`) — composable subpriors with
  sampled<->standard transforms, registered in a `prior_registry`.
- **`Posterior`** (`posterior.py`) — pairs a prior + likelihood, supports folding.
- **`Sampler`** subclasses (`sampling.py`) — wrap dynesty/nautilus/zeus/PyMultiNest;
  write `samples.feather` to run directories.
- Most stateful objects subclass `utils.JSONMixin` for JSON (de)serialization.

### External interfaces & dependencies

numpy, scipy, pandas (>=2.0), numba, lalsuite (lal, lalsimulation), gwpy, gwosc,
astropy, matplotlib, pyarrow; samplers dynesty, nautilus-sampler, zeus-mcmc,
pymultinest (optional). Docs: Sphinx + numpydoc + furo (`docs/source/`, Read the
Docs). Packaging: setuptools + setuptools_scm, GPL-3.0-or-later, Python >=3.9.

### Conventions

- Units: frequencies Hz, times GPS seconds, masses solar masses, distances Mpc,
  angles radians.
- Tests live in `cogwheel/tests/` (stdlib `unittest`), not a top-level `tests/`.
- `largedata/` (top-level) holds large run outputs, injection HDF5, and detector
  `.gwf` frame files used by tutorials — not part of the installed package.
- Numerically hot paths (relative binning, coherent-score marginalization) use
  numba and lookup tables and must remain numerically accurate.

## Constraints

- Numerical accuracy is paramount: relative-binning and marginalized likelihoods
  must agree with exact/brute-force references within tolerance.
- Phase/spin conventions must be respected across waveform and coordinate code.
- numba-accelerated code must stay numba-compatible.
