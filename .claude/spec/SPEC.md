---
spec_version: 0.3.0
last_updated: 2026-07-17
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
| Microlensing engine (Chang–Refsdal wave optics) | Complete wave-optics amplification engine for microlensed-PE: double-double arithmetic substrate (`_dd.py`), exact gauge/cluster-split channel algebra (`_gauge.py`), image geometry (`geometry.py` — quartic solver, delays, magnifications, stationary-phase kernels), dd-accumulated complex-1F1 kernel (`_hyp1f1.py`), contour-free operator `F_op` (`operator.py`), and topology-stable `ChangRefsdalChannels` (`channels.py`, the public entry point). Limitations: positive-parity macro images only (`1-kappa > \|gamma\|`; macro saddles out of scope); certified frequency ceiling `w <= 500`; double-double product ceiling `w*sqrt(s) <= 60`; geometric branch taken only above `w*delta_min >= 4.0` and `L > 48`; wave-branch contraction oracle-certified at 1e-10 only to `L ~ 25-30`, and above it the overflow-safe contraction raises a named `CancellationError` (never a silent `nan` or finite-but-wrong number) so the band `L in [~30, 48]` is certified-or-refused rather than silently wrong — the accuracy extension remains open (FINDINGS F005, NARROWED). | `cogwheel/lensing/chang_refsdal/channels.py` (public), `operator.py`, `_hyp1f1.py`, `geometry.py`, `_gauge.py`, `_dd.py` |
| Microlensed waveform & likelihood (Chang–Refsdal PE) | Apply the engine's amplification to CBC waveforms and evaluate a fast relative-binning likelihood for microlensed events. `LensedWaveformGenerator` composes an ordinary `WaveformGenerator` and multiplies every harmonic mode by the shared factor `F(w(f))`, with `w = 8*pi*G*M_L*(1+z_L)*f/c^3` dimensionless and linear in `f`; it exposes the per-image `(tau_a, K_a)` decomposition the likelihood consumes, not only the collapsed total. `LensedRelativeBinningLikelihood` (subclass of `BaseLinearFree`) heterodynes against an *unlensed* reference `par_dic_0` and reconstructs the lensed `(d\|h)`/`(h\|h)` from delay-continuous frequency-moment summaries contracted mode-then-image (additive `M^2 + n_img^2` cost, no FFTs on the hot path); image-delay phases stay analytic, smooth kernels are interpolated, and a lens-aware bin guard raises `LensedBinningError` if `pi*Delta_f_bin*delta_t_max` breaches tolerance. Positive-parity macro images only: `1-kappa > \|gamma\|` is enforced by raising `geometry.LensDomainError` at the API boundary (constructor and every strain/likelihood path), never a warning or `nan`; the wave branch is certified-or-named-refusal everywhere (WP1, F005 NARROWED). Fast path (Builds 3/3b): the smooth channel kernels `K_a(w)` are engine-evaluated only on a deterministic coarse `w` grid — a log-spaced base of `_DEFAULT_KERNEL_NODES = 100` nodes unioned with FULL-CLUSTER smootherstep/branch transition frequencies (real + parked virtual labels, the F008 rule) — then cubic-splined (not-a-knot, real/imag separately) to the bin sub-samples; the dd/1F1 ladder and operator contraction are numba-njit (`fastmath=False`, refusal logic and thresholds untouched in Python), and the frequency-independent nearest-caustic search is njit-accelerated (value-preserving to rel `1e-10`, branch-invariant). Certified by `cogwheel/tests/test_lensing_fast_path.py`: numba-vs-mpmath preservation, a null-safe production-grid interpolation gate (`max\|dF\|/max\|F\| < 1e-3`, worst regime two-image `4.2e-4`), RB-vs-brute agreement on every lens regime, and single-thread timing guards (warm lnlike ~0.3 s/eval, ~50x over brute force; the few-ms target is deferred to a 2D surrogate-table decision). | `cogwheel/lensing/waveform.py` (`LensedWaveformGenerator`), `cogwheel/lensing/likelihood.py` (`LensedRelativeBinningLikelihood`) |

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
