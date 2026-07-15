# References Index

Curated gravitational-wave parameter-estimation literature for the Professor.
PDFs are stored as `<arxiv_id>.pdf`. The Professor marks a paper read by citing
its arxiv ID in a `professor/<topic>` memory (the
`.claude/hooks/professor-auto-mark-read.sh` hook then writes a marker in
`.serena/memories/professor/read.d/`); `python scripts/sync_professor_papers.py`
reconciles read status. Seeded from the PE-relevant subset of the
gw_detection_ias library; add more with the Professor's paper-reading workflow.

| ArXiv ID | Title | PDF | cogwheel modules | Notes |
|---|---|---|---|---|
| 1806.08792 | Relative Binning and Fast Likelihood Evaluation for GW Parameter Estimation | [1806.08792.pdf](./1806.08792.pdf) | `cogwheel/likelihood/relative_binning.py` | Core: the relative-binning (heterodyne) likelihood cogwheel implements. |
| 2404.02435 | Fast marginalization algorithm for optimizing GW detection, parameter estimation, and sky localization | [2404.02435.pdf](./2404.02435.pdf) | `cogwheel/likelihood/marginalization/, relative_binning.py` | Mode-by-mode relative binning + the coherent-score extrinsic marginalization. |
| 2210.16278 | Factorized Parameter Estimation for Real-Time Gravitational Wave Inference | [2210.16278.pdf](./2210.16278.pdf) | `cogwheel/likelihood/marginalized_extrinsic*.py, marginalization/` | Factorized / extrinsic-marginalized PE. |
| 1908.05644 | Detecting gravitational waves in data with non-stationary and non-Gaussian noise | [1908.05644.pdf](./1908.05644.pdf) | `cogwheel/data.py, cogwheel/likelihood/likelihood.py` | Noise model / data conditioning underlying the likelihood. |
| 1904.01683 | Template bank for CBC searches: a general geometric placement algorithm | [1904.01683.pdf](./1904.01683.pdf) | `cogwheel/waveform.py, cogwheel/gw_prior/` | Waveform/coordinate metric background. |
| 1809.02293 | Thrane & Talbot: An introduction to Bayesian inference in GW astronomy | [1809.02293.pdf](./1809.02293.pdf) | `(foundations)` | Canonical intro — likelihood, priors, sampling, PP-tests. Grounds professor/likelihood_and_inference + validation. |
| 1409.7215 | Veitch et al.: LALInference | [1409.7215.pdf](./1409.7215.pdf) | `(foundations)` | The standard GW PE framework cogwheel is an alternative to; MCMC/nested sampling PE. |
| 1811.02042 | Ashton et al.: Bilby | [1811.02042.pdf](./1811.02042.pdf) | `(foundations)` | Modern GW PE library; priors, likelihood, sampler interfaces. |
| 1904.02180 | Speagle: dynesty (dynamic nested sampling) | [1904.02180.pdf](./1904.02180.pdf) | `cogwheel/sampling.py` | Sampler cogwheel wraps; nested-sampling evidence + posteriors. |
| 2306.16923 | Lange: nautilus (importance nested sampling + deep learning) | [2306.16923.pdf](./2306.16923.pdf) | `cogwheel/sampling.py` | Sampler cogwheel wraps. |
| 2105.03468 | Karamanis et al.: zeus (ensemble slice sampling) | [2105.03468.pdf](./2105.03468.pdf) | `cogwheel/sampling.py` | Sampler cogwheel wraps; MCMC convergence. |
| 0809.3437 | Feroz et al.: MultiNest | [0809.3437.pdf](./0809.3437.pdf) | `cogwheel/sampling.py` | Sampler cogwheel wraps (PyMultiNest). |
| 2004.06503 | Pratten et al.: IMRPhenomXPHM | [2004.06503.pdf](./2004.06503.pdf) | `cogwheel/waveform.py` | Default precessing higher-mode waveform model; professor/waveform_conventions. |
