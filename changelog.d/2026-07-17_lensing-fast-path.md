---
date: 2026-07-17
---
### Microlensed relative-binning likelihood: ~50x faster at unchanged accuracy

`LensedRelativeBinningLikelihood.lnlike` drops from ~15 s/eval to
~0.3 s/eval (warm, single-thread) with all accuracy gates at their
original tolerances. Three levers: the double-double 1F1 derivative
ladder and the operator-series contraction are numba-compiled
(`fastmath=False`; every `CancellationError`/`LensDomainError` refusal
path and threshold is byte-identical); the frequency-independent
nearest-caustic search is njit-accelerated (distance preserved to
relative 1e-10, wave/geometric branch selection invariant); and the
smooth channel kernels `K_a(w)` are engine-evaluated only on a coarse
deterministic `w` grid — a log-spaced base of 100 nodes
(`n_kernel_nodes` constructor knob) unioned with full-cluster
smootherstep/branch transition frequencies — and cubic-splined to the
bin sub-samples. Node-count provenance is a measured convergence sweep
(worst regime two-image: null-safe reconstruction error 4.2e-4 vs the
1e-3 ceiling). New test module `cogwheel/tests/test_lensing_fast_path.py`
certifies the fast path end-to-end, including RB-vs-brute agreement on
every lens regime (the kappa-config leak found in review is fixed by
full-cluster transition-node placement).
