## 2026-07-17 — Build 3/3b (lensing fast path) doc-sync audit

Scope: audit docs/source/{overview,api,installation,crash_course}.rst against the
Build 3/3b landing (numba fast path, `n_kernel_nodes` kwarg on
LensedRelativeBinningLikelihood, FINDINGS F010). SPEC.md/DATA_CONTRACTS/
changelog fragments were already updated by the build itself this cycle.

Result: no-op. Nothing needed changing.
- `scripts/sync_derived_docs.py` (5 checks) — all OK, zero drift.
- `docs/source/overview.rst` lensing paragraph (lines 85-91) describes
  LensedWaveformGenerator/LensedRelativeBinningLikelihood at the
  architecture level (heterodyne + delay-continuous frequency-moment
  contraction) and never asserted a per-eval cost/timing number, so the
  fast-path landing (numba engine, coarse kernel-node grid, ~0.3s/eval)
  didn't invalidate anything written there. Nothing to sync.
- `docs/source/api.rst` uses `:recursive:` autosummary over the bare
  `cogwheel` package name (confirmed again, per prior note) — new public
  kwarg `n_kernel_nodes` and `_DEFAULT_KERNEL_NODES` need no manual entry.
- `docs/source/installation.rst` never enumerates dependencies by name
  (delegates to `environment.yaml` / `pip install -e .`); numba is already
  in `pyproject.toml` install_requires — nothing to add.
- `docs/source/crash_course.rst` has zero lensing/ChangRefsdal/
  LensedRelativeBinning references — build's API surface doesn't touch it.
- Cross-ref check: SPEC.md's fast-path paragraph cites F005 and F008;
  both exist and are consistent in FINDINGS.md (F005 NARROWED, F008
  supersedes F006). Module attribution in SPEC.md
  (`cogwheel/lensing/likelihood.py`: `_DEFAULT_KERNEL_NODES`,
  `n_kernel_nodes` ctor kwarg; `cogwheel/lensing/waveform.py`:
  `LensedWaveformGenerator`) verified exact against code via grep.

Pattern for future cycles: when a build's SPEC.md paragraph adds
low-level implementation detail (numba, grid sizes, timing numbers) but
the user-facing overview.rst paragraph is already pitched at the
architecture/API level, there is usually nothing to propagate downward —
check whether overview.rst makes any per-eval/perf claim before assuming
it needs a sync edit. Don't manufacture a performance blurb just because
the spec grew one.

No fragile cross-references newly discovered this cycle beyond the
already-known SPEC_CHANGELOG.md alphabetical-bump quirk (see long-term
memory). No files touched; no commit made (nothing to commit).
