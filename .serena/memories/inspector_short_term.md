# Inspector Short-Term Observations

## 2026-07-20 — Build 8a review (amplification surrogate)

Scope: uncommitted working-tree changes for Build 8a (additive
`LensAmplificationSurrogate` speed layer). Files: new
`cogwheel/lensing/surrogate.py` + `test_lensing_surrogate.py`; edits to
`channels.py` (WP2 `ChangRefsdalGeometryPartition` + `geometry_partition`),
`likelihood.py` (WP3 surrogate dispatch), `marginalized_likelihood.py`
(kwarg threading). Plus agent-state/memory/handoff docs (non-code).

Verified GOOD:
- `geometry_partition` reproduces `evaluate`'s geometry lines verbatim,
  stopping before `_min_delay_separation`/`_exact_total`. Continues labels
  identically; callers use a fresh `ChangRefsdalChannels` per call so
  marker state is deterministic.
- Surrogate reconstruction path is structurally identical to the direct
  path: `reconstruct_from_envelope(dense_w, E, geom.delays,
  geom.saddle_kernels, geom.switch, geom.critical_delay)` mirrors
  `_kernels_from_dense_envelope`; saddle/switch evaluated at dense_w inside
  `geometry_partition`. Same `_reduce_dense_kernels`/`_image_delays`.
  4th return element (partition/geom) is discarded by both production
  callers, so returning a GeometryPartition is fine.
- `_amplification_coefficients` surrogate intercept only short-circuits on
  a non-None served result; None default = byte-identical (crown byte-id
  test vs HEAD side-by-side PASSES across 4 configs incl. saddle/4-image).
- Named refusals propagate unswallowed (geometry LensDomainError not
  caught in surrogate path). `dense_w<=0` returns None → exact path raises
  LensedBinningError unchanged.
- surrogate.py imports only chang_refsdal + numpy/scipy (no likelihood →
  no circular import, WP1 constraint honored). No absolute paths/secrets.
- get_init_dict overrides match JSONMixin(**kwargs); None-drop keeps JSON
  byte-identical; fitted surrogate raises NotImplementedError (deferred).
- pickle drops derived caches (_fid_cache, _surrogate_region_nimg);
  surrogate __getstate__ stores flat ndarrays only, rebuilds interpolants.
- Full suite: 23 passed, 1 skipped (timing smoke), 301s.

FINDINGS (both NON-blocking, latent/doc):
- INS-8a-001 (design): surrogate serve gate does NOT check kappa.
  `in_domain(gamma,y1,y2,beta)` has no kappa axis; surrogate trained
  kappa=0 only. A candidate with kappa!=0 but (gamma,y1,y2,beta) in-box
  would be served a kappa=0 envelope while `geometry_partition` builds
  geometry at the candidate's kappa → silently-wrong F. NON-TRIGGERING in
  production (mass-sheet eliminates kappa; sampled space kappa=0 always)
  but violates the conservative-serve contract. Fix: return None when
  lens['kappa'] != 0.
- INS-8a-002 (trivial → Librarian): SPEC.md not updated though the plan
  expected it; new surrogate layer (LensAmplificationSurrogate,
  geometry_partition, amplification_surrogate kwarg) undocumented in spec.

New pattern: new .npz artifact (surrogate.save/load) is offline-only, no
shipped file yet → no DATA_CONTRACTS entry required this build; revisit if
a surrogate file is shipped/consumed by pipeline scripts.
