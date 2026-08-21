# Inspector Short-Term — 2026-08-21 (low_w_chart_cusp_fallback RE-REVIEW, pass 3 FINAL)

Scope: Re-review of the Pearcey (cusp) fallback build after the Coder's fixes to
INS-2-001/002. Re-checked both previously-open findings plus a fresh pass over
the whole uncommitted surface (_pearcey_cusp.py, low_w_diffractive_chart.py,
likelihood.py, train_low_w_diffractive_chart.py, test file).

RESOLVED (verified in code + grep, not trusted):
- INS-2-001: `_CuspUniformGeometry` no longer carries dead fields. Current
  fields = matrix, images, nearest, c4, tau_c, delta_parallel, delta_perp,
  abs_c4, curvature — all 9 are read. `branch`/`vertex`/`phi_ssr` are computed
  locally inside `_cusp_uniform_geometry` and used there (vertex for
  soft/hard axes + hard_eigenvalue; phi_ssr for curvature; branch passed to
  _cusp_vertex) but never stored. grep `.branch`/`.vertex`/`.phi_ssr` = only
  unrelated `arc.branch` in surrogate_training.
- INS-2-002: scalar `cusp_uniform_reference` + `_cusp_uniform_form` wrapper
  are GONE (the Coder took the "drop the scalar accessor" branch). Only
  `cusp_uniform_reference_grid` remains, consumed by `_pearcey_cusp_reference`
  -> `fold_cusp_reference`. `_CuspUniformForm` is now a dataclass bundle, not
  a function. grep confirms zero scalar-accessor refs.

VERIFIED GREEN (fresh runs): test_lensing_low_w_diffractive_chart.py 47 passed
(56s, incl. all 6 new cusp methods collected); test_lensing_airy_fold.py
128 passed/7 skipped/2 xfailed (40s) — the cusp_amplification refactor
(geometry/controls/per-w split) is behavior-preserving. Imports clean;
`cusp_uniform_reference_grid` in _pearcey_cusp.__all__.

CONFIRMED CORRECT (audited, no findings):
- `_cusp_uniform_geometry`/`_cusp_controls`/`_cusp_uniform_at_w` are a faithful
  extraction: x/y/handoff_radius/reflected/phase_sign math byte-identical to
  pre-diff; handoff_radius stays in pre-F074 gauge; c4 -> reflected; curvature
  = phi_ssr/(2*hard_eigenvalue).
- `cusp_uniform_reference_grid` solves geometry ONCE per cell, loops only w,
  calls `_cusp_uniform_at_w` per node WITHOUT ppGO rung / F074 gate /
  calibration certificate (documented; cluster_sum->0 collapses caught by the
  non-vanishing guard). `_consult_pearcey(x,y,None)` -> live `pearcey` (no
  table), matches the "no Pearcey table" docstring.
- `fold_cusp_reference` guard `min|F_ref|/max|F_ref| >= _NON_VANISHING_MIN_RATIO
  = 1e-3`: Airy Wronskian trivially passes; Pearcey can hit 0 via BOTH P~0
  (interior) AND cluster_sum->0 (far-exterior cluster resolves above w~7, the
  self-falsification witness). max==0 -> nan -> np.isfinite guard declines.
- `_airy_fold_form` refuses b3->0 via `_fold_amplitudes` (|b3| <= _B3_MIN=1e-6),
  so the fold->cusp handoff is real; test premise `assertIsNone(airy)` is
  self-guarding even though the literal `1e-6` is re-typed (adjacent premise
  assertion re-derives from live _B3_MIN).
- Rename `airy_fold_reference`->`_airy_fold_form`/`fold_cusp_reference` complete:
  zero source refs to the old name (only a stale .pyc). All consumers (likelihood
  serve, train script, test helpers) updated.

NEW FINDINGS:
- INS-3-001 (design -> Librarian, carries INS-2-003/INS-1-004 lineage): SPEC.md
  line 54 and DATA_CONTRACTS.yaml line 389 STILL describe the low-w diffractive
  chart anchor as `prefactor_c(w) = C(w)` / `r_pure = f_pure/(sqrt(mu_pure)*
  prefactor_c(w))` / serve `F = mass_sheet_phase * prefactor_c(w) * sqrt_mu_full
  * r_pure`. Code now uses `fold_cusp_reference` (Airy fold + Pearcey cusp) as
  F_ref. Plan listed SPEC.md + DATA_CONTRACTS.yaml as expected-to-change; they
  did NOT change. Bidirectional divergence (recurring doc-staleness lineage).
- INS-3-002 (trivial): `fold_cusp_reference` docstring's guard rationale
  ("exterior cusp cells have P != 0 ... interior cusp cells can hit P ~ 0")
  describes ONLY the P~0 mechanism; the far-exterior decline the
  self-falsification test actually exercises is cluster_sum->0 (cusp cluster
  resolves, matched->0), a SECOND mechanism the docstring omits. A reader could
  wrongly conclude exterior cusp cells are never guard-declined. Clarify.

NOTE (not a finding): the non-vanishing guard ratio is grid-dependent —
computed over the bake w_grid at train time, over the likelihood dense_w at
serve. Failure mode is a safe decline (-> exact engine); per-node F_ref values
are grid-independent. Serve/bake decline asymmetry to remember if a future
census shows unexpected Pearcey-cell declines at serve.
