# Librarian Short-Term Observations

## 2026-08-09 — cusp ppGO fast rung build (post-build doc sync)

Scope: completed `todo.d/lensing_cusp_ppgo_at_high_w.md` (tagged `[→ spec]`)
after the "Add ppGO fast rung inside cusp_amplification" build landed
`_R_PPGO_ERROR_CONST = 50.0`, `_W_PPGO_FLOOR = 50.0`, `_PPGO_BAR_DIVISOR =
10` + the `fold_ppgo_correction` fast path in `_pearcey_cusp.cusp_amplification`
(private module; no public-API change). Added SPEC.md sentence to the
`_pearcey_cusp.py` description, `spec_changelog.d/2026-08-09_cusp_ppgo_high_w.md`
(`bump: patch`, SPEC 0.35 -> 0.36.1 — renderer assigned the .1 suffix by
alphabetical order within the date, no surprise), `completed.d` fragment, and
repointed the dependent `depends_on:`.

Patterns learned/confirmed:
- **`depends_on` repointing is mandatory on completion.** A dependent open
  fragment's `depends_on: [<old todo stem>]` dangles the moment the fragment
  moves to `completed.d` under a date-prefixed name — the renderer's validator
  then warns. The established convention (confirmed against
  `2026-08-07_polar_rechart`, `2026-08-07_subdivision-recursion-wedge-v3-r-caustic`)
  is to repoint `depends_on` to the NEW date-prefixed completed stem. I missed
  this at first; the renderer's dangling-dep warning caught it.
- **`delete_lines` empties a file but does not delete it** — `rm` the now-empty
  todo fragment afterwards.
- The recurring `lens_amplification_surrogate` test-only-consumer warning from
  `sync_derived_docs.py` surfaced AGAIN this run (fifth+). Escalation fragment
  `todo.d/surrogate_contract_test_consumer_warning.md` verified still open; no
  duplicate created. Still unresolved.

New fragile cross-references worth watching:
- SPEC.md's `_pearcey_cusp.py` sentence now names `_R_PPGO_ERROR_CONST`,
  `_W_PPGO_FLOOR`, `_PPGO_BAR_DIVISOR` — same rename-preserved staleness family
  as the schema constants; a rename in code must touch SPEC + the new completed
  fragment. The phrase "returns before any table or quadrature lookup" breaks
  if a future build moves the rung after the table consult.
- The completed fragment records `_R_PPGO_ERROR_CONST = 50.0` as PROVISIONAL
  (driver post-build measurement owed). SPEC intentionally carries the
  mechanism, not the provisional marker — if the constant is tightened later,
  the SPEC gate sentence stays valid; only the completed record ages.

Surprises: none major — the build was a private-module-only change, so
overview.rst / api.rst / installation.rst / DATA_CONTRACTS / data_registry all
stayed clean (no new disk artifact; `fold_ppgo_correction` pre-exists).
