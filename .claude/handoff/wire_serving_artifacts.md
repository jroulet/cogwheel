# Build: wire the shipped serving artifacts (F077 — the chart layer is dead code)

## Mission

Two shipped artifacts and two rungs are unreachable from any production
entry (F077, serve-path-traced): `born_residual_chart.npz` (7,990 B,
2026-08-04, CLEAN per F075) and `certified_ppgo_map.npz` (15,632 B,
2026-08-03). Nothing auto-loads them; `get_certified_ppgo_map()` reads a
process global no production code sets. Make both reachable, each behind
its content-hash/schema refusals. ACCEPTANCE IS A TRACE, not an inventory:
per artifact, a serve-path trace from `LensedRelativeBinningLikelihood`
production entry (`lnlike_and_metadata -> ... ->
_amplification_coefficients`) to the artifact actually serving a draw.

## The structural problem the plan must solve first (measured at HEAD a4ba536)

The Born rung ("fact-4 slot", `likelihood.py:1775-1787`) and BOTH certified-
map consult sites (`_ppgo_band_split:1487/1521`, `_ppgo_cell_ceiling:1532/
1564`) live INSIDE `_surrogate_coefficients`, which is entered only when
`self.amplification_surrogate is not None` (`:2220`) — and no surrogate npz
ships until the training campaign (confirmed: `cogwheel/data/` has no
`lens_amplification_surrogate.npz`; nothing calls its `load()` without a
path). Attach-at-construction of the born chart + map alone therefore
reaches NOTHING. The Architect must lift reachability: the candidate shape
is a first-class intercept in `_amplification_coefficients` (the pattern
`_saddle_farfield_analytic:2254-2257` already uses), serving the Born rung
(rho > 2 exterior, both parities, its existing gates) and consulting the
map without a surrogate. The Professor rules on serve-order correctness
(the existing intercepts' mutual exclusivity comment at `:2245-2253`).
Byte-identity everywhere off the newly-served path.

## Measured facts (HEAD a4ba536; line numbers re-locate by symbol)

1. `BornResidualChart` (`born_residual_chart.py`, 139 lines) has NO
   loader — `covers()` and `evaluate()` only. The npz is written raw by
   `scripts/train_born_residual.py:133-145` with provenance as `str(dict)`
   (not JSON). Wiring requires a NEW `load()` mirroring
   `CertifiedPpgoMap.load` (`ppgo_map.py:365-425`): `allow_pickle=False`,
   content hash over grids, schema-key hard-refusal, `ValueError` naming
   the regenerating script. Provenance needs `ast.literal_eval` or a
   one-time re-save (decide; a re-save changes the artifact hash —
   record it).
2. Certified map machinery exists and is production-ready:
   `use_certified_ppgo_map(path=None) -> bool` (`:608-624`) with hash +
   schema 0.2.0 refusals; it is called by ZERO production code (only
   tests). The Pearcey table (`_pearcey_cusp.py:119-133`) is the pattern:
   process-global opt-in via function switch, per-call override beating
   the global.
3. JSON round-trip blocker: BOTH `get_init_dict` overrides raise
   `NotImplementedError` on a non-None artifact (`likelihood.py:938-952`,
   `marginalized_likelihood.py:205-212`). Auto-attach must teach them a
   serializable form (artifact path or a default-sentinel), or every
   `Posterior.to_json` breaks. This is in scope and needs tests.
4. `LensedMarginalizedExtrinsicLikelihood` does NOT thread
   `born_residual_chart` at all (ctor `:100-106`, `_set_summary:~243`
   rebuilds the inner engine passing only `amplification_surrogate`).
   Thread it, or the marginalized path stays chart-less — say which.
5. F075 retroactive-label advisory (REPORT only, no artifact edits):
   `certified_ppgo_map.npz` has 32 positive-parity exterior cells measured
   against the contaminated fold-arm oracle — OVER-conservative direction
   (refuses ppGO where it is fine; correctness is not at risk). Retraining
   (`scripts/train_ppgo_map.py --production`) is the training campaign's
   job, NOT this build's. The build's evidence report must state the
   advisory next to the map-serving trace.
6. `DATA_CONTRACTS.yaml:361-370` contains a FALSE claim: born chart
   "attaches it at construction time". Fix it to describe the shipped
   behavior, with a `contracts_changelog.d/` fragment (`bump:` per rules).
7. Test debt (enumerated; None-default assertions):
   - `test_lensing_surrogate.py:1333` `DefaultSurrogatePathTestCase` —
     `test_default_surrogate_attribute_is_none` AND (critical) its
     default-constructed likelihood is the ORACLE for `LnlikeAccuracyTestCase`
     served-lnL gates: if construction auto-attaches, the oracle changes.
     The oracle must construct with explicit `None`s (or an equivalent
     pure-engine switch) — a one-line fix with a comment saying why.
   - `test_lensing_born_residual_wiring.py` (`NoChartByteIdentityTestCase`
     :185-270 et al.) encodes the None-default contract via a local probe.
   - `test_lensing_ppgo_bandsplit.py` save/restore discipline
     (`addCleanup(set_certified_ppgo_map, None)`; F078 rule: any suite
     installing a process global owes save-restore). If construction
     installs the map, suites assuming a None global can leak — audit the
     enumerated sites (`test_lensing_fold_ppgo_handoff.py:918-961` patches
     the census getter to None).
8. Census consult site: `surrogate_census.py:413` mirrors
   `_ppgo_band_split`. served == counted: whatever reachability the build
   gives the production path, the census mirror moves in the same build.

## Scope

IN: the reachability lift (Architect-designed, Professor-ruled); the new
`BornResidualChart.load` with refusals; attach-at-construction (explicit
kwarg default changing from `None` to auto-load-with-opt-out — design the
opt-out: explicit `None` stays pure-engine); `get_init_dict` JSON fix both
classes; marginalized threading (fact 4); census mirror; DATA_CONTRACTS
fix + changelog fragment; the enumerated test re-points; fast
decision-level tests (serve-path trace tests per artifact — the trace IS
the test); evidence report with the F075 advisory.

OUT: retraining anything (F075 advisory only); the map's saddle rho<1
API-guard relaxation (5/7, next build — do NOT touch `w_cert`'s guard);
`lens_amplification_surrogate` wiring beyond leaving its `None` path
intact; the training campaign; slow tiers.

## Acceptance

- Per artifact, a serve-path trace test: a production-entry draw whose
  served value changes when the artifact is detached (byte-equality
  comparison identifies the route — F077's lesson). Born chart: a rho > 2
  exterior draw served by the Born rung; map: a draw whose band split /
  cell ceiling consults the map.
- Byte-identity off the newly-served path: a battery of draws NOT owned by
  the artifacts serves bit-identically to HEAD.
- JSON round-trip green with attached artifacts, both likelihood classes.
- A corrupted/tampered artifact refuses loudly at construction (hash and
  schema each get a test) and construction falls back per the designed
  opt-out semantics — never silently serving a bad artifact.
- Full fast suite green; no test left asserting the retired None-default
  without the explicit-opt-out rationale.

## Constraints

Branch claude-dev; fragments (`[→ spec]`: this closes
`todo.d/lensing_wire_serving_artifacts.md`; SPEC/DATA_CONTRACTS surfaces
per rules); values-not-paths; in-build tests FAST; no engine sweeps
in-build; measurement belongs driver-side — if a WP needs a new
measurement, escalate, do not iterate.
