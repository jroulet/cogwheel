# Inspector Short-Term Observations

## 2026-08-18 — Born far-field completion Build 2 (trained-floor band split)

Scope: uncommitted working-tree diff (cwd = /home/tejaswi/Work/cogwheel-claude-dev
worktree). WP1 `likelihood.py` `_born_residual_analytic` gains Route 2 (trained-floor
band split) + `_born_reconstruct` gains `engine_envelope`/`engine_mask` overlay params.
WP2 `serve_route_census.py` gains `_born_trained_floor_route` mirror + Route-2 branch in
`classify_draw` intercept-5. Test file test_lensing_born_certificate.py (+1155 lines,
full new suite). NOTE: the headline closure #1 (two-image GO carrier) was DROPPED per the
mandatory reachability check in the handoff — this build ships closures #2/#3 only.

VERDICT: PASS. resolved_ids: [INS-1-001, INS-1-002].

### Why PASS — production WP1 verified correct
- Three-route serve, in specificity order: Route 1 (fully in box -> interpolated
  residual, returns via `_born_reconstruct` with NO engine_envelope — BYTE-IDENTICAL to
  HEAD); Route 2 NEW (box-covered but host sub-band drops below the trained log_w floor —
  low-edge escape); Route 3 (beyond-box/high-edge/disjoint -> certificate-gated
  carrier-only via `_born_carrier_certificate_serves`, else fall through — byte-identical).
- `trained_floor = math.exp(float(born_chart.log_w_grid[0]))` — DERIVED from the shipped
  artifact, NEVER a literal (satisfies the DERIVED-not-pinned rule).
- Route 2 four-way gate: `band_split_floor and engine_mask.any() and chart_mask.any() and
  born_chart.covers(gamma, rho, dense_w[chart_mask])`. A high-edge/disjoint escape leaves
  chart_mask uncovered -> skips to Route 3 (byte-identical). The `band_split_floor` term
  rejects the null-fallback all-True below_floor (null-split identity preserved).
- Mask disjointness VERIFIED: host_mask = below_mask & ~bottom_mask; engine_mask =
  host_mask & below_floor (⊆ below_mask, so NOT zeroed by envelope[~below_mask]=0; and ⊆
  host so disjoint from bottom_mask/F_P overwrite). chart_mask = host_mask & ~below_floor.
- `_born_reconstruct` overlay ORDER correct: F_P bottom overwrite (in f_total) -> envelope
  zeroed above below_mask -> `if engine_envelope is not None: envelope[engine_mask] =
  engine_envelope[engine_mask]`. Non-conflicting.
- Gauge consistency: `_engine_envelope_below_split` (PRE-EXISTING helper, reused per plan;
  NOT in this diff) returns the FARFIELD_KERNEL_SUM-gauge exact-Schwinger envelope,
  full-length, zero above split — same gauge as the chart carrier, so the two tiers stitch
  with no field discontinuity. It calls `self._evaluate_envelope` (an engine door).

### Why PASS — census WP2 is a faithful mirror
- `_born_trained_floor_route` reads trained_floor from `log_w_grid[0]`, delegates to
  `mods.band_split_mask` + `born_chart.covers`, same four-way conjunction as production.
  No decision logic re-typed.
- `classify_draw` intercept-5 Route order: Route 1 -> born_analytic; Route 2
  (trained_band_escape AND `_born_trained_floor_route`) -> born_analytic BEFORE Route 3;
  Route 3 (`born_carrier_serves`) -> born_carrier_only. host_mask/chart_w derived
  line-for-line from production (ppgo band split ceiling-capped, nested diffractive-bottom
  split). This RESOLVES prior carry-forward INS-1-002 (census previously did not model
  trained_band_escape).
- SERVE_ROUTES UNCHANGED (12 labels; born_analytic/born_carrier_only pre-existing) -> NO
  contract-widening laggard. `_ProductionModules` needed no new accessor.

### Tests
- test_lensing_born_certificate.py + test_lensing_serve_route_census.py +
  test_lensing_born_analytic_reachability.py = 117 passed (43s). The latter two are
  UNCHANGED in the working tree (only the certificate suite is in the diff) -> independent
  green corroboration that the SERVE_ROUTES contract + Born dispatch chain aren't broken.
- Certificate suite quality: fixtures DERIVED with premise assertions (rho>2, strict inner
  sub-band, mask non-emptiness) + self-falsification classes + delegation spies
  (`wraps=`) + engine-door tripwires (BornCensusEngineFreeTestCase: booby-traps
  ChangRefsdalChannels.evaluate / _schwinger.f_schwinger / _f_schwinger_mpmath, asserts 0
  calls and no fresh mpmath import) + a census-vs-production route-agreement matrix
  (BornCensusMirrorFaithfulnessTestCase). Route 2 IS covered here via `_make_floor_probe`
  (spies `_engine_envelope_below_split` returning engine_value, engine-free).

### resolved_ids rationale
- INS-1-001 (probe missing `_born_reconstruct`): resolved by a PRIOR commit — the
  reachability `_BornAnalyticProbe` now binds `_born_reconstruct` (line 399) and the suite
  is green. Not fixed by THIS diff but confirmed closed.
- INS-1-002 (census did not model trained_band_escape): resolved by THIS build's WP2.

### Carry-forward observations (NOT findings this diff)
- LATENT / not-a-finding: `_BornAnalyticProbe` binds `_born_residual_analytic` + deps but
  NOT the newly-called `_engine_envelope_below_split`. This is the recurring
  stub-hides-new-attribute shape — BUT Route 2 is engine-dependent (calls
  `self._evaluate_envelope`), and the probe is engine-free BY DESIGN, so no faithful
  engine-free reachability test can drive Route 2 on it; Route 2 is instead covered by the
  certificate suite's engine-free spy. If a future Route-2 reachability test is ever added
  it must stub BOTH `_engine_envelope_below_split` AND `_evaluate_envelope`. Noted, not
  flagged.
- -> Librarian (doc-staleness, pre-existing, NOT actionable this diff): SPEC.md /
  DATA_CONTRACTS.yaml still do not name ANY Born serve route (born_analytic /
  born_carrier_only / the new trained-floor split). The plan listed SPEC.md +
  DATA_CONTRACTS.yaml as expected-to-change but they were not touched. The born routes
  were ALREADY undocumented before this build (see 2026-08-18 prior entry), so this build
  introduces no NEW spec-accuracy divergence — it refines behavior within an
  already-undocumented rung. Pure doc-sync item.
