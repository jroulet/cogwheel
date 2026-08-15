# Inspector Short-Term Observations

## 2026-08-14 FINAL PASS: certified_map_guard_relaxation (F080 per-cell saddle rho<1 relaxation) — VERDICT PASS

Worktree: /home/tejaswi/Work/cogwheel-claude-dev.
Scope: 3rd/final pass on the per-cell evidence-keyed relaxation of the F073
blanket `saddle rho<1 -> UNKNOWN` guard in CertifiedPpgoMap.

### INS-2-001 (stale SITE-4 docstrings) — RESOLVED
test_lensing_saddle_rho_guards.py CensusBandSplitMirrorIntegrityTestCase
docstrings were rewritten this pass:
- Class doc (L515-536): now says "The former SITE 4 guard" (explicitly
  removed), "Saddle rho<1 sources are no longer suppressed", and the
  no-band-split cause is correctly "w_trust (28.746) lies well above the
  test's tiny w-band (max ~1.24)... split condition w_lo<w_trust<w_hi is
  False, not because rho was suppressed."
- test_corridor_source_no_band_split doc (L580): "Corridor source:
  w_trust (28.746) exceeds the tiny w-band."
- `_M_LENS` comment (L550): corrected to "gives w in [~0.25, ~1.24] for a
  20-100 Hz band" (was the wrong "w in [10,50]"). Matches actual w_grid
  [0.2476, 1.2379].

### Production code (likelihood.py / surrogate_census.py / ppgo_map.py)
Unchanged from my prior verified pass. Re-diffed the two small consumers:
- likelihood._ppgo_cell_coords: SITE1 pre-guard `if parity=='saddle' and
  rho<1.0: return None` deleted, replaced with delegation comment. Delegates
  to CertifiedPpgoMap.w_trust/w_ceiling.
- surrogate_census.characterize_sample: SITE4 `rho=None` pre-guard deleted,
  delegation comment added; census mirror routes through same map methods.
Both are faithful mirror-deduplications (De Morgan-equivalent: a still-
refused cell returns UNKNOWN downstream so band-split/ceiling unchanged;
only the allowlisted Cell 1 newly flows through).

### Tests
test_lensing_saddle_rho_guards.py + test_lensing_ppgo_map.py => 77 passed
(7.71s). test_lensing_surrogate_census.py not changed this pass; prior pass
confirmed no stranded pins on the removed SITE4/SITE5 behavior.

### Carried-forward (all RESOLVED across the 3 passes)
- INS-1-001 (stranded 6 red guard tests): fixed pass-2.
- INS-1-002 (unused _surrogate_census import): fixed pass-2.
- INS-2-001 (stale SITE-4 docstrings): fixed this pass.

### Patterns reaffirmed
- STALE-DOCSTRING-AFTER-GUARD-REMOVAL: when a build removes a guard and
  rewrites SOME sibling test docstrings, the missed sibling can pass for a
  DIFFERENT reason than its docstring claims (latent vacuity). This build
  closed it cleanly on the follow-up pass.
- A trivial docstring finding is worth carrying: the corridor test passes
  because w_trust exceeds the w-band, NOT because of any suppression — the
  old docstring named dead machinery (SITE 4) as the cause.
