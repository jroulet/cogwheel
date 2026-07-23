# Inspector Short-Term Observations

## 2026-07-23 (Build 8h-a-fin) — VERDICT: ISSUES (1 blocker regression + 6 missing binding tests)

Scope: all uncommitted working-tree changes completing approved Build 8h-a
(WP1 ppgo_map, WP2 likelihood band-split, WP3 interior/strata trimming, WP4
subdivision). Reviewed cogwheel/lensing/{ppgo_map.py(NEW),likelihood.py,
surrogate_training.py}, scripts/train_ppgo_map.py(NEW), registry/DATA_CONTRACTS,
test_lensing_farfield_envelope.py(+615). Imports OK (all three modules import).

### INS-8haf-001 (implementation, BLOCKER) — REGRESSION: far-field packing deleted
In `_train_band_charts` the WP4 edit REMOVED the two trailing non-gated packing
lines (`charts.append(chart)` / `chart_reports.append(chart_report)`). git diff
tail literally shows `-        charts.append(chart)` / `-        chart_reports.
append(chart_report)`. The `for tile in admitted:` loop body now ENDS at the
`continue` inside `if gated:` (file ends at L2084). Consequence: a far-field
tile that PASSES gating is never packed into `charts` → ZERO far-field charts
built/served. Only gated (failing) tiles subdivide; their passing CHILDREN are
packed inside `_subdivide_farfield_tile(charts=charts,...)`, but a directly-
passing parent tile is dropped silently.
Breaks 2 PRE-EXISTING (unchanged file) tests in test_lensing_surrogate_training.py:
- TilingRecordTestCase::test_built_tile_boxes_are_pairwise_disjoint
  ("{} is not true : no built far-field charts recorded")
- ResidueBucketPartitionTestCase::test_chart_served_bucket_is_nonempty
  ("0 not greater than 0 : no draw was chart-served")
Full earlier run: 2 failed, 102 passed, 1 skipped, 2 errors (the 2 errors are in
UNCHANGED test_lensing_surrogate.py = pre-existing env, out of scope).
FIX: restore the two packing lines after the `if gated: ... continue` block.

### INS-8haf-002 (design) — 7 of 8 BINDING domain tests missing
Plan has 8 binding domain_test_descriptions; only #7 (WP4 EDGE-ANNULUS
SUBDIVISION) was implemented (the +615 lines in test_lensing_farfield_envelope.py).
MISSING: #0 WP2 band-split node-match; #1 WP3 telescoping interior 4-image;
#2 WP1 sup-over-w floor non-monotone; #3 WP1 safety margin; #4 WP1+WP2
corrupt/absent/UNKNOWN map refusal F010; #5 WP3 interior admission geometry +
morse-sign mask; #6 WP3 strata trimming record. Plan expected changes to
test_lensing_ppgo_map.py (DOES NOT EXIST) and test_lensing_surrogate_training.py
(UNCHANGED). New public API CertifiedPpgoMap and the band-split dispatch ship
UNCERTIFIED. Owner: Test Dev.

### INS-8haf-003 (trivial) — private-symbol import
likelihood.py L99-100 imports private `_caustic_geometry` from ppgo_map
(alongside UNKNOWN, get_certified_ppgo_map). Same subpackage so tolerable, but
prefer a public accessor for the caustic reach.

### Carried open (NOT this build)
- INS-8gb-005 (Librarian): SPEC farfield_eps_max=3e-3 vs code 1e-3;
  DATA_CONTRACTS lens_amplification_surrogate lacks REQUIRED per-chart npz meta
  `envelope_definition`. DATA_CONTRACTS.yaml WAS touched this build (added
  certified_ppgo_map stanza) but the pre-existing divergence was not addressed.
- INS-4-001 (design): TrainingConfig.max_farfield_regions default.

### Positives verified this build
- ppgo_map.py mirrors PearceyTable cleanly; SHA1 hash-pinned load,
  allow_pickle=False; w_trust=max(1.5 w_cert, w_cert+2.0) single-source rule.
- Band-split certification transfer confirmed: E_ff=0 reconstruction telescopes
  to image-kernel sum == geometric_amplification (certified object). Byte-
  identical when map is None (opt-in).
- Registry end-to-end resolves: pipeline_graph resolve certified_ppgo_map ->
  cogwheel/data/certified_ppgo_map.npz, matches CertifiedPpgoMap.load loader.

### Lessons
- A guard-clause refactor that ends a loop body on `continue` inside the guard
  is a red flag: check the FALL-THROUGH (non-guarded) tail still packs/returns.
  Here the passing-case packing was the deleted tail — silent coverage loss.
- Re-run PRE-EXISTING suites over UNCHANGED test files: a production-only edit
  regressed them; the failure text ("no built far-field charts recorded") points
  straight at the deleted producer line.
