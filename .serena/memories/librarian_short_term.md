Build: lobe subdivision + cusp carve-out (2026-08-08, uncommitted working tree; HEAD fd84cea). Clean.

Scope: WP-1 `_subdivide_lobe_tile` wired into gated-lobe branch of `_train_band_charts`; WP-2 deltoid-cusp carve-out RESOLVED as "no carve-out needed" (existing eta_max tube-shell nearest-distance test in `_SaddleLobeAdmission.admits` already rejects near-cusp tiles; `_LOBE_CUSP_EXCLUSION_DISTANCE=0.1` recorded as redundant constant). New test file test_lensing_lobe_subdivision.py (19 tests incl. GhostKernelSaddleTestCase).

What went stale & fixed:
- todo.d/lensing_chart_kinds_should_share_one_tiling_machine.md item (a): "LobeInteriorChart still has NO subdivider" → lobe now covered; fragment stays OPEN (TubeChart, OOP shape, region-scoped entry, byte-identical probe test still owed).
- todo.d/lensing_saddle_forensics.md items (b,c,f): (b) NO→YES subdivider; (c) carve-out question ANSWERED (no carve-out needed, Professor ruling); (f) "no test calls ghost_kernel at gamma>1" → GhostKernelSaddleTestCase exists (structural only), branch VALUE still unpinned, item stays open.
- SPEC.md REGISTRATION GATE sentence: "its window falls through to the serving ladder" → now "subdivided recursively where the kind has a subdivider (far-field, wedge, lobe); only all-children-fail windows are ladder gaps". patch spec_changelog fragment (undated, empty-date bucket 0.11.8 — known quirk, top stays 0.34.0, don't fix).
- Added completed.d/2026-08-08_lobe-subdivision-cusp-carveout.md as build record; both tracking fragments STAY OPEN (no premature closure).

New fragile cross-refs worth watching: the `[[2026-08-08_lobe-subdivision-cusp-carveout]]` backlink from the chart_kinds fragment; the SPEC.md gate-sentence now names "far-field, wedge, lobe" as the subdividing kinds — if TubeChart gains a subdivider, that list and item (a)'s "TubeChart still has none" both need touching (same rename-preserved-staleness family as the polar-re-chart case).

Surprises:
- SPEC.md backtick trap again: a `MAX_SUBDIVISION_DEPTH` backtick inside a double-quoted bash python -c was eaten by command substitution. Used a heredoc temp script to repair. Verify SPEC.md edits by raw-bytes read (already known, bit me again).
- The recurring lens_amplification_surrogate test-only-consumer warning STILL fires; the escalation TODO fragment surrogate_contract_test_consumer_warning.md already exists (created by a prior librarian) — no duplicate needed, it remains open awaiting contract owner. sync_derived_docs.py used conda env python (jedi present, rg absent).
- render_fragments.py left no tidy_advisory/foreman_lite side-effects this time (the tidy_advisory.json diff in the tree is the build's own).
- No docs/source or docstring edits → no Sphinx rebuild needed.
