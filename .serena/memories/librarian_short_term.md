# Librarian Short-Term Observations

## Run: 2026-08-12 — post-commit sync for {2eeab69, 34035eb, 4c7dc92} (PRIMARY: 4c7dc92 / INS-5-001)

**Scope**: 3 commits in `.claude/sync_issues.json`.

**Triage**:
1. `2eeab69` (build_monitor.sh) and `34035eb` (orchestrator.py) — pure
   `.claude/sdk/` agent-infra commits, each with its own already-rendered
   `TODO.md`/`COMPLETED.md` fragments committed in the same commit. No
   `cogwheel/` or `docs/source/` touch. NO-OP for doc surfaces (agent-only
   paths, out of Librarian's tracked scope per CLAUDE.md EXCLUDE_PATHS).
2. `4c7dc92` (lensing Build WP2, deltoid exterior fix) — PRIMARY. Verified
   INS-5-001 independently against the actual diff (surrogate.py,
   surrogate_training.py) before editing, per standing practice: confirmed
   `LobeExteriorChart` (frozen dataclass sibling of `LobeInteriorChart`,
   `image_count=2`, `FARFIELD_KERNEL_SUM` envelope, no `other_centroid`/
   `corridor_half`/fold-carrier, `rho_lobe` domain `(1, rho_outer]`),
   `from_lobe_exterior_engine`, `_lobe_exterior_serves`, NPZ kind
   `lobe_exterior`, and the explicit code comment "the origin-polar
   saddle-exterior tiler is RETIRED" in `surrogate_training.py` (~line 4899).
   Confirmed `_deltoid_cusp_axis_map` still DEFINED in surrogate.py but no
   longer imported/called by `surrogate_training.py` (orphaned for the
   macro-saddle-exterior path it used to serve; still referenced by tests).

**Fixed** (SPEC.md, 7 targeted edits; DATA_CONTRACTS.yaml, 3 targeted edits;
fragments `spec_changelog.d/2026-08-12_lobe_exterior_chart.md` +
`contracts_changelog.d/2026-08-12_lobe_exterior_chart.md`, both `bump: patch`
— confirmed by precedent, e.g. `2026-08-03_interior_wedge_chart.md`, that
even a whole-new-chart-type addition is `patch` in this repo's convention,
not minor):
- `ExteriorPolarChart` now documented positive-parity (astroid) ONLY in both
  files (was "positive-parity astroid and macro-saddle alike").
- `LobeExteriorChart` added alongside `LobeInteriorChart`: pipeline table row,
  "Key abstractions" far-field coordinate contract (SPEC.md), and the
  `lens_amplification_surrogate` description (DATA_CONTRACTS.yaml).
- Corridor-serve correction: the old sentence "a source inside the corridor
  falls through to the exact-engine ladder as a named refusal" is now WRONG
  — a corridor source is served by the canonical `+y1` lobe's
  `LobeExteriorChart` via the D2 reflection fold (`_lobe_exterior_serves`
  drops the corridor test entirely). Fixed in SPEC.md; this is a genuine
  BEHAVIOR-CHANGE staleness pattern (an outcome clause, not just a missing
  entity) — worth watching for again: "X falls through to Y" sentences need
  re-verification whenever a build adds a new serving tier between the named
  refusal point and Y.
- GATED-subdivider kind list "(far-field, wedge, lobe; bounded by
  MAX_SUBDIVISION_DEPTH)" was ambiguous now that there are two lobe kinds —
  clarified to `lobe-interior`; lobe-exterior (like tube) has NO subdivider,
  a gated/flipped tile there is a ladder-served gap (verified via the
  `surrogate_training.py` code comment at the `region == 'lobe_exterior'`
  branch, ~line 5596-5611). NEW instance of the "ENUMERATED-KIND-LIST
  CROSS-REF" fragile-pair pattern already in long-term memory.
- Extended the `test_lensing_surrogate_lobe.py` certified-by sentence: that
  file now also carries 7 `LobeExterior*` test classes (verified via
  `get_symbols_overview`, not assumed from the file being in `changed_files`).

**Skipped / out of scope, recorded for a future pass (do NOT re-derive from
scratch — verify against current code first)**:
- **theta_to_u REQUIRED-vs-OPTIONAL contradiction, LobeInteriorChart**
  (pre-existing, NOT touched by 4c7dc92): SPEC.md and DATA_CONTRACTS.yaml
  both currently claim the lobe-interior loader "reads theta_to_u
  unconditionally... an absent map hard-refuses (KeyError)". I read the
  ACTUAL CURRENT `_chart_from_npz` body (fresh `find_symbol`, this session)
  and the `'lobe'` branch uses `theta_to_u = data.get(prefix + 'theta_to_u')`
  — a SOFT read, never raises KeyError. `mem:lobe_interior_chart` (2026-08-08)
  independently corroborates this WAS a known open bug at the time ("Open
  non-blocking note: the cusp_angle=None raw-theta fallback builds charts
  that cannot survive an NPZ round-trip (`_chart_from_npz` reads
  `theta_to_u` unconditionally while `_chart_to_npz` writes it only when not
  None)") — so a commit between 2026-08-08 and now likely fixed the reader
  to be soft (fixing the round-trip bug) WITHOUT updating SPEC.md/
  DATA_CONTRACTS.yaml's "REQUIRED/hard-refuse" sentence. I did NOT fix this
  — out of my assigned scope (INS-5-001 only) and not flagged by Inspector
  this run. I deliberately wrote the NEW `LobeExteriorChart` theta_to_u
  description WITHOUT any comparison to LobeInteriorChart's contract (no
  "unlike/matching the lobe-interior loader" language) specifically to avoid
  propagating this unresolved contradiction. NEXT LIBRARIAN/INSPECTOR PASS:
  verify `_chart_to_npz`'s `'lobe'` branch write-side too, then fix the
  "REQUIRED... hard-refuses (KeyError)" sentence in both SPEC.md (near
  "Lobe-interior artifacts carry a SINGLE axis-schema tag") and
  DATA_CONTRACTS.yaml (near "mapped from theta_local at serve time through
  the REQUIRED theta_to_u map").
- **`_to_caustic_fixed` astroid-exterior multiplicative-vs-additive
  contradiction** (pre-existing, unrelated to lobe_exterior): SPEC.md's
  "CAUSTIC-FIXED RADIAL COORDINATE" section says the astroid EXTERIOR arm is
  directional-MULTIPLICATIVE (only the saddle exterior arm is scalar
  ADDITIVE), but DATA_CONTRACTS.yaml's `ExteriorPolarChart` paragraph says
  "additive exterior on both parities... multiplicative only on the astroid
  interior arm" (i.e. astroid exterior = additive too). Did not verify
  against code or fix — deep, unrelated tangent, flag for a dedicated pass.
- Verified `docs/source/` has ZERO mentions of `ExteriorPolarChart`,
  `LobeInteriorChart`, `lens_amplification_surrogate`, or
  `LensAmplificationSurrogate` — confirms the surrogate speed layer is
  correctly absent from the Sphinx narrative (internal/offline, no
  user-facing doc page needed). Did not invent one.
- `.claude/spec/todo.d/tests_cross_class_attribute_borrowing.md` (in
  4c7dc92's changed_files) is test-only housekeeping, already
  self-rendered into TODO.md within the same commit — no action needed.

**Verification method note**: for two huge single-line SPEC.md/
DATA_CONTRACTS.yaml prose blocks (the pipeline-table cell and the
`lens_amplification_surrogate` YAML description are each ONE physical line
of several KB), used a Python script with `text.count(old) == 1` assertions
per edit BEFORE writing, all edits batched into one script, applied only
after every count verified — avoids the backslash-pipe (`\|`) escaping traps
already in long-term memory and catches an accidental double-match before
it corrupts the file.
