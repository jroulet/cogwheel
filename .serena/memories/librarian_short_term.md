# Librarian Short-Term Observations

## 2026-08-08 — post-build doc sync for ExteriorPolarChart cusp-adapted u=d^(2/3) coordinate

**Scope**: sync SPEC.md + DATA_CONTRACTS.yaml to the exterior-polar axis-schema
change (`'exterior_polar_rho_theta_c'` -> `'exterior_polar_rho_u_v1'`, optional
cusp-adapted `theta_to_u` on parity == 1; raw `theta_c` on the macro-saddle).
Inspector findings INS-4-001/INS-4-002 (the INS-3-001/002 carry-forwards) were
both Librarian-scope design findings; both fixed.

**What went stale and why**: the build touched only `cogwheel/lensing/` +
tests; no doc surface. SPEC.md's "Far-field surrogate coordinate contract"
paragraph (Key abstractions) and the DATA_CONTRACTS.yaml lens_amplification_
surrogate description still named the OLD tag and "no arc-length map is needed"
as a standalone claim. Same silent-staleness family as the lobe/wedge schema
bumps — a schema rename in code with zero doc diff.

**Fixes**: SPEC.md Key abstractions paragraph (tag + retired-tag hard-refuse +
optional theta_to_u on parity == 1 + raw-theta on parity == -1), SPEC.md
GLOBAL MULTI-CHART ARTIFACT summary sentence (4th-axis note), DATA_CONTRACTS.yaml
ExteriorPolarChart description (tag, map shape (2, 2001), uniform-in-u build,
conditional write / None-load, parity split). New fragments:
spec_changelog.d + contracts_changelog.d `2026-08-08_exterior_polar_cusp_axis`
(minor / major). Render: spec 0.36.0, contracts 3.0.0. Closed TODO
`todo.d/lensing_exterior_polar_cusp_coordinate.md` -> completed.d fragment.

**Pattern worth remembering (fragile cross-references)**:
- `_EXTERIOR_POLAR_AXIS_SCHEMA` constant name is now cited in BOTH SPEC.md and
  DATA_CONTRACTS.yaml (same family as the `_LOBE_AXIS_SCHEMA*`/schema-constant
  note in librarian_knowledge). Rename in code => touch both surfaces.
- The retained sentence "No arc-length map is needed" is now paired with the
  cusp-adapted u map in both surfaces. If a future build adds an arc-length or
  s-map to exterior-polar, that sentence breaks.
- DIFFERENT from wedge/lobe: the exterior-polar `theta_to_u` is OPTIONAL
  (parity == -1 charts are raw-theta, written conditionally, loaded as None),
  NOT REQUIRED like wedge v3 / lobe v1. A mechanical copy of the wedge/lobe
  "REQUIRED, read unconditionally, KeyError on absence" phrasing would have
  been WRONG. When syncing a cusp-adapted contract, verify required-vs-optional
  per chart kind before writing the sentence.

**Surprises**:
- The TODO fragment's acceptance ("4x4x4 probe ~70 charts not 500", "cusp-vertex
  tile clears eps bar") is a BULK-TRAINING sweep — not measured in-build (never
  in a build per AGENTS.md). Closed the TODO but recorded the two training-scale
  items as driver post-build verification in the completed.d fragment.
- sync_derived_docs.py again flagged the same four test-only consumers of
  `LensAmplificationSurrogate.load`. Escalation fragment
  `todo.d/surrogate_contract_test_consumer_warning.md` verified still OPEN —
  not duplicated.
- render_fragments.py version-numbering quirk observed again: my new
  `2026-08-08_exterior_polar_cusp_axis` fragment (alphabetically earlier on the
  same date) took the LOWER version (spec 0.35.0, contracts 2.0.0) while the
  chronologically-earlier lobe fragment rendered at the higher version
  (spec 0.36.0, contracts 3.0.0). Flagged, not fixed — known behavior.
