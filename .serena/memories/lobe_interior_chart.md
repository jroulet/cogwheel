# Lobe interior chart — cusp-adapted coordinate design record (2026-08-08)

Design record for the lobe cusp-adapted coordinate build (salvaged commit
b18e6a8 after inspector quota death; final feature commit 98c4e7f).
Canonical detail lives in the agent knowledge files; this memory is the
index/pointer:

- `mem:architect_knowledge` (Lobe cusp-adapted coordinate build section) —
  design decisions, u-midpoint subdivision, carve-out retirement,
  INS-3-001 trap (u never activated without the shared `_lobe_nearest_cusp`
  helper), QUOTA-DEATH SALVAGE PATTERN.
- `mem:professor_code_observations` (LOBE CUSP-ADAPTED COORDINATE PHYSICS) —
  A3/2/3-exponent physics, smoothing verification, u-midpoint subdivision
  correctness, schema hard-refuse, and the non-blocking
  `_chart_from_npz`/`_chart_to_npz` asymmetry concern.
- `mem:coder_knowledge` (LOBE CUSP-ADAPTED COORDINATE IMPLEMENTATION) —
  theta_to_s -> theta_to_u rename, `lobe_caustic_relative_v1` schema
  contract (sole known tag, old tags hard-refuse), `_lobe_cusp_axis_map`,
  `_lobe_nearest_cusp` single-source wiring, INS-4-003 sibling-insertion
  clobber.
- `mem:test_dev_knowledge` (LOBE U-COORDINATE MIGRATION TEST PATTERNS) —
  migration, new test classes, measured rho_lobe/eta and smoke-vs-
  production facts.
- `mem:inspector_knowledge` (QUOTA-DEATH SALVAGE RE-AUDIT) — salvage audit
  invariants (single schema tag, zero theta_to_s in lobe paths, cusp
  threading through all tiers).
- `mem:foreman_knowledge` (CARRY-FORWARD DOC-FINDING GREP LESSON) —
  INS-4-001 exact-needle docstring grep lesson.

Durable summary: `LobeInteriorChart`'s sqrt-edge angular coordinate
(s = sqrt(span) - sqrt(theta_max - theta), A2-fold-designed, WRONG for the
deltoid's A3 cusps) was replaced by the cusp-adapted u = d**(2/3), d the
angular distance to the nearest deltoid cusp vertex — the universal A3
fold-cusp caustic-reach scaling (r_deltoid ~ const - c*d**(2/3)), mirroring
the `InteriorWedgeChart` v3 pattern. `lobe_caustic_relative_v1` is the ONLY
known lobe schema tag: both old tags (raw-theta V1, sqrt-edge) hard-refuse
at load and `theta_to_u` is read unconditionally (absent map hard-refuses).
`_lobe_child_boxes` splits at the U-MIDPOINT mapped back to theta_local
(NOT the raw theta midpoint). `_LOBE_CUSP_EXCLUSION_DISTANCE` retired: the
eta_max nearest-caustic-distance admission test alone excludes near-cusp
tiles. Open non-blocking note: the cusp_angle=None raw-theta fallback
builds charts that cannot survive an NPZ round-trip (`_chart_from_npz`
reads `theta_to_u` unconditionally while `_chart_to_npz` writes it only
when not None).
