# Architect Short-Term Observations

(## Lobe cusp coordinate build (2026-08-08)
- Plan designed for brief_lobe_cusp_coordinate.md @ fd84cea
- Inputs: Simplifier (trim _lobe_cusp_angles duplicate, keep WPs 1-4 split, u-midpoint subdivision gap, schema naming), Professor (2/3 exponent universal, 1e-3 bar, no rho axis change, tube-shell sufficient)
- Key decisions: thread cusp angles through tile dict (wedge axis_origin precedent), _lobe_cusp_axis_map mirrors _wedge_cusp_axis_map, schema `lobe_caustic_relative_v1` hard-refuses both old schemas, retire identity (theta_to_u=None) path
- WP-1: _lobe_cusp_axis_map + training thread + u-midpoint subdivision
- WP-2: LobeInteriorChart theta_to_s→theta_to_u migration
- WP-3: from_lobe_engine s→u coordinate
- WP-4: retire _LOBE_CUSP_EXCLUSION_DISTANCE
- ~30 test references need Test Developer migration — largest scope item)
- TRIAGE INS-3-001 (coder_fix, 2026-08-08): lobe build's cusp-adapted u NEVER activated in production. build_lobe closure + _subdivide_lobe_tile.build_child call _build_lobe_chart WITHOUT cusp_angle/cusp_side; from_lobe_engine raw-theta fallback wins; _build_lobe_chart docstring says \"unused until WP-3\". _lobe_child_boxes u-midpoint split IS wired; tile dicts carry lobe_cusps (inherited to children by _subdivide_tile). Fix: derive nearest-cusp+side via SHARED helper (single-source with _lobe_child_boxes) at both build sites.
