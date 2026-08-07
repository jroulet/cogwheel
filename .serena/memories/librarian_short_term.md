## Run: 2026-08-07 — polar re-chart librarian pass (range 4d59a6d..5859a78, HEAD 5960ceb)

**Scope**: Full doc-surface update for the polar re-chart build (ExteriorPolarChart replaces FarFieldChart; charts in polar (rho, theta_c) instead of (s,d); ~1064 lines deleted in 0a31fcf; saddle exterior now chartable in polar; cusp carve-out 0.2 y-units; m_lens_range override; _KNOWN_ENVELOPE_DEFINITIONS widened).

**Changed**:
- `changelog.d/2026-08-07_polar_rechart.md` (NEW) → rendered into CHANGELOG.md under the existing 2026-08-07 section (next to m_lens_range entry).
- `.claude/spec/SPEC.md`: GLOBAL MULTI-CHART ARTIFACT now names `ExteriorPolarChart` + (rho, theta_c) explicitly; Key-abstractions contract now covers BOTH parities (saddle-exterior fall-through sentence replaced with additive scalar-reach rho).
- `.claude/spec/spec_changelog.d/2026-08-07_polar_rechart.md` (NEW, bump: patch) + `.claude/spec/contracts_changelog.d/2026-08-07_polar_rechart.md` (NEW, bump: patch).
- `.claude/spec/DATA_CONTRACTS.yaml`: lens_amplification_surrogate — removed "exterior positive-parity only" and the "Macro-saddle ... remain exact-engine fall-through" sentence; clarified rho is additive exterior / multiplicative only on astroid interior arm.
- Closed `todo.d/lensing_exterior_should_chart_in_polar_not_sd.md` → `completed.d/2026-08-07_polar_rechart.md` (NEW, cites 0a31fcf + 5859a78).
- Repointed `[[lensing_exterior_should_chart_in_polar_not_sd]]` → `[[2026-08-07_polar_rechart]]` in 3 todo.d fragments (saddle_forensics, exterior_followup_four_items, d2_fold_unexploited) AND in the already-completed `2026-08-07_driver_probes_exterior_wedge.md` (prose + depends_on).
- Ran `render_fragments.py`; `--check` clean, zero dangling links.

**Surprises / patterns**:
- The "Macro-saddle exterior ... exact-engine fall-through" sentence in SPEC.md AND DATA_CONTRACTS.yaml was STALE versus the code (`_build_farfield_chart` docstring: both parities chartable; `_to_caustic_fixed` handles gamma>=1 with additive scalar reach). The user's brief listed "saddle exterior now chartable in polar" as a major change — the doc sentences had survived the 337ac15 sync that introduced ExteriorPolarChart. This is the recurring "SPEC STATUS SENTENCE STALE SILENTLY" pattern: the 337ac15 sync updated the class name but preserved the old parity restriction verbatim.
- SPEC_CHANGELOG version quirk reconfirmed: my new spec fragment rendered at `0.11.7` (no date field → empty date bucket), while the top version stays 0.34.0 (wedge-axis fragment carries `date:`). Patch-bump fragments without `date:` sort into the empty-date bucket — harmless, known quirk.
- No `docs/source/` staleness: FarFieldChart zero hits (generated/ is gitignored autosummary output; api.rst uses :recursive:).
- `render_fragments.py` had NO stray tidy_advisory/foreman_lite side effects this run. `.claude/agent_state/librarian.json` diff is the post-commit hook's ambient state (last_commit 5960ceb = HEAD), not renderer output — left untouched.

**Fragile cross-reference to watch**: `lensing_farfield_sd_coordinate_degenerates` (todo.d) is superseded by the closed polar fragment ("Supersedes the open direction in [[...]]") — it remains open as the measurement record; do NOT close it without confirming the sd-coordinate measurement acceptance criteria are met. Also `lensing_farfield_name_spans_three_regimes` still open (rename deferral).
