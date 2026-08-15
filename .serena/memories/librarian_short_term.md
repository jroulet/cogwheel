## 2026-08-15 run: lobe_cusp_axis_edge_tolerance (WP1) doc-sync audit

Scope: diff = `_lobe_cusp_axis_map` edge-coincidence-tolerance fix
(`_CUSP_EDGE_COINCIDENCE_ULPS = 8` guard relaxation, keep-map semantics) in
`cogwheel/lensing/surrogate.py` + new test suites in
`test_lensing_surrogate_lobe.py`. Handoff
`.claude/handoff/lobe_cusp_axis_edge_tolerance.md` explicitly gated the
doc-impact question: "[housekeeping] unless the semantics choice moves a
documented boundary -- then [-> spec]".

VERDICT: genuine no-op for all doc surfaces. Checked:
- SPEC.md's lobe-chart paragraph (~line 61-67) describes theta_to_u's
  REQUIRED-at-load / cusp-adapted-u mechanism but never asserted a strict
  (non-tolerant) cusp-vs-edge inequality -- there was no claim to
  contradict. The new ULP tolerance is pure guard-arithmetic, fully
  self-documented in the function's own docstring + a comment on the
  constant (float-noise band, 2.8e-17 observed gap, 8 ULPs comfortably
  covers it without absorbing a real geometric offset) -- exactly the
  kind of implementation-level detail that belongs in the module
  docstring, not SPEC.md (per the standing "SPEC carries mechanism, not
  provisional values" rule).
- DATA_CONTRACTS.yaml's lobe_caustic_relative_v1 description already only
  describes theta_to_u's required/optional-ness and construction site
  (`_lobe_cusp_axis_map`), not boundary-inequality strictness -- untouched,
  correctly so.
- docs/source/ has ZERO references to "lobe" or any cusp_axis internals
  (surrogate lensing internals aren't part of the public Sphinx narrative)
  -- confirmed via grep, no rebuild needed.
- Inspector's own review (read via inspector_short_term) independently
  concluded "No SPEC/DATA_CONTRACTS impact ... Nothing for Librarian from
  this build" -- concurring verdict from the accuracy-owner.
No todo.d fragment existed for this work (it originated from a same-build
smoke-crash handoff, not a backlog item), so there is nothing to move to
completed.d either; a private-helper guard-tolerance bugfix does not meet
the "new pipeline step / changed public interface / new core logic" bar
that triggers the full spec-bump workflow. Left as-is.

## Carried-forward item RESOLVED (verify before re-flagging)

INS-1-002/003 ("exterior_polar_rho_log_carrier_v1 'ONLY known tag' stale
since V5 2D fold-carrier shipped") -- Inspector's short-term memory still
lists this as carried-forward for Librarian, but SPEC.md line ~61-62 and
DATA_CONTRACTS.yaml line ~198 BOTH already correctly describe the two-tag
V4/V5 set (_EXTERIOR_POLAR_AXIS_SCHEMA_V4 retained-for-back-compat /
_V5 current-write-tag) and the 2-D (n_rho, n_theta_c) rho_u_carrier array
with the 1-D-broadcast backward-compat note. This was fixed by some
earlier librarian pass between 2026-08-10 and today without a memory
update recording the closure. Do NOT re-fix; if Inspector re-flags this
exact pair again, point back to this note and re-verify with a fresh grep
rather than assuming staleness from memory.
