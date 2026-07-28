Librarian run — 2026-07-28 (INS-10-001 doc-sync, saddle Born carrier)

Scope: single Inspector finding INS-10-001 (trivial): SPEC.md's Born-rung
Conventions bullet (lines 88-112) still described positive-parity-only
serving after commit 31ee133 landed macro-saddle serving in _born.py,
channels.py, and surrogate_census.py. Foreman-Lite twice explicitly
declined to touch SPEC.md (correctly — it's Librarian-owned) and left the
finding for me. Confirmed via git diff that SPEC.md itself had zero changes
since 31ee133 while the three source files gained ~150 lines of saddle
machinery.

What I fixed:
- SPEC.md Born bullet: rewrote to describe BOTH parities served —
  positive-parity minimum image (unchanged gamma<3/4 fence) AND macro-saddle
  (det A<0, F026 closed-form `saddle_caustic_max_y` fence, serving band
  1.0502342<gamma<3). Added: the exact Morse phase -1j applied by
  `born_lead_carrier` on the saddle (F024/F009-S, explicitly NOT
  cmath.exp(-1j*pi/2) which injects a ~6e-17 real-part rotation error —
  this distinction is load-bearing in the code's own comment, so I kept it
  in SPEC too); `born_gate`'s three named guards (guard B two-sided
  parity-wall margin, the parity-split exterior fence, guard A band split);
  `channels.born_carrier_from_partition`'s macro-saddle above-split branch
  (pure two-real-image ppGO, complex ghost explicitly REFUSED for det A<0,
  positive-parity path unchanged); and the census's mirroring saddle arm.
  Called out that `born_amplification`/`born_envelope` remain
  positive-parity-only diagnostics (the a0/b1 correction is NOT derived on
  the saddle — I checked this is still true by reading both docstrings,
  which raise BornDomainError on det_a<=0 unconditionally). Bump: minor
  (fragment `spec_changelog.d/2026-07-28_born_saddle_carrier_sync.md`),
  rendered to spec_version 0.24.0 — landed in alphabetical order after last
  session's 0.23.0 fragment, no reordering surprise this time.
- `todo.d/lensing_saddle_born.md`: this fragment's 4-item ordered plan (1.
  b1/a0 derivation, 2. saddle expansion origin+guard, 3. saddle fence, 4.
  wire into likelihood) had items 2 and 3 fully landed this build (tests:
  11 carrier/gate tests + 12 band-split tests + 6 census tests = 29, cross-
  checked against the 3 separate coder change-reports' claimed counts,
  which sum correctly). Rewrote the fragment to mark 1-3 done with specifics
  and keep 4 open, per the standing "multi-part program stays open until
  every part finishes" rule — same pattern as last session's
  lensing_born_b1_derivation.md edit, just the sibling fragment this time.
  Cross-referenced the two TODOs (both converge on the same TRAIN_TIER
  residual-chart wiring blocker) so wiring doesn't get duplicated between
  them.
- changelog.d/2026-07-28_born_saddle_carrier.md: new entry. Caught my own
  arithmetic slip while drafting — first wrote "19 new tests" without
  actually summing the three coder reports (11+12+6=29); recomputed and
  corrected before finalizing. Lesson: always sum test counts from the
  literal per-report numbers, don't eyeball a round total.

What I verified and left alone:
- docs/source/**: grepped for "born"/"Born" — zero hits, confirming (again)
  that no Sphinx page enumerates census categories or Born-rung internals;
  no rebuild needed since nothing under docs/source/ was touched this run.
- DATA_CONTRACTS.yaml: grepped for "born" — zero hits; no disk artifact
  changed (the Born carrier is in-memory analytic, no save/load), so no
  contract entry needed.
- sync_derived_docs.py: ran clean, zero new diff beyond the SPEC/TODO edits
  already made by render_fragments.py; only flagged the same 4 test-file-
  only `lens_amplification_surrogate` consumers as prior sessions — left
  off DATA_CONTRACTS per the standing production-only convention.
- Did not touch .claude/agent_state/foreman_lite.json (pre-existing dirty
  state, not mine, not a doc surface).
- Did not rebuild Sphinx docs this run — correctly skipped per the rule
  (only required when docs/source/ or surfaced docstrings change); this
  run touched neither.

Process note: this was the cleanest possible Librarian trigger — a single
named finding, already fully diagnosed by Inspector down to file/lines, with
Foreman-Lite explicitly deferring rather than guessing at spec prose. No
ambiguity about scope or ownership boundary this time.
