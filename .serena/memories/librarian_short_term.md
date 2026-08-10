# Librarian Short-Term Observations

## 2026-08-10 — Post-commit sync: ghost-exclusion filter (cf81d66)

### Scope
Commit cf81d66 (2026-08-10). Changed files in cogwheel/:
- `cogwheel/lensing/surrogate_training.py` — modified (+100 lines)
- `cogwheel/tests/test_lensing_exterior_admission.py` — new (759 lines, test-only)
- `cogwheel/tests/test_lensing_surrogate_training.py` — new (212 lines, test-only)

### Outcome
Complete no-op. ALL doc sync work was already done by a PRIOR Librarian session
in commit 008cd83 (1 minute after cf81d66), with message:
"docs: post-commit sync (ghost-exclusion filter section in SPEC.md +
rho_phase_carrier completion)"

008cd83 already contained:
- GHOST-EXCLUSION FILTER section added to SPEC.md FAR-FIELD TILING row
  (`_exclude_ghost_dominated`, positive-parity scope, `_GHOST_DECAY_IM_THRESHOLD`
  reuse, gamma-band probe mirroring `_exclude_near_cusp`, `ghost_excluded_tiles`
  counter in region report, DT-8 ref in `test_lensing_surrogate_training.py`)
- completed.d fragment `2026-08-10_exterior_rho_phase_carrier.md`
- spec_changelog.d fragment `2026-08-10_ghost_exclusion_filter.md`
- todo.d fragment `lensing_exterior_rho_phase_carrier.md` deleted
- depends_on in `lensing_exterior_fold_carrier_demodulation.md` repointed to
  `2026-08-10_exterior_rho_phase_carrier`
- librarian_short_term.md updated

My session recreated all these operations, producing zero net diff (files overwrote
with same content, render_fragments.py reported "All surfaces up to date").

### Key pattern learned
CHECK `git log --oneline -5` BEFORE DOING ANY WORK in post-commit mode, not just
the session-start snapshot. When a post-commit sync already ran and committed,
the snapshot's "M tidy_advisory.json" is the only clue — the spec changes were
already committed. Confirmed: running the operations again is harmless but wastes
a full session.

### Sphinx docs
No check needed: surrogate_training.py changes are private functions only
(`_exclude_ghost_dominated`, `_farfield_exterior_tiles` params). No new module,
no public API change, no signature change visible to Sphinx autosummary.

### sync_derived_docs.py result
Clean except the KNOWN recurring test-consumer warnings for `lens_amplification_surrogate`
(4 test-only callers in test_lensing_surrogate.py). Escalation fragment
`todo.d/surrogate_contract_test_consumer_warning.md` confirmed still open.

### Fragile cross-references to watch
- GHOST-EXCLUSION FILTER SPEC paragraph references `_GHOST_DECAY_IM_THRESHOLD` by name
  and `geometry.ghost_kernel` by function — rename in code breaks both surfaces
- DT-8 certification references `test_lensing_surrogate_training.py` — if this
  file is renamed, SPEC.md needs updating
