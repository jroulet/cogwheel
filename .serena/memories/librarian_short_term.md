# Librarian Short-Term Observations

## 2026-08-07 post-commit sync (commits d711934, 67338d6, 2704723)

**Scope**: spec: record driver probe findings + correction (wedge coordinates WORK,
NaN was probe-config artifact); scripts: probe_wedge_v3.py re-run at production gamma.

**Changed files in scope**:
- `.claude/spec/completed.d/2026-08-07_driver_probes_exterior_wedge.md` — corrected
  (wedge coords valid; NaN was giant gamma_band_halfwidth=0.48 probe-config mistake;
  re-run at 0.04 shows good eps; residual NaN is known carrier-flip fragment)
- `.claude/spec/COMPLETED.md` / `.claude/spec/TODO.md` — regenerated
- `.claude/spec/todo.d/lensing_exterior_recursion_never_measured.md` — deleted (done in 2704723)
- `.claude/spec/todo.d/lensing_exterior_should_chart_in_polar_not_sd.md` — depends_on
  updated from deleted fragment to `2026-08-07_driver_probes_exterior_wedge` (done in 2704723)
- `scripts/probe_wedge_v3.py` — scripts-only, no-op per SCRIPTS/ REWRITE NO-OP RULE
- Memory files — no doc surfaces

**Result**: no doc surfaces stale. All three commits are no-ops for Sphinx/RST/SPEC.md/
DATA_CONTRACTS.yaml. FINDINGS.md has no wedge-probe-invalid entry to retract (correction
was confined to the completed.d fragment). sync_derived_docs.py: same four pre-existing
`lens_amplification_surrogate` test-consumer warnings (already escalated via
`surrogate_contract_test_consumer_warning.md`); "auto-fixed" message was the known
internal-state-flush no-op (trust git diff, not the script message).

**Pattern noted — TREE-GATE-STRANDING VARIANT CONFIRMED**: the previous librarian
session (d711934/67338d6 sync) wrote its librarian_short_term.md and doc fixes
(deleted lensing_exterior_recursion_never_measured.md, updated depends_on) but DID NOT
commit them. The driver then ran git add -A when committing 2704723, which swept in the
doc-fragment changes WITHOUT the memory file — leaving librarian_short_term.md as the
only uncommitted diff entering THIS session. This is the exact pattern described in
librarian_knowledge under "TREE-GATE-STRANDING + git add -A MISLABELING VARIANT".

**Fragile cross-references to watch**:
- `lensing_exterior_should_chart_in_polar_not_sd.md` depends_on includes
  `2026-08-07_driver_probes_exterior_wedge` — if that completed.d fragment is renamed,
  the reference breaks silently.
- `lensing_wedge_centre_carrier_flips_in_gamma.md` is the open fragment for the
  astroid-centre carrier-flip; the completed.d fragment references it correctly as still-open.
- `lensing_exterior_followup_four_items.md` and
  `lensing_d2_fold_unexploited_in_three_of_four_regions.md` both depend on
  `lensing_exterior_should_chart_in_polar_not_sd` — will remain blocked until polar
  recharting ships.
