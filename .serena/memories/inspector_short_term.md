# Inspector Short-Term Observations

## 2025-08-03: Build ppgo_interior_handoff review (RE-REVIEW #2)

### Scope
WP1: Add a fold-ppGO serve path for interior draws (rho < 1) above the
InteriorWedgeChart w-ceiling. When the chart declines (w > DD cap) and
the fold pair's Airy parameter ξ_min >= 4.0 AND the per-pair uniform
error estimate <= CERTIFICATION_BAR (1e-4), serve via fold_ppgo_correction
instead of falling through to exact quadrature.

Files changed:
- cogwheel/lensing/likelihood.py (new _XI_FOLD_THRESHOLD=4.0, new
  fold-ppGO handoff block inside _surrogate_coefficients after the
  `rho <= 1.0 or not born_chart.covers(...)` condition)
- cogwheel/lensing/surrogate_census.py (new _XI_FOLD_THRESHOLD=4.0,
  new ppgo_fold census classification block between select_chart and
  classify_fallthrough)
- cogwheel/tests/test_lensing_fold_ppgo_handoff.py (new test file,
  17 tests: 14 pass, 3 skipped behind COGWHEEL_TRAIN_TIER)

### Findings
- **PASS** — Implementation is correct and complete.
- Gate structure is conservative: xi coarse gate → fine error-estimate
  gate → serve. Multiple structural refusals via _merging_fold_pair
  returning None, delta_tau <= 0, images not found, error_est > bar.
- Reconstruction mirrors the Born rung exactly: demodulate, extract
  envelope, reconstruct_farfield with FARFIELD_KERNEL_SUM.
- fold_ppgo_correction returns in ABSOLUTE frame; correctly demodulated
  by exp(-1j*dense_w*geom.t_min) to min-relative frame.
- All exception types (LensDomainError, ValueError, ZeroDivisionError)
  are caught, matching the inspector knowledge pattern.
- beta=0 and kappa=0 guaranteed by upstream guards in _surrogate_coefficients.
- Census code correctly restricts to image_count==4 (astroid interior
  or saddle lobe interior — physically correct for fold pairs).
- _XI_FOLD_THRESHOLD duplicated between likelihood.py and census.py
  with explicit mirror comment — follows established DD_PRODUCT_MARGIN
  pattern.
- SPEC.md does NOT describe this new fold-ppGO interior handoff serve
  path (it says draws above the DD ceiling fall through to exact). This
  is expected for a new feature — Librarian scope to update.
- No DATA_CONTRACTS impact (runtime serve decision, no new artifact).
- All existing test suites pass (test_lensing_born_residual_wiring 34/34,
  test_lensing_ghost_gate 18/18, test_lensing_surrogate 69/69,
  test_lensing_surrogate_census 14 pass 13 skip, test_lensing_ppgo_bandsplit
  62 pass 4 skip, new test 14 pass 3 skip).
- Census MECE invariant preserved: ppgo_fold records have served=True,
  so they don't enter the fallthrough bucket.
- New test suite has good coverage: accuracy vs exact engine (skipped
  in fast tier), xi gate refusal geometry (fast), round-trip identity
  (fast), self-falsification (fast), error-estimate fine gate (fast),
  census integration (fast), default-path-unaffected (fast).
- `rho is not None` check at line 1688 is defensive-correct (rho cannot
  be None at that point since the except block returns early). Harmless.
- Saddle-lobe interior (gamma > 1) case: fold-ppGO correctly falls through
  because _merging_fold_pair returns None when no (Morse 0, Morse 1)
  adjacent pair exists. For saddle-lobe configs that DO have a fold pair,
  the correction is valid and will serve correctly.
- Phase convention verified: fold_ppgo_correction returns absolute-frame
  total, demodulated to min-relative by exp(-1j*w*t_min), envelope
  extracted by exp(+1j*w*t_min), fed to reconstruct_farfield which
  internally re-modulates via _frame_phase. Consistent with the Born rung.
- External callers (Born rung and fold-ppGO block) both use raw `w*t_min`
  for pre-call demodulation, NOT _frame_phase. This is consistent — the
  _frame_phase helper is internal to reconstruct_farfield.

### No new bug patterns discovered.
### Open issues carried forward:
- INS-1-001: SPEC.md does not describe the fold-ppGO interior handoff
  serve path. Librarian scope, trivial doc debt, not a code defect.
  STILL OPEN — SPEC.md has zero mentions of fold_ppgo/ppgo_fold/interior
  handoff (confirmed by grep).
