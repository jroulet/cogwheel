# Inspector Short-Term Observations

## 2026-08-17 — tube angular axis graduation, PASS-2 (re-review), VERDICT: PASS

Scope: re-review of uncommitted working tree for the tube-chart 4th-axis
graduation from arc length `s` to delay-uniformized `s'(theta) = TV(Δτ)`
(schema tag `tube_delay_tv_v1`). This pass fixed the test sweep that PASS-1
flagged as INS-1-001.

### INS-1-001 — RESOLVED (re-verified, did not trust the diff)
- Fast tier fresh run (121 s): test_lensing_surrogate.py +
  test_lensing_low_w_extrapolation.py + test_lensing_wedge_dd_arclength.py +
  test_lensing_tube_nyquist_coordinate.py + test_lensing_log_reach_gamma.py =
  **236 passed, 12 skipped, 0 fail/0 error** (was 9 fail + 3 error in PASS-1).
- Mechanical `theta_to_s` -> `theta_to_s_prime` rename applied at every
  from_values kwarg + `.theta_to_s` attribute access in surrogate.py tests,
  low_w, log_reach (the 6 previously-skipped-latent kwargs now fixed).
- FieldExposureTestCase renamed test_tube_still_exposes_theta_to_s ->
  test_tube_exposes_theta_to_s_prime; now asserts 'theta_to_s_prime' IN and
  'theta_to_s' NOT IN field names. Correct.
- Train-tier `ShippedArcLengthTubeGridTestCase` (checklist-5b strand)
  REWRITTEN and DERIVED, not deleted/loosened:
  * test_shipped_nodes_uniform_under_carried_map: uniformity measured over
    the span the nodes actually cover (s_at_nodes[0]..[-1]) via
    chart.theta_to_s_prime — correct for servable-subrange semantics.
  * test_shipped_endpoints_equal_servable_subrange (replaces
    endpoints_equal_arc_bounds): calls live
    `training._tube_delay_map(rep_gamma, arc, float(chart.eta_max))` and pins
    theta_grid[0]/[-1] == theta_fine[i_lo]/[i_hi]. DERIVED from production,
    not a literal.
  * Old independent-polyline-oracle test dropped with an explicit parsimony
    rationale (invariant now owned by the fast-tier nyquist DRY delay pin).
  * Verified `_tube_delay_map` signature `(gamma, arc, eta_ref, n_map=...)`
    returns 4-tuple `(theta_fine, s_fine, i_lo, i_hi)` — the test's 3-arg
    call + 4-tuple unpack MATCHES.
- Collection clean: test_lensing_surrogate_training.py --collect-only =
  179 collected, 0 errors (class bodies execute at collection; no strand).
- Tree-wide grep `theta_to_s(?!_prime)`: ALL residual hits are legitimate —
  the shared validator FUNCTION name `_validate_theta_to_s` (surrogate.py
  1358, produces the theta_to_s_prime field at 2801, shared numeric core with
  wedge validator), historical comments, NEGATIVE assertions confirming the
  old name is gone, and deliberate stale-artifact hard-refusal tests
  (nyquist 894/898 builds `chart0_theta_to_s` to prove the schema refuses).
  No functional strand remains.

### NON-BLOCKING observations (not flagged)
- surrogate.py `_validate_theta_to_s` is now a mild misnomer (validates the
  delay-TV `s'` map, shared with the wedge cusp-axis validator). Pre-existing
  shared-core name; cosmetic only. Candidate for a future Tidier/Librarian
  naming sweep, not a Coder defect.
- test_lensing_surrogate.py section-header comments still say "ARC LENGTH s"
  though the field is renamed; the helpers use arbitrary non-affine maps to
  test the MAP MECHANISM (any monotone map), so the terminology is stale but
  not a correctness issue.

### Carry-forward -> Librarian (doc-sync, NOT a code defect)
- DATA_CONTRACTS.yaml was in the plan's expected-changed files but is NOT in
  the actually-changed set. Verify SPEC.md + DATA_CONTRACTS.yaml name the new
  tube axis schema tag `tube_delay_tv_v1` and the s'(theta)=TV(Δτ)
  coordinate. Doc surfaces owned by Librarian.

### Production physics (WP1-WP4): verified correct in PASS-1, unchanged this
pass (working tree not committed between passes); fast-tier green re-confirms
behavior. See prior PASS-1 entry notes on Δτ sign, cusp-tail extrapolation,
Nyquist node count, census mirror faithfulness.
