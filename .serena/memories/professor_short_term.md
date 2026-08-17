# Professor short-term (F083 tube delay-uniform axis — inference REVIEW, 2026-08-17)

Reviewed the delay-uniformized tube angular coordinate build (graduation of the
tube 4th/angular axis from arc-length s to fold-delay TV s').
Test file: `cogwheel/tests/test_lensing_tube_nyquist_coordinate.py`
(46 tests, all PASS in ~19s on cogwheel-newlal). Fast, engine-mostly-mocked.

## Verdict: CONCERN (invariants correct; deliverable's reason-for-being unmet)

### The 7 pinned invariants are physics-correct and pass
1. DRY delay-equality: s' reconstructed from oracle Δτ=0.5(τ₋−τ₊) matches
   shipping to 1e-12; builder calls `_merging_fold_pair` once/node. The
   DT/2 convention with Nyquist /(2π) is SELF-CONSISTENT (Ω=(w/2)|dDT/dθ|
   =w|dΔτ/dθ|).
2. Monotone TV: s'=cumtrapz(|dΔτ/dθ|) strictly increasing while Δτ folds
   2-to-1 (interior turnover). Correct — signed Δτ would be non-invertible.
3. Nyquist count: ceil(8·w·TV/(2π)) capped at n_theta_cap=32; saturates at
   cap under 10×/1e5× w. PPP=8. Matches spec.
4. A3 cusp slope: Δτ~d^{2/3} near cusp; rejects slope=1 (arclength) & 1/2.
5. Stale-schema refusal: `tube_delay_tv_v1` validated BEFORE map read; absent/
   old `theta_to_s` layout hard-refuses ValueError; no identity fallback.
6. `_heldout_eps` unserved-as-coverage: unserved-but-referenceable folds in as
   miss (eps=max(eps,1.0)); nan ONLY on zero served; float preserved. Correct.
7. Census no-explosion: engine-free Nyquist ceiling; EXPLOSION/SILENT_EMPTY/
   IN_BAND with mpmath+engine booby-trapped (never hit).

### Two material CONCERNS (both independently verified this session)
A. F083 acceptance HALF-TESTED / accuracy target UNMET. Spec demanded
   N<48 AND held-out eps≤0.0237. Only N<48 is pinned and it is TAUTOLOGICAL
   (cap=32<48 by construction). Accuracy half is NOT run as a gate; the build's
   OWN measurement (test docstring) is eps≈0.145 at 30 nodes (6× over 0.0237)
   and ≈0.56 at the 4-node floor. So on the buildable sub-arc the delay-uniform
   axis does NOT reach target accuracy — node economy rests on an unmet premise.
   Full build+engine eval ~171s => operator-deferred, but numbers point to FAIL
   of the accuracy target, not merely "unverified".
B. CONFIRMED production crash. Full cusp-to-cusp `_tube_delay_map` raises
   "not strictly increasing" for EVERY band probed (γ∈{0.2,0.4,0.6,0.8}):
   near-cusp tails reach Δτ~1e-7, `_fill_cusp_tails` clamps to flat s'. And
   `_load_or_build` guards only artifact LOADING, NOT `build_fn()` — so
   production `_train_band_charts` would crash on the first astroid tube arc.
   Tests honestly pin invariants only on a hand-trimmed servable sub-arc and
   flag this for the driver. (`surrogate_training.py` ~L3132, ~L4017, ~L5184.)

Reason-for-being (fewer nodes at target accuracy, buildable in production) is
NOT demonstrated. The coordinate math itself is correct and well-pinned.
