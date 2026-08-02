# Inspector Short-Term Observations

## Build 5 C8 — Far Zone Becomes Caustic-Relative (2026-08-02, pass 7)

### Scope
Re-reviewed (pass 7, same diff): Annulus retirement (delete ANNULUS_INNER_RADIUS,
GAMMA_FENCE, saddle fence 1.0502342; rename annulus_rho → caustic_rho; re-express
census classify_fallthrough in caustic-relative coordinates).

### Findings
1. **INS-5-001 STILL OPEN (flag to Librarian)**: SPEC.md lines 53 (engine table)
   and 97-137 still reference the old annulus (far annulus 3.0<|y|<=4.2426,
   gamma<3/4 exterior fence, saddle_caustic_max_y, serving band 1.0502342<gamma<3,
   "three guards", "shared closed-form fences" in census, "Born far-annulus carrier").
   Code correctly deletes these; SPEC needs sync.

2. **INS-5-003 STILL OPEN (flag to Librarian)**: DATA_CONTRACTS.yaml line 228
   uses 'caustic-frame annulus rho' and 'annulus radius' terminology; code now
   uses 'caustic-relative rho'. Functionally identical definition
   (rho = |y|/caustic_reach), purely cosmetic.

### Verdict
PASS. All changed code is correct. Rename is complete (zero residual `annulus_rho`
in .py). Tests pass: test_lensing_born (53), test_lensing_ppgo_map (37),
test_lensing_ppgo_bandsplit (62+4skip), test_lensing_surrogate_census (14+13skip).
born_gate correctly reduced from 3 guards to 2. Census classify_fallthrough
correctly uses caustic_rho > 1 on both parities (with proper exception handling
for degenerate gamma). No production impact (serve slot is NOT wired). Edge
cases verified:
- gamma=0.80 (positive, formerly fenced): now admitted by born_gate.
- gamma=0.90 (positive, formerly fenced): now admitted by born_gate.
- gamma=1.04 (saddle, formerly under fence root 1.0502342): now admitted.
- gamma=0.998 (parity wall): correctly refused by Guard B.
- gamma=1.003 (parity wall, saddle): correctly refused by Guard B.
- CENSUS_NONANNULUS_Y1_EIG reduced from 2.0 to 0.5 (rho=0.41 < 1 at gamma=0.45).
- SADDLE_CENSUS_NONANNULUS_Y1_EIG reduced from 2.0 to 1.0 (rho=0.62 < 1 at gamma=1.2).
- C8FenceRetirementTestCase (new, 3 tests) correctly pins the fence deletion.
- CausticRelativeClassificationTestCase (new, 5 tests) correctly pins rho>1
  on both parities.
- All `caustic_rho` consumers use correct (gamma, |y|, kappa) arg order.
- Pipeline graph confirms all certified_ppgo_map consumers are intact.
- Exception handling in census: (ValueError, LensDomainError) -> rho=0.0
  is correct — gamma-guard fires first for degenerate gammas, so the reach
  exception path is only reachable for very rare edge cases.
- All imports verified successfully.

### Convention / pattern learned
- gitignored auto-generated doc stubs do NOT constitute a committed staleness
  finding — they regenerate from the live module on the next Sphinx build.
