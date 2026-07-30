# Inspector Short-Term Observations

## 2026-07-29 — Build 1c THIRD PASS (analytic _cusp_vertex + y''' cascade) — PASS

Scope: uncommitted worktree cogwheel-claude-dev. Prod: _pearcey_cusp.py
(_cusp_vertex rewrite), geometry.py (_CausticCascade NamedTuple +
_caustic_cascade helper + caustic_third_derivative). Tests:
test_lensing_airy_fold.py, test_lensing_caustic_derivatives.py. SPEC.md
NOT touched (INS-1c-001 still open).

### Fresh evidence this pass (not carried)
- MATH re-derived by hand from scratch this pass, all correct:
  * u''' = 8 e s + b*d''' with d''' = 32e²N/d - 24e⁴c4 N/d³ - 24e⁶N³/d⁵,
    N=sc. Derived d'' = -4e²c4/d - 4e⁴N²/d³ then D() → exact match to code.
  * r''' = r(-15 u'³/8u³ + 9 u'u''/4u² - u'''/2u): verified via f=u^{-1/2}
    successive derivatives. Exact match.
  * y''' full 10-term triple-product multinomial (coeffs 1,3,3,3,6,3,1,3,
    3,1): all present & correct; p'''=-lam u''', T'''=(sinθ,-cosθ)/(−cosθ)
    tangent chains correct for both components.
- Independent central-difference of caustic_derivatives' y'' vs analytic
  caustic_third_derivative at 5 configs incl. both saddle branches:
  relerr 5.8e-11/1.2e-10/1.4e-11/5.2e-12/1.2e-11 (FD floor). Correct.
- Suite: pytest both files → 88 passed, 7 skipped, 2 xfailed, 23.4s.
  Matches prior two passes.
- Imports OK; geometry.caustic_third_derivative + _CausticCascade present;
  NamedTuple imported (typing L75).
- Caller: find_referencing_symbols → single prod caller _pearcey_cusp:720,
  6-arg `_cusp_vertex(gamma,beta,kappa,source,nearest.theta,branch)` matches
  def sig. `source` param UNUSED in analytic body (also unused in retired FD
  body — PRE-EXISTING, not new; seed_theta carries the proximity). Not flagged.
- caustic_third_derivative: only test consumers (independent oracle_third_
  derivative via mpmath.diff on _oracle_y_component, AST-guarded non-circular
  by OracleIndependenceTestCase). No production consumer — ships for the
  cascade/API completeness, per brief.
- _cusp_vertex wedge-edge refusal logic checked: candidates {center,
  center±theta_max}; refuse if |nearest-center|>0.5·theta_max (i.e. nearest
  is a diverging deltoid edge). Twin gates (a) upward g sign change, (b)
  speed(root) < 1e-4·off-cusp scale. Sound.

### Carried-open finding (NOT resolved)
- INS-1c-001 (trivial, Librarian doc-sync): SPEC.md row 54 enumerates the
  cascade as caustic_derivatives/caustic_speed/caustic_curvature_radius/
  fold_opening_direction — still OMITS the new public
  caustic_third_derivative. List is representative not exhaustive
  (critical_point/macro_matrix/r_caustic also unlisted), so SPEC is NOT
  false — completeness gap only. Confirmed by grep of the actual row this
  pass. Verdict stays PASS.

### Non-issues checked & cleared
- SPEC row 54 certifier names test_lensing_caustic_cusps.py; that file
  EXISTS alongside the changed test_lensing_caustic_derivatives.py (both
  found via find_file). No divergence.

## Pattern reinforced
- Re-deriving the math by hand each pass (even on a stable diff) is cheap
  and catches nothing-here but proves the y''' Leibniz coefficients — the
  FD cross-check alone would miss a wrong-order term the oracle also has.
