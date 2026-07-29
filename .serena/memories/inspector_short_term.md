# Inspector Short-Term Observations

## 2026-07-29 — Build 1a (F038) analytic caustic cascade — RE-REVIEW pass 3 (diff byte-identical to pass 2) — ISSUES (1 trivial carried: INS-1-001 doc-sync)

Scope: uncommitted working tree. geometry.py +195 (unchanged from pass 2);
test_lensing_caustic_derivatives.py still untracked. 4 new PUBLIC fns after
r_caustic: caustic_derivatives, caustic_speed, caustic_curvature_radius,
fold_opening_direction. Diff is the SAME +195 as pass 2 (d_root==0 wedge-edge
guard present). Per standing rule, re-ran suite + re-derived math by hand anyway.

### RE-VERIFICATION THIS PASS (all fresh, not trusted from prior pass)
- pytest test_lensing_caustic_derivatives.py: 20 passed (4.89s).
- MATH re-derived with independent from-scratch parametric-curve FD oracle
  y(theta)=p_i r T_i: pos-parity & saddle y' err 2e-11..1.5e-10, y'' err
  1.5e-4..3.5e-4 (= eps/h^2 roundoff floor, h=1e-6). PASS. caustic_speed
  delegate == hypot(y') to 1e-12.
- Out-of-wedge case (g=2,k=0.3,th=0.3; 0.7/2=0.35 < sin0.6=0.56) correctly
  REFUSES. Wedge-edge (g=2,k=0,th=0.5*arcsin(1/eff)) refuses under
  -W error::RuntimeWarning with NO divide-by-zero warning → INS-1-002
  still resolved.
- fold_opening_direction: independent lens-map y(x)=Ax-x/|x|^2 FD D2y[e,e]
  vs served: |dot|=1.0 (err 2e-14, 7e-15). PASS. e->-e invariance confirmed
  by construction (xe^2, 4xe*e).
- CONSUMERS: grep geometry.<fn> across cogwheel/**/*.py → ONLY the test file.
  _pearcey_cusp.py:476 caustic_speed = NESTED LOCAL def (not geometry's);
  test_lensing_fast_path._caustic_speed = unrelated classmethod. Additive/
  unwired; no caller breakage.
- Test oracle NON-CIRCULAR: primary gate compares to independent
  oracle_derivatives; line 417 `real=geometry.caustic_derivatives` is a
  MUTATION test (scales y' by 1.01, asserts gate raises) — has teeth.
- git diff .claude/spec/ EMPTY.

### FINDINGS
- INS-1-001 STILL OPEN (trivial, doc-sync): plan listed SPEC.md +
  todo.d/lensing_analytic_derivatives.md as expected-to-change; spec diff
  empty; 4 new public fns absent from SPEC. FLAG TO LIBRARIAN (spec-code
  divergence / doc sync), NOT a Coder code defect.
- INS-1-002 RESOLVED (confirmed 3rd time): saddle wedge-edge divide-by-zero
  guarded; raises LensDomainError before u_p/u_pp compute.

No NEW findings. Code correct; only doc-sync remains open.
