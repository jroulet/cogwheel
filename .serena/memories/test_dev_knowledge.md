# Test Dev Long-Term Knowledge

- Premise repair, not tolerance repair: if a fixture assumes a wrong
  physical premise (e.g. F->1 at w->0 under shear), fix the fixture to a
  case where it holds; keep the inconvenient original as a companion
  test PREDICTING its nonzero offset via an independent closed form.
- If the WP under test never landed (blocked), write an honest contract
  suite — never fabricate tests against a phantom method. Guard the
  planned API with an @expectedFailure hasattr test: it flips to
  unexpected-success (RED) the moment the API lands, forcing the update.
- To inject a buggy/old variant without editing source, mock.patch the
  MODULE GLOBAL the function resolves internally. Test-side
  reproductions of production helpers break when production signatures
  drift (e.g. a new 4th arg) — align arity. Neighbor-suite reds from
  such drift belong to the owning run: report, don't touch.
- Extend AST/name-forbidding guards for every new mutation/oracle
  helper (oracle independence, F002); pure-mpmath oracles for phase
  gates.
- For rules differing only in edge cases, assert a sub-case where old
  and new logic must agree bit-for-bit as a cheap regression control.
- Fully revert probe/mutation edits; verify by read-back plus a pattern
  search for residue.
- Shell gate: prefer the plainest command shape (`python -m pytest
  <file> -q`; heredocs/`python -c`/pipes may be denied); run from the
  WORKTREE root; retry a bare unexplained denial once, but a denial
  WITH a stated reason binds.
- Falsification under numba: patch through the FULL .py_func chain
  (F010); define "gate RED" as (refusal raised) OR (error > tol). Test
  refusals at the production default operating point, not the
  accuracy-study setting.
- When a plan-anticipated gate exposes a production shortfall, leave it
  RED/xfail (no tolerance widening, no production edits); pair with a
  green converged positive control so the falsification is non-vacuous.
- Gate each pipeline path and each component at its own numerical floor;
  an aggregate downstream gate can pass while a component gate fails —
  keep both.
- Prefer machine-independent structural timing gates (speedup ratio,
  component subdominance); absolute ms ceilings only as
  arithmetic-derived secondary guards (expect machine-dependent xfail).
- np.exp(1j*x) range-reduces large args ACCURATELY — float64 phase loss
  lives in the w*tau MULTIPLICATION. Phase-loss demos need irrational-
  scaled factors (e.g. pi*1e6, e*1e6) so the product carries >53 bits;
  exact-power-of-ten products are exactly representable and show none.
- If a gate's claimed band (e.g. thousands of radians of phase) is
  unreachable from realistic fixtures, build SYNTHETIC inputs that
  actually hit the band, checked against an independent oracle.
- When probing an internal path exposed only through reduced outputs,
  prove your reproduction reduces bit-identically to production first;
  assemble the independent observable from shipped components via a
  documented identity, not the function's own convenience return.
- ChangRefsdalChannels requires a >=2-point strictly-increasing
  positive w grid (np.unique/sort probes; no scalar fixtures).
