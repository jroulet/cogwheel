# Professor short-term — Build 8b-levers consultation (2026-07-20)

## Session summary
Detailed rulings on the two Build 8b engine micro-accelerations:
- Lever 1: Newton nearest-caustic shortcut in `geometry.nearest_caustic_point`
- Lever 2: weight-vector contraction fusion in `operator.py`

## Key observations

### Lever 1 (nearest_caustic_point)
- `theta` is NOT a physically certified output. Downstream consumers in
  `channels.py` use only `.image` (for label assignment and delay computation)
  and `.distance` (for the caustic_distance field). `.theta` is stored in the
  partition's `critical_theta` field but is metadata/diagnostic — it does not
  feed the likelihood or any gate.
- The `.image` and `.source` fields ARE consumed: `.image` feeds `_assign_labels`
  and `geometry.delay(caustic.image, ...)`, and `.source` is stored but not
  re-consumed for any arithmetic.
- The `critical_point()` call at the winning theta produces the frame fields via
  `np.linalg.eigh(hessian(image, matrix))`, so a ~1e-11 theta perturbation
  propagates SMOOTHLY to `.image` (via cos/sin at ~1e-11 * radius ~ 1e-11),
  thence to all downstream fields. Distance is at a minimum, so the distance
  perturbation is ~(g''/(2g)) * dtheta^2 ~ 1e-20, far below ULP.
- The HEAD_NEAREST_CAUSTIC_PINS test uses assertEqual on BOTH theta and distance.
  Theta will break under Newton. Distance will survive. My ruling: theta is an
  internal coordinate; the bit-exact pin on theta is overly tight and should be
  replaced by a 1e-10 value-preservation gate. This is a legitimate re-certification,
  not a pin weakening, because the pin's purpose was to freeze the positive-parity
  path against the saddle extension, and the Newton acceleration is a SANCTIONED
  engine edit.

### Lever 2 (operator fusion)
- The existing F010 py_func self-falsification patches `operator._contract_grid`
  and `operator._weight_vectors` SEPARATELY. If these are fused into one njit
  function, the py_func chain must still expose the SAME perturbation entry points
  (the fused function's .py_func must re-read _SERIES_TOLERANCE and accept a
  half_sum argument). If the signature changes, F010 tests need updating.
- Bit-identity feasible fusion: merging the two njit calls into one (removing
  intermediate array handoff) while keeping identical accumulation order is the
  only safe fusion. The j-loop already runs identically for both v[n]·derivs and
  v_abs[n]·|derivs| — merging these is also safe. Any reassociation of the (a,b)
  scatter with the j-contraction would change bits.
