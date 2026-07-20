# Coder Short-Term Observations

- WP2 (Build 7a): operator.py cross-parity Schwinger fallback.
  Added private `_positive_parity_grid_with_fallback(w_array,y,gamma,*,
  beta,kappa,max_order)` next to _grid_certified: FIRST try batched
  _grid_certified (all-certified hot path returns 5-tuple UNCHANGED,
  byte-identical); ONLY on CancellationError fall to per-node loop —
  retry single-element _grid_certified, on its CancellationError if
  w<=W_CEILING_SCHWINGER reconstruct via _mass_sheet_map (pos-parity,
  succeeds since caller guards lam>|gamma|) + f_schwinger, exactly
  mirroring _saddle_grid's wave prefactor; if w>60 re-raise; let
  SchwingerCertificationError propagate. Fallback-node diagnostics =
  zeros/converged=True (saddle-arm convention). Wired F_op_grid & F_op
  pos-parity arms to helper (saddle arm untouched). _schwinger.py:
  relaxed _validate_inputs guard `gamma_prime>1.0`->`>0.0` + rewrote
  msg; updated f_schwinger docstring param + ValueError lines
  (gamma'<=1 -> gamma'<=0). Verified ast.parse OK both files.
  UNVERIFIED (sandbox std::bad_alloc on heavy numba/lal import):
  runtime bit-freeze + fallback value + w>60 refusal — reasoned
  correct by inspection (per-node single-elem _grid_certified is
  bit-identical to batched per-node arithmetic).
- WP1 (Build 7a): added `_check_image_census(images, matrix)` next to
  `morse_index` in geometry.py; called once at end of
  `find_images_quartic` after sort. Refuses (LensDomainError, 'Image
  census defect') when `sum((-1)**morse_index) != sign(det A) - 1`.
  No tolerance band, no count check (Professor: signed sum is complete;
  tr(Hess)=2*lam>0 forbids maxima). Verified: pos-parity 2/4-img
  signed=0 (det>0), saddle signed=-2 (det<0), no spurious raise;
  dropped-pair raises. Env note: numpy C-ext fails when python launched
  from source-tree cwd — run with cwd=/home/tejaswi to smoke-test.
