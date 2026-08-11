2026-08-11 WP-1 (operator_routing_one_home): Added dual gate to the ppGO fast rung in `cusp_amplification`:

## Changes
1. Added `_PPGO_RESOLUTION_GATE = 4.0` module-level constant after `_PPGO_BAR_DIVISOR`, documenting that it mirrors `operator.RHO_END` (circular-import barrier prevents a direct import).

2. In the ppGO rung block (inside `if (radius >= r_ppgo_min and ...):`), BEFORE the existing `try:`, compute `delta_min` from already-available `images` list: `delays = sorted(geometry.delay(image, source, matrix) for image in images)`, `delta_min = min(b-a for a,b in zip(delays[:-1],delays[1:])) if len(delays) >= 2 else 0.0`.

3. The existing try/except now wraps an additional guard: `if (_airy_fold._merging_fold_pair(images, source, matrix) is not None or w * delta_min >= _PPGO_RESOLUTION_GATE):` — the `fold_ppgo_correction` call is only made inside this new conditional. The `_merging_fold_pair` call is inside the try/except (morse_index can raise LensDomainError). On gate miss, `result = None` so the rung falls through to the Pearcey uniform form.

## UNVERIFIED
- Fast-tier test gate (not executed — role constraint)
