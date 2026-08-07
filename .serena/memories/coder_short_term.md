# Coder Short-Term Observations

Added `regions` filter to training entry points (WP1):
- `_train_band_charts` signature: added `regions: tuple[str, ...] | None = None`; `None` → `('tube', 'exterior', 'wedge_interior', 'lobe_interior')`
- Tube section: `if 'tube' in regions:` wraps `tube_w_range` computation; for-loop iterates empty when tube not in regions
- Exterior section: `if 'exterior' in regions:` wraps exterior tiles, ppgo boundary, and region report; `else` sets defaults (`exterior_tiles=None`, `region_exclusion_rho=exclusion_rho`)
- Lobe section: `if 'lobe_interior' in regions:` wraps the body of `if parity != 1:`
- Wedge section: `if 'wedge_interior' in regions:` wraps the body of `else:` (wedge branch)
- Dispatch loop at bottom is unguarded (iterates `admitted`, which is empty for skipped regions)
- `exterior_admission = None` initialized before exterior guard (avoid NameError in dispatch loop)
- `train()`: added `regions` kwarg, threads to `_train_band_charts`
- `scripts/train_lens_surrogate.py`: added `--regions` with `nargs='*'`, `choices=[...]`, converts to tuple and threads to `train()`
