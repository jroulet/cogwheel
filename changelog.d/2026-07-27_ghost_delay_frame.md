---
date: 2026-07-27
---

### Fixed: ghost complex-saddle carrier was in the wrong delay frame

`channels.farfield_ghost_term` carried the decaying complex-saddle ghost with
the RAW absolute complex Fermat delay `tau_c`, while the real-image kernels it
is subtracted alongside are carried in the partition's min-subtracted frame
`tau_a - t_min`. The ghost was therefore off by `exp(-1j*w*t_min)`, so the
mid-band `FARFIELD_KERNEL_SUM_MINUS_GHOST` label subtracted a mis-phased ghost
and left the residual LARGER than subtracting nothing.

The mismatch was identified from the phase of `R/G` (with
`R = F - sum(real kernels)`): `|R|/|G| = 1.00` with decay rates agreeing to
four digits, and `arg(R/G)` linear in `w` with intercept zero mod `2*pi` and
slope equal to `-min(raw image delay)` to five significant figures.

The correction lives in the composition layer: `tau_c` is a holomorphic,
gauge-independent property of the ghost saddle, whereas `t_min` is the
partition's gauge choice, unknown to the single-saddle primitive.
`geometry.ghost_kernel` and `geometry._ghost_delay` stay RAW and unchanged; a
new module-private `channels._frame_t_min` recomputes the frame origin through
the same deterministic path the partition uses, and the ghost is returned as
`C_c(w) * exp(1j*w*(tau_c - t_min))`. The ghost gate still reads the raw
`Im tau_c`, which is frame-invariant, so admit/reject is byte-identical.

Measured mid-band residual collapse through the production label: 174x, 31x
and 607x at the three probe configurations.
