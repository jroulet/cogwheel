---
date: 2026-07-27
bump: minor
---

Record the lensing delay-frame convention under Conventions: all
`chang_refsdal` channel kernels are carried in the partition's min-subtracted
frame (`t_min = min(absolute real-image Fermat delays)`), the `geometry`
primitives return RAW absolute delays and stay frame-agnostic, and the
min-subtraction is applied at the composition layer in `channels` via the
single authoritative `_frame_t_min`.

Added because a frame mismatch in the ghost carrier was invisible to every
amplitude-based check: it corrupts phase only, leaving magnitudes and decay
rates correct. The convention is now stated so any future term added to a
kernel sum is required to declare its frame.
