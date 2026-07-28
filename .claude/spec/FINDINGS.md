# Findings

Empirical discoveries, numerical-accuracy notes, and non-obvious gotchas
uncovered while working on cogwheel. Each finding gets an ID (`F001`, `F002`, ...),
a date, and a short writeup. Mark superseded findings rather than deleting them.

Use this for things like: tolerance levels at which relative binning diverges
from the exact likelihood, sampler convergence quirks, ASD-drift sensitivities,
waveform phase-convention pitfalls, numba compatibility traps.

---

## F001 — The two-channel cancellation law (2026-07-16)

The Chang–Refsdal amplification loses precision to alternating-series
cancellation through TWO channels. They are INDEPENDENT: they live at
SEPARATE code sites and do NOT compound into a single summed exponent.

- `L_1F1 = w * |y'|` — the confluent-hypergeometric (1F1) kernel series in
  `cogwheel/lensing/chang_refsdal/_hyp1f1.py`. Its partial terms reach
  `e**(w*|y'|)` while the sum is O(1). Double-double arithmetic is required
  HERE, and only here: Kahan summation does not help, because its error
  bound also carries `sum|term_i|`. dd holds the 1e-10 target out to
  `w*|y'| ~ 50` and degrades to ~1e-6 at the ceiling `w*|y'| = 60`.
- `L_op = w * gamma'/2` — the operator power series in `operator.py`. This
  channel is NOT rescued with extended precision; instead `F_op` MEASURES
  its own cancellation ratio `max_partial_term / |total|` and REFUSES
  (raising `CancellationError`) once it exceeds ~1e13. That runtime refusal
  is the operational form of the law: past ~13 lost digits the double-double
  substrate no longer protects the sum, so returning a plausible-but-wrong
  amplification is the failure mode being avoided.

Because the channels are independent, treating them as one summed exponent
`w*(|y'| + gamma'/2)` (as an earlier `_dd.py` docstring did) overstates the
precision demand and misplaces the dd requirement.

## F002 — The oracle-tautology trap in the lens-engine tests (2026-07-16)

A test fixture built by the very code it is meant to judge cannot fail, no
matter how broken that code is. Two concrete instances shaped the lens-engine
test design:

- The fold/cusp crossing-scenario builders that the label-continuity test
  judges `channels.py` against are constructed from `geometry`, `operator`,
  and `_gauge` ONLY. They must never import, call, or derive a value from
  `channels.py`; otherwise the ground truth is the tracker's own output and
  the test is vacuous. This is enforced with an AST import guard in the idiom
  the committed `test_lensing_gauge.py` already uses.
- A mass-sheet identity checked by comparing `F_op` against its own kappa-
  rescaling path is equally vacuous — the code agrees with itself by
  construction. Such identities are asserted on OBSERVABLES (the delay
  differences `Delta tau` and flux ratios `|K_a/K_c|`, which are exactly
  kappa-invariant) or gated against an INDEPENDENT mpmath computation, never
  against the code's own rescaling path.

## F003 — mpmath is an undeclared test dependency (2026-07-16)

The committed lens-engine test suites import `mpmath` as a high-precision
oracle, but `mpmath` is declared nowhere in `pyproject.toml` (no runtime
dependency, no test/dev extra). It is present only because it happens to be
installed in the `cogwheel_310` environment; a clean install would fail to
collect these tests. Recorded here per the Build 1b brief as an observation
to be resolved deliberately (e.g. a test extra), NOT fixed in this build.

## F004 — boundary-domain tests need float64-exact boundary points (2026-07-16)

`macro_matrix` rejects `(kappa, gamma)` iff `1 - kappa <= |gamma|` (strict
positive-parity `1 - kappa > |gamma|` required). A test intending to hit the
EQUALITY boundary must choose values where `1 - kappa == |gamma|` holds
bit-for-bit in float64. `(kappa=0.7, gamma=0.3)` does NOT: `1 - 0.7` evaluates
to `0.30000000000000004`, a hair above `0.3`, so that point is genuinely just
inside the domain and correctly does not raise — a test asserting it must raise
fails against correct code. Use powers-of-two endpoints (`0.5/0.5`, `0.75/0.25`)
where `1 - kappa` equals `|gamma|` exactly. Caught in the first delivered
lens-engine test suite; the code was right, the test's boundary point was not.

## F005 — wave-branch contraction: silent-nan closed, gap NARROWED to a named refusal (2026-07-16, NARROWED)

Original defect (OPEN): the float64 operator contraction in `F_op`
(`z_powers @ (table*radial) @ zbar_powers`, complex128 — deliberately NOT
double-double, per the two-channel error model F001) drifted below the 1e-10
target for cancellation exponents `L = w*sqrt(s)` above roughly 30 at high `w`,
and near `L ~ 40` the intermediate products overflowed to a SILENT `nan` — no
`CancellationError`, no named refusal — violating the module's "named error,
never a silently wrong number" contract inside the nominally certified
`L <= 48` band. `estimated_relative_tail` does NOT bound this error (it tracks
the kernel series tail, not contraction round-off/overflow).

Resolution shipped (WP1). `F_op` now does two things, WITHOUT adding
double-double to the contraction (F001's two-channel model is preserved — dd
stays in the 1F1 kernel only):

1. Overflow-safe contraction. Before the matmuls it factors the peak radial
   magnitude `max|derivs|` out as an EXACT power of two (`np.frexp` picks the
   exponent, `np.ldexp` rescales with no rounding). Every scaled entry is then
   O(1), so no intermediate overflows; the whole summation runs in units of
   `2**scale_exp` and the total is scaled back exactly. For the previously
   certified region this is bit-for-bit identical (the power-of-two factor
   commutes through multiply/add without rounding, and `scale_exp` is small
   enough there that no meaningful entry underflows), so the `L <= 25`
   oracle behaviour is unchanged.
2. Named certification refusal, via TWO independent cuts for two independent
   error sources (revised 2026-07-16 after direct 70-dps-oracle calibration;
   see below). (a) TRUNCATION cut: `estimated_relative_tail`
   (max of the operator-series last-term ratio and the kernel per-order tail)
   `> _CONTRACTION_TARGET = 1e-10` — binding at small `max_order`, where the
   shear series has not converged (this closed a pre-existing SEPARATE silent
   hole: `large-shear` w=40 at the default `max_order=42` returned 1.26e-4 with
   `converged=False` and no refusal). (b) CONTRACTION round-off cut: the
   first-order float64 round-off `eps * (sum|term| / |total|)`
   (`positive_total / |total|`, an all-positive companion contraction)
   `> _CONTRACTION_GUARD = 2e-9` — binding once the series has converged and the
   float64 derivative-ladder cancellation dominates near `L ~ 45`, the regime
   the truncation cut goes BLIND to (its tail collapses to ~1e-14). Both name
   `w, y, gamma, kappa`. A non-finite-total backstop (and a non-finite
   reconstructed-value backstop) catch any residual overflow, since
   `nan > threshold` is False and would otherwise slip past the ratio gates. The
   gamma-channel refusal (`max_partial_term / |total| > _CANCELLATION_REFUSAL =
   1e13`, F001) is unchanged and still fires first.

Net status — NARROWED, not RESOLVED. WP1's overflow-safe rescaling did MORE
than remove the nan: direct 70-dps-oracle measurement (2026-07-16) shows the
returned-and-accurate ceiling is now near `L ~ 45` (true relative error stays
below 1e-10 out to `L ~ 44` on the `y=(0.9,0)`, gamma=0.2 sweep, crossing 1e-10
near `L ~ 45-46`), NOT the pre-WP1 `L ~ 30`. The accuracy of the wave branch
WAS extended by the rescaling (no dd added; F001 preserved). What the refusal
adds on top is the CONTRACT: the residual band `L in [~45, 48]` (below the
geometric branch's `L > 48` onset) now exits through a named
`CancellationError` — never a silent `nan`, never a finite-but-wrong number.

Calibration caveat (why the guard is 2e-9, not the 1e-10 target). The
round-off bound `eps * (sum|term| / |total|)` is a WORST-CASE upper bound, loose
by ~20-30x and NOT a rigorous 1e-10 proof: across shear it can INVERT (an
accurate `y=(0.9,0)` L=45 config reads bound 5e-9 while an accurate
`large-shear` w=40 config, true 2.7e-11, reads 7.4e-10; a low-shear gamma=0.1
config at `L ~ 50` is inaccurate at 3e-10 yet reads bound 4e-13 — that last is
kernel degradation past `L ~ 48`, out of the wave band and handed to the
geometric branch). So the round-off cut is a MEASURED COARSE NET, not a proof;
`_CONTRACTION_GUARD = 2e-9` is pinned to the clean 2.7x gap between the largest
must-return bound (CERT L=43.5 at 1.12e-9; large-shear at 7.4e-10) and the
smallest must-refuse bound (CERT L=45 at ~3-5e-9) on the tested configs. An
interim driver value of 1e-8 was too loose and let `L ~ 45-48` leak as
finite-but-wrong returns (rel err ~1.3-1.5e-10) — retightened to 2e-9.

Consequence for callers and tests. `channels.py` reaches the wave branch only
when `select_branch` returns `'wave'` (i.e. NOT both resolved and `L > 48`), so
an unresolved config with `L in [~45, 48]` now propagates a named
`CancellationError` instead of a silent nan — the intended "certified or
named-refusal everywhere" contract the Build-2 likelihood relies on.
`ContractionCertificationTestCase` asserts the certify-XOR-refuse contract
across `L in [24, 48]`; `GeometricOpticsSlopeTestCase` keeps `L <= 24.3` (w to
27 at `|y'| = 0.9`) so it exercises only certified returns. Do NOT reopen the
gap by widening tolerances.

## F006 — near-cusp (h|h) blow-up: edge-secant kernel coefficients alias the caustic-sharpened amplification (2026-07-16, SUPERSEDED by F008 2026-07-17)

> **SUPERSEDED — mechanism attribution sign-disproven (F008, 2026-07-17).**
> History preserved below; do NOT act on the fix rationale in this entry.
>
> What F006 got RIGHT: densely sub-sampling each bin and reducing the kernel
> by per-bin least squares (raising `kernel_subsamples` 2 → 8) changed the
> physics not at all once the real cause was fixed — nsub=2 and nsub=8 agree
> to `< 1e-4`. That null result was the correct diagnostic signal that the
> defect lay ELSEWHERE than the contraction algebra, the frequency moments, or
> the `F→1` normalization (all of which F006 correctly cleared and are still
> correct). The symptom (`|RB lnL - brute lnL| = 6.43e8` on `near-cusp`,
> bit-stable, deterministic) is also accurately recorded.
>
> What F006 got WRONG: the attribution to an edge-secant slope-squaring
> aliasing failure. It is sign-disproven — the spurious `(h|h)` drove the norm
> term NEGATIVE-huge (RB − brute = **+6.43e8**, i.e. brute ≫ RB after the sign
> in `lnL = (d|h) - (h|h)/2`), and squaring a real per-bin slope `k1` can only
> ADD a positive quantity to `(h|h)`; it cannot produce a negative-huge
> excursion. The dense sub-sampling that F006 shipped was therefore
> COMPENSATING for the true bug (an unbounded upstream kernel) at 8× the engine
> cost, not correcting an aliasing artifact of the reduction. The real cause —
> the `_channel_switch` real-only neighbourhood bug versus the paper's
> Eq. (delay-separation) — is recorded in **F008**, and Build 2c reverts the
> sub-sampling default to 2.

Symptom (Build 2b crown gate): `LensedRelativeBinningLikelihood` disagreed
with its brute-force oracle by `|RB lnL - brute lnL| = 6.43e8` (tolerance 1.5)
on the `near-cusp` config, BIT-STABLE across runs — a deterministic defect, not
sampling noise. The RB `(d|h)`/`(h|h)` summaries and their mode→image
contraction algebra (`_data_term`/`_norm_term`) were verified CORRECT term by
term against the truncated linear-kernel × linear-ratio × linear-phase model;
the fault was upstream, in how the per-bin amplification kernel `K_a(f)` was
reduced to coefficients.

Confirmed mechanism. Near a cusp the merged-image regime is *unresolved*: in
`channels.exact_transition_channels` the smootherstep switch
`smootherstep(w*delay_sep, 0.5, 4)` goes to 0, so each channel kernel collapses
to the artificial single-image split `K_a = alpha_a * exp(-i w tau_a) * F(w)`,
which carries the FULL amplification oscillation `F(w)` — a rapidly varying,
caustic-sharpened function of `f` within one relative-binning bin. The old hot
path (`_edge_linear_coefficients`) built the per-bin `(k0, k1)` from the TWO bin
EDGES only (a secant). When the two edges happen to straddle a phase alias of
that oscillation, the secant midpoint value `k0` collapses toward zero while the
secant slope `k1` blows up; `_norm_term` then SQUARES `k1`, manufacturing the
~6.43e8 spurious `(h|h)`. This is an aliasing/undersampling failure of the
edge-secant reduction, NOT an error in the contraction, the moments, the
delay-phase model, or the normalization.

Fix (WP1, `cogwheel/lensing/likelihood.py`). New hot-path method
`_amplification_coefficients` evaluates `ChangRefsdalChannels` on a per-bin
sub-sample grid (`kernel_subsamples = 8` interior midpoints per bin, strictly
increasing and positive in `w`) and reduces each channel kernel to `(k0, k1)`
by a per-bin least-squares line — offsets symmetric about `f_center`, so
`k0 = mean` and `k1 = <offset, K>/sum(offset^2)`. Densely sampling within the
bin resolves the caustic-sharpened `F(w)` that the edge secant aliased, so the
fitted slope is bounded and the squared-slope blow-up disappears. The
contraction, frequency moments, ratio path, image-delay guard
(`LensedBinningError`), and all three design decisions are UNCHANGED — only the
kernel-coefficient reduction improved. `_edge_linear_coefficients` is
retained (the ratio path uses it); `_amplification_at_bins` was removed as
dead code (INS-3-002: nothing in the hot path, `lnlike_bruteforce`, or the
test suite referenced it). Engine refusals stay symmetric across the RB and brute-force
paths: `geometry.LensDomainError` (macro saddle) and `operator.CancellationError`
(uncertifiable contraction) propagate unswallowed on both.

F→1 normalization: AUDITED, NO code change. The unlensed-limit floor readings
(0.10–0.33 across runs in the failing report) were traced to an unseeded
`EventData` noise draw (nondeterminism, seeded in the test suite), NOT a
normalization bug. The moment prefactor `4*df`, blued strain, `wht_filter**2`,
and `asd_drift**-2` match `CBCLikelihood._compute_d_h`/`_h_h` and
`RelativeBinningLikelihood`; at `F→1` the `p=0` moments sum over bins to the
exact integral, so `(d|h)`/`(h|h)` are exact to ~1e-8. Because WP1 changed no
normalization code, this finding records the audit outcome only — the F005
overflow-safe contraction and its refusal contract are untouched.

> Correction (F007, 2026-07-16): the "exact to ~1e-8" bound above holds only
> for the `p=0` moment with an exactly-constant per-bin ratio and kernel and
> matched fiducial/candidate frequency sets. The full `F→1` evaluation path is
> NOT exact to 1e-8: `_set_summary` builds `_h0_edges` with precession forced on
> and `_stall_ringdown` applied, while `_candidate_bin_ratios` builds the
> candidate `h_edges` with neither, so at `r ~ 1` the ratio is not identically
> `1` in the in-band ringdown of the fixture and a template-construction
> asymmetry `delta-h/h ~ 1e-3` survives. See F007 for the mechanism and the
> zero-noise anchor that gates it test-side.

## F007 — the timing gate was mis-specified, and the F→1 floor is a template-construction asymmetry, not a normalization bug (2026-07-16)

Two Build-2b diagnoses that are NOT defects in the shipped
`LensedRelativeBinningLikelihood` math, recorded so neither is re-opened as a
correctness bug. Both concern `cogwheel/lensing/likelihood.py`; cited by symbol
name. F006 (near-cusp secant-aliasing) is the separate, real correctness fix and
is unchanged by this entry.

### Timing gate — the baseline was RB's co-cost, not its competitor

`ContractionTimingTestCase` originally asserted `t_contract < t_coarse_waveform`,
where `t_coarse_waveform` is the cost of the coarse `get_strain_at_detectors`
call on the relative-binning `fbin` grid. That baseline is WRONG: the coarse
strain call is a per-eval CO-COST that relative binning itself incurs, not the
thing RB competes against. RB exists to eliminate the FULL-GRID matched filter —
that full-grid evaluation is exactly what `lnlike_bruteforce` does. So the coarse
call is not a meaningful ceiling for the contraction, and the measured `23x`
overshoot (`1.47e-3 s` contraction vs `6.4e-5 s` coarse call) is the additive
`M^2 + n_img^2` design behaving AS DESIGNED, not a regression: with IMRPhenomXPHM
(`n_m` up to 4) the contraction runs `~n_m^2` mode-pair einsums plus an
`(n_img, n_img, n_det, n_bins)` image reduction — roughly an order of magnitude
more numpy-dispatch work than a single cached coarse higher-mode strain call.
Beating one cached strain call was never the design's subdominance requirement.

The gate additionally EXCLUDED the F006 `_amplification_coefficients` cost from
measurement entirely, so it left the genuinely new per-eval work (the
Chang–Refsdal / 1F1 special-function engine, now evaluated at
`n_bins * kernel_subsamples` points every eval) completely unguarded.

The two correct gates:
1. `lnlike` faster than `lnlike_bruteforce` by a conservative margin — the
   actual RB speedup claim, stated against the PUBLIC entry points so it
   survives internal refactors.
2. `t_contract < t_amplification` — the pure `_data_term` + `_norm_term`
   contraction must be subdominant to the `_amplification_coefficients` call.
   The special-function engine at `n_bins * kernel_subsamples` points is the
   unavoidable per-eval cost of microlensed evaluation; a likelihood whose
   contraction is dwarfed by its own special-function evaluation is the correct
   shape for a special-function-dominated microlensing runtime. Baseline (2)
   also closes the hole the old gate left by excluding
   `_amplification_coefficients` from measurement.

### F→1 floor — a template-construction asymmetry, left in place for Build 2b

> Scope note (F009, 2026-07-17): this section is the UNLENSED limit
> (`gamma = kappa = 0`), where `F -> 1` exactly and the residual floor is
> construction noise. The SHEARED small-``w`` floor (`0.1214`/`0.1307`, similar
> magnitude, different mechanism) is NOT construction noise — it is the exact
> macro-magnification limit `F(w->0) = sqrt(mu_macro) != 1`. See F009; do not
> conflate the two.

The unlensed-limit floor readings (`0.10`–`0.33` across runs) are NOT a
normalization error. `_compute_d_h` / `_compute_h_h` apply `asd_drift**-2` once
per detector and BOTH oracles route through them; the `4*df` prefactor, blued
strain, and `wht_filter**2` match `CBCLikelihood` / `RelativeBinningLikelihood`.
The residual is a template-construction asymmetry: `_set_summary` builds
`_h0_edges` with `disable_precession=False` forced and `_stall_ringdown`
applied, while `_candidate_bin_ratios` builds the candidate `h_edges` with
NEITHER. So at `r ~ 1` the per-bin ratio is not identically `1` in the in-band
ringdown of the 60+45 M_sun fixture (`delta-h/h ~ 1e-3`), and beaten against the
(previously unseeded) noise this reads as `~ rho * (delta-h/h) ~ 0.05`–`0.3`.
For scale, the physically expected lensing residual at `w ~ 1e-7` is `~1e-4` —
about four orders of magnitude BELOW the observed floor, confirming the floor is
construction noise, not lensing signal or a normalization slip.

This construction asymmetry is a KNOWN residual mechanism left in place for
Build 2b: aligning the two template builders risks the currently-green
brute-force suite and is out of scope here. The crown gate handles it test-side
with a ZERO-NOISE `F→1` anchor (the deterministic unlensed-limit check that
removes the noise projection so the `delta-h` term cannot be amplified into a
spurious floor). This entry also corrects F006's "at `F→1` … exact to ~1e-8"
line: that bound holds only for the `p=0` moment with exactly-constant
ratio/kernel and matched frequency sets; the full `F→1` path carries the
construction-asymmetry `delta-h`.

### Static verification of the crown-gate deliverables (WP1 re-dispatch, 2026-07-16)

Recorded to resolve INS-3-001, which was an Inspector-SESSION access failure
(Bash false-denial, Serena/Read timeouts) — not a code or test defect. A
session with working file access statically confirmed that the shipped
artifacts exist and encode exactly what F006/F007 document:

- `cogwheel/lensing/likelihood.py`: `_amplification_coefficients` (the F006
  dense per-bin sub-sample + least-squares kernel reduction) is present as a
  `LensedRelativeBinningLikelihood` method, alongside its subsampling machinery
  (`_build_kernel_subsampling`, `_kernel_dense_f`, `_kernel_fit_value`,
  `_kernel_fit_slope`), the contraction (`_data_term`, `_norm_term`), and the
  retained `_edge_linear_coefficients` (`_amplification_at_bins` was later
  removed as dead code, INS-3-002).
- `cogwheel/tests/test_lensing_likelihood.py`: `ContractionTimingTestCase`
  asserts the two F007 baselines — (a) `lnlike` faster than `lnlike_bruteforce`
  by `SPEEDUP_MIN`, and (b) the `_data_term`+`_norm_term` contraction
  subdominant to the `_amplification_coefficients` engine call — and its
  docstring explicitly declines the old coarse-strain baseline as a per-eval
  co-cost. `NearCuspRegressionPinTestCase`, `NormalizationFloorZeroNoiseTestCase`
  (the zero-noise `F→1` anchor), and `DeterminismTestCase` are all present.

UNVERIFIED (handed to a runtime-capable reviewer, NOT the Coder — code and the
suite that blesses it must not share an author): the actual pass/fail state of
`python -m pytest cogwheel/tests/test_lensing_likelihood.py
cogwheel/tests/test_lensing_operator.py` at the stated tolerances. The static
read establishes the deliverables are real and structurally consistent with the
documentation; runtime green-ness must still be confirmed before Build 2b is
committed, and no tolerance may be widened to achieve it.

## F008 — the real near-cusp cause: `_channel_switch` measures delay separation against real channels only, not the full cluster (2026-07-17)

> **ADDENDUM (2026-07-18, Build 3f / SACR-C): switch keying superseded, F008's
> lesson preserved.** The SACR-C channel construction replaced the full-cluster
> pairwise switch keying with the CRITICALITY separation `delta_a = \|tau_a -
> tau_c\|` (each image's delay distance from the parked critical carrier). The
> switch scale is now the demodulation distance of the single transition
> envelope by construction, so the bounded-phase theorem applies (envelope
> phase <= RHO_END = 4 rad — beats impossible). For genuine mergers
> `tau_a -> tau_c`, so the new gate is AT LEAST as conservative as F008's
> full-cluster rule (measured `max\|S_a H_a\| <= 1.21` through fold/cusp
> crossings at eta = +-0.002); ACCIDENTAL delay degeneracies between
> non-merging images no longer stall the switch (small carrier separation =
> no beat, so staying unswitched is harmless). F008's underlying lesson —
> never key a cluster decision on a real-only neighbour set — remains valid
> and is embodied in `tau_c` coming from the parked/critical carrier.

This is the actual mechanism behind the Build-2b crown-gate accuracy failures
that F006 mis-attributed (F006 now SUPERSEDED). The defect is one line in the
lens engine, not in the relative-binning likelihood.

Root cause. `_channel_switch` (`cogwheel/lensing/chang_refsdal/channels.py`)
ramps a real channel over to the divergent saddle-kernel branch based on that
channel's delay separation from its neighbours. It computed the neighbour set
as `others = real_ids[real_ids != channel]` — REAL channels only. The paper's
Eq. (delay-separation) takes the minimum over ALL members of the image's
cluster, INCLUDING the labels parked at the critical point (the "virtual"
labels). On the 2-image side of a caustic a near-critical image's actual
cluster-mates ARE those parked virtual labels: the measured gap to a virtual
label was `5.5e-5` at the crown `near-cusp` config, versus `0.856` to the
nearest persistent real image. Keying only on real neighbours, the switch saw
the large `0.856` gap, spuriously ramped to 1, and handed the channel to the
divergent saddle kernel `H` (`|H_0| ~ 1.8e8` there, growth `~ gap^-2` toward the
cusp), which then flooded all four channels through the residual projection
(`|K_a| ~ 5.2e5`, cancelling coherently to `|F| ~ 3`). Squaring that in the
norm term is what produced the negative-huge `(h|h)` excursion — an UNBOUNDED
upstream kernel, not an aliased reduction slope (which is why F006's
slope-squaring story was sign-wrong).

Fix (Build 2c, WP1). Replace the neighbour set with all cluster labels except
self (`np.delete(np.arange(_N_CHANNELS), channel)`), so parked virtual labels
count as the legitimate neighbours Eq. (delay-separation) intends. The fix is
one-directional (it can only LOWER a switch value) and a no-op wherever all
four labels are real (4-image regions, near-fold-inside): there `real_ids` is
the full set, so old and new neighbour sets are identical and the result is
bit-for-bit unchanged.

Measured effect (two independent agents, scratch probes; repo untouched):

| config    | switch        | max\|k0\| | RB − brute lnL | 1.5 gate |
|-----------|---------------|-----------|----------------|----------|
| two-image | current (bug) | 40.9      | +9.768         | FAIL     |
| two-image | fixed         | 0.922     | +0.080         | PASS     |
| near-cusp | current (bug) | 5.22e5    | +6.43e8        | FAIL     |
| near-cusp | fixed         | 0.975     | +0.329         | PASS     |

- `kernel_subsamples = 2` under the fixed switch: `+0.069` (two-image) /
  `+0.316` (near-cusp) — both PASS. This is why F006's dense-subsampling
  compensation is unnecessary and Build 2c reverts the default 8 → 2 (WP2).
- Moment orders are irrelevant to the fix: `p+s ≤ 3 == p+s ≤ 4 == p+s ≤ 5` to
  `< 1e-4` once the kernels are bounded — no norm-moment change is needed; the
  contraction algebra is correct as shipped.
- Brute force is switch-independent (it uses `exact_total`; reconstruction
  `~1e-16` under either switch) — the oracle never moved, so the disagreement
  was entirely the RB path riding the unbounded kernel.
- Reconstruction error IMPROVES under the fix: `2.5e-10 → 5e-16`.

Sibling audit — `_min_delay_separation` NOT implicated. The sibling
`_min_delay_separation` (feeds the wave/geometric branch gate) carries the same
real-only pattern but is deliberately kept real-only: the geometric branch
replaces `F_op` with the stationary-phase sum over REAL images only (virtual
labels carry no saddle), so its resolution gate must key on real-image
separation. `exact_total` is unaffected by the switch bug, confirming this
sibling is not implicated in the crown failures. Recorded here so the two
real-only patterns are not "fixed" together by mistake.

Cross-references. F005 (wave-branch contraction overflow/refusal, NARROWED)
and F007 (timing-gate spec and the `F→1` template-construction asymmetry) are
UNAFFECTED by this fix — the switch correction touches neither the operator
contraction refusal thresholds nor the template builders. It does, however,
retire F006's dense-subsampling rationale and lets the crown gate pass at the
original tolerances (`RB_ATOL = 1.5`) with the sub-sampling default back at 2.

## F009 — the small-``w`` floor is the exact macro-magnification limit, not a `gamma/(2w)` singularity (2026-07-17)

The Build-2d episode. After Build 2c landed (F008), three tiny-``w``
observations were read as a fresh engine defect; a consult misdiagnosed them
as a series singularity and prescribed an engine short-circuit; the Architect
rediagnosed them as an EXACT closed-form physical limit, and the test ratified
that to 16 digits. No engine code changed. Recorded so the short-circuit is
never re-proposed.

### 1. Symptom (three post-2c "failures")

* Zero-noise likelihood floors of `0.1214` and `0.1307` against a `0.01`
  expectation, in the sheared small-``w`` regime.
* A small-mass `|F| - 1` that sat FLAT at `2.062e-2` — refusing to vanish as
  the mass (and hence ``w``) shrank, where a diffraction signal was expected
  to switch off.
* The channel-tracker "flat-gate" premise (that `|F| -> 1` as `w -> 0`) was
  apparently void.

### 2. Misdiagnosis (consult) — FALSIFIED

A consult attributed the flat `|F| - 1` to a `gamma/(2*w)` small-``w`` series
singularity in the operator power series (the prefactor `i*gamma/(2*w)` grows
as `w -> 0`), and prescribed an ENGINE SHORT-CIRCUIT: detect tiny ``w`` and
return `F -> 1 + O(w)`. This is sign- and scale-disproven below and must not be
implemented. The operator series is summed in the shear eigenframe and its
value is bounded as `w -> 0`; the growing prefactor does not make `F` diverge,
it makes `F` approach a NON-UNIT constant.

### 3. Actual physics (Architect rediagnosis, ratified by test)

`F` is normalized to NO LENS AT ALL, not to the macro image. As `w -> 0` the
point-mass diffraction switches off, but the smooth QUADRATIC macro potential
(convergence + shear) integrates EXACTLY — a quadratic Fermat phase makes the
diffraction integral a Gaussian — so

    F(w -> 0) -> 1 / sqrt((1 - kappa)**2 - gamma**2) = sqrt(mu_macro),

a real, positive, MASS- and FREQUENCY-INDEPENDENT constant, not `1`. The flat
`|F| - 1 = 2.062e-2` at `gamma = 0.20`, `kappa = 0` is exactly
`1/sqrt(0.96) - 1` to 16 digits — the closed form, not roundoff. The
zero-noise floor is then the exact projection of that constant offset,
`0.5 * <|F - 1|**2> * (h0|h0) = 0.1214`, not a defect. `|F| - 1` vanishes as
`w -> 0` ONLY when `gamma = kappa = 0` (the unsheared control), which is why
F007's `F -> 1` premise holds only there.

Gate (source of truth, `operator.py` "NORMALIZATION AND THE w -> 0 MACRO LIMIT"
docstring): `test_lensing_operator.py::MacroMagnificationLimitTestCase` pins
`|F_op|` to the LITERAL closed form `1/sqrt((1-kappa)**2 - gamma**2)` — never
built from `F_op`/`channels`/`geometry` (F002 oracle-tautology trap) — to
relative `7.85e-9` across a 48-point positive-parity grid (4 shears x 2
convergences x 2 shear orientations x 3 frequencies) spanning THREE DECADES of
tiny ``w`` (`1e-8, 1e-10, 1e-12`); the plateau's frequency-independence is the
signature that separates the exact limit from a `1/w` singularity. The
prescribed short-circuit would inject a real 2% DISCONTINUITY at the crossover
and DESTROY the exact pure-shear limit — never add one.

### 4. Residual pin (what the RB anchor is, and is not)

The RB zero-noise anchor decomposes as an INHERITED `8.962e-3` standard
`RelativeBinningLikelihood` stall floor plus a `2.676e-3` lensing-layer
increment (gated at `5e-3`); their sum is the `1.164e-2` fft-comparison
regression PIN — a pin, not a physical claim. The `8.962e-3` term is upstream,
not a lensing defect: the stall-ringdown is applied to the REFERENCE
(`_h0_edges`) only, BY DESIGN, so at fiducial parameters and zero noise the
per-bin ratio `r = h/h0 != 1` in the ringdown band and its linear-model
residual IS that floor. The stall is LOAD-BEARING (it buys a smooth
interpolable reference) and MUST NOT be removed or weakened; the requirement is
lnL accuracy, driven upstream through the normal build workflow
(`todo.d/likelihood_standard-rb-zero-noise-floor.md`), not by touching the
stall.

### 5. Lesson (verbatim, the Architect)

> a "numerical artifact" that is FLAT across many decades of the
> supposedly-singular parameter is almost never roundoff — match it against a
> closed form before planning around it.

## F010 — numba compilation silently voids Python-level test instrumentation (2026-07-17)

When the Build 3/3b fast path njit-compiled the dd primitives, the 1F1
ladder, and the operator contraction, three previously-green tests in the
OLD suites broke — none of them accuracy regressions, all of them
Python-introspection assumptions that compilation invalidates. Recorded
so the next numba migration re-checks this list instead of rediscovering
it:

1. **Module-global patching does not reach compiled code.** numba freezes
   module globals at compile time, so `mock.patch.object(_dd, '_SPLITTER',
   ...)` never reaches the njit `_split`. Worse, patching only the OUTER
   function to its `py_func` is not enough when it calls another njit
   function via module globals: `_two_prod.py_func` resolved `_dd._split`
   to the frozen njit dispatcher, so the dd splitter falsification
   (`test_broken_splitter_breaks_two_prod`) passed vacuously-green in
   reverse — the broken splitter no longer broke anything. IDIOM: patch
   the ENTIRE call chain to `py_func` bodies (`_split` AND `_two_prod`)
   for the duration of the sensitivity check.
2. **Call-counting wrappers count zero.** `LadderComplexityTestCase`
   replaced `_hyp1f1.dd_mul`/`dd_complex_mul` with counting wrappers; the
   njit ladder core binds them at compile time, so the counters read 0.
   Same idiom: swap `_shared_numerator`/`_ladder_sum`/`_ladder_core` to
   their `py_func` bodies while counting — the py bodies are the same
   algorithm numba compiles, so the counts still measure the shipped
   design.
3. **Internal-shape assumptions break silently.** `_amplification_
   coefficients` now returns the COARSE-node partition (~`n_kernel_nodes`
   points), not the dense sub-sample partition; a diagnostic helper that
   zipped it against `_kernel_dense_f` (506) crashed shape-wise. Tests
   reading engine profiles must evaluate `ChangRefsdalChannels` directly
   at the grid they want, not scrape likelihood internals.

The falsifiability lesson generalizes: after ANY compilation/JIT change,
re-run the self-falsification tests and confirm they can still go RED —
a falsification test that cannot fail anymore is worse than no test, and
nothing else in the suite will tell you.

## F011 — the certification-blind eps64 class in paired-rule quadrature (2026-07-19, Build 6)

A paired-rule (N-vs-2N) quadrature certificate is BLIND to any error
that is bit-identical in both rules. In the Schwinger saddle evaluator
two such errors shipped, each a single float64 operation amplified by
the `e^{pi w/4}` prefactor into silent fabrication above `w ~ 20`
(O(1) wrong by w=45, |F| ~ 8.5e3 at w=59.9 — returned WITH a green
certificate):
1. The IBP endpoint evaluated at `t_cap` while both rules split at
   `e^{fl(log t_cap)}` — the identity is split-point-arbitrary but
   demands ONE split point; fixed by deriving endpoint and domains from
   the same `u_mid`.
2. `1/s` computed in float64, entering the endpoint and A-pieces but
   not B, so it can never cancel; fixed with a dd reciprocal.
Extinction was PROVEN, not assumed: post-fix error vs the independent
oracle is 6.6e-15 (w=30) and 1.6e-11 (w=59.9), while any surviving
class member would measure ~3e+4 at the ceiling — fifteen orders
excluded. Audit rule for any future paired-rule certificate: every
float64 quantity that (a) enters the certified identity inconsistently
or (b) multiplies the accumulated result, and is identical across both
rules, is a candidate silent fabricator — carry it in dd or prove it
benign (parameter roundings consumed self-consistently are benign; the
domain-truncation margin is N/2N-invisible but analytically bounded).

## F012 — near-axial quartic dead zone: silent wrong-finite wedge above the wave ceiling (2026-07-19, pre-existing, both parities)

`find_images_quartic` silently drops the symmetric near-degenerate
image pair for sources ~1e-10..1e-9 (rel. angle; wider on positive
parity, to 1e-8) off a macro-matrix eigenaxis inside the 4-image
region — the dead band between the axial-path threshold and the
generic path's removable-singularity guard. Index-theorem violation
(signed Morse sum wrong). Consequences: for `w <= 60` the wave branch
is IMMUNE (it never consumes images); but above the ceiling,
`_real_delay_min_separation` computed from the surviving images
misreports a truly-unresolved config as resolved and the geometric
branch returns an O(1)-wrong finite value — a certify-or-refuse
violation confined to {dead zone} x {w > 60} (~1e-10 hit probability
per proposal, but silent). Tracked by
`NearAxialQuarticDefectTestCase` (@expectedFailure). REQUIRED Build-7
precondition: a runtime index-theorem check (signed Morse sum ==
sign(det A) - 1) in every image-consuming path, converting the dead
zone into a named refusal on both parities.

GUARDED (Build 7a): `_check_image_census` now runs at the end of
`find_images_quartic` (the single image producer — `find_images` is a
pure alias and `_centered_source_images` is internal to the quartic
path), enforcing the parity-agnostic invariant
`sum (-1)^{n_a} == sign(det A) - 1` with no tolerance band (a dropped
mirror pair shifts the sum by an even +-2; a fold-merging pair
contributes 0 and cannot false-positive). The dead zone is now a
named `LensDomainError` census refusal on BOTH parities;
`NearAxialQuarticDefectTestCase` was flipped from @expectedFailure to
a positive assertRaises with a positive-parity twin.

## F013 — the negative-parity (macro-saddle) branch: certified summary (2026-07-19, Build 6)

Full story in `.claude/handoff/lensing/negative_parity_research.md`
(design authority) and the two suites
`test_lensing_saddle_geometry.py` / `test_lensing_schwinger.py`.
Measured certification: census (index sum -2, multisets
{1,1}/{0,1,1,1}) over 200+ sources incl. both deltoid lobes; Schwinger
vs independent AST-guarded mpmath oracle 9.1e-14 (w=20) to 1.6e-11
(w=59.9); deep band |F| -> 1/sqrt(gamma^2-(1-kappa)^2) at 4.4e-5
(w=1e-4) with Morse intercept -pi/2 to 1.5e-7 (F009-S); mass-sheet on
observables 1e-13-class, lam <= 0 refused (F004 float64-exact
boundaries); positive parity pinned bit-for-bit to pre-extension HEAD.
The single cancellation channel is `L_S = pi*w/4`, y-independent
(F001-S); ceiling w <= 60 with certify-XOR-refuse (F005-S). Warm cost
~30-125 ms/point (w-linear) — the surrogate is load-bearing for the
homogenized architecture. INTERIM LAYER CONTRACT: the channel/waveform
layer refuses saddles by name (guards in `channels.evaluate` and the
`LensedWaveformGenerator` constructor) until Build 7 delivers the
saddle-domain channel layer.

## F014 — `lnl_marginalized` integrates arrival time with a unit-density (1 per second) prior — DELIBERATE convention (2026-07-19, stock cogwheel)

The coherent-score marginalized likelihood
(`MarginalizationInfo.lnl_marginalized`, consumed by
`MarginalizedExtrinsicLikelihood.lnlike` and the lensed subclass) is
not normalized over arrival time: the only time factor in the QMC
numerator is `sky_prior = (dOmega/4pi) * (1/sky_dict.f_sampling)`,
which carries units of SECONDS. Per the owner, this is a DELIBERATE
design choice (Javier's): with the integrand exp(<d|h> - <h|h>/2), a
silent stretch of window contributes e^{-<h|h>/2} ~ 0, so
integral(L dt) is invariant to how much data the analyst happened to
include — evidences stay comparable across analyses with different
window lengths, which a proper 1/T prior would break. Sky
(dOmega/4pi), orbital phase (1/2pi), polarization (1/pi), and
distance (uniform-in-volume to `lookup_table.d_luminosity_max`;
verified exact against direct quadrature) are all proper.
Consequences:
- `lnl_marginalized` = ln[ integral L dt <proper angles/distance> ] —
  it is ln(seconds)-offset relative to any oracle that uses a proper
  uniform t prior 1/T. A validation oracle must ADD ln(T_oracle) to
  compare. The offset is intrinsic-independent and cancels in
  posterior sampling; for model comparison, evidences computed under
  this convention are mutually consistent (that is its purpose) but
  differ from a proper-1/T-prior evidence by ln(T/1s). In particular
  the program's headline lensed-vs-unlensed Bayes factor is safe by
  construction: both hypotheses use the same coherent-score
  machinery, so the convention cancels identically in the ratio —
  only mixing in an evidence from a proper-1/T-prior code requires
  the ln(T/1s) adjustment.
- Verified empirically to 1.5 mnat: widening an importance-sampling
  oracle's t window 3.906x moved the gap by ln 3.906 while the
  marginalized value did not move (n_eff ~ 19-21k of 40k draws,
  Student-t proposal; probes marg_oracle_probe5*.py, 2026-07-19).
- After the convention correction a residual remains: marg LOW by
  0.146 +- ~0.03 (unlensed control) and 0.193 +- ~0.03 (lensed
  MAIN_LENS point) — both inside the 0.3-nat oracle gate, and their
  difference (0.05 +- 0.04) shows no lensed-fold defect. The ~0.15
  residual is an upstream stock-cogwheel effect (candidate: sky-delay
  discretization at f_sampling); track before using marginalized
  values as absolute evidences.
- Beware numerology: the uncorrected unlensed gap (2.0795) matched
  ln 8 to 1e-4 by pure coincidence — the T-scaling test is what
  discriminates, not constant-matching.

## F016 — the rescued strong-shear lnlike gap is RB-binning/noise-limited, NOT envelope-limited (2026-07-20, Build 7b falsification)

Build 7a's rescued-node measurement (ratio/direct paths agreeing with
each other but ~0.9 nats from brute force at gamma' ~ 0.94) was
root-caused during 7b planning as envelope under-resolution (the
max|F| error-currency normalization under-weighting deep-cancellation
troughs) and WP4 tightened the LOO stop to fix it. Test authorship
FALSIFIED that mechanism: the gap is INSENSITIVE to the stop (1e-3 ->
1e-5 moves 0.72 -> 0.75 nats) and SWINGS WITH THE NOISE SEED
(gamma'=0.8: 0.004 vs 0.150 nats across two seeds; gamma'=0.94: 0.72
vs 1.35). It is an RB-binning / data-noise effect of the same class
the crown gate prices at RB_ATOL=1.5 — larger here because the
strong-shear F(w) has more unresolved structure per bin. Consequences:
- The gamma'-keyed two-tier LOO stop (_LOO_STOP_STRONG=1e-3 for
  gamma' >= 0.5) is JUSTIFIED ONLY by the research's saddle-side
  envelope gate (reconstruction eps < 1e-3, enforced in
  test_lensing_saddle_channels), not by the nat-gap story.
- Rescued/strong-shear likelihood accuracy is gated at the standard
  RB tolerance; saddle-family configs (gamma' ~ 1.25-1.3) measure
  ~0.04 nats and pass the tight 0.1-nat gate comfortably.
- A principled strong-shear RB-binning audit (bin density vs F(w)
  structure) belongs to the Build 8 program alongside the surrogate.
- S2 node-count note: under the tightened saddle stop, genuine
  2-decade saddle windows converge at N ~ 40-42 envelope nodes (below
  the 48 cap; true reconstruction error 2-4e-4) — the research's
  N <= 30 was calibrated on <=1-decade windows.

## F017 — theta of the nearest caustic point is gauge, and the old Brent was the less accurate party (2026-07-20, Build 8b-levers)

The Newton reimplementation of `nearest_caustic_point` (analytic
g'/g'' on the stationarity condition; wedge-clamped per lobe/branch;
bounded-Brent fallbacks at cusps/discriminant clamp) preserves the
PHYSICAL observable to certification grade: distance within 9.3e-12
relative over 5677 both-parity configs, the HEAD pins at ~1e-16, zero
global-min basin misses, branch/lobe selection identical off
degeneracies. Theta, however, drifts up to ~5e-9 at SHALLOW minima —
and the evidence shows the drift is dominated by the OLD path's error
(Brent at xatol=1e-12 on a near-flat objective under-converges; the
Newton iterate sits closer to the independent dense oracle).
Dispositions to reuse:
- Theta is internal parametrization (gauge), not a certified
  observable (Professor ruling): its bit-exact pin component was
  re-certified at 1e-10 (well-conditioned) / 1e-8 (shallow-minimum)
  vs the pinned values, distance kept at places=14.
- The right theta gate currency is ARC LENGTH (theta_gap x
  caustic_speed) against the independent oracle — cusp-safe (speed ->
  0 tolerates the genuinely ambiguous angle) and immune to the
  old-code-was-worse trap that a HEAD-referenced theta gate falls
  into. Falsified by a forged non-global theta going red.
- General lesson: when a reimplementation is MORE accurate than the
  incumbent, value-preservation gates against the incumbent must be
  scoped to the physically certified quantities; gauge quantities get
  gated against an independent oracle, or the gate punishes the
  improvement. Timing: caustic search 1.23 -> 0.095 ms (positive
  parity), 4.54 -> 0.99 ms (saddle; the residual is 4 cusp-Brent
  fallbacks). The operator contraction fusion alongside it is 0-bit
  different across the certified sweep with refusal parity exact.

## F018 — the tube chart's measured advantage is against the extrapolating far-field, not a fair same-band raw chart; design-claim currencies matter (2026-07-20, Build 8c census tests)

The 8c design falsifiable was stated as "tube beats an equal-budget
raw chart by >= 3x at eps_95; raw fold-approach slope ~ -1/2". The
independent census test developer measured extensively: in the
census's MAX-NORMALIZED error currency (max|dE|/max|E|, the
F016-correct currency for lnL impact) a FAIR raw Cartesian chart
trained ON the same narrow band TIES the tube at fixture scale
(ratio ~0.8-1.1) and its fold-approach error IMPROVES inward
(slope ~ +0.5) — because the fold divergence of |E| itself inflates
the denominator and masks absolute-error growth. The tube's measured
advantage appears against the EXTRAPOLATING far-field chart (trained
outside the caustic, eta_overlap_min = 0.05) — which is the actual
production alternative the tube replaces: p95 ratio 2.93, deep-
caustic gap 3.01, raw slope saturating at ~ -0.19 (not -1/2; same
denominator effect). Dispositions:
- The Professor's 3x / -1/2 numbers came from derivative counting in
  a POINTWISE currency; any design-advantage claim must name its
  error currency, and gates must use the currency that tracks lnL.
- The tube design still stands on (a) the curved-band tiling
  argument (a fair raw band chart exists only on a short arc segment;
  globally, axis-aligned boxes fight the curved caustic and straddle
  image-count changes) and (b) the asymptotic convergence-rate gap
  (h^4 in u vs h^{1/2}-limited in eta), which fixture-scale grids are
  too coarse to expose.
- In-build bars (binding, coarse-fixture): tube-vs-extrapolant >= 2x
  at p95 / >= 1.5x at max; tube ray slope |s| < 0.15; raw-extrapolant
  slope < -0.05; deep-ratio >= 2. The literal 3x figure moves to the
  production-density training report (post-8e), measured in BOTH
  currencies.

## F019 — the two exact evaluators' ceilings live in different variables; a homogenization plan missed it and the census caught it pre-commit (2026-07-21, Build 8d)

The legacy operator series refuses on the PRODUCT w*sqrt(s) <= 60
(the y-dependent dd channel, F001/F005); the Schwinger quadrature
refuses on the frequency ALONE, w <= 60 (its cancellation channel
L_S = pi*w/4 is y-independent, F011/F013). The sampled prior box
bounds the PRODUCT (w*sqrt(s) <= 58 by construction) while w itself
runs to ~443 — so the two domains are NOT nested, and "route
everything through Schwinger" (the approved 8d plan, incl. Professor
Q2(b) "Schwinger-refuses => production refuses", driver-co-signed)
would have wholesale-changed the disposition of the ~25% of prior
draws carrying w > 60 non-geometric nodes. The WP3 geometry census
measured this BEFORE commit. Dispositions:
- OWNER RULING: 8d ships as the PURE homogenization anyway — the
  corner refuses by name until Build 8e serves it, because sampling
  is PARKED (ruling A): nothing production evaluates the corner in
  the interim, so coverage parity with pre-8d was a non-goal. (A
  driver per-band legacy revival was implemented and REVERTED on
  this ruling.)
- The legacy series also truncation-refuses part of the corner
  (shear-ladder degradation with w, measured at w=100 gamma'=0.2),
  so the 25% is an UPPER BOUND on what 8e must serve, not a
  pre-8d-served fraction; the truly-refused sub-fraction needs the
  8e scoping census (per-node evaluation).
- Standing lessons: domain claims comparing two evaluators must name
  the VARIABLE each ceiling lives in; reverse-coverage (B-refuses-
  where-A-served) is part of any evaluator-swap verification; and a
  cheap geometry census run BEFORE commit is what turned this from a
  shipped regression into a plan amendment.
- Cost note: homogenization re-prices the exact path (~90 ms/node,
  crown lnlike ~751 ms default; brute comparisons ~138 s/call) —
  brute-heavy accuracy tests are gated behind COGWHEEL_BRUTE_ACCURACY
  (driver post-build tier); the production hot path (surrogate/
  geometry) is unaffected.

## F015 — fold-degenerate images crashed the geometric kernel with a raw LinAlgError (2026-07-19, surfaced in production, fixed Build 7a)

The headline marginalized sampling run died mid-flight (bound 17+)
when a nautilus proposal produced an image on a fold:
`geometry._saddle_metric` inverted the projected Fermat Hessian with a
bare `np.linalg.inv`, the exactly-singular matrix raised
`numpy.linalg.LinAlgError`, and the unnamed exception sailed past
`LensedPosterior`'s refusal net (which maps `LensDomainError` /
`CancellationError` to -inf) and killed the whole sampler. This is
the unresolved-near-caustic corner (the fold/cusp Airy-patch gap)
manifesting as a CRASH class rather than the known accuracy class.
Fix: `_saddle_metric` catches the `LinAlgError` itself and re-raises
it as the named `LensDomainError` ('Fold-degenerate image') — refusing
EXACTLY the crash class and nothing more. Two threshold attempts were
wrong before this landed: `1e-13 * ||P||_F^2` amputated near-fold
configs the channel sweep legitimately serves (det ~ 40*eps), and even
`4*eps * ||P||_F^2` broke the channel layer's on-cusp rows (det ~
2*eps), because the SACR-C/F008 switch design DELIBERATELY consumes
huge near-singular metrics and multiplies the divergent
stationary-phase target away. Lesson: a guard for a crash class must
be scoped to the crash condition itself, not to a nearby-looking
numerical neighborhood. Two open notes:
1. NEAR-singular (not machine-singular) projected Hessians still
   return finite, increasingly divergent SPA kernels — the principled
   near-fold validity bound belongs to the fold/cusp uniform (Airy)
   asymptotics program, not to an ad-hoc threshold.
2. Audit rule: any raw `np.linalg.inv/solve/cholesky` on the physics
   path is a latent member of this class (a repo grep found exactly
   this one; `_newton_polish` already guards its solve with an lstsq
   fallback).

## F020 — a far-field chart trained below `w_floor` fits a divergent object; and a tile centred on a cusp ray fits a slope kink (2026-07-27, Build 8h-d triage)

Two INDEPENDENT causes of an inflated held-out `eps` in exterior chart
fixtures, both invisible to inspection and both worth orders of magnitude.
Recorded because a prior pass searched only box PLACEMENT (three `rho_c`
values, halves down to `(0.1, 0.03)`, narrower gamma bands), found nothing,
and recorded the degradation as unexplained. Neither cause is reachable by
varying tile geometry.

1. WRONG LABEL BELOW `w_floor`. `FARFIELD_KERNEL_SUM` subtracts the real
   image kernels, which is the correct label only in the mid band
   `[w_floor, w_trust)`. Below `w_floor` no real pair separates and the
   residual is the divergent diffractive-bottom object; the correct label
   there is `FARFIELD_DIFFRACTIVE` (subtract nothing). A fixture training the
   kernel-sum label from `w = 0.0248` against a region `w_floor = 0.661` put
   ~11 of ~15 log-`w` nodes under the floor and measured `eps = 7.67` against
   a `3e-3` bar. Correcting the band bottom to the region floor
   (`_farfield_region_w_floor`) alone took `eps` to `9.0e-3` — a factor 847.

2. CUSP-RAY TILE CENTRES. For the astroid the cusps sit at
   `theta_c = 0, pi/2, pi, 3pi/2`, where `r_caustic` has a slope KINK. A tile
   centred on one asks a cubic spline to represent a non-smooth map. Moving
   the same fixture's centres off the rays (to 0.5 / 0.95 / 1.4, all inside
   `(0, pi/2)`) took `eps` from `9.0e-3` to `3.4e-4`. Combined with (1):
   `7.67 -> 3.4e-4`, a factor 22,500, with the bar and the poison factor
   UNCHANGED.

DIAGNOSTIC SIGNATURE for (2): the ordering inverts. A deliberately POISONED
chart scored BETTER than the "healthy" one (2.61 vs 7.67) because the healthy
tile straddled a cusp ray while the poisoned tile sat between cusps. An
incoherent ordering between a control and its degraded twin means neither is
fitting anything — look for a structural cause, not a tolerance.

The production tiler already addresses (2) with cusp-ALIGNED columns
(`_cusp_aligned_theta_tiles`), which put the ray on a column EDGE; see
`OnCuspColumnEdgeTestCase`. Any fixture or chart path that does NOT route
through that alignment inherits this defect.

## F021 — agent short-term memories are tail-capped, so a build's findings can be evicted before the Dreamer consolidates them (2026-07-27)

Inlined short-term memories are truncated to a 24 KB tail. When several
builds run between Dreamer passes, the OLDEST entries are dropped first — so
a finding recorded early in a busy day never reaches long-term memory, and
its absence is silent: consolidation reports success having promoted only
what survived.

Observed 2026-07-27: after three builds in one day, two substantive findings
(F020 above, and the deletion of 25 structural test classes with a restore
SHA) were absent from every short-term memory at consolidation time. The
Dreamer correctly REFUSED to promote them rather than inventing content, and
flagged the gap — which is the right behaviour and the only reason the loss
was noticed.

Mitigations, in order of preference: run the Dreamer between builds on a busy
day rather than at the end; write durable findings to `FINDINGS.md` (this
file) at the time they are measured, not via memory; and treat a Dreamer
report that flags "pattern not present" as a signal to recover from git or
the build handoffs, never as a reason to drop the finding.

## F022 — the far-field carrier guard measures `arg`, but the interpolant splines re/im; a phase flip at an amplitude null is smooth in re/im and refinement cannot remove it (2026-07-28, Build 8h-d2 triage)

`_assert_farfield_carrier_continuity` (`lensing/surrogate.py`) rejects an
exterior tile whose frame-invariant label `E_tilde` winds by `>= pi/2` in
`arg` between adjacent spatial nodes at the top of the band. Three coarse
fixtures trip it: `test_lensing_surrogate_census.py::_pos_farfield_dense`,
`test_lensing_ppgo_bandsplit.py::BandSplitReconstructionTestCase.setUpClass`,
and `test_lensing_exterior_admission.py::_build_guard_chart`.

**The refinement test is the discriminator, and it must be run before any
conclusion about this guard.** A carrier / under-resolution story predicts the
per-gap step falls like `1/n` as nodes are added. An amplitude-null story
predicts it stays pinned at `pi` forever. Measured on the two `gamma`-wall
guard boxes:

| n_gamma | max wind (rad) | rel. amplitude at that pair | `d|re|/span` | `d|im|/span` |
|---|---|---|---|---|
| 4 | 2.68 | 0.157 | 0.430 | 0.239 |
| 6 | 3.00 | 0.051 | 0.153 | 0.015 |
| 8 | 3.07 | 0.621 | 0.392 | 1.572 |
| 12 | 2.97 | 0.948 | 0.119 | 1.938 |
| 16 | 3.12 | 0.0027 | 0.0036 | 0.0160 |

(box `gamma in (0.5, 1.5)` behaves identically, reaching 3.14 rad at relative
amplitude 8.8e-6.) The step does not shrink — it converges to `pi` — while the
amplitude at the offending pair collapses and the re/im increments go smooth.
That is a null, not a carrier. This reproduces independently what the 8h-d2
Coder found by instrumenting the census and band-split boxes (refine
`n_gamma` 4 -> 6 -> 8 -> 12, step does not shrink; census trips even at
`w_max = 4`), and it confirms Inspector finding INS-5-001.

**Root cause.** `FarFieldChart` stores and splines `envelope_real` and
`envelope_imag` as separate real fields. Near an amplitude null the complex
label passes close to the origin: `arg` swings by `pi` while `re` and `im`
both pass smoothly through zero. The guard therefore measures a quantity the
interpolant never sees. `pi/2` is the right Nyquist bound for a *carrier*, but
`arg`-winding is not the right observable for a re/im spline — the failure is
in the metric, not the threshold.

**Two wrong turns taken here, both from reasoning instead of measuring:**

1. *Parity-flip story* (driver): the `(0.5, 1.5)` box straddles the `gamma = 1`
   parity wall and showed a ~`pi` step, which looked conclusive. Refuted by the
   `(1.0, 1.6)` box, which lies entirely in the saddle region with the
   `gamma = 1` node refused out of every pair, and still winds 2.5-2.7 rad.
2. *"Genuinely too coarse" story* (driver): supported by one measurement at one
   resolution showing the flips at 9-29 % relative amplitude — healthy, so not
   noise. Refuted by the refinement table above: at `n = 16` the same flip sits
   at 0.27 % amplitude with smooth re/im. Measuring a single grid cannot
   distinguish a carrier from a null; only the `n`-sweep can.

**Why the prescribed floor was not enough.** A relative-magnitude floor is the
right direction and INS-5-001 stands, but a floor alone does not retire the
bypasses: at the fixtures' actual coarseness the worst pair sits at 0.157
relative amplitude, far above any sane floor, because a coarse grid straddles
the null at moderate amplitude. A 1e-3 floor was implemented and excluded
nothing.

**FIXED 2026-07-28.** The guard now measures the complex increment
`|E_lead - E_trail|` normalized by the peak `|E_tilde|` over the WHOLE grid,
against `_FARFIELD_CARRIER_STEP_MAX = 1.0`, replacing
`_FARFIELD_CARRIER_WIND_MAX = pi/2`. All three `_skip_carrier_guard=True`
bypasses are removed and the kwarg is deleted from `from_engine`, so the
escape hatch cannot be reached for again.

Calibration across every known fixture (this is what the bound is set from,
and re-deriving it is how to change it safely):

| fixture | slice-norm | whole-norm | all-slices | verdict |
|---|---|---|---|---|
| synthetic continuous | 0.1997 | 0.1997 | 0.1997 | pass |
| synthetic zeroed-flip | 0.1997 | 0.1997 | 0.1997 | pass |
| `gamma1 (0.5,1.5)` | 1.0861 | 0.1160 | 1.0619 | pass |
| `gamma1 (1.0,1.6)` | 1.2928 | 0.1556 | 1.3703 | pass |
| band-split box | 1.3542 | 0.0000 | 1.0004 | pass |
| census dense box | 1.5289 | 0.0000 | 0.2881 | pass |
| synthetic pathological | 1.8980 | 1.8980 | 1.8980 | **raise** |

Whole-grid normalization is the only column with a real margin: worst
must-pass 0.1997 against must-raise 1.8980, a 9.5x gap, bound placed at 1.0.
Two alternatives were measured and REJECTED — top-slice normalization (margin
1.24x) and scanning all `w` slices (margin 1.38x). Do not re-propose either
without re-running the calibration.

The bound is tuning-free to state: a violation means the label changed by more
than the entire chart's peak magnitude across one node gap. At full amplitude
that is `pi/3` of winding, i.e. STRICTER than the retired `pi/2` where the
label is strong, and permissive only where it has decayed to noise. The change
does not weaken the guard; it re-points it at the quantity the spline sees.

**What the all-slices experiment taught, and why it is recorded.** Scanning
every `w` slice flags the census box at 1.37 — yet that chart meets its
accuracy bar (1.08e-3 against 5e-3). So large mid-band increments are
COMPATIBLE with an accurate chart, which means this guard is a cheap
GROSS-ALIASING SCREEN and not an accuracy proxy. The held-out eps gate
(`_gate_chart`) is the real falsifier. Treating the carrier guard as an
accuracy check is what produced both over-strict designs.

**Lesson.** For any guard that measures the phase of a complex field, the
`n`-refinement sweep is the cheap discriminator and should be run *first*: a
carrier shrinks like `1/n`, a null pins at `pi`. Reporting the relative
amplitude at the offending pair (the guard now does this in its error message)
tells you which regime you are in without a probe, but only the sweep proves
it. Two plausible mechanisms were adopted here without that sweep and both
were wrong.

## F023 — the Born rung's stated rationale is backwards, its series was missing a term, and its `b1` had the wrong sign; the chart absorbs `ln w` for free (2026-07-28, Professor commission)

`chang_refsdal/_born.py` shipped DORMANT with `b1 = 1.0`, a placeholder, and a
stated purpose that measurement contradicts. Four separate results, all
measured against the exact engine rather than argued.

**1. `b1` is derived, and the placeholder had the WRONG SIGN.**

    b1 = -lam * (x0 . A^-1 . x0) / |x0|**2        # reduced: -(1 + g'P)/(1 - g'**2)
    a0 = -lam * gamma * P / det_a                  # P = cos 2(theta_x0 - beta)

with `P` referred to the MACRO IMAGE direction, not the source. A pure point
mass gives `b1 = -1`, not `+1`. Both collapse onto `r0_sq` and `x0_dot_y`,
which `_born_factors` already computes — no new geometry, no fifth convention
site. Three algebraic forms agree to 2.2e-14 over 4000 both-parity draws.

**2. The mandated series structure was INCOMPLETE.** Expanding the Kirchhoff
integral leaves TWO terms at order `1/q2r`, not one: the imaginary `b1` term
the ansatz had, and a real, `w`-independent `a0` term it did not. Worth
1-20 % depending on shear. The build implemented its mandate faithfully; the
mandate was wrong. A structure handed down as fixed is still a hypothesis.

**3. The WHY premise is measurably backwards.** The module says the low-`w`
far zone "varies on the Einstein scale, so trained tiles there are
prior-sized". Once `exp(1j*w*phi_geo)` is demodulated out, that variation is
ENTIRELY GONE — it lives in the closed-form phase. Measured: the demodulated
residual needs 4 y-nodes across the whole annulus at `w <= 0.2`, versus 9-17
at `w >~ 1`. Einstein-scale fringe motion is a MID/HIGH-`w` problem, the
opposite of the documented rationale. The rung is worth having, for the
reverse reason.

**4. The chart absorbs `ln(w/2)` at ZERO node cost, so no low-`w` analytic
rung is needed.** The term the mid-`w` carrier omits is
`(1j*w/2)*[ln(w*r0_sq/2) + EULER_GAMMA + 2*ln(Lam)] + pi*w/4`. With `u = ln w`,
`w*ln w = exp(u)*u` is ENTIRE in `u`, and the charts already carry a `log_w`
axis. Node counts to a given held-out error are IDENTICAL with and without the
log term at every tolerance from 4e-3 down to 1e-5. The residual is also
bounded as `w -> 0` (tends to `-a0/q2r`).

**Measured ladder (positive parity), residual as an interpolation object:**

| band | serve | residual / `max|F|` | nodes (`log_w`, y) |
|---|---|---|---|
| `w < 0.5` | carrier `(a0,b1)` ALONE | 2.4e-2 - 8.7e-2 | 4-15, 4 |
| `w >= 0.5` | ppGO (both real images, full C1/C2) + complex ghost where admitted | 1.6e-3 - 2.5e-2 | 4-8, 4 |

No gap: the seam `[0.05, 0.5]` is covered by the carrier at 7-15 nodes with
prior-universal tiles. Mixing the bands the other way is catastrophic — adding
ppGO below `w = 0.05` inflates the residual by FIVE ORDERS OF MAGNITUDE through
the `1/w**2` kernel.

**Terminology, resolved because it nearly caused a double-count.** The faint
near-lens SECOND REAL IMAGE (`x_c ~ -y/|y|**2`, Morse index 1, from
`find_images`, worth 4.4e-2 - 8.4e-2 here) is NOT the COMPLEX-saddle ghost that
`farfield_ghost_term` implements (conjugate quartic pair, `Im tau_c > 0`,
Picard-Lefschetz, gated on geometric separation >= 0.7). Both are real
contributions and must be counted ONCE EACH. The complex ghost is large below
`w ~ 0.2` but UNUSABLE there — its own stationary-phase kernel diverges as
`1/w`, `1/w**2` — and is worth 2.5e-3 -> 1.6e-3 in `[0.5, 8]`.

Also measured: `ghost_kernel` raises `GhostDomainError` at
`(|y|=3.6, theta=0.5, gamma=0.25, kappa=0.3, beta=0.5)` while `find_images`
returns 2 real images, so the complex ghost is NOT universally available in the
annulus and the serve path must tolerate its absence (it does — the error is a
`LensDomainError`).

**A methodological correction worth keeping.** The first pass reported that
adding the second image made the chart WORSE (121-241 nodes). That was an
artifact of using the leading `sqrt|mu|` amplitude only; with the full C1/C2
`image_kernel` the same residual collapses to 4 nodes. An approximation used
while EVALUATING an architecture can condemn the architecture. When a
measurement says a component hurts, check that the component was measured at
full fidelity before believing it.

**What `b1`/`a0` actually buy:** ~10-25 % smaller residual in the low band at
the same node count, superseded by ppGO above `w ~ 0.5`. The load-bearing
results of this commission were the sign fix, the `a0` omission, the regime
diagnosis, and the `log_w` absorbability measurement — not an accuracy gain.
The rung's own T1 target of 1e-3 was never the right bar: in the
carrier-plus-chart architecture the criterion is how CHEAPLY THE RESIDUAL
SPLINES, not how accurate the analytic term is standing alone.

## F024 — the band-split currency `w*r0_sq` was a positive-parity coincidence; the invariant is `w*Delta_tau`, and the annulus is only "exterior" for part of the prior (2026-07-28, Professor saddle commission)

Three results, each of which invalidated something already written down.

**1. The split currency.** F023 set the carrier/ppGO band split at
`w * r0_sq ~ 8`. That is correct ONLY at positive parity, and only because
`Delta_tau ~ r0_sq / 2` happens to hold there — a coincidence of the regime,
not an identity. On the macro saddle `x0_i = y_i / a_i` with `a1 = lam - gamma
< 0`, so `r0_sq` swings 1700x with angle (3721 -> 2.2 at gamma = 1.05) while
`Delta_tau` does not; measured `r0_sq / (2*Delta_tau)` spans 0.16 to 35.6.

The invariant is `w * Delta_tau ~ 4`, with `Delta_tau` the Fermat-delay
difference of the two real images — already available from the partition via
`geometry.delay`. It coincides with SACR-C's own switch scale `RHO_END = 4`.
Confirmed both directions: at `gamma=1.6, |y|=4.24, theta=0.9`,
`Delta_tau = 0.294` gives `w_split = 13.6` and the carrier still works at
`w = 8` (N=7); at `gamma=1.2, theta=0.3`, `Delta_tau = 35.3` gives
`w_split = 0.113` and the carrier has already failed by `w = 0.5` (N=161).
`w * r0_sq` mispredicts both by two orders of magnitude.

**2. The annulus is not always exterior.** Measured caustic extent on a 241^2
source grid: max `|y|` inside the caustic is 1.85 at `gamma=0.60`, 2.95 at
0.75, 4.35 at 0.85, and `>= 6.00` for `gamma >= 0.95`; on the saddle side
`>= 6.00` at `gamma = 1.005`, 3.71 at 1.02, 2.49 at 1.05, 1.70 at 1.30.

So the target annulus `3.0 < |y| <= 4.2426` is a FAR EXTERIOR region only for
`gamma <~ 0.75` and `gamma >~ 1.03`. Between them — roughly 17 % of the
prior's uniform `(0, 1.6)` shear range — it straddles or lies INSIDE the
caustic, with fold crossings in the tile and a (0,1,1,1) census. The exterior
ladder cannot close that band; it is either an interior-chart problem or a
named refusal.

This RETRO-SCOPES F023, whose positive-parity ladder was measured at
`gamma` in {0.2, 0.25, 0.3, 0.45} only. F023's conclusions stand for
`gamma <~ 0.75` and are NOT established above it. The Professor flagged this
against its own prior report rather than letting it stand.

**3. The saddle carrier is LEAD-ONLY, and the complex ghost is harmful there.**
`|a0|, |b1| ~ 1/(gamma' - 1)` (a0 = 10.24 at gamma = 1.05), so `a0/q2r` is
O(1) wherever `q2r` is small, and `q2r` falls to 1.4 near the
positive-eigenvalue axis. Measured on `[1e-3, 0.05]`: lead-only
`sqrt|mu_macro| * exp(-1j*pi/2) * exp(1j*w*phi_geo)` gives residual
1.0e-2 - 7.4e-2 at N=4 on both `log_w` and the y-axes, while the full
`(a0,b1)` carrier gives 1.7e-2 - 1.42 and needs N = 23-65 in the y-plane. The
correction injects theta-structure that is not in `F`. Drop it on the saddle.

The complex ghost is worse than useless there: at `gamma=1.6, |y|=4.243,
w=5`, ppGO alone gives residual 1.4e-3 at N(theta)=4; adding the admitted
ghost gives 4.2e-2 at N(theta)=14. Two causes — the admission set flips across
theta inside a tile (43-54 of 65 points admitted), and, more fundamentally,
`geometry.ghost_kernel` pins its sqrt branch with
`reference_amplitude = exp(-0.5j*pi)` justified in its own docstring by "the
two real images continue into a Morse-index-1 saddle". THAT IS A
POSITIVE-PARITY STATEMENT: on the macro saddle both real images are ALREADY
index-1. The branch reference has not been re-derived for `det A < 0`. Refuse
the complex ghost on the saddle branch until it is.

**Method caveat recorded by the Professor against its own numbers:** its
theta-direction node counts demodulate by the SINGLE carrier
`exp(1j*w*phi_geo)`. Above the split the residual inherits the other image's
carrier (`Delta_tau` varies ~25 per radian, so at `w=5` the fringe spacing is
~0.05 rad), which is why ppGO's theta counts look bad at `gamma=1.2` despite a
residual of 8.2e-3. The correct object above the split is the SACR-C switched
envelope. Residual SIZES are demodulation-independent and stand; the
above-split theta NODE COUNTS are pessimistic and must be re-measured through
`switched_analytic_channels` before they size a tile.

**Pattern.** Both F023 and F024 corrected a fact that had already been written
down and acted on — F023 the `b1` sign and the missing `a0`, F024 the split
currency and the annulus's exteriority. In each case the wrong version was not
a guess but a measurement taken over too narrow a slice. A measured fact
carries the scope of its measurement; F023 said "the split is at
`w * r0_sq ~ 8`" when what was measured was "the split is at
`w * r0_sq ~ 8` FOR gamma in [0.2, 0.45]". State the sweep next to the result.

## F025 — `a0` violates F009 and belongs in no serving path; the exterior fence is `gamma < 3/4` exactly; and F023's node counts were swept in one direction only (2026-07-28)

Third correction from the same Professor thread, and the one that most
simplifies the build. Sweep for every number: `gamma` in {0.45, 0.50, 0.55,
0.60, 0.65, 0.70, 0.75}, `|y|` in {3.05, 4.2426}, `theta` in {0.3, 0.9, 1.35},
`kappa = 0`, `beta = 0` (production pins both), azimuthal sweeps at 65 points.

**1. `a0` hurts at EVERY gamma, including 0.45 — the value F023 reported on.**
Azimuthal node counts at eps 4e-3, `w = 0.01`:

| gamma | N(F) | N(lead-only) | N(a0+b1) |
|---|---|---|---|
| 0.45 | 4 | 4 (size 1.9e-2) | 11 (size 1.2e-1) |
| 0.60 | 4 | 4 (size 2.0e-2) | 20 (size 2.5e-1) |
| 0.75 | 4 | 4 (size 2.3e-2) | 44 (size 5.5e-1) |

`N(F) == N(lead)` EXACTLY at all 18 sampled (gamma, w, radius) combinations.

**2. The mechanism is an F009 violation, not a coefficient blow-up.**
`F(w -> 0) = sqrt(mu_macro)` exactly (F009), but the carrier returns
`sqrt(mu_macro) * (1 + a0/q2r)`. `a0` is a RESOLVED-IMAGE amplitude
correction, valid only for `w * Delta_tau >> 1`; below the split it is a
constant offset of size `|a0|/q2r`. `b1` alone is indistinguishable from
lead-only — its term carries `w` and vanishes correctly — so it is harmless
but buys nothing.

CONSEQUENCE: `(a0, b1)` serve NOWHERE. Below the split lead-only wins; above
it ppGO wins. Use the lead-only carrier
`sqrt(mu_macro) * exp(1j*w*phi_geo)` below the split at ALL gamma on BOTH
parities — one rule, no branch, matching the saddle recommendation. The
coefficients remain correct physics and the right macro-limit diagnostic, and
should stay in the module as such; they should not be in the serving path.

**3. Why F023 missed it — a METHOD error, not a sampling one.** F023's y-plane
node counts swept `|y|` RADIALLY at fixed `theta = 0.3`. The `a0` pathology is
AZIMUTHAL: `q2r` varies strongly with angle through
`x0 = (y1/(1-gamma), y2/(1+gamma))`. Same gamma, same radii — the radial sweep
gives N=4 and size 2.4e-2; the azimuthal sweep gives N=11 and size 1.2e-1. One
direction was swept and reported as "per y-axis".

**4. The exterior fence, in closed form.** With `u = 1/|x|**2` on the critical
curve `u = gamma*cos2theta +- sqrt(1 - gamma**2 sin**2 2theta)`, the outermost
astroid cusp is

    max |y| on the astroid = 2*gamma / sqrt(1 - gamma)             (kappa = 0)
                           = sqrt(lam) * 2*gp / sqrt(1 - gp)       (gp = gamma/lam)

verified against the measured extent to 4 decimals at five gammas (0.60 ->
1.8974, 0.70 -> 2.5560, 0.75 -> 3.0000, 0.80 -> 3.5777, 0.85 -> 4.3894).
Solving `2 s**2 + R s - 2 = 0` with `s = sqrt(1 - gamma)`:

* the annulus INNER edge `|y| = 3.0` is breached at `gamma = 3/4` EXACTLY;
* the outer edge `|y| = 3*sqrt(2)` at `gamma = 0.8423291`.

So the annulus is fully exterior for `gamma < 3/4`, straddles the caustic for
`3/4 <= gamma < 0.84233`, and is fully interior above. The earlier "`<~ 0.75`"
was a geometric guess that happened to land on the right number; encode the
derived form.

**5. F023's node counts were floors, not ceilings.** Over `gamma <= 3/4` the
true ceilings are 5 on `[1e-3, 0.05]`, **31** on `[0.05, 0.5]` (F023 said
4-15), and **27** on `[0.5, 8]` (F023 said 4-8), with 4 and 14 per y-axis in
the two sub-split bands. The `[0.5, 8]` ppGO residual rises to 2.0e-1 near the
`y2`-axis cusp and PLATEAUS rather than diverging — that cusp reaches
`|y| = 3.0` exactly at `gamma = 3/4`, which is the same fence.

**The pattern, now three deep.** F023 corrected the placeholder; F024
corrected F023's split currency and scope; F025 corrects F023's node counts
and retires its headline recommendation. None of the wrong versions was a
guess — each was a real measurement whose SWEEP was narrower than the claim
built on it. F023 swept four gammas and one angular direction. The discipline
that would have caught all three: state the sweep in the same sentence as the
result, and treat any claim that outruns it as unmeasured.

## F026 — one closed form gives the caustic extent on BOTH parities; the saddle fence is a BAND; and F024's measured extent table under-reported the spike (2026-07-28)

Sweep: `kappa = 0`, `beta = 0`; closed form checked against a direct caustic
parametrisation (2e6 points per branch) at 16 gammas spanning both parities
(0.45 to 3.0) — agreement to 4 decimals at all 16. `kappa != 0` enters only
through the mass-sheet reduction (`gp = gamma/lam`, `|y| -> sqrt(lam)*|y'|`).

**The closed form.** The critical curve satisfies
`u**2 - 2*gamma*cos(2*theta)*u - (1 - gamma**2) = 0` with `u = 1/|x|**2`, and
the caustic is `y = ((a1-u)x1, (a2-u)x2)`, `a1,a2 = 1 -+ gamma`. Eliminating
`theta`, `|y|**2` becomes a function of `u` ALONE:

    |y|**2 (u) = 2*u - 3 + 2*gamma**2/u + (1 - gamma**2)/u**2,
                                       u in [abs(1-gamma), 1+gamma]
    f'(u) = (2/u**3) * (u - 1) * (u**2 + u + 1 - gamma**2)
        -> stationary at u = 1 and u_c = (sqrt(4*gamma**2 - 3) - 1)/2

`u_c > 0` IFF `gamma > 1`. That single fact is the whole difference between the
two parities.

* POSITIVE PARITY: `u**2 + u + 1 - gamma**2 > 0` for `u > 0`, so `f` falls then
  rises and the maximum is at the endpoint `u = 1 - gamma`:
  `max|y| = 2*gamma/sqrt(1 - gamma)`, UNCONDITIONALLY for all `gamma < 1`.
  The fence recorded in F025 is right with no caveat.
* SADDLE: `u_c` is always interior (`u_c > gamma - 1` iff `gamma > 1`), and
  there `gamma**2 = u_c**2 + u_c + 1`, collapsing `f` to

      max|y|_saddle = sqrt(max( 4*u_c + 1/u_c - 2,  4*gamma**2/(gamma + 1) ))

  the two candidates being the OFF-AXIS cusp (`u_c`) and the ON-AXIS cusp
  (`u = gamma + 1`, giving `2*gamma/sqrt(gamma + 1)`).

**The non-monotonicity is real: a cusp switch.** The outermost point sits on
the off-axis cusp for `gamma < 1.177651` and on the on-axis cusp above; the
`u_c` branch falls while `2*gamma/sqrt(gamma+1)` rises, so the extent MINIMISES
at the switch: `max|y| = 1.596072` at `gamma = 1.177651`.

**F024's extent table under-reported and is RETIRED.** Its 241^2 source grid
misses the thin spike. True vs grid: `gamma=1.005` -> 9.939 vs 6.002
(grid-capped); `1.02` -> 4.886 vs 3.712; `1.05` -> 3.008 vs 2.491. Use the
closed form.

**The saddle fence is a BAND, not a one-sided inequality.** Solving
`4v + 1/v - 2 = R**2` then `gamma = sqrt(v**2 + v + 1)`:

| edge | exact | value |
|---|---|---|
| inner, `\|y\| = 3.0` | `sqrt((189 - 15*sqrt(105))/32)` | 1.0502342 |
| outer, `\|y\| = 3*sqrt(2)` | `sqrt(63 - 24*sqrt(6))/2` | 1.0261879 |

and the RISING branch reaches the inner edge again at `4*gamma**2/(gamma+1) = 9`,
i.e. `4*gamma**2 - 9*gamma - 9 = 0`, `gamma = 3` EXACTLY (outer edge at
`gamma = 5.342329`). So the annulus is exterior for
`1.0502342 < gamma < 3`. The prior stops at `gamma = 1.6` where
`max|y| = 1.9846`, clear of 3.0 by a factor 1.51 — safe, but write the fence as
a band in case the prior is ever widened.

**Both sides diverge as `|gamma - 1|**(-1/2)`** — positive `2/sqrt(1-gamma)`,
saddle `1/sqrt(2*(gamma-1))` — so they join at the wall in power but not in
amplitude: the astroid is `2*sqrt(2) ~ 2.83x` larger at equal distance from it.

**The deltoid is spiky too, but a scalar fence costs far less there.** Angular
width beyond radius R (upper bound on the inside-fraction): `gamma=1.02` gives
<= 3.95 % beyond `|y| = 3.0`, `1.05` gives <= 0.029 %. On a uniform `(0, 1.6)`
prior the POSITIVE fence `gamma < 3/4` discards 15.6 % of the shear range while
the SADDLE fence `gamma > 1.0502342` discards only 3.1 %. Ship the scalar fence
on the saddle; spend per-theta admission on the positive branch, where it is
worth 5x more.

**METHOD WARNING, and it applies to the driver's own probe.** A 721-point ring
scan reported 0.00 % inside at `gamma=1.03, |y|=3.6` where the true spike is
NARROWER than the 0.0087 rad sampling. The driver's 1441-point scan
(0.0044 rad) returned 0.42 % at `gamma=0.80` and 2.91 % at `0.90`, just under
the analytic bounds 0.58 % and 2.98 % — it held, but only just. TRUST THE
PARAMETRISATION, NOT RING SCANS, FOR SPIKE GEOMETRY. A caustic that is a
directional spike cannot be characterised by sampling angles; sample the curve
itself.

**Fourth correction in one thread.** F023 fixed the placeholder; F024 fixed
F023's split currency and scope; F025 fixed F024's node counts and retired
F023's headline; F026 retires F024's extent table. Every wrong version was a
real measurement whose resolution or sweep was narrower than the claim built on
it. Grid at 241^2, sweep at four gammas, radial-only, ring scan at 0.0087 rad —
four different ways to under-resolve, four wrong tables.
