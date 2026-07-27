# Professor short-term (session: Build 8h-d2 four-defect rulings)

Consultation on the exterior far-field tiling/training path. Four rulings derived
from code + physics. Key durable findings (Dreamer: promote the starred ones):

## Q4 * — astroid cusp SOURCE angles are gauge-exact {0, ±pi/2, pi} (kappa=0, pos parity)
DERIVED in closed form (not just measured). For kappa=0, beta=0 eigenframe,
A = diag(1-gamma, 1+gamma). Astroid cusps sit at LENS-plane angles theta = 0,
pi/2, pi, 3pi/2 (the shear axes). Mapping through y = A x - x/|x|^2:
- theta=0: v = 1+gamma, r=1/sqrt(1+gamma), y = (-2 gamma r, 0) -> SOURCE angle pi.
- theta=pi/2: v = 1-gamma, r=1/sqrt(1-gamma), y = (0, +2 gamma r) -> SOURCE angle pi/2.
By the C4v symmetry the four cusps land at source angles EXACTLY {0, pi/2, pi, -pi/2},
INDEPENDENT of gamma. The magnitude |y_cusp| = 2 gamma / sqrt(1±gamma) IS gamma-dependent,
but the ANGLES are not. So `from_engine` can cusp-align theta_c nodes by inserting
whichever of {0, ±pi/2, pi} fall in theta_c_range as exact nodes — pure closed form,
no surrogate_training import, no circular dependency. This matches
`_cusp_source_angles` output (which detects them numerically via branch-speed sweep).
NOTE: this is the POSITIVE-parity astroid only. Saddle (two deltoids, 3 cusps each)
is NOT on-axis in general — the closed-form shortcut is positive-parity-specific.

## Q3 * — far-field carrier IS recomputed fresh at serve; residual defect is winding in E
Serve path (likelihood.py ~1640): `geom = ChangRefsdalChannels(dense_w).
geometry_partition(...)` rebuilds delays FRESH from the query source (min-relative,
t_min-subtracted), and `reconstruct_farfield` re-applies exp(1j w tau_a) with those
fresh delays. So the CARRIER is query-fresh (analytic), like the interior tau_c.
BUT the far-field label E_ff = F - sum_a H_a exp(1j w (tau_a - t_min(x))) is built in
EACH node's own t_min(x) frame at train time. The spline interpolates E_ff across
nodes with different t_min frames -> E_ff carries a fast winding phase exp(-1j w t_min(x))
that varies node-to-node -> E non-smooth even though the carrier is re-applied fresh.
This is the SAME class of pathology the interior avoids because tau_c-demodulation is
algebraically frame-invariant, whereas the far-field label parks carrier at tau_c=0
(constant) and leaves t_min(x) in the delays only.
Measured: d t_min/d rho ~ -1.03 at gamma=0.30 theta_c=0.4; n_rho=4 -> node gap ~5e-2;
w_max~60 -> per-gap winding ~ 60*1.03*5e-2 ~ 3.1 rad. That EXCEEDS a pi/2 continuity
bound, so a t_min-continuity assertion on current ship tiling would FAIL -> this defect
is NOT guard-only; it needs the frame-invariant relabel (demodulate each node's E by
exp(+1j w t_min(node)) at train, re-apply exp(-1j w t_min(query)) at serve) OR much
finer rho tiling. Frame-invariant relabel is the physically-correct, lower-risk route
(mirrors the interior's frame-invariance; t_min is real so pure phase, no magnitude change).

## Q1/Q2 — ppGO annulus gauge conversion + one converter
Q2: annulus_rho(gamma,|y|,kappa=0) := |y| / caustic_geometry(gamma,kappa)[0] is the
single correct ppGO-gauge def. likelihood.py:1375 hypot(y1,y2)/reach is byte-equivalent
(reach = _caustic_reach = caustic_geometry(...)[0], same call) -> pure refactor.
kappa!=0 caveat: caustic_geometry accepts kappa but production pins kappa=0; annulus_rho
must forward kappa, and callers that hard-code kappa=0 stay correct.
Q1: to recover the annulus rho the region actually covers from region_exclusion_rho
(additive gauge, rho_add = 1 + |y|_inner - coordinate_radius_min):
|y|_inner = region_exclusion_rho - 1 + coordinate_radius_min; then annulus rho =
|y|_inner / reach_scalar. reach_scalar = MAX directional radius, coordinate_radius_min =
MIN directional radius. Dividing |y|_inner by the MAX reach gives the SMALLEST annulus
rho -> reads w_cert at a cell CLOSER to the caustic (harder, higher w_cert) -> conservative
(never certifies a cell easier than reality). The mixed use (min radius additively,
max radius as divisor) makes recovered |y|_inner a LOWER bound on the true per-column
inner physical radius -> annulus rho is a lower bound -> conservative. GOOD.
