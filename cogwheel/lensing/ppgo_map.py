"""
Certified post-point-mass-geometric-optics (ppGO) frequency-floor map.

WHAT
----
`CertifiedPpgoMap` is a load-time, hash-pinned lookup table of the
*certified-ppGO frequency floor* ``w_cert`` per
``(parity, gamma-band, caustic-frame annulus)`` cell.  ``w_cert`` is the
smallest dimensionless frequency at (and above) which the bare
point-mass geometric-optics reconstruction -- the plain image-kernel sum
``geometric_amplification`` (``sum_a exp(1j w tau_a) * image_kernel_a``,
no envelope) -- reproduces the exact engine total ``F`` to the F-normalized
certification bar

    max_over_band( |F - ppGO_full| / max|F| )  <  BAR   (BAR = 1e-4)

(``F`` and the ppGO sum are first put on a common time origin -- the engine
demodulates ``exact_total`` at the minimum image delay ``t_min``, so the
absolute-delay ppGO glue is demodulated by ``exp(-1j w t_min)`` before
differencing; otherwise the residual is dominated by a global winding
phase, not the geometric-optics defect).

Below ``w_cert`` the wave zone and the near-caustic region are *chart*
territory (Build 8h-a lever 1's band-split dispatch, WP2); ``w_cert``
tells the dispatch where the split lies.  ``w_cert`` is **measured**, never
asserted: `scripts/train_ppgo_map.py` runs an offline validation sweep
against exact references and records the floor as a hash-pinned data
product, mirroring the `pearcey_table` pattern (DATA_CONTRACTS + registry
+ loader hash check + refuse-to-certify when the map is absent / corrupt).

SUP-OVER-W FLOOR (why not the first downward crossing)
------------------------------------------------------
The ppGO error is *non-monotone* in ``w``: the neglected higher-order
stationary-phase terms carry the image-delay phases ``exp(1j w tau_a)``,
which beat at the pairwise delay differences ``Delta tau_ab`` and can push
the residual back above the bar after a first crossing.  ``w_cert`` for a
cell is therefore the **sup-over-w floor** -- the smallest ``w`` such that
the error stays below the bar for *all* sampled ``w'`` in ``[w, w_wall]``
(equivalently the last upward re-crossing), not the first downward
crossing.  The raw measured onset is stored; the safety margin is applied
at query / dispatch time (`w_trust`).

BEYOND THE WALL
---------------
Above the Schwinger wall (astroid ``w = 443.7``, saddle ``w = 58``; mass
``m > ~458 Msun``) the exact reference does not exist -- the engine raises
by name.  A cell whose ppGO error never clears the bar below the wall has
no certified floor: it is marked **UNKNOWN** (`UNKNOWN` sentinel), never
certified and never extrapolated.  Dispatch (WP2) then refuses / charts
rather than serving ppGO there.

TRUNCATION ON REFUSAL AND MEASURED CEILINGS
-------------------------------------------
An engine refusal in the middle of a cell's ``w`` sweep does *not*
invalidate the whole cell.  The dominant refusal -- the saddle-image
wave branch above ``W_CEILING_SCHWINGER = 60`` (positive-parity cells
sweep to ``w = 443.7`` and cross it) -- is MONOTONE in ``w``: it refuses
for *all* ``w`` above a per-cell ceiling, so the accepted set is a prefix
``[w_min, w_ceiling]``.  `_measure_cell` finds that maximal accepted
prefix per angle by bisection on the prefix-endpoint index (O(log n_w)
engine calls, not a re-sweep), certifies the sup-over-w floor on the
accepted range only, and stores a per-cell ``w_ceiling`` (the min over
angles of each angle's maximal accepted ``w``).  A cell certifies on its
measured range even when its top-``w`` refuses; **beyond the ceiling stays
UNKNOWN** -- never extrapolated.  A genuine refusal at the *lowest* ``w``
(no accepted prefix at all) still invalidates the cell (`STATUS_INVALID`).

OUTER-ANNULUS MEASURED-RHO CAP
------------------------------
The outermost annulus ``[4.0, inf)`` is sampled at a single finite
representative radius (``rho = lo * 1.5``), yet its open outer edge would
imply certification out to ``rho = inf`` -- unsound from one finite
sample (the same soundness rule as an uncertified ``w`` ceiling).  Each
cell therefore carries a finite ``rho_measured_max`` (the finite band top
edge for inner annuli; the sampled radius for the open outer annulus), and
*every* cell accessor returns `UNKNOWN` for a query ``rho`` beyond it via
the same out-of-grid fall-through (no new sentinel).  A
Professor-certified monotone-outward argument may later replace this cap;
until then the infinite tail is UNKNOWN.

SERIALIZATION AND INTEGRITY
---------------------------
The artifact is a single ``.npz`` of plain float64 arrays (the parity
codes, the gamma-band and annulus edges, and the per-cell ``w_cert``,
diagnostic ``w_cert`` at the 1e-3 probe bar, measured ``w_ceiling``,
status codes, interpolability flags, and measured-``rho`` cap
``rho_measured_max``) plus a JSON provenance scalar carrying the grid
bounds, the certification bar, the safety-margin rule, the walls and a
SHA1 content hash.  `CertifiedPpgoMap.load` reads it with
``allow_pickle=False``, recomputes the content hash and refuses (raises
``ValueError``) on mismatch.  No new exception class is introduced --
integrity failures are plain ``ValueError`` / ``OSError`` / ``KeyError``
that the opt-in `use_certified_ppgo_map` switch turns into a
refuse-to-certify (the process-global map stays ``None`` and every query
returns ``UNKNOWN``).
"""
from __future__ import annotations

import hashlib
import json
import math
import warnings
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Sequence

import numpy as np

__all__ = ['CertifiedPpgoMap', 'UNKNOWN',
           'set_certified_ppgo_map', 'get_certified_ppgo_map',
           'use_certified_ppgo_map', 'certified_w_cert', 'certified_w_trust',
           'certified_w_ceiling', 'caustic_geometry', 'annulus_rho',
           'build_map', 'save_map', 'map_summary',
           'CERTIFICATION_BAR', 'DIAGNOSTIC_BAR',
           'W_TRUST_MULTIPLIER', 'W_TRUST_ADDITIVE', 'MAX_CELL_JUMP',
           'ASTROID_WALL', 'SADDLE_WALL']


# ----------------------------------------------------------------------
# UNKNOWN sentinel.
# ----------------------------------------------------------------------

class _UnknownSentinel:
    """Singleton returned by a query with no certified answer.

    Distinct from any ``float`` so a consumer tests ``result is UNKNOWN``
    unambiguously (a certified boundary is always a plain ``float``).
    """

    _instance: '_UnknownSentinel | None' = None

    def __new__(cls) -> '_UnknownSentinel':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return 'UNKNOWN'

    def __bool__(self) -> bool:
        return False


#: Sentinel returned when a cell is uncertified / beyond the wall, or when
#: no map is installed.  WP2 dispatch refuses / charts on this, never serves
#: ppGO.
UNKNOWN = _UnknownSentinel()


# ----------------------------------------------------------------------
# Constants (single authoritative source for the certification contract).
# ----------------------------------------------------------------------

#: Shipped package-data artifact name (under ``cogwheel/data/``).
_DEFAULT_MAP_NAME = 'certified_ppgo_map.npz'

#: Provenance schema version for the artifact.  Bumped 0.1.0 -> 0.2.0 with
#: the per-cell ``w_ceiling`` (truncation-on-refusal) and
#: ``rho_measured_max`` (outer-annulus cap) grids; old ceiling-less
#: artifacts lack these arrays and the loader hard-refuses them.
_SCHEMA_VERSION = '0.2.0'

#: F-normalized ppGO certification bar (Professor: 1e-3 is a probe / onset
#: bar only -- per-node ppGO error accumulates coherently across the band
#: as ~eps * SNR^2 and 1e-3 blows the 0.05-nat lnL target).
CERTIFICATION_BAR = 1e-4

#: Diagnostic (probe / onset) bar kept as a reported column only; never the
#: certified boundary.
DIAGNOSTIC_BAR = 1e-3

#: Safety-margin rule (applied at query / dispatch time, raw ``w`` units):
#: ``w_trust = max(W_TRUST_MULTIPLIER * w_cert, w_cert + W_TRUST_ADDITIVE)``.
W_TRUST_MULTIPLIER = 1.5
W_TRUST_ADDITIVE = 2.0

#: Max-jump guard: adjacent cells whose ``w_cert`` differ by more than this
#: (== the additive margin) are non-interpolable -- WP2 falls that region to
#: charts rather than interpolating across the discontinuity.
MAX_CELL_JUMP = 2.0

#: Schwinger walls (dimensionless ``w``): the exact reference does not exist
#: above these (mass ``m > ~458 Msun``).
ASTROID_WALL = 443.7
SADDLE_WALL = 58.0

#: Parity axis encoding (integer codes stored as float64).
_PARITY_CODES = {'positive': 0.0, 'saddle': 1.0}

#: Per-cell status codes (stored as float64).
STATUS_CERTIFIED = 0.0
STATUS_BEYOND_WALL = 1.0      # ppGO error never clears the bar below the wall
STATUS_INVALID = 2.0          # engine refusal / parity-gamma mismatch / no ref

#: Default caustic-frame annulus edges ``rho = |y| / caustic_reach``.  The
#: single ``1.0`` edge splits the interior 4-image band (``0.9 <= rho < 1``)
#: from the exterior 2-image band (``1 <= rho < 1.5``): interior and exterior
#: are DIFFERENT reconstruction objects and never share a cell.
_DEFAULT_RHO_EDGES: tuple[float, ...] = (0.0, 0.5, 0.9, 1.0, 1.5, 2.5, 4.0,
                                         math.inf)


# ----------------------------------------------------------------------
# Content hash (integrity).
# ----------------------------------------------------------------------

def _content_hash(parity_codes: np.ndarray, gamma_edges: np.ndarray,
                  rho_edges: np.ndarray, w_cert: np.ndarray,
                  w_cert_diagnostic: np.ndarray, w_ceiling: np.ndarray,
                  cell_status: np.ndarray, interpolable: np.ndarray,
                  rho_measured_max: np.ndarray,
                  scalars: Sequence[float]) -> str:
    """SHA1 over the stored arrays and the certification scalars (float64).

    ``w_ceiling`` and ``rho_measured_max`` are hashed alongside the other
    grids so a ceiling-less or tampered artifact fails the hash and
    `use_certified_ppgo_map` refuses (the 8g-b tag philosophy).
    """
    hasher = hashlib.sha1()
    for array in (parity_codes, gamma_edges, rho_edges, w_cert,
                  w_cert_diagnostic, w_ceiling, cell_status, interpolable,
                  rho_measured_max):
        hasher.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
    hasher.update(np.asarray(scalars, dtype=np.float64).tobytes())
    return hasher.hexdigest()


def _hash_scalars(provenance: dict) -> list[float]:
    """The scalar certification parameters folded into the content hash."""
    return [float(provenance['certification_bar']),
            float(provenance['diagnostic_bar']),
            float(provenance['w_trust_multiplier']),
            float(provenance['w_trust_additive']),
            float(provenance['max_cell_jump']),
            float(provenance['astroid_wall']),
            float(provenance['saddle_wall']),
            float(provenance['kappa'])]


# ----------------------------------------------------------------------
# The map.
# ----------------------------------------------------------------------

@dataclass
class CertifiedPpgoMap:
    """Hash-pinned lookup of the certified-ppGO frequency floor per cell.

    Construct one from a shipped artifact with `CertifiedPpgoMap.load`, or
    build a fresh one offline with `build_map`.

    Attributes
    ----------
    parity_codes : ndarray, shape (2,)
        The parity axis, ``[0.0, 1.0]`` = ``['positive', 'saddle']``.
    gamma_edges : ndarray, shape (n_gamma + 1,)
        Strictly increasing gamma-band edges (an edge sits exactly at the
        parity boundary ``gamma = 1.0`` so no band spans it).
    rho_edges : ndarray, shape (n_rho + 1,)
        Strictly increasing caustic-frame annulus edges (last edge ``inf``).
    w_cert_grid : ndarray, shape (2, n_gamma, n_rho)
        Raw measured sup-over-w floor per cell (``nan`` where uncertified;
        read `cell_status_grid`, not ``nan``, to decide UNKNOWN).
    w_cert_diagnostic_grid : ndarray, same shape
        The floor at the 1e-3 diagnostic bar (reported column only).
    w_ceiling_grid : ndarray, same shape
        Per-cell measured ``w`` ceiling -- the min over angles of each
        angle's maximal accepted ``w`` (truncation-on-refusal).  A
        certified cell is trusted only on ``[w_cert, w_ceiling]``; beyond
        the ceiling is UNKNOWN.  ``nan`` where the cell is invalid.
    cell_status_grid : ndarray, same shape
        ``STATUS_CERTIFIED`` / ``STATUS_BEYOND_WALL`` / ``STATUS_INVALID``.
    interpolable_grid : ndarray, same shape
        ``1.0`` where every certified neighbour is within `MAX_CELL_JUMP`,
        else ``0.0`` (WP2 must not interpolate across a flagged cell).
    rho_measured_max_grid : ndarray, same shape
        Finite ``rho`` upper bound at which the cell was actually sampled
        (the finite band top edge for inner annuli; ``lo * 1.5`` for the
        open outer annulus).  Every accessor returns UNKNOWN for a query
        ``rho`` above this bound -- an infinite annulus is never certified
        from one finite sample.
    provenance : dict
        Grid bounds, certification bar, safety-margin rule, walls,
        ``content_hash`` and build metadata.
    """

    parity_codes: np.ndarray
    gamma_edges: np.ndarray
    rho_edges: np.ndarray
    w_cert_grid: np.ndarray
    w_cert_diagnostic_grid: np.ndarray
    w_ceiling_grid: np.ndarray
    cell_status_grid: np.ndarray
    interpolable_grid: np.ndarray
    rho_measured_max_grid: np.ndarray
    provenance: dict

    # -- construction -------------------------------------------------

    @classmethod
    def from_arrays(cls, parity_codes: np.ndarray, gamma_edges: np.ndarray,
                    rho_edges: np.ndarray, w_cert_grid: np.ndarray,
                    w_cert_diagnostic_grid: np.ndarray,
                    w_ceiling_grid: np.ndarray,
                    cell_status_grid: np.ndarray,
                    interpolable_grid: np.ndarray,
                    rho_measured_max_grid: np.ndarray,
                    provenance: dict) -> 'CertifiedPpgoMap':
        """Validate shapes and assemble a `CertifiedPpgoMap`."""
        parity_codes = np.ascontiguousarray(parity_codes, dtype=np.float64)
        gamma_edges = np.ascontiguousarray(gamma_edges, dtype=np.float64)
        rho_edges = np.ascontiguousarray(rho_edges, dtype=np.float64)
        grids = {
            'w_cert_grid': np.ascontiguousarray(w_cert_grid, dtype=np.float64),
            'w_cert_diagnostic_grid': np.ascontiguousarray(
                w_cert_diagnostic_grid, dtype=np.float64),
            'w_ceiling_grid': np.ascontiguousarray(
                w_ceiling_grid, dtype=np.float64),
            'cell_status_grid': np.ascontiguousarray(
                cell_status_grid, dtype=np.float64),
            'interpolable_grid': np.ascontiguousarray(
                interpolable_grid, dtype=np.float64),
            'rho_measured_max_grid': np.ascontiguousarray(
                rho_measured_max_grid, dtype=np.float64),
        }
        if parity_codes.ndim != 1 or gamma_edges.ndim != 1 \
                or rho_edges.ndim != 1:
            raise ValueError('ppGO map axes must be 1-D.')
        if gamma_edges.size < 2 or rho_edges.size < 2:
            raise ValueError('ppGO map needs at least one gamma and one rho '
                             'band.')
        if np.any(np.diff(gamma_edges) <= 0.0):
            raise ValueError('gamma_edges must be strictly increasing.')
        if np.any(np.diff(rho_edges) <= 0.0):
            raise ValueError('rho_edges must be strictly increasing.')
        expected = (parity_codes.size, gamma_edges.size - 1, rho_edges.size - 1)
        for name, array in grids.items():
            if array.shape != expected:
                raise ValueError(
                    f'{name} shape {array.shape} does not match the grid '
                    f'{expected}.')
        return cls(parity_codes, gamma_edges, rho_edges,
                   grids['w_cert_grid'], grids['w_cert_diagnostic_grid'],
                   grids['w_ceiling_grid'], grids['cell_status_grid'],
                   grids['interpolable_grid'], grids['rho_measured_max_grid'],
                   dict(provenance))

    # -- load ---------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path | None = None) -> 'CertifiedPpgoMap':
        """Load and hash-verify a ppGO-map artifact.

        Parameters
        ----------
        path : str or Path, optional
            Explicit artifact path; ``None`` resolves the shipped
            package-data default under ``cogwheel/data/``.

        Returns
        -------
        CertifiedPpgoMap
            The reconstructed, hash-verified map.

        Raises
        ------
        ValueError
            If the recomputed content hash does not match the stored one
            (corrupt / stale artifact).  `use_certified_ppgo_map` treats
            this -- and any ``OSError`` / ``KeyError`` from a missing /
            malformed file -- as a refuse-to-certify (global stays ``None``).
        """
        if path is None:
            path = cls._default_artifact_path()
        with np.load(path, allow_pickle=False) as data:
            parity_codes = np.asarray(data['parity_codes'], dtype=np.float64)
            gamma_edges = np.asarray(data['gamma_edges'], dtype=np.float64)
            rho_edges = np.asarray(data['rho_edges'], dtype=np.float64)
            w_cert_grid = np.asarray(data['w_cert'], dtype=np.float64)
            w_cert_diagnostic_grid = np.asarray(data['w_cert_diagnostic'],
                                                dtype=np.float64)
            # A ceiling-less (pre-0.2.0) artifact lacks these keys; the
            # direct item access raises KeyError, which `use_certified_ppgo_map`
            # turns into a refuse-to-certify (global stays None).
            w_ceiling_grid = np.asarray(data['w_ceiling'], dtype=np.float64)
            cell_status_grid = np.asarray(data['cell_status'],
                                          dtype=np.float64)
            interpolable_grid = np.asarray(data['interpolable'],
                                           dtype=np.float64)
            rho_measured_max_grid = np.asarray(data['rho_measured_max'],
                                               dtype=np.float64)
            provenance = json.loads(str(data['provenance']))

        expected = provenance.get('content_hash')
        actual = _content_hash(parity_codes, gamma_edges, rho_edges,
                               w_cert_grid, w_cert_diagnostic_grid,
                               w_ceiling_grid, cell_status_grid,
                               interpolable_grid, rho_measured_max_grid,
                               _hash_scalars(provenance))
        if expected != actual:
            raise ValueError(
                f'Certified-ppGO map content-hash mismatch: stored '
                f'{expected!r}, recomputed {actual!r}. The artifact is '
                f'corrupt or stale; regenerate with '
                f'scripts/train_ppgo_map.py.')
        return cls.from_arrays(parity_codes, gamma_edges, rho_edges,
                               w_cert_grid, w_cert_diagnostic_grid,
                               w_ceiling_grid, cell_status_grid,
                               interpolable_grid, rho_measured_max_grid,
                               provenance)

    @staticmethod
    def _default_artifact_path() -> Path:
        """Resolve the shipped package-data artifact under cogwheel/data."""
        return Path(str(files('cogwheel').joinpath('data',
                                                    _DEFAULT_MAP_NAME)))

    # -- safety-margin rule (single authoritative source) -------------

    @staticmethod
    def w_trust_from_cert(w_cert: float) -> float:
        """Apply the Professor-authorized safety margin to a raw floor.

        ``w_trust = max(1.5 * w_cert, w_cert + 2.0)`` in raw ``w`` units.
        WP2 dispatch consumes this one rule so the margin lives in exactly
        one place.
        """
        return max(W_TRUST_MULTIPLIER * w_cert, w_cert + W_TRUST_ADDITIVE)

    # -- cell indexing ------------------------------------------------

    def _parity_index(self, parity: str) -> int:
        """Row index of ``parity`` (``'positive'`` / ``'saddle'``)."""
        try:
            code = _PARITY_CODES[parity]
        except KeyError:
            raise ValueError(
                f"parity must be 'positive' or 'saddle', got {parity!r}.")
        matches = np.flatnonzero(self.parity_codes == code)
        if matches.size == 0:
            raise ValueError(f'parity {parity!r} absent from this map.')
        return int(matches[0])

    @staticmethod
    def _band_index(edges: np.ndarray, value: float) -> int | None:
        """Band index of ``value`` in ``edges`` (``None`` if out of range)."""
        if not math.isfinite(value):
            return None
        if value < edges[0] or value > edges[-1]:
            return None
        index = int(np.searchsorted(edges, value, side='right') - 1)
        # A value exactly on the top edge lands in the last band.
        return min(index, edges.size - 2)

    def _cell(self, parity: str, gamma: float, rho: float
              ) -> tuple[int, int, int] | None:
        """``(p, gi, ri)`` index of the cell, or ``None`` if out of grid.

        A query ``rho`` above the cell's finite ``rho_measured_max`` (the
        open outer annulus was sampled at one finite radius) returns
        ``None`` via the same out-of-grid fall-through -- the infinite
        tail is never certified from that single sample, so every accessor
        yields UNKNOWN there.
        """
        p = self._parity_index(parity)
        gi = self._band_index(self.gamma_edges, float(gamma))
        ri = self._band_index(self.rho_edges, float(rho))
        if gi is None or ri is None:
            return None
        if float(rho) > self.rho_measured_max_grid[p, gi, ri]:
            return None
        return p, gi, ri

    # -- queries ------------------------------------------------------

    def w_cert(self, parity: str, gamma: float, rho: float
               ) -> float | _UnknownSentinel:
        """Raw certified-ppGO floor for the cell, or `UNKNOWN`.

        Returns the stored sup-over-w floor (a plain ``float``) only for a
        ``STATUS_CERTIFIED`` cell.  Out-of-grid, beyond-wall and invalid
        cells return the `UNKNOWN` sentinel -- WP2 refuses / charts there
        and never extrapolates.

        Parameters
        ----------
        parity : str
            ``'positive'`` or ``'saddle'``.
        gamma : float
            External shear magnitude.
        rho : float
            Caustic-frame annulus coordinate ``|y| / caustic_reach``.
        """
        cell = self._cell(parity, gamma, rho)
        if cell is None:
            return UNKNOWN
        if self.cell_status_grid[cell] != STATUS_CERTIFIED:
            return UNKNOWN
        value = float(self.w_cert_grid[cell])
        if not math.isfinite(value):
            return UNKNOWN
        return value

    def w_trust(self, parity: str, gamma: float, rho: float
                ) -> float | _UnknownSentinel:
        """Margin-inflated dispatch floor `w_trust_from_cert(w_cert)`.

        `UNKNOWN` passes straight through (an uncertified cell has no
        trusted floor).
        """
        floor = self.w_cert(parity, gamma, rho)
        if floor is UNKNOWN:
            return UNKNOWN
        return self.w_trust_from_cert(float(floor))

    def w_ceiling(self, parity: str, gamma: float, rho: float
                  ) -> float | _UnknownSentinel:
        """Measured ``w`` ceiling for the cell, or `UNKNOWN`.

        Returns the stored per-cell ceiling (a plain ``float``) only for a
        ``STATUS_CERTIFIED`` cell -- the top of the trusted range
        ``[w_cert, w_ceiling]``; WP2 dispatch splits the ppGO band at
        ``min(parity_wall, w_ceiling)`` and refuses / charts above it.
        Out-of-grid, beyond-``rho_measured_max``, beyond-wall and invalid
        cells return the `UNKNOWN` sentinel (mirrors `w_cert` exactly; no
        new sentinel).

        Parameters
        ----------
        parity : str
            ``'positive'`` or ``'saddle'``.
        gamma : float
            External shear magnitude.
        rho : float
            Caustic-frame annulus coordinate ``|y| / caustic_reach``.
        """
        cell = self._cell(parity, gamma, rho)
        if cell is None:
            return UNKNOWN
        if self.cell_status_grid[cell] != STATUS_CERTIFIED:
            return UNKNOWN
        value = float(self.w_ceiling_grid[cell])
        if not math.isfinite(value):
            return UNKNOWN
        return value

    def is_interpolable(self, parity: str, gamma: float, rho: float) -> bool:
        """Whether the cell may be interpolated across (max-jump guard).

        ``False`` for any UNKNOWN cell and for a certified cell flagged
        non-interpolable (a certified neighbour differs by more than
        `MAX_CELL_JUMP`); WP2 then falls that region to charts.
        """
        cell = self._cell(parity, gamma, rho)
        if cell is None:
            return False
        if self.cell_status_grid[cell] != STATUS_CERTIFIED:
            return False
        return bool(self.interpolable_grid[cell] > 0.0)

    def cell_status(self, parity: str, gamma: float, rho: float) -> str:
        """Human-readable status of the cell (diagnostic)."""
        cell = self._cell(parity, gamma, rho)
        if cell is None:
            return 'out_of_grid'
        code = self.cell_status_grid[cell]
        return {STATUS_CERTIFIED: 'certified',
                STATUS_BEYOND_WALL: 'beyond_wall',
                STATUS_INVALID: 'invalid'}.get(code, 'invalid')


# ----------------------------------------------------------------------
# Process-global map (opt-in switch, mirrors the Pearcey-table pattern).
# ----------------------------------------------------------------------

_CERTIFIED_PPGO_MAP: CertifiedPpgoMap | None = None


def set_certified_ppgo_map(ppgo_map: CertifiedPpgoMap | None) -> None:
    """Install (or clear, with ``None``) the process-global ppGO map."""
    global _CERTIFIED_PPGO_MAP
    _CERTIFIED_PPGO_MAP = ppgo_map


def get_certified_ppgo_map() -> CertifiedPpgoMap | None:
    """Return the process-global ppGO map (``None`` if unset)."""
    return _CERTIFIED_PPGO_MAP


def use_certified_ppgo_map(path: str | Path | None = None) -> bool:
    """Load and install the process-global ppGO map (opt-in switch).

    Returns ``True`` on success.  On ANY load / hash anomaly the global is
    left cleared (``None``) and ``False`` is returned, so every
    `certified_w_cert` query returns `UNKNOWN` and WP2 refuses / charts --
    the map is never installed in a state that could certify a wrong floor.
    """
    try:
        ppgo_map = CertifiedPpgoMap.load(path)
    except (OSError, ValueError, KeyError) as error:
        warnings.warn(f'Certified-ppGO map unavailable ({error}); ppGO '
                      f'dispatch will refuse / chart.', RuntimeWarning)
        set_certified_ppgo_map(None)
        return False
    set_certified_ppgo_map(ppgo_map)
    return True


def certified_w_cert(parity: str, gamma: float, rho: float
                     ) -> float | _UnknownSentinel:
    """Global-map ``w_cert`` query; `UNKNOWN` when no map is installed."""
    if _CERTIFIED_PPGO_MAP is None:
        return UNKNOWN
    return _CERTIFIED_PPGO_MAP.w_cert(parity, gamma, rho)


def certified_w_trust(parity: str, gamma: float, rho: float
                      ) -> float | _UnknownSentinel:
    """Global-map ``w_trust`` query; `UNKNOWN` when no map is installed."""
    if _CERTIFIED_PPGO_MAP is None:
        return UNKNOWN
    return _CERTIFIED_PPGO_MAP.w_trust(parity, gamma, rho)


def certified_w_ceiling(parity: str, gamma: float, rho: float
                        ) -> float | _UnknownSentinel:
    """Global-map ``w_ceiling`` query; `UNKNOWN` when no map is installed."""
    if _CERTIFIED_PPGO_MAP is None:
        return UNKNOWN
    return _CERTIFIED_PPGO_MAP.w_ceiling(parity, gamma, rho)


# ----------------------------------------------------------------------
# Build helpers (offline; used by scripts/train_ppgo_map.py).
# ----------------------------------------------------------------------

def _gamma_band_edges() -> np.ndarray:
    """Log-spaced gamma-band edges on ``[0.05, 1.55]``.

    Densified near ``gamma = 0.5`` (the shear-series cancellation edge) and
    ``gamma = 1.0`` (the parity boundary), with an edge sitting exactly at
    ``1.0`` so no band spans the boundary (positive parity below, macro
    saddle above).
    """
    below = np.geomspace(0.05, 1.0, 6)          # positive-parity side
    above = np.geomspace(1.0, 1.55, 4)          # macro-saddle side
    extra = np.array([0.45, 0.55, 0.9, 1.1])    # cancellation / boundary
    edges = np.unique(np.concatenate([below, above, extra, [1.0]]))
    return np.ascontiguousarray(edges, dtype=np.float64)


def _gamma_band_valid(parity: str, lo: float, hi: float) -> bool:
    """Whether a gamma band lies wholly on ``parity``'s side of ``1.0``."""
    if parity == 'positive':
        return hi <= 1.0
    return lo >= 1.0


def _rho_center(rho_edges: np.ndarray, ri: int) -> float:
    """Representative annulus radius for band ``ri`` (midpoint; ``lo*1.5``
    for the open outer band)."""
    lo = float(rho_edges[ri])
    hi = float(rho_edges[ri + 1])
    if math.isinf(hi):
        return lo * 1.5 if lo > 0.0 else 1.5
    return 0.5 * (lo + hi)


def caustic_geometry(gamma: float, kappa: float = 0.0, n_theta: int = 720
                      ) -> tuple[float, np.ndarray]:
    """Max source-plane caustic radius and its direction for ``gamma``.

    Sweeps the critical curve over polar angle (both square-root branches,
    so both parities are covered) via `geometry.critical_point`, skipping
    the wedge-forbidden angles of a macro saddle.  Returns
    ``(reach, unit_direction)`` where ``unit_direction`` points to the
    farthest caustic point, so a source placed at ``rho * reach`` along it
    is interior for ``rho < 1`` and exterior for ``rho > 1``.
    """
    from cogwheel.lensing.chang_refsdal import geometry

    reach = 0.0
    direction = np.array([1.0, 0.0])
    thetas = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    for branch in (1, -1):
        for theta in thetas:
            try:
                source = geometry.critical_point(
                    gamma, float(theta), 0.0, kappa, branch).source
            except geometry.LensDomainError:
                continue
            radius = float(math.hypot(source[0], source[1]))
            if radius > reach:
                reach = radius
                direction = np.asarray(source, dtype=float) / radius
    if reach <= 0.0:
        raise geometry.LensDomainError(
            f'No caustic reach found for gamma={gamma}, kappa={kappa}.')
    return reach, direction


def annulus_rho(gamma: float, y_magnitude: float, kappa: float = 0.0) -> float:
    """Authoritative ppGO annulus coordinate ``rho`` for a source magnitude.

    Converts a physical source-plane offset magnitude ``|y|`` into the ONE
    scalar-reach ppGO annulus gauge in which the certified-ppGO map is built
    and queried::

        rho = |y| / caustic_reach(gamma, kappa)

    where ``caustic_reach`` is element 0 of `caustic_geometry` -- the MAXIMUM
    source-plane caustic radius over polar angle (a single scalar per
    ``gamma``).  ``rho`` is dimensionless (Einstein-radius-normalised): the
    caustic sits at ``rho = 1``, the interior at ``rho < 1`` and the exterior
    at ``rho > 1``.

    This is the SINGLE authoritative converter into the ppGO annulus gauge.
    It is DISTINCT from the additive directional interior/exterior gauge used
    by the far-field charts (``rho = 1 + |y| - r_caustic``); the two gauges
    must not be conflated.  Both `likelihood._ppgo_cell_coords` and
    `surrogate_training._train_band_charts` obtain their ppGO ``rho``
    exclusively through this function, so the map is always queried in the
    gauge it was certified in.

    Parameters
    ----------
    gamma : float
        External shear magnitude.
    y_magnitude : float
        Physical source-plane offset magnitude ``|y|`` (dimensionless ``y``
        units); must be non-negative.
    kappa : float, optional
        Convergence.  Defaults to ``0.0`` (the ppGO map's ``kappa = 0``
        certified surface).

    Returns
    -------
    float
        The ppGO annulus coordinate ``rho``.

    Raises
    ------
    ValueError
        If ``y_magnitude`` is negative or the caustic reach is non-positive.
    LensDomainError
        Propagated from `caustic_geometry` at the ``det A = 0`` parity
        boundary or an over-critical convergence.
    """
    if y_magnitude < 0.0:
        raise ValueError(
            f'y_magnitude must be non-negative, got {y_magnitude}.')
    reach, _direction = caustic_geometry(gamma, kappa)
    if not reach > 0.0:
        raise ValueError(
            f'Non-positive caustic reach {reach} for gamma={gamma}, '
            f'kappa={kappa}.')
    return float(y_magnitude) / float(reach)


def _w_nodes(wall: float, nodes_per_decade: int = 12) -> np.ndarray:
    """Log-spaced ``w`` nodes on ``[1, wall]`` at >= 12 nodes/decade."""
    n_nodes = max(2, int(math.ceil(nodes_per_decade * math.log10(wall))) + 1)
    return np.geomspace(1.0, float(wall), n_nodes)


def _sup_over_w_floor(w_nodes: np.ndarray, error: np.ndarray, bar: float
                      ) -> float | None:
    """Sup-over-w certified floor: smallest ``w`` above the last violation.

    ``w_nodes`` ascending.  Returns the smallest ``w`` such that ``error``
    stays below ``bar`` for all ``w' >= w`` (the last upward re-crossing).
    Returns ``w_nodes[0]`` if the whole band is below the bar, and ``None``
    if the top node (nearest the wall) still violates -- the floor lies
    beyond the wall, so the cell is uncertified.
    """
    violated = np.flatnonzero(error >= bar)
    if violated.size == 0:
        return float(w_nodes[0])
    last = int(violated[-1])
    if last == w_nodes.size - 1:
        return None
    return float(w_nodes[last + 1])


def _max_accepted_prefix(evaluate, n_nodes: int, refusal_types: tuple):
    """Largest accepted ``w``-prefix length by bisection on the prefix index.

    The named engine refusals -- chiefly the saddle-image wave branch above
    ``W_CEILING_SCHWINGER`` -- are treated as MONOTONE in ``w``: the accepted
    set is a prefix ``w_nodes[:k]`` (refuse for all ``w`` above a per-cell
    ceiling).  This finds the largest ``k`` in ``[0, n_nodes]`` for which
    ``evaluate(k)`` (the engine run on the first ``k`` nodes) does not raise
    one of ``refusal_types``, in ``O(log n_nodes)`` engine evaluations --
    NOT a full re-sweep per step.

    Parameters
    ----------
    evaluate : callable
        ``evaluate(k)`` runs the engine on the first ``k`` ``w``-nodes and
        returns its result, or raises one of ``refusal_types`` when the
        accepted prefix is shorter than ``k``.
    n_nodes : int
        Total number of ``w``-nodes (the full prefix length).
    refusal_types : tuple of type
        Exception classes that mark a refused prefix (the truncation
        vocabulary); anything else propagates and invalidates the cell.

    Returns
    -------
    (k_max, result) : (int, object or None)
        ``k_max`` is the largest accepted prefix length; ``result`` is the
        value ``evaluate(k_max)`` returned, or ``None`` when even the
        single-node prefix refuses (``k_max == 0``) -- which the caller
        maps to ``STATUS_INVALID`` (a genuine refusal at the lowest ``w``,
        not a ceiling).

    Notes
    -----
    A non-monotone refusal can only make ``k_max`` SMALLER than the true
    accepted set (bisection probes a subset), which is conservative:
    ``result`` is always from a prefix that genuinely evaluated without a
    refusal.
    """
    try:
        return n_nodes, evaluate(n_nodes)
    except refusal_types:
        pass
    lo, hi = 1, n_nodes - 1
    best_k, best_result = 0, None
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            result = evaluate(mid)
        except refusal_types:
            hi = mid - 1
        else:
            best_k, best_result = mid, result
            lo = mid + 1
    return best_k, best_result


def _measure_cell(parity: str, gamma: float, rho_center: float, kappa: float,
                  wall: float) -> tuple[float, float, float, float]:
    """Measure one cell against the exact engine (truncation-on-refusal).

    Returns ``(status_code, w_cert, w_cert_diagnostic, w_ceiling)``.
    ``w_cert`` / ``w_cert_diagnostic`` are ``nan`` when uncertified at the
    respective bar; ``w_ceiling`` is the min over angles of each angle's
    maximal accepted ``w`` (``nan`` only for ``STATUS_INVALID``).

    A named engine refusal (`LensDomainError`, `CancellationError`,
    `SchwingerCertificationError`) part-way up an angle's ``w`` sweep
    TRUNCATES that angle at its maximal accepted ``w``-prefix rather than
    invalidating the cell -- the saddle-image branch ceiling is monotone in
    ``w`` (accepted set is a prefix).  The sup-over-w floor is then measured
    on the accepted prefix only, so a cell certifies on its measured range
    even when its top-``w`` refuses.  A refusal at the *lowest* ``w`` (no
    accepted prefix) or a failed caustic placement still yields
    ``STATUS_INVALID``; an angle whose accepted prefix never clears
    `CERTIFICATION_BAR` yields ``STATUS_BEYOND_WALL``.
    """
    from cogwheel.lensing.chang_refsdal import geometry
    from cogwheel.lensing.chang_refsdal.channels import ChangRefsdalChannels
    from cogwheel.lensing.chang_refsdal.operator import (
        geometric_amplification, CancellationError)
    from cogwheel.lensing.chang_refsdal._schwinger import (
        SchwingerCertificationError)

    refusal_types = (geometry.LensDomainError, CancellationError,
                     SchwingerCertificationError)

    w_nodes = _w_nodes(wall)
    n_nodes = w_nodes.size

    # Caustic placement is w-independent: a failure here means the source
    # cannot be placed at all, so the whole cell is invalid (not truncated).
    try:
        reach, direction = caustic_geometry(gamma, kappa)
    except geometry.LensDomainError:
        return STATUS_INVALID, math.nan, math.nan, math.nan

    # ANGULAR SWEEP (driver fix 2026-07-23): a cell certified from a single
    # axial source point is blind to the measured angular anisotropy of the
    # ppGO error (axis-good/diagonal-bad, factor ~500 at fixed radius in the
    # v2 tile diagnosis; for the saddle the axis point sits BETWEEN the two
    # deltoid lobes while near-lobe angles degrade).  Certify each cell
    # against the WORST of several angles: sup of per-angle floors, min of
    # per-angle ceilings.
    angles = (0.0, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2)
    floors_cert: list = []
    floors_diag: list = []
    angle_ceilings: list = []
    for angle in angles:
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        source = rho_center * reach * (rot @ np.asarray(direction))

        def evaluate(k: int, source=source):
            """Engine total and demodulated ppGO glue on ``w_nodes[:k]``."""
            w_prefix = w_nodes[:k]
            partition = ChangRefsdalChannels(w_prefix).evaluate(
                gamma=gamma, y=source, beta=0.0, kappa=kappa)
            # Align the time origin before differencing.  The engine
            # expresses ``exact_total`` RELATIVE to the minimum image delay
            # ``t_min`` (the operator series is demodulated at that carrier;
            # see ``ChangRefsdalChannels.evaluate``), whereas
            # ``geometric_amplification`` sums the ABSOLUTE image delays
            # ``exp(1j w tau_a)``.  Demodulating the ppGO glue by
            # ``exp(-1j w t_min)`` removes the pure global winding phase that
            # would otherwise dominate the residual and grow with ``w`` --
            # without it the F-normalized error never falls (every cell would
            # be spuriously beyond-wall).  After alignment the residual is
            # the genuine geometric-optics defect (-> machine precision at
            # high w).
            ppgo = np.asarray(geometric_amplification(
                w_prefix, source, gamma, beta=0.0, kappa=kappa))
            ppgo = ppgo * np.exp(-1j * w_prefix * float(partition.t_min))
            return w_prefix, np.asarray(partition.exact_total), ppgo

        best_k, best = _max_accepted_prefix(evaluate, n_nodes, refusal_types)
        if best_k == 0:
            # Even the single-node prefix refuses: a genuine refusal at the
            # lowest w (or a w-independent geometry error), not a ceiling.
            return STATUS_INVALID, math.nan, math.nan, math.nan
        w_prefix, exact, ppgo = best
        denominator = float(np.max(np.abs(exact)))
        if not (denominator > 0.0):
            return STATUS_INVALID, math.nan, math.nan, math.nan
        error = np.abs(exact - ppgo) / denominator
        floors_cert.append(
            _sup_over_w_floor(w_prefix, error, CERTIFICATION_BAR))
        floors_diag.append(
            _sup_over_w_floor(w_prefix, error, DIAGNOSTIC_BAR))
        angle_ceilings.append(float(w_prefix[-1]))

    # The cell is trusted only up to the tightest per-angle ceiling.
    w_ceiling = min(angle_ceilings)

    # Worst angle governs: any angle that never clears the bar within its
    # accepted range makes the cell beyond-wall; otherwise the cell floor is
    # the sup over angles.
    if any(f is None for f in floors_diag):
        diagnostic = math.nan
    else:
        diagnostic = max(floors_diag)
    if any(f is None for f in floors_cert):
        return STATUS_BEYOND_WALL, math.nan, diagnostic, w_ceiling
    floor = max(floors_cert)
    # Degenerate interplay: the worst angle only clears above a w that the
    # tightest-ceiling angle no longer accepts, so no w is simultaneously
    # certified for all angles -- the certified interval is empty.  Refuse
    # (uncertified), never certify an empty range.
    if floor > w_ceiling:
        return STATUS_BEYOND_WALL, math.nan, diagnostic, w_ceiling
    return STATUS_CERTIFIED, floor, diagnostic, w_ceiling


def _compute_interpolable(w_cert_grid: np.ndarray, cell_status_grid: np.ndarray
                          ) -> np.ndarray:
    """Flag certified cells whose certified gamma / rho neighbours are all
    within `MAX_CELL_JUMP` (per parity slice)."""
    interpolable = np.zeros_like(w_cert_grid)
    n_parity, n_gamma, n_rho = w_cert_grid.shape
    for p in range(n_parity):
        for gi in range(n_gamma):
            for ri in range(n_rho):
                if cell_status_grid[p, gi, ri] != STATUS_CERTIFIED:
                    continue
                here = w_cert_grid[p, gi, ri]
                ok = True
                for dgi, dri in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    gj, rj = gi + dgi, ri + dri
                    if not (0 <= gj < n_gamma and 0 <= rj < n_rho):
                        continue
                    if cell_status_grid[p, gj, rj] != STATUS_CERTIFIED:
                        continue
                    if abs(here - w_cert_grid[p, gj, rj]) > MAX_CELL_JUMP:
                        ok = False
                        break
                interpolable[p, gi, ri] = 1.0 if ok else 0.0
    return interpolable


def build_map(*, kappa: float = 0.0, astroid_wall: float = ASTROID_WALL,
              saddle_wall: float = SADDLE_WALL,
              gamma_edges: Sequence[float] | None = None,
              rho_edges: Sequence[float] | None = None) -> CertifiedPpgoMap:
    """Run the offline validation sweep and assemble a `CertifiedPpgoMap`.

    For each ``(parity, gamma band, annulus)`` cell a representative
    below-the-wall config is measured (`_measure_cell`): the sup-over-w
    floor at `CERTIFICATION_BAR`, the diagnostic floor at `DIAGNOSTIC_BAR`,
    the measured ``w_ceiling`` (truncation-on-refusal), and a status code.
    Parity-invalid gamma bands (a positive-parity band above ``1.0`` or a
    saddle band below it) stay ``STATUS_INVALID``.  Each cell also records
    ``rho_measured_max`` -- the finite radius at which it was sampled (the
    band top edge for inner annuli; ``lo * 1.5`` for the open outer annulus)
    -- so the accessors never certify the infinite tail beyond it.

    Parameters
    ----------
    kappa : float
        External convergence (production driver may sweep it; the shipped
        model pins ``kappa = 0``).
    astroid_wall, saddle_wall : float
        Per-parity Schwinger walls; the in-build coarse sweep may pass
        reduced ceilings for speed (the production sweep uses the true
        walls ``443.7`` / ``58``).
    gamma_edges, rho_edges : sequence of float, optional
        Override the default band edges (coarse synthetic grids for the
        in-build acceptance).
    """
    gamma_axis = (np.ascontiguousarray(gamma_edges, dtype=np.float64)
                  if gamma_edges is not None else _gamma_band_edges())
    rho_axis = (np.ascontiguousarray(rho_edges, dtype=np.float64)
                if rho_edges is not None
                else np.asarray(_DEFAULT_RHO_EDGES, dtype=np.float64))
    n_gamma = gamma_axis.size - 1
    n_rho = rho_axis.size - 1
    parity_codes = np.asarray([_PARITY_CODES['positive'],
                               _PARITY_CODES['saddle']], dtype=np.float64)

    shape = (parity_codes.size, n_gamma, n_rho)
    w_cert_grid = np.full(shape, np.nan, dtype=np.float64)
    w_cert_diagnostic_grid = np.full(shape, np.nan, dtype=np.float64)
    w_ceiling_grid = np.full(shape, np.nan, dtype=np.float64)
    cell_status_grid = np.full(shape, STATUS_INVALID, dtype=np.float64)

    # Per-cell finite rho cap: the band top edge for inner annuli; the
    # finite sampled radius (``_rho_center``) for the open outer annulus.
    # Depends only on the annulus index, but is stored per cell (in the
    # (2, n_gamma, n_rho) grid family) and hashed like ``w_ceiling``.
    rho_measured_max_grid = np.empty(shape, dtype=np.float64)
    for ri in range(n_rho):
        hi = float(rho_axis[ri + 1])
        rho_cap = hi if math.isfinite(hi) else _rho_center(rho_axis, ri)
        rho_measured_max_grid[:, :, ri] = rho_cap

    walls = {'positive': float(astroid_wall), 'saddle': float(saddle_wall)}
    for p, parity in enumerate(('positive', 'saddle')):
        for gi in range(n_gamma):
            lo, hi = float(gamma_axis[gi]), float(gamma_axis[gi + 1])
            if not _gamma_band_valid(parity, lo, hi):
                continue
            gamma_center = float(math.sqrt(lo * hi))
            for ri in range(n_rho):
                status, floor, diagnostic, ceiling = _measure_cell(
                    parity, gamma_center, _rho_center(rho_axis, ri), kappa,
                    walls[parity])
                cell_status_grid[p, gi, ri] = status
                w_cert_grid[p, gi, ri] = floor
                w_cert_diagnostic_grid[p, gi, ri] = diagnostic
                w_ceiling_grid[p, gi, ri] = ceiling

    interpolable_grid = _compute_interpolable(w_cert_grid, cell_status_grid)

    provenance = {
        'schema_version': _SCHEMA_VERSION,
        'certification_bar': CERTIFICATION_BAR,
        'diagnostic_bar': DIAGNOSTIC_BAR,
        'w_trust_multiplier': W_TRUST_MULTIPLIER,
        'w_trust_additive': W_TRUST_ADDITIVE,
        'w_trust_rule': 'w_trust = max(1.5 * w_cert, w_cert + 2.0)',
        'floor_rule': 'sup-over-w (smallest w above the last upward '
                      're-crossing of the bar), not the first crossing',
        'w_ceiling_rule': 'truncation-on-refusal: per angle the maximal '
                          'accepted w-prefix (the monotone saddle-image '
                          'branch ceiling), w_ceiling = min over angles; a '
                          'certified cell is trusted only on '
                          '[w_cert, w_ceiling], beyond it is UNKNOWN',
        'rho_measured_max_rule': 'finite sampled-rho cap: band top edge for '
                                 'inner annuli, lo*1.5 for the open outer '
                                 'annulus; queries beyond it return UNKNOWN '
                                 '(no infinite-annulus certification from one '
                                 'finite sample)',
        'max_cell_jump': MAX_CELL_JUMP,
        'astroid_wall': float(astroid_wall),
        'saddle_wall': float(saddle_wall),
        'kappa': float(kappa),
        'parity_codes': {'positive': 0.0, 'saddle': 1.0},
        'gamma_edges': [float(edge) for edge in gamma_axis],
        'rho_edges': [float(edge) for edge in rho_axis],
        'n_gamma': int(n_gamma),
        'n_rho': int(n_rho),
    }
    provenance['content_hash'] = _content_hash(
        parity_codes, gamma_axis, rho_axis, w_cert_grid,
        w_cert_diagnostic_grid, w_ceiling_grid, cell_status_grid,
        interpolable_grid, rho_measured_max_grid, _hash_scalars(provenance))

    return CertifiedPpgoMap.from_arrays(
        parity_codes, gamma_axis, rho_axis, w_cert_grid,
        w_cert_diagnostic_grid, w_ceiling_grid, cell_status_grid,
        interpolable_grid, rho_measured_max_grid, provenance)


def save_map(ppgo_map: CertifiedPpgoMap, path: str | Path) -> None:
    """Write a map to ``path`` as an ``allow_pickle=False`` ``.npz``.

    The provenance dict (including the content hash) is stored as a JSON
    scalar so the artifact carries only plain float64 arrays and a unicode
    scalar (no pickled objects).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        parity_codes=ppgo_map.parity_codes,
        gamma_edges=ppgo_map.gamma_edges,
        rho_edges=ppgo_map.rho_edges,
        w_cert=ppgo_map.w_cert_grid,
        w_cert_diagnostic=ppgo_map.w_cert_diagnostic_grid,
        w_ceiling=ppgo_map.w_ceiling_grid,
        cell_status=ppgo_map.cell_status_grid,
        interpolable=ppgo_map.interpolable_grid,
        rho_measured_max=ppgo_map.rho_measured_max_grid,
        provenance=np.asarray(json.dumps(ppgo_map.provenance)),
    )


def map_summary(ppgo_map: CertifiedPpgoMap) -> dict:
    """Cell-status tallies for the regeneration script's report."""
    status = ppgo_map.cell_status_grid
    return {
        'n_cells': int(status.size),
        'n_certified': int(np.sum(status == STATUS_CERTIFIED)),
        'n_beyond_wall': int(np.sum(status == STATUS_BEYOND_WALL)),
        'n_invalid': int(np.sum(status == STATUS_INVALID)),
        'n_interpolable': int(np.sum(ppgo_map.interpolable_grid > 0.0)),
    }
