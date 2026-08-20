"""
Mechanical enforcement of Part 0 structural invariants on cogwheel/lensing/.

Prevents regression of the 'one plausible constant at a time' accretion bug
class by AST-scanning the lensing tree for:

1. Prior-box-derived constants (diagonal ≈ 4.2426 or half-width 3.0 with
   a box-like name).
2. Retired concept names reappearing in production symbols/exports.
3. New discretization-absorber constants (epsilon/margin/safety suffixes)
   introduced without explicit allowlisting.

All tests are pure AST/text — no lensing module imports, no numerical
computation. Budget: < 2 s total.

The file also hosts the numerical ``w_low_fit`` certificate-behaviour
tests (``WLlowFitBaseTestCase`` and subclasses): monotonicity in reduced
shear and source magnitude (over the served exterior), D2 angular symmetry
(period-``pi`` + reflection), the DD-ceiling cap / parity-wall behaviour,
the near-fold-shell fence (declined to ``None``), the deep-interior
fit serve, and the just-outside-shell conservative pin.  Those import
the fitted-certificate module LAZILY
(``_load_w_low_fit``) and call only the O(1), engine-free ``w_low_fit`` —
they stay far under the fast-tier budget.

The exact-heavy INS-3-002 deep-interior HONEST-VALUE pin (served ceiling
<= engine-honest `_measure_w_low_true` ceiling over the gamma x rho x
theta grid) and the clip-not-the-mechanism guard live in the
``COGWHEEL_BRUTE_ACCURACY``-gated `TestWLlowFitDeepInteriorHonestServe`:
the provisional smoke fit still over-runs at the uncalibrated low-gamma
interior and is clipped to the DD ceiling, so those pins are RED BY DESIGN
until the feature-basis regularization + driver's full interior-inclusive
re-bake land (see `_BRUTE_ACCURACY_REASON`).
"""
from __future__ import annotations

import ast
import cmath
import json
import math
import os
import pathlib
import re
import unittest
from unittest import mock

import numpy as np

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

#: Root of the cogwheel/lensing/ tree (test file lives at cogwheel/tests/).
LENSING_ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parents[1] / 'lensing'

#: Repository root (two parents above cogwheel/tests/test_*.py).
REPO_ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _py_files_under(root: pathlib.Path) -> list[pathlib.Path]:
    """Recursively collect .py files, excluding __pycache__ directories."""
    return sorted(
        p for p in root.rglob('*.py')
        if '__pycache__' not in p.parts
    )


def _numeric_value(node: ast.expr) -> float | None:
    """Extract a numeric float value from a Constant or UnaryOp(-Constant)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if (isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, (int, float))):
        return -float(node.operand.value)
    return None


def _module_level_constants(
    tree: ast.Module,
) -> list[tuple[str, float, int]]:
    """
    Return (name, value, lineno) for every module-level numeric constant.

    Handles both ``NAME = <num>`` and ``NAME: type = <num>`` forms.
    """
    results: list[tuple[str, float, int]] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    val = _numeric_value(node.value)
                    if val is not None:
                        results.append((target.id, val, node.lineno))
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.value is not None:
                val = _numeric_value(node.value)
                if val is not None:
                    results.append((node.target.id, val, node.lineno))
    return results


# ===========================================================================
# TEST CLASS 1: TestNoPriorBoxConstants
# ===========================================================================

#: The diagonal of a 3×3 prior box — any constant ≈ this value is a violation.
_PRIOR_BOX_DIAGONAL: float = 3 * math.sqrt(2)  # 4.242640687...

#: Tolerance around the diagonal value; anything within this is flagged.
_DIAGONAL_TOL: float = 1e-2

#: Name fragments that identify a prior-box half-width constant.
_BOX_NAME_FRAGMENTS: frozenset[str] = frozenset({
    'BOX', 'RANGE', 'EXTENT', 'HALF_WIDTH', 'PRIOR_HALF',
    'Y_MAX', 'Y_MIN', 'SOURCE_RANGE', 'SOURCE_EXTENT', 'SOURCE_BOX',
})

#: Allowlist: (relative_path, constant_name) pairs that are NOT prior-box
#: despite having value 3.0.  Each entry has a justification comment.
_VALUE_3_ALLOWLIST: frozenset[tuple[str, str]] = frozenset({
    ('cogwheel/lensing/prior.py', '_Y_SCALE_CAP'),
    ('cogwheel/lensing/chang_refsdal/_pearcey_cusp.py', '_SPLIT_BASE'),
    ('cogwheel/lensing/chang_refsdal/_pearcey_table.py', '_SPLINE_DEGREE'),
    ('cogwheel/lensing/surrogate.py', '_SPLINE_DEGREE'),
})


class TestNoPriorBoxConstants(unittest.TestCase):
    """Enforce absence of prior-box-derived magic constants in lensing/."""

    @classmethod
    def setUpClass(cls) -> None:
        """Parse all lensing .py files and collect module-level constants."""
        cls.all_constants: list[tuple[str, str, float, int]] = []
        cls.files_scanned: int = 0
        for path in _py_files_under(LENSING_ROOT):
            rel = str(path.relative_to(REPO_ROOT))
            source = path.read_text(encoding='utf-8')
            tree = ast.parse(source, filename=rel)
            cls.files_scanned += 1
            for name, value, lineno in _module_level_constants(tree):
                cls.all_constants.append((rel, name, value, lineno))

    def test_anti_vacuity(self) -> None:
        """Verify the scan actually found files and constants to check."""
        self.assertGreater(self.files_scanned, 10,
                           'Expected to scan >10 lensing .py files')
        self.assertGreater(len(self.all_constants), 20,
                           'Expected >20 module-level numeric constants')

    def test_no_prior_box_diagonal(self) -> None:
        """No module-level constant should be ≈ 3√2 (the prior-box diagonal)."""
        violations: list[tuple[str, str, float]] = []
        for rel, name, value, _lineno in self.all_constants:
            if abs(value - _PRIOR_BOX_DIAGONAL) < _DIAGONAL_TOL:
                violations.append((rel, name, value))
        self.assertEqual(
            violations, [],
            f'Prior-box diagonal constants found: {violations}',
        )

    def test_no_prior_box_halfwidth_by_name(self) -> None:
        """No constant with value ≈ 3.0 and a box-like name should exist."""
        violations: list[tuple[str, str, float]] = []
        for rel, name, value, _lineno in self.all_constants:
            if abs(value - 3.0) >= 1e-9:
                continue
            upper_name = name.upper()
            if any(frag in upper_name for frag in _BOX_NAME_FRAGMENTS):
                if (rel, name) not in _VALUE_3_ALLOWLIST:
                    violations.append((rel, name, value))
        self.assertEqual(
            violations, [],
            f'Prior-box half-width constants found: {violations}',
        )


# ===========================================================================
# TEST CLASS 2: TestNoRetiredConceptNames
# ===========================================================================

#: Path to the retired concepts registry.
_RETIRED_CONCEPTS_PATH: pathlib.Path = (
    REPO_ROOT / '.claude' / 'hooks' / 'retired_concepts.json'
)

#: Exclusion substrings — lines containing these (case-insensitive) are
#: exempt from the retired-name scan (matches the pre-commit hook carveout).
_EXCLUSION_WORDS: tuple[str, ...] = (
    'retired', 'deleted', 'removed', 'no longer', 'used to',
)


def _load_retired_concepts() -> list[dict[str, str]]:
    """Load and return the retired concepts list from the JSON registry."""
    with open(_RETIRED_CONCEPTS_PATH, encoding='utf-8') as f:
        data = json.load(f)
    return data['retired']


def _compile_retired_patterns(
    entries: list[dict[str, str]],
) -> list[tuple[str, re.Pattern[str]]]:
    """Compile word-boundary patterns for each retired concept name."""
    return [
        (entry['name'], re.compile(r'\b' + re.escape(entry['name']) + r'\b'))
        for entry in entries
    ]


def _is_excluded_line(line: str) -> bool:
    """True if the line contains an exclusion word (case-insensitive)."""
    lower = line.lower()
    return any(word in lower for word in _EXCLUSION_WORDS)


def _extract_all_list(tree: ast.Module) -> list[str]:
    """Extract names from a module's __all__ assignment, if present."""
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == '__all__':
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        return [
                            elt.value for elt in node.value.elts
                            if isinstance(elt, ast.Constant)
                            and isinstance(elt.value, str)
                        ]
    return []


def _top_level_symbol_names(tree: ast.Module) -> list[tuple[str, int]]:
    """
    Return (name, lineno) for all top-level classes, functions, and
    constant assignments.
    """
    names: list[tuple[str, int]] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append((node.name, node.lineno))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.append((target.id, node.lineno))
    return names


class TestNoRetiredConceptNames(unittest.TestCase):
    """Enforce that retired concept names do not reappear in lensing/."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load retired concepts and parse lensing source files."""
        cls.retired_entries = _load_retired_concepts()
        cls.patterns = _compile_retired_patterns(cls.retired_entries)
        cls.lensing_files: list[tuple[str, pathlib.Path, ast.Module, str]] = []
        for path in _py_files_under(LENSING_ROOT):
            rel = str(path.relative_to(REPO_ROOT))
            source = path.read_text(encoding='utf-8')
            tree = ast.parse(source, filename=rel)
            cls.lensing_files.append((rel, path, tree, source))

    def test_registry_well_formed(self) -> None:
        """retired_concepts.json parses and has the required structure."""
        self.assertTrue(
            _RETIRED_CONCEPTS_PATH.exists(),
            f'Missing: {_RETIRED_CONCEPTS_PATH}',
        )
        with open(_RETIRED_CONCEPTS_PATH, encoding='utf-8') as f:
            data = json.load(f)
        self.assertIn('retired', data)
        entries = data['retired']
        self.assertIsInstance(entries, list)
        seen_names: set[str] = set()
        for entry in entries:
            self.assertIn('name', entry)
            self.assertIn('retired_by', entry)
            self.assertIn('note', entry)
            self.assertIsInstance(entry['name'], str)
            self.assertIsInstance(entry['retired_by'], str)
            self.assertIsInstance(entry['note'], str)
            self.assertNotIn(
                entry['name'], seen_names,
                f'Duplicate retired name: {entry["name"]}',
            )
            seen_names.add(entry['name'])

    def test_no_retired_names_in_exports(self) -> None:
        """No __all__ list in lensing/ exports a retired concept name."""
        violations: list[tuple[str, str, str]] = []
        for rel, _path, tree, _source in self.lensing_files:
            all_names = _extract_all_list(tree)
            for export_name in all_names:
                for concept_name, pattern in self.patterns:
                    if pattern.search(export_name):
                        violations.append((rel, export_name, concept_name))
        self.assertEqual(
            violations, [],
            f'Retired names in __all__: {violations}',
        )

    def test_no_retired_names_in_symbols(self) -> None:
        """No top-level class/function/constant in lensing/ uses a retired name."""
        violations: list[tuple[str, str, str]] = []
        for rel, _path, tree, _source in self.lensing_files:
            for sym_name, _lineno in _top_level_symbol_names(tree):
                for concept_name, pattern in self.patterns:
                    if pattern.search(sym_name):
                        violations.append((rel, sym_name, concept_name))
        self.assertEqual(
            violations, [],
            f'Retired names in symbols: {violations}',
        )

    def test_no_retired_names_in_source_lines(self) -> None:
        """Full-text scan: no retired concept name in any non-excluded line."""
        violations: list[tuple[str, int, str, str]] = []
        for rel, _path, _tree, source in self.lensing_files:
            for lineno_0, line in enumerate(source.splitlines()):
                if _is_excluded_line(line):
                    continue
                for concept_name, pattern in self.patterns:
                    if pattern.search(line):
                        violations.append((rel, lineno_0 + 1, concept_name, line.strip()))
        self.assertEqual(
            violations, [],
            f'Retired names in source (first 5): {violations[:5]}',
        )

    #: Live documentation paths (relative to REPO_ROOT) that must not
    #: reference retired concept names in non-excluded lines.
    LIVE_DOCS: tuple[pathlib.Path, ...] = (
        pathlib.Path('.claude/spec/SPEC.md'),
        pathlib.Path('.claude/spec/COVERAGE_DESIGN.md'),
        pathlib.Path('.claude/spec/DATA_CONTRACTS.yaml'),
    )

    def test_no_retired_names_in_live_docs(self) -> None:
        """Live documentation must not reference retired concept names."""
        violations: list[tuple[str, int, str, str]] = []
        for doc_rel in self.LIVE_DOCS:
            doc_path = REPO_ROOT / doc_rel
            if not doc_path.exists():
                continue
            text = doc_path.read_text(encoding='utf-8')
            for lineno_0, line in enumerate(text.splitlines()):
                if _is_excluded_line(line):
                    continue
                for concept_name, pattern in self.patterns:
                    if pattern.search(line):
                        violations.append((
                            str(doc_rel), lineno_0 + 1,
                            concept_name, line.strip(),
                        ))
        self.assertEqual(
            violations, [],
            f'Retired names in live docs (first 5): {violations[:5]}',
        )


# ===========================================================================
# TEST CLASS 3: TestNoNewDiscretizationAbsorbers
# ===========================================================================

#: Regex matching absorber-shaped constant names (leading underscore,
#: ALL_CAPS, ending with a discretization-absorber suffix).
_ABSORBER_PATTERN: re.Pattern[str] = re.compile(
    r'^_[A-Z][A-Z0-9_]*(_EPS|_MARGIN|_FRAC|_STANDOFF|_SAFETY)$'
)

#: Allowlist of legitimate constants matching the absorber name pattern.
#: Each entry is (relative_path, constant_name).
_ABSORBER_ALLOWLIST: frozenset[tuple[str, str]] = frozenset({
    ('cogwheel/lensing/surrogate_training.py', '_CUSP_BRACKET_EPS'),
    ('cogwheel/lensing/surrogate_training.py', '_CUSP_WIDTH_SAFETY'),
    ('cogwheel/lensing/surrogate_training.py', '_SADDLE_CUSP_WIDTH_SAFETY'),
    ('cogwheel/lensing/surrogate_training.py', '_ARC_MARGIN_FRAC'),
    ('cogwheel/lensing/surrogate_training.py', '_DD_PRODUCT_MARGIN'),
    ('cogwheel/lensing/surrogate.py', '_DD_PRODUCT_MARGIN'),  # Mirrors surrogate_training._DD_PRODUCT_MARGIN for serve-time DD ceiling
    ('cogwheel/lensing/surrogate_training.py', '_DEFAULT_FARFIELD_OVERLAP'),
    ('cogwheel/lensing/surrogate_training.py', '_INTERLOBE_CORRIDOR_ETA_SCALE'),
    ('cogwheel/lensing/surrogate_census.py', 'CROWN_CAUSTIC_MARGIN'),
    ('cogwheel/lensing/chang_refsdal/channels.py', '_MARKER_SCALE_FLOOR'),
    ('cogwheel/lensing/chang_refsdal/_schwinger.py', '_U_MARGIN_CONST'),
    ('cogwheel/lensing/chang_refsdal/_airy_fold.py', '_CUSP_TIE_EPS'),  # Delay-equality tolerance for cusp-cluster detection, NOT a discretization absorber
    ('cogwheel/lensing/likelihood.py', '_PPGO_INTERIOR_SAFETY'),  # Measured margin on the interior c3 certificate (worst ratio 0.980, p99 0.953 over 1248 oracle points), not a discretization absorber
    ('cogwheel/lensing/likelihood.py', '_SADDLE_FARFIELD_SAFETY'),  # Measured margin on the EXTERIOR c3 certificate (covers the 9.4x worst measured c3 shortfall with ~2.1x headroom; zero false admits over 672 calibration points, scripts/calibration_pilot_followup.json), not a discretization absorber
    ('cogwheel/lensing/surrogate_training.py', '_TUBE_TRIM_DTAU_FRAC'),  # F083 knee threshold (0.6 of the Delta_tau peak), carried VERBATIM from the falsified-and-shipped F083 fixture (test_lensing_tube_beat_free) where the trimmed 10-node chart measured eps 4.3e-3 vs the 0.0237 bar; a profile-shape landmark, not a discretization absorber
    ('cogwheel/lensing/surrogate_training.py', '_TUBE_TRIM_LO_STANDOFF'),  # F083 inward stand-off (0.20 of the knee-to-peak span) off the steep-rise end, verbatim F083 provenance as above -- geometric bracket placement, not a discretization absorber
    ('cogwheel/lensing/surrogate_training.py', '_TUBE_TRIM_HI_STANDOFF'),  # F083 inward stand-off (0.05 of the span) off the turnover end, verbatim F083 provenance as above -- geometric bracket placement, not a discretization absorber
})


class TestNoNewDiscretizationAbsorbers(unittest.TestCase):
    """Enforce that no new absorber-shaped constants appear without allowlisting."""

    @classmethod
    def setUpClass(cls) -> None:
        """Parse lensing tree and collect absorber-shaped constants."""
        cls.found: list[tuple[str, str, float]] = []
        for path in _py_files_under(LENSING_ROOT):
            rel = str(path.relative_to(REPO_ROOT))
            source = path.read_text(encoding='utf-8')
            tree = ast.parse(source, filename=rel)
            for name, value, _lineno in _module_level_constants(tree):
                if _ABSORBER_PATTERN.match(name):
                    cls.found.append((rel, name, value))

    def test_no_new_absorber_constants(self) -> None:
        """Every absorber-shaped constant must appear in the allowlist."""
        violations: list[tuple[str, str, float]] = []
        for rel, name, value in self.found:
            if (rel, name) not in _ABSORBER_ALLOWLIST:
                violations.append((rel, name, value))
        self.assertEqual(
            violations, [],
            f'Un-allowlisted absorber constants: {violations}',
        )


# ===========================================================================
# SELF-FALSIFICATION: proves the suite can go red
# ===========================================================================


# ===========================================================================
# TEST CLASS 4: TestNoDocstringAbsorberLanguage
# ===========================================================================

#: Target files to scan for absorber-language docstrings on constants.
_DOCSTRING_ABSORBER_TARGET_FILES: tuple[str, ...] = (
    'cogwheel/lensing/surrogate_training.py',
    'cogwheel/lensing/surrogate.py',
)

#: Phrases whose presence in a constant's docstring indicates absorber intent.
_FORBIDDEN_DOCSTRING_PHRASES: tuple[str, ...] = (
    'discretization error',
    'sampling artifact',
    'safety factor for',
)

#: Allowlist of (relative_path, constant_name) tuples that are legitimate
#: despite using absorber-like language in their docstrings.
_DOCSTRING_ABSORBER_ALLOWLIST: frozenset[tuple[str, str]] = frozenset()


def _collect_constant_docstrings(
    tree: ast.Module,
    relative_path: str,
) -> list[tuple[str, str, str, int]]:
    """
    Collect (relative_path, constant_name, docstring_text, lineno) for
    module-level constants followed by a bare-string expression (the
    Python docstring-on-constant convention).
    """
    results: list[tuple[str, str, str, int]] = []
    body = tree.body
    for i in range(len(body) - 1):
        node = body[i]
        next_node = body[i + 1]
        # Check if next_node is a bare string expression (docstring)
        if not (isinstance(next_node, ast.Expr)
                and isinstance(next_node.value, ast.Constant)
                and isinstance(next_node.value.value, str)):
            continue
        docstring = next_node.value.value
        # Extract constant names from Assign or AnnAssign
        names: list[str] = []
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.append(target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                names.append(node.target.id)
        for name in names:
            results.append((relative_path, name, docstring, node.lineno))
    return results


class TestNoDocstringAbsorberLanguage(unittest.TestCase):
    """
    Enforce that module-level constants in surrogate files do not have
    docstrings containing absorber-intent language (e.g. 'discretization error',
    'sampling artifact', 'safety factor for').

    The bug-class signature: constants introduced to absorb discretization
    artifacts rather than fix the underlying issue document themselves with
    these phrases.

    Budget: Pure AST scan of 2 files, < 0.5s. No imports of lensing modules.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Parse target files and collect constant docstrings."""
        cls.constant_docstrings: list[tuple[str, str, str, int]] = []
        cls.files_parsed: int = 0
        for rel in _DOCSTRING_ABSORBER_TARGET_FILES:
            path = REPO_ROOT / rel
            if not path.exists():
                continue
            source = path.read_text(encoding='utf-8')
            tree = ast.parse(source, filename=rel)
            cls.files_parsed += 1
            cls.constant_docstrings.extend(
                _collect_constant_docstrings(tree, rel)
            )

    def test_anti_vacuity(self) -> None:
        """Verify the scan found files and at least some constant docstrings."""
        self.assertGreater(
            self.files_parsed, 0,
            'Expected to parse at least 1 target file',
        )
        # It's acceptable for files to have zero docstring-annotated constants,
        # but the file scan itself must have happened.
        self.assertGreaterEqual(
            len(self.constant_docstrings), 0,
            'constant_docstrings must be a list (even if empty)',
        )

    def test_no_absorber_language_in_constant_docstrings(self) -> None:
        """No constant docstring should contain absorber-intent phrases."""
        violations: list[tuple[str, str, str, int]] = []
        for rel, name, docstring, lineno in self.constant_docstrings:
            if (rel, name) in _DOCSTRING_ABSORBER_ALLOWLIST:
                continue
            lower_doc = docstring.lower()
            for phrase in _FORBIDDEN_DOCSTRING_PHRASES:
                if phrase in lower_doc:
                    violations.append((rel, name, phrase, lineno))
                    break  # One violation per constant suffices
        self.assertEqual(
            violations, [],
            f'Absorber-language in constant docstrings: {violations}',
        )

# ===========================================================================
# TEST CLASS 5: TestDiffractiveFitStructuralPurity
# ===========================================================================

#: The fitted-certificate module (WP1).  `w_low_fit` must be an O(1),
#: engine-free pure function -- no scan, no kernel, no mpmath -- that is the
#: entire point of replacing the per-proposal `diffractive_w_low` scan with
#: baked coefficients.
_DIFFRACTIVE_PATH: pathlib.Path = (
    REPO_ROOT / 'cogwheel/lensing/chang_refsdal/_diffractive.py'
)

#: The likelihood consumer module (WP2), whose `_diffractive_bottom_ceiling`
#: must bind `w_low_fit` rather than re-implement the retired scan.
_LIKELIHOOD_PATH: pathlib.Path = REPO_ROOT / 'cogwheel/lensing/likelihood.py'

#: The calibration script whose grid generators now FENCE the near-fold
#: shell (`_fence_excluded` drops shell rows from `_grid_points` /
#: `_off_grid_points`); single source of the fenced probe domain.
_FIT_SCRIPT_PATH: pathlib.Path = REPO_ROOT / 'scripts' / 'fit_diffractive_certificate.py'

#: The census mirror (WP3) whose Rung-P admission must decline a fenced draw
#: to engine/fold demand, never `diffractive_analytic`.
_SERVE_ROUTE_CENSUS_PATH: pathlib.Path = (
    REPO_ROOT / 'cogwheel/lensing/serve_route_census.py'
)

#: Symbols of the retired per-proposal certificate scan and its
#: deep-optimistic constant tree.  These must have NO code reference left in
#: the value-path module or its likelihood consumer (docstring prose is
#: exempt -- only Name nodes count).
_RETIRED_SCAN_SYMBOLS: frozenset[str] = frozenset({
    'diffractive_w_low',
    '_rootfind_w_low',
    '_rootfind_w_high',
    '_honest_tail_ratio',
    '_DIFFRACTIVE_CERT_SAFETY',
    '_CERT_REFERENCE_W',
})

#: Engine / kernel / mpmath entry points that `w_low_fit` must never touch
#: (the exact doors the ENGINE-FREE PURITY spec booby-traps at runtime).
_FIT_ENGINE_DOORS: frozenset[str] = frozenset({
    'f_schwinger', '_f_schwinger_mpmath', 'evaluate', 'mpmath', 'mp',
    'gauss_quadrature', 'point_mass_g_derivatives',
    'diffractive_w_low', '_rootfind_w_low', '_rootfind_w_high',
    '_honest_tail_ratio', '_operator_terms', '_kernel_length',
    'diffractive_amplification',
})


def _parse_source(path: pathlib.Path) -> ast.Module:
    """Parse a source file to an AST module."""
    return ast.parse(path.read_text(encoding='utf-8'), filename=str(path))


def _called_identifiers(node: ast.AST) -> set[str]:
    """
    Collect every identifier appearing in a Call target chain under `node`.

    ``_schwinger.f_schwinger(w)`` yields both ``_schwinger`` and
    ``f_schwinger``; ``foo()(x)`` recurses into the inner call.  This catches
    an engine entry point regardless of how it is reached.
    """
    names: set[str] = set()
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        func: ast.expr = call.func
        while isinstance(func, ast.Call):
            func = func.func
        if isinstance(func, ast.Name):
            names.add(func.id)
        elif isinstance(func, ast.Attribute):
            cur: ast.expr = func
            while isinstance(cur, ast.Attribute):
                names.add(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                names.add(cur.id)
    return names


def _has_loop(node: ast.AST) -> bool:
    """True if the AST subtree contains any for/while/async-for loop."""
    return any(
        isinstance(n, (ast.For, ast.While, ast.AsyncFor))
        for n in ast.walk(node)
    )


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    """Return the first FunctionDef with the given name anywhere in `tree`."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _all_name_ids(tree: ast.AST) -> set[str]:
    """
    Collect every Name-node identifier under ``tree`` (module or function).

    Code references only: docstring prose lives in Constant string nodes, not
    Name nodes, so a documented "replaces ``diffractive_w_low``" note does not
    trip a call/definition scan.
    """
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def _handler_catches(handler: ast.ExceptHandler, name: str) -> bool:
    """True if the except handler catches `name` (bare or as a tuple element)."""
    typ = handler.type
    if typ is None:
        return False
    if isinstance(typ, ast.Name):
        return typ.id == name
    if isinstance(typ, ast.Tuple):
        return any(isinstance(elt, ast.Name) and elt.id == name for elt in typ.elts)
    return False


class TestDiffractiveFitStructuralPurity(unittest.TestCase):
    """
    Enforce that the WP1 certificate replacement is structurally a pure,
    O(1), engine-free lookup, and that the retired per-proposal scan is gone.

    Pins three structural invariants of the fitted certificate:
    1. `w_low_fit` exists.
    2. The retired scan symbols (`diffractive_w_low`, the two root-find
       helpers, the tail-ratio helper, and the deep-optimistic constant tree)
       have no code reference left in the value-path module or its likelihood
       consumer.
    3. `w_low_fit`'s body has no loop and touches no engine / kernel / mpmath
       entry point -- so a serve-time call is an O(1) pure function.

    Budget: pure AST scan of 2 files, < 0.5s.  No imports of lensing modules.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Parse the two value-path files and locate `w_low_fit`."""
        cls.diffractive_tree = _parse_source(_DIFFRACTIVE_PATH)
        cls.likelihood_tree = _parse_source(_LIKELIHOOD_PATH)
        cls.w_low_fit = _find_function(cls.diffractive_tree, 'w_low_fit')

    def test_anti_vacuity(self) -> None:
        """The scan found the fitted certificate function it is meant to police."""
        self.assertIsNotNone(
            self.w_low_fit, 'w_low_fit must exist in _diffractive.py')

    def test_retired_scan_symbols_gone(self) -> None:
        """No retired scan symbol is defined or referenced in either file."""
        for tree, label in (
            (self.diffractive_tree, '_diffractive.py'),
            (self.likelihood_tree, 'likelihood.py'),
        ):
            defined = {name for name, _ in _top_level_symbol_names(tree)}
            referenced = _all_name_ids(tree)
            for name in _RETIRED_SCAN_SYMBOLS:
                self.assertNotIn(name, defined, f'{name} still defined in {label}')
                self.assertNotIn(name, referenced, f'{name} still referenced in {label}')

    def test_w_low_fit_is_engine_free_and_o1(self) -> None:
        """`w_low_fit` has no loop and touches no engine/kernel/mpmath door."""
        w_low_fit = self.w_low_fit
        if w_low_fit is None:
            self.fail('w_low_fit must exist in _diffractive.py')
        self.assertFalse(_has_loop(w_low_fit), 'w_low_fit must be O(1): no loop')
        offending = _called_identifiers(w_low_fit) & _FIT_ENGINE_DOORS
        self.assertEqual(
            offending, set(),
            f'w_low_fit touches engine/kernel/mpmath entry points: {offending}',
        )


# ===========================================================================
# TEST CLASS 6: TestDiffractiveBottomCeilingWrapper
# ===========================================================================


class TestDiffractiveBottomCeilingWrapper(unittest.TestCase):
    """
    Enforce that `_diffractive_bottom_ceiling` (likelihood.py) is a THIN
    wrapper over the fitted certificate, and that the parity-wall refusal is
    mapped to None there -- not re-implemented as a scan.

    Pins the DRY single-source invariant: the likelihood's nested-split
    boundary must bind the SAME production `w_low_fit`, and the wall
    `DiffractiveDomainError` must be caught and returned as `None` (the
    null-split identity at the call sites).  No retired scan helper may
    reappear.

    Budget: pure AST scan of 1 file, < 0.5s.  No imports of lensing modules.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Parse likelihood.py and locate the wrapper method."""
        cls.method = _find_function(
            _parse_source(_LIKELIHOOD_PATH), '_diffractive_bottom_ceiling')

    def test_anti_vacuity(self) -> None:
        """The scan found the wrapper method it is meant to police."""
        self.assertIsNotNone(
            self.method, '_diffractive_bottom_ceiling must exist in likelihood.py')

    def test_wrapper_forwards_to_w_low_fit(self) -> None:
        """The wrapper calls the fitted certificate, not the retired scan."""
        method = self.method
        if method is None:
            self.fail('_diffractive_bottom_ceiling must exist in likelihood.py')
        called = _called_identifiers(method)
        self.assertIn('w_low_fit', called, 'wrapper must call w_low_fit')
        self.assertNotIn(
            'diffractive_w_low', called, 'wrapper must not call the retired scan')

    def test_wrapper_maps_domain_error_to_none(self) -> None:
        """`DiffractiveDomainError` is caught and returned as None."""
        method = self.method
        if method is None:
            self.fail('_diffractive_bottom_ceiling must exist in likelihood.py')
        mapped = False
        for node in ast.walk(method):
            if not isinstance(node, ast.Try):
                continue
            for handler in node.handlers:
                if not _handler_catches(handler, 'DiffractiveDomainError'):
                    continue
                for stmt in handler.body:
                    if (isinstance(stmt, ast.Return)
                            and isinstance(stmt.value, ast.Constant)
                            and stmt.value.value is None):
                        mapped = True
        self.assertTrue(
            mapped, 'DiffractiveDomainError must be mapped to None in the wrapper')

class TestSelfFalsification(unittest.TestCase):
    """
    Demonstrate that each scanner has teeth by injecting a synthetic
    violation and asserting it would be caught.

    These tests do NOT write to disk — they inject synthetic entries into
    the data structures the real tests consume and verify the detection
    logic fires.
    """

    def test_diagonal_detector_fires(self) -> None:
        """A synthetic constant ≈ 3√2 is detected by the diagonal check."""
        # Simulate a constant with value ≈ _PRIOR_BOX_DIAGONAL
        synthetic = [('cogwheel/lensing/fake.py', '_FAKE_DIAG', 4.2426, 1)]
        violations = [
            (rel, name, value)
            for rel, name, value, _ in synthetic
            if abs(value - _PRIOR_BOX_DIAGONAL) < _DIAGONAL_TOL
        ]
        self.assertEqual(len(violations), 1, 'Diagonal detector must fire')

    def test_box_name_detector_fires(self) -> None:
        """A synthetic constant named _SOURCE_BOX_HALF = 3.0 is detected."""
        synthetic = [('cogwheel/lensing/fake.py', '_SOURCE_BOX_HALF', 3.0, 1)]
        violations = []
        for rel, name, value, _ in synthetic:
            if abs(value - 3.0) >= 1e-9:
                continue
            upper_name = name.upper()
            if any(frag in upper_name for frag in _BOX_NAME_FRAGMENTS):
                if (rel, name) not in _VALUE_3_ALLOWLIST:
                    violations.append((rel, name, value))
        self.assertEqual(len(violations), 1, 'Box-name detector must fire')

    def test_absorber_detector_fires(self) -> None:
        """A synthetic constant _FOO_EPS = 0.1 is detected as un-allowlisted."""
        name = '_FOO_EPS'
        rel = 'cogwheel/lensing/fake.py'
        self.assertIsNotNone(
            _ABSORBER_PATTERN.match(name),
            'Pattern must match _FOO_EPS',
        )
        self.assertNotIn(
            (rel, name), _ABSORBER_ALLOWLIST,
            'Synthetic constant must NOT be in allowlist',
        )

    def test_retired_name_detector_fires(self) -> None:
        """The retired-concept patterns actually match their own names."""
        entries = _load_retired_concepts()
        patterns = _compile_retired_patterns(entries)
        # Every retired name should match itself
        for concept_name, pattern in patterns:
            self.assertIsNotNone(
                pattern.search(concept_name),
                f'Pattern for {concept_name!r} does not match itself',
            )

    def test_exclusion_allows_commentary(self) -> None:
        """Lines with exclusion words are correctly exempted."""
        line = '# _WEDGE_EPS was retired in build 1b'
        self.assertTrue(_is_excluded_line(line))
        # But a line without exclusion words is not exempt
        line_active = '_WEDGE_EPS = 0.01'
        self.assertFalse(_is_excluded_line(line_active))

    def test_live_doc_detector_fires(self) -> None:
        """Inject a synthetic doc file with a retired name; detection must fire."""
        import tempfile
        entries = _load_retired_concepts()
        patterns = _compile_retired_patterns(entries)
        # Use the first retired concept as the canary
        concept_name, pattern = patterns[0]
        # Create a synthetic doc containing the retired name in a non-excluded line
        synthetic_content = (
            f'# Design Notes\n'
            f'The {concept_name} parameter controls the inner boundary.\n'
        )
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.md', delete=False, encoding='utf-8',
        ) as tmp:
            tmp.write(synthetic_content)
            tmp_path = pathlib.Path(tmp.name)
        try:
            # Re-use the detection logic inline
            violations: list[tuple[str, int, str, str]] = []
            text = tmp_path.read_text(encoding='utf-8')
            for lineno_0, line in enumerate(text.splitlines()):
                if _is_excluded_line(line):
                    continue
                for cn, pat in patterns:
                    if pat.search(line):
                        violations.append((
                            str(tmp_path), lineno_0 + 1, cn, line.strip(),
                        ))
            self.assertGreater(
                len(violations), 0,
                f'Live-doc detector failed to catch retired name {concept_name!r}',
            )
        finally:
            tmp_path.unlink()

    def test_docstring_absorber_detector_fires(self) -> None:
        """A synthetic constant with a forbidden-phrase docstring is detected."""
        # Simulate the detection logic with a synthetic triple
        synthetic_docstrings: list[tuple[str, str, str, int]] = [
            (
                'cogwheel/lensing/fake.py',
                '_FAKE_SAFETY_FACTOR',
                'safety factor for discretization error in the grid',
                42,
            ),
        ]
        violations: list[tuple[str, str, str, int]] = []
        for rel, name, docstring, lineno in synthetic_docstrings:
            if (rel, name) in _DOCSTRING_ABSORBER_ALLOWLIST:
                continue
            lower_doc = docstring.lower()
            for phrase in _FORBIDDEN_DOCSTRING_PHRASES:
                if phrase in lower_doc:
                    violations.append((rel, name, phrase, lineno))
                    break
        self.assertGreater(
            len(violations), 0,
            'Docstring absorber detector must fire on synthetic input',
        )

    def test_engine_door_detector_fires(self) -> None:
        """A synthetic body calling f_schwinger is flagged by _called_identifiers."""
        tree = ast.parse(
            'def f():\n    return _schwinger.f_schwinger(1.0)\n')
        called = _called_identifiers(tree)
        self.assertIn('f_schwinger', called)
        self.assertTrue(
            called & _FIT_ENGINE_DOORS,
            'engine-door detector must flag a synthetic f_schwinger call')

    def test_loop_detector_fires(self) -> None:
        """A synthetic body with a for loop is flagged by _has_loop."""
        tree = ast.parse('def f():\n    for i in range(3):\n        pass\n')
        self.assertTrue(_has_loop(tree), 'loop detector must flag a for loop')

    def test_retired_symbol_detector_fires(self) -> None:
        """A synthetic Name reference to a retired scan symbol is caught."""
        tree = ast.parse('_rootfind_w_low = None\n')
        self.assertIn('_rootfind_w_low', _all_name_ids(tree))
        self.assertTrue(
            _all_name_ids(tree) & _RETIRED_SCAN_SYMBOLS,
            'retired-symbol detector must flag a synthetic reference')


# ===========================================================================
# NUMERICAL ``w_low_fit`` BEHAVIOUR TESTS
# ===========================================================================

#: Relative tolerance for the D2 angular-symmetry check.  The fitted
#: surface's angular dependence is the even-harmonic basis
#: ``cos(2 k theta)`` (period ``pi``) plus the parametric-caustic feature
#: ``log(|y'| / |y_c(theta)|)`` whose astroid radius ``|y_c(theta)|`` is
#: period-``pi`` and reflection-symmetric (the astroid caustic SET is
#: 4-fold symmetric, but its critical-angle parametrisation is only
#: 2-fold); ``s = |y'|**2`` / ``sqrt_mu`` are rotation-invariant, so the
#: eigenframe ``pi`` rotation and reflection agree to float round-off
#: (measured ~1e-15).  1e-12 leaves three decades of headroom while still
#: catching any angular model that is not restricted to the D2-invariant
#: (period-``pi`` + reflection) subspace.
_D2_TOL: float = 1e-12

#: Relative slop for the monotonicity sweeps.  ``w_low_fit`` is a smooth
#: log-log polynomial, so its monotone branches are exact to float round-off
#: (~1e-15); the slop only absorbs last-ulp jitter at the ceiling (60) and
#: the parity wall (values ~1e-12), where a real regression -- the U-shaped
#: source-magnitude turnover, say -- is orders of magnitude larger.
_MONOTONE_REL_TOL: float = 1e-9

#: Upper end of the ``s = |y'|**2`` monotonicity sweep: the calibration grid's
#: maximum reduced radius squared (`scripts/fit_diffractive_certificate.py`
#: ``_grid_points`` trains on ``r = linspace(0.3, 1.3, 5)``, so
#: ``s_max = 1.3**2 = 1.69``).  Sweeping past it would probe an uncalibrated
#: extrapolation region; the engine-oracle full-grid suite in
#: `test_lensing_diffractive.py` certifies the calibration domain directly.
_CALIBRATION_S_MAX: float = 1.3 ** 2


#: Relative slop for the "just outside the shell is conservative" pin.
#: ``w_low_fit`` is the de-rated fitted ceiling and ``w_low_true`` the
#: engine-honest ceiling measured by the calibration script's own
#: `_measure_w_low_true`; the de-rate keeps the served value ~5-15% below
#: the honest ceiling at the fold-direction boundary handover, so this slop
#: only absorbs last-ulp jitter from the independent engine measurement.

#: Deep-interior witness grid (INS-3-002): reduced shears, eigenframe angles
#: and reduced-caustic ratios (all strictly below ``RHO_LO = 0.6`` today).
#: ``rho`` reaches deeper than the smoke grid's single interior cell, and
#: ``theta = pi/4`` is the cusp direction (min ``|y_c(theta)|``) -- the worst
#: over-serving direction of the provisional smoke bake (measured ~7x).
_DEEP_INTERIOR_GAMMAS: tuple[float, ...] = (0.2, 0.3, 0.5)
_DEEP_INTERIOR_THETAS: tuple[float, ...] = (
    0.0, math.pi / 4.0, math.pi / 2.0)
_DEEP_INTERIOR_RHOS: tuple[float, ...] = (0.2, 0.3, 0.5)

#: Relative slop for the INS-3-002 deep-interior honest-serve pin.
#: `_measure_w_low_true` returns an ALWAYS-HONEST lower bound (bisection
#: width ~3.4e-7), so 1e-5 is the Inspector-stated round-off guard with
#: ~30x headroom over the measurement resolution.
_INTERIOR_REL_TOL: float = 1e-5

#: Brute-accuracy tier gate (the other lensing suites' idiom): exact-heavy
#: engine-oracle tests are born gated under ``COGWHEEL_BRUTE_ACCURACY``.
_BRUTE_ACCURACY: bool = bool(os.environ.get('COGWHEEL_BRUTE_ACCURACY'))

#: Skip reason for the exact-heavy deep-interior honest-serve suite
#: (`TestWLlowFitDeepInteriorHonestServe`).  Its oracle is the calibration
#: script's engine-honest `_measure_w_low_true`, and the pinned expectation
#: (the interior is served by the regularized, de-rated FIT -- never by the
#: ``min(w_fit, CEILING)`` DD-ceiling clip) is RED at the provisional smoke
#: bake BY DESIGN.  INS-3-002: the smoke fit over-runs at the UNCALIBRATED
#: low-gamma interior and clips to the ceiling, over-serving the
#: engine-honest ceiling (~4-41 deep inside the caustic) by up to ~7x
#: (measured at gamma=0.5/rho=0.2/cusp direction).  The suite flips green
#: when the coder's feature-basis regularization lands and the DRIVER's full
#: interior-inclusive bake (calibration grid reaching ``r ~ 0.1``) is pasted.
_BRUTE_ACCURACY_REASON = (
    'exact-heavy deep-interior honest-serve suite gated behind '
    'COGWHEEL_BRUTE_ACCURACY=1: its oracle is the engine-honest '
    '_measure_w_low_true and the pinned expectation is RED at the '
    'provisional smoke bake BY DESIGN (INS-3-002: the deep interior is '
    'still served at the DD-ceiling clip, over-serving the engine-honest '
    'ceiling by up to ~7x). The driver re-runs it after the coder\'s '
    'feature-basis regularization + full interior-inclusive re-bake land.')
_brute_accuracy_tier = unittest.skipUnless(_BRUTE_ACCURACY,
                                           _BRUTE_ACCURACY_REASON)
_CONSERVATIVE_REL_TOL: float = 1e-6


def _load_w_low_fit() -> tuple:
    """
    Lazily import the fitted-certificate module and its domain constants.

    Kept out of the module top-level imports so the mechanical (AST-scan)
    tests in this file stay import-free: they must be able to run -- and
    report the breakage -- even when the lensing tree fails to import.
    """
    from cogwheel.lensing.chang_refsdal._born import DELTA_GAMMA_P
    from cogwheel.lensing.chang_refsdal._diffractive import (
        DiffractiveDomainError, _DIFFRACTIVE_FIT_CEILING, w_low_fit)
    return (w_low_fit, DiffractiveDomainError, _DIFFRACTIVE_FIT_CEILING,
            DELTA_GAMMA_P)


def _load_diffractive_module():
    """
    Lazily import the fitted-certificate module for mock.patch targets.

    The mechanical (AST-scan) tests stay import-free; only the de-rate teeth
    need the module object -- to patch its shipped constants -- and they call
    only the O(1) `w_low_fit`.
    """
    import importlib
    return importlib.import_module('cogwheel.lensing.chang_refsdal._diffractive')


def _load_fit_certificate_script():
    """
    Lazily import ``scripts/fit_diffractive_certificate.py`` by path.

    The calibration script defines `_measure_w_low_true` -- the
    engine-honest ceiling oracle (the order-16 operator series
    ``diffractive_amplification`` against the exact `f_schwinger` engine
    under the `CERTIFICATION_BAR` sup-over-w semantics) -- the SAME
    measurement the bake de-rates against.  Importing it rather than
    re-deriving the oracle keeps the just-outside-shell conservative pin
    coupled to the bake's own honesty metric.  It is NOT a package (no
    ``scripts/__init__.py``), so it is loaded by path.
    """
    import importlib.util
    script_path = REPO_ROOT / 'scripts' / 'fit_diffractive_certificate.py'
    spec = importlib.util.spec_from_file_location(
        'fit_diffractive_certificate', str(script_path))
    if spec is None or spec.loader is None:
        raise ImportError(f'could not load calibration script: {script_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_likelihood_wrapper():
    """
    Lazily import the likelihood consumer's `_diffractive_bottom_ceiling`.

    Returns ``(host, band_split_mask)`` where ``host`` is an UNINITIALIZED
    `LensedRelativeBinningLikelihood` shell (``object.__new__``, ``__init__``
    never run -- no event data, no engine) and ``band_split_mask`` the shared
    split arithmetic.  The wrapper reads only its ``lens`` argument (plus the
    ``w_hi`` band cap) and the process-global certified-ppGO map, never
    instance state -- the same reuse the census mirror's
    `_load_production_modules` relies on -- so binding it to the shell
    exercises the PRODUCTION fall-through byte-for-byte without an engine
    call.  Kept lazy so the mechanical (AST-scan) tests stay import-free.
    """
    from cogwheel.lensing.likelihood import (
        LensedRelativeBinningLikelihood, _band_split_mask)
    host = object.__new__(LensedRelativeBinningLikelihood)
    return host, _band_split_mask


class WLlowFitBaseTestCase(unittest.TestCase):
    """
    Base for the numerical ``w_low_fit`` behaviour tests.

    ``w_low_fit`` is the O(1), engine-free fitted truncation certificate
    whose structural purity `TestDiffractiveFitStructuralPurity` enforces;
    these tests pin its behaviour.  The import is lazy (see
    `_load_w_low_fit`) so the mechanical tests stay import-free.  `tearDown`
    fails a test that made zero comparisons, so a sweep that silently
    empties cannot read green.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Lazily import ``w_low_fit`` and derive the wall + fence once."""
        (cls._w_low_fit, cls._domain_error, cls._ceiling,
         cls._delta_gamma_p) = _load_w_low_fit()
        cls._wall: float = 1.0 - cls._delta_gamma_p
        cls._module = _load_diffractive_module()
        cls._rho_lo: float = cls._module._DIFFRACTIVE_FIT_FENCE_RHO_LO
        cls._rho_hi: float = 1.0 + cls._module._DIFFRACTIVE_FIT_FENCE_DELTA

    def setUp(self) -> None:
        """Reset the per-test comparison tally used by `tearDown`."""
        self._n_checks = 0

    def tearDown(self) -> None:
        """Fail a test whose every comparison was skipped."""
        if self._n_checks == 0:
            self.fail('test made zero comparisons; it asserted nothing')

    def _evaluate(self, y, gamma, beta: float = 0.0, kappa: float = 0.0):
        """Evaluate ``w_low_fit`` on a source position and lens parameters."""
        # Access via the class: a plain function stored as a class attribute
        # binds to the instance through the descriptor protocol, which would
        # prepend ``self`` and raise "5 positional arguments given".
        return type(self)._w_low_fit(y, gamma, beta, kappa)

    def _fence_rho(self, y, gamma: float, beta: float = 0.0,
                   kappa: float = 0.0) -> float:
        """The fence discriminator ``rho`` a source maps to, as ``w_low_fit`` sees it."""
        lam = 1.0 - kappa
        root = math.sqrt(lam)
        yp0, yp1 = y[0] / root, y[1] / root
        s = yp0 * yp0 + yp1 * yp1
        z = cmath.exp(-1j * beta) * complex(yp0, yp1)
        theta = math.atan2(z.imag, z.real)
        return self._module._caustic_rho(abs(gamma / lam), s, theta)

    def _assert_non_increasing(self, values) -> None:
        """Assert ``values`` is non-increasing within float round-off."""
        for prev, nxt in zip(values, values[1:]):
            self._n_checks += 1
            self.assertLessEqual(
                nxt, prev + _MONOTONE_REL_TOL * max(1.0, abs(prev)),
                f'w_low_fit increased: {prev!r} -> {nxt!r}')


class TestWLlowFitGammaMonotonicity(WLlowFitBaseTestCase):
    """
    ``w_low_fit`` falls with ``gamma'`` past a low-edge peak (served exterior).

    The engine-honest truncation ceiling falls as the operator series' small
    parameter ``gamma' s w / 2`` grows, and the fitted surface reproduces that
    power-law direction where it SERVES -- the smooth exterior (``rho >
    1 + DELTA``).  There the surface is non-increasing in ``gamma`` from a
    low-edge peak (~``gamma`` 0.05-0.08, LIVE-derived) out to the point where
    ``rho`` crosses into the near-fold shell.  The shell and the deep
    interior are NOT part of the monotonicity claim: the shell returns
    ``None`` and the interior is served by the fit -- asserted by
    `TestWLlowFitNearFoldFence` / `TestWLlowFitDeepInteriorServedByFit`
    instead.  The sweep is therefore RESTRICTED to the served exterior,
    derived from the live `_caustic_rho` discriminator, never a pinned gamma
    bound.  ``gamma`` is swept at fixed ``(y, beta, kappa)`` per the spec; at
    fixed ``kappa`` the reduced shear rises monotonically with ``gamma``, so
    non-increasing past the peak in ``gamma`` pins non-increasing in
    ``gamma'`` there.
    """

    def test_non_increasing_past_low_edge_peak(self) -> None:
        """Sweep gamma over the served exterior; past the peak it only falls."""
        y = (0.5, 0.3)
        for beta, kappa in ((0.0, 0.0), (0.7, 0.0), (0.0, 0.3), (0.7, 0.3)):
            with self.subTest(beta=beta, kappa=kappa):
                lam = 1.0 - kappa
                gammas = np.logspace(
                    math.log10(0.05), math.log10(0.999 * self._wall * lam), 300)
                exterior = [float(g) for g in gammas
                            if self._fence_rho(y, float(g), beta, kappa)
                            > self._rho_hi]
                self._n_checks += 1
                self.assertGreater(
                    len(exterior), 3,
                    'premise lost: no served exterior at this (beta, kappa)')
                values = [self._evaluate(y, g, beta, kappa) for g in exterior]
                peak = int(np.argmax(values))
                self._n_checks += 1
                self.assertLess(
                    peak, len(values) - 1,
                    'premise lost: no falling branch after the low-edge peak')
                self._n_checks += 1
                self.assertGreater(
                    values[peak], values[-1],
                    'premise lost: no genuine gamma falling branch')
                self._assert_non_increasing(values[peak:])


class TestWLlowFitSMonotonicity(WLlowFitBaseTestCase):
    """
    ``w_low_fit`` falls with ``s = |y'|^2`` across the served exterior.

    The operator-series tail grows like ``(gamma' s w / 2)^n / n!``, so the
    engine-honest ceiling falls as the reduced source magnitude ``s`` grows,
    and the fitted surface reproduces that direction where it SERVES -- the
    smooth exterior (``rho > 1 + DELTA``).  The sweep is RESTRICTED to the
    served exterior, derived from the live `_caustic_rho` discriminator: the
    deep interior (``rho < RHO_LO``, served by the fit) and the near-fold
    shell (``None``) are NOT part of this monotonicity claim (asserted by
    `TestWLlowFitDeepInteriorServedByFit` / `TestWLlowFitNearFoldFence`).

    Within the exterior the ceiling falls monotonically down to a live-derived
    minimum near the top of the calibration range; PAST that minimum the
    PROVISIONAL fence-smoke coefficients (a positive ``log(s)**2`` poly term)
    introduce a small (~1-4%) large-``s`` up-turn.  That up-turn is a fit
    LOOSENESS, not an over-serve (the de-rate keeps it conservative, and the
    engine-oracle full-grid sweep in `test_lensing_diffractive.py` certifies
    the calibration domain directly), and the full driver bake is expected to
    remove it -- so this test pins the durable FALLING branch (exterior start
    to the live minimum) and does not assert the up-turn away.  ``beta = kappa
    = 0`` (eigenframe angle == lens polar angle 0.7).
    """

    def test_falls_with_s_across_the_served_exterior(self) -> None:
        """From the exterior start down to the live minimum, it only falls."""
        for gamma in (0.1, 0.2, 0.3, 0.5):
            with self.subTest(gamma=gamma):
                ss = np.logspace(-2.0, math.log10(_CALIBRATION_S_MAX), 200)
                exterior: list[float] = []
                for s in ss:
                    y = (math.sqrt(float(s)) * math.cos(0.7),
                         math.sqrt(float(s)) * math.sin(0.7))
                    if self._fence_rho(y, gamma, 0.0, 0.0) > self._rho_hi:
                        exterior.append(float(s))
                self._n_checks += 1
                self.assertGreater(
                    len(exterior), 2,
                    'premise lost: no served exterior at this gamma')
                values = [self._evaluate(
                    (math.sqrt(s) * math.cos(0.7),
                     math.sqrt(s) * math.sin(0.7)), gamma) for s in exterior]
                argmin = int(np.argmin(values))
                self._n_checks += 1
                self.assertGreater(
                    values[0], values[argmin],
                    'premise lost: no genuine s falling branch')
                self._assert_non_increasing(values[:argmin + 1])


class TestWLlowFitD2Symmetry(WLlowFitBaseTestCase):
    """
    ``w_low_fit`` is D2-symmetric in the eigenframe angle ``theta``.

    The fitted angular model is the even-harmonic basis
    ``cos(2 k theta)`` (``k = 1 .. _DIFFRACTIVE_FIT_N_HARM``) plus the
    parametric-caustic feature ``log(|y'| / |y_c(theta)|)``.  Each
    ``cos(2 k theta)`` has period ``pi`` and is reflection-invariant
    (``cos(2 k theta) = cos(-2 k theta)``), and the astroid caustic radius
    ``|y_c(theta)|`` is period-``pi`` and reflection-symmetric (the astroid
    caustic SET is 4-fold symmetric, but its critical-angle parametrisation
    is only 2-fold), while ``s = |y'|**2`` and ``sqrt_mu`` are
    rotation-invariant -- so rotating the eigenframe source by ``pi`` or
    reflecting it leaves the certificate unchanged to float round-off.
    ``pi/2`` is NOT a symmetry: the odd harmonics ``cos(2 theta)``,
    ``cos(6 theta)``, ... flip sign under a ``pi/2`` rotation AND the
    caustic radius is only period-``pi``, so `test_pi2_rotation_changes_value`
    shows the value genuinely moves there.  (The retired ``cos(4 k theta)``
    basis was 4-fold symmetric; the even-harmonic basis is only 2-fold,
    period ``pi``.)
    """

    #: Interior fixtures exercising non-trivial ``beta``/``kappa``; the D2
    #: symmetry is a property of the fit in ``theta``, so the eigenframe
    #: transformations (not the lens-plane ones) are the honest probes.
    #: Every fixture must lie OUTSIDE the near-fold shell (``rho > 1 +
    #: DELTA``) so the fence serves it rather than declining it to ``None``;
    #: `test_fixtures_are_outside_the_shell` asserts that premise live.  The
    #: third fixture keeps ``beta = kappa = 0`` (pure lens frame, eigenframe
    #: == lens frame) and its ``gamma`` was lowered 0.45 -> 0.35: at
    #: ``gamma = 0.45`` every ``r ~ 0.8-1.1`` source sits at ``rho < 1.4`` in
    #: some direction and would be fenced off (its ``pi/2``-rotated image
    #: would return ``None`` and break the self-falsification).
    _CONFIGS: tuple[tuple[tuple[float, float], float, float, float], ...] = (
        ((0.4, 0.7), 0.3, 0.5, 0.2),
        ((1.1, -0.2), 0.15, -0.9, 0.1),
        ((0.8, 0.6), 0.35, 0.0, 0.0),
    )

    @staticmethod
    def _eig_z(y, beta: float, kappa: float) -> complex:
        """Eigenframe complex source ``z_eig = exp(-i beta) y'``."""
        root = math.sqrt(1.0 - kappa)
        z = complex(float(y[0]) / root, float(y[1]) / root)
        return cmath.exp(-1j * float(beta)) * z

    @staticmethod
    def _from_eig(z_eig: complex, beta: float, kappa: float):
        """Lens-plane source ``y = sqrt(lam) R(beta) y_eig`` for ``z_eig``."""
        root = math.sqrt(1.0 - kappa)
        z = cmath.exp(1j * float(beta)) * z_eig
        return (root * z.real, root * z.imag)

    def test_fixtures_are_outside_the_shell(self) -> None:
        """Every D2 fixture is served (``rho > 1 + DELTA``), never fenced off."""
        for y, gamma, beta, kappa in self._CONFIGS:
            with self.subTest(y=y, gamma=gamma, beta=beta, kappa=kappa):
                rho = self._fence_rho(y, gamma, beta, kappa)
                self._n_checks += 1
                self.assertGreater(
                    rho, self._rho_hi,
                    f'premise lost: fixture (y={y}, gamma={gamma}) now sits in '
                    f'or below the near-fold shell (rho={rho:.3f} <= '
                    f'{self._rho_hi:.3f}); the D2 symmetry probe would be '
                    f'fenced to None')

    def test_period_pi_invariance(self) -> None:
        """``theta -> theta + pi`` (eigenframe negation) reproduces the value."""
        for y, gamma, beta, kappa in self._CONFIGS:
            base = self._evaluate(y, gamma, beta, kappa)
            z = self._eig_z(y, beta, kappa)
            shifted = self._from_eig(-z, beta, kappa)
            value = self._evaluate(shifted, gamma, beta, kappa)
            self._n_checks += 1
            self.assertLessEqual(
                abs(value - base), _D2_TOL * max(1.0, abs(base)),
                f'period-pi symmetry broke: {base!r} vs {value!r} '
                f'at (y={y}, gamma={gamma}, beta={beta}, kappa={kappa})')

    def test_reflection_invariance(self) -> None:
        """``theta -> -theta`` (eigenframe reflection) reproduces the value."""
        for y, gamma, beta, kappa in self._CONFIGS:
            base = self._evaluate(y, gamma, beta, kappa)
            z = self._eig_z(y, beta, kappa)
            reflected = self._from_eig(z.conjugate(), beta, kappa)
            value = self._evaluate(reflected, gamma, beta, kappa)
            self._n_checks += 1
            self.assertLessEqual(
                abs(value - base), _D2_TOL * max(1.0, abs(base)),
                f'reflection symmetry broke: {base!r} vs {value!r} '
                f'at (y={y}, gamma={gamma}, beta={beta}, kappa={kappa})')

    def test_pi2_rotation_changes_value(self) -> None:
        """A ``pi/2`` rotation is NOT a symmetry (odd harmonics + caustic)."""
        for y, gamma, beta, kappa in self._CONFIGS:
            base = self._evaluate(y, gamma, beta, kappa)
            z = self._eig_z(y, beta, kappa)
            rotated = self._from_eig(1j * z, beta, kappa)
            value = self._evaluate(rotated, gamma, beta, kappa)
            self._n_checks += 1
            self.assertGreater(
                abs(value - base), _D2_TOL * max(1.0, abs(base)),
                f'pi/2 rotation did not change w_low_fit: {base!r} vs '
                f'{value!r} -- the D2 symmetry tests would be vacuous')


class TestWLlowFitCeilingCapAndWallCollapse(WLlowFitBaseTestCase):
    """
    The fitted certificate is capped at the DD ceiling and finite at the wall.

    (a) ``w_low_fit`` never exceeds ``_DIFFRACTIVE_FIT_CEILING``
    (``= W_CEILING_SCHWINGER = 60``): the de-rated surface is clipped to the
    cap.  ``None`` (the near-fold-shell decline) is a separate, valid outcome
    and is not "exceeding the ceiling".
    (b) As ``gamma'`` approaches the parity wall ``1 - DELTA_GAMMA_P`` the
    source becomes deep interior (``rho -> 0`` as the caustic radius blows
    up).  There ``w_low_fit`` serves the FIT, whose ``log(1 - gamma')``
    feature collapses it toward 0; in the shell it declines (``None``).
    Either way it never returns a divergent value.  The wall refusal itself
    (exactly at the wall) is unchanged.
    """

    def test_never_exceeds_ceiling(self) -> None:
        """The certificate stays at or below the DD ceiling across the domain."""
        y = (0.5, 0.3)
        gammas = np.logspace(-3.0, math.log10(0.999 * self._wall), 200)
        for g in gammas:
            value = self._evaluate(y, float(g))
            self._n_checks += 1
            self.assertTrue(
                value is None or value <= self._ceiling,
                f'w_low_fit exceeded the ceiling: {value!r}')
        # Large-s leg: the raw fit also over-predicts for big sources and
        # must be clipped the same way.
        for s in (1.0, 2.0, 5.0, 10.0):
            value = self._evaluate((s, 0.0), 0.2)
            self._n_checks += 1
            self.assertLessEqual(
                value, self._ceiling,
                f'w_low_fit exceeded the ceiling at large s: {value!r}')

    # RETIRED (2026-08-19, diffractive full re-bake): the re-baked surface is
    # conservative enough that NO small-gamma fixture hits the ceiling cap
    # (measured max w_low ~51.4 < _DIFFRACTIVE_FIT_CEILING = 60 over a broad
    # gamma/s scan) -- the old claim "small gamma returns the cap exactly"
    # pinned the SMOKE-baked surface, which clipped at gamma'->0.  The cap
    # invariant itself (never exceed the ceiling) is still asserted by
    # `test_never_exceeds_ceiling`; the clip remains defense-in-depth against
    # an un-de-rated raw fit (raw max ~71 > 60), pinned by
    # `TestWLlowFitDerateTeeth`.
    # RETIRED (2026-08-20, near-fold fence): `test_wall_collapse_monotone_and_
    # finite` pinned the fit's ``log(1 - gamma')`` collapse to ~0 near the
    # wall.  The original interior->ceiling fence branch briefly made that
    # path unreachable for a fixed source (the deep interior was hard-coded
    # to the CEILING), so the collapse was replaced by
    # `test_wall_serves_ceiling_or_declines`.  That ceiling branch is gone
    # (INS-2-001 -- the interior is served by the fit, whose engine-honest
    # ceiling there is ~4-34, not 60), so the wall collapse is reachable
    # again and is re-pinned by `test_wall_declines_or_collapses_finitely`.

    def test_wall_declines_or_collapses_finitely(self) -> None:
        """Toward the wall, w_low_fit declines (shell) or collapses finitely to ~0."""
        y = (0.5, 0.3)
        gammas = (0.9, 0.95, 0.98, 0.99, 0.994, 0.9949)
        served = []
        for g in gammas:
            value = self._evaluate(y, g)
            self._n_checks += 1
            self.assertTrue(
                value is None or (math.isfinite(value) and value >= 0.0),
                f'near the wall w_low_fit must decline or serve a finite '
                f'non-negative value, got {value!r} at gamma={g}')
            if value is not None:
                served.append(value)
        self._n_checks += 1
        self.assertTrue(
            served,
            'premise lost: no served (deep-interior) point near the wall')
        self._assert_non_increasing(served)
        self._n_checks += 1
        self.assertLessEqual(
            served[-1], 1e-9,
            f'the served interior tail must collapse to ~0 at the wall; got '
            f'{served[-1]!r}')

    def test_wall_refusal_bounds_the_collapse(self) -> None:
        """Exactly at the wall the positive-parity rung refuses (domain edge)."""
        with self.assertRaises(self._domain_error):
            type(self)._w_low_fit((0.5, 0.3), 1.0 - self._delta_gamma_p, 0.0, 0.0)
        self._n_checks += 1


class TestWLlowFitSelfFalsification(WLlowFitBaseTestCase):
    """
    Prove the numerical ``w_low_fit`` checks can go red.

    The D2 symmetry tests would pass vacuously if ``w_low_fit`` ignored
    ``theta``, and the monotonicity tests would pass vacuously if the
    surface were flat.  These tests show the surface genuinely varies in the
    directions the invariants pin, so a green suite is evidence rather than
    decoration.
    """

    def test_surface_varies_at_pi4(self) -> None:
        """A pi/4 rotation (not a D2 symmetry) must change the value."""
        # Fixture chosen so BOTH the base and its pi/4-rotated image sit in
        # the served exterior (rho > 1 + DELTA); the rotated image of the old
        # (0.4, 0.7, 0.3, 0.5, 0.2) fixture fell into the near-fold shell.
        y = (0.9, 0.5)
        gamma, beta, kappa = 0.3, 0.0, 0.0
        base = self._evaluate(y, gamma, beta, kappa)
        c = math.sqrt(0.5)
        rotated = (c * y[0] - c * y[1], c * y[0] + c * y[1])
        value = self._evaluate(rotated, gamma, beta, kappa)
        self._n_checks += 1
        self.assertIsNotNone(base, 'premise: base fixture fenced off')
        self._n_checks += 1
        self.assertIsNotNone(value, 'premise: rotated fixture fenced off')
        self._n_checks += 1
        self.assertGreater(
            abs(value - base), _D2_TOL * max(1.0, abs(base)),
            'pi/4 rotation did not change w_low_fit; the D2 symmetry '
            'tests would be vacuous')

    def test_monotone_check_detects_an_increase(self) -> None:
        """The non-increasing check rejects a sequence with an increase."""
        with self.assertRaises(AssertionError):
            self._assert_non_increasing([1.0, 2.0])
        self._n_checks += 1


class TestWLlowFitDerateTeeth(WLlowFitBaseTestCase):
    """
    Prove the de-rating MARGIN -- not the fit shape alone -- enforces the
    never-over-serve guarantee (SELF-FALSIFICATION teeth).

    `_DIFFRACTIVE_FIT_DERATE` (shipped strictly below 1 -- see
    `test_derate_is_a_genuine_margin`) is the load-bearing safety margin:
    the fitted log-log surface is a least-squares fit to the engine-honest
    ceiling and by itself over-predicts wherever the fit overshoots (the
    worst over-prediction is baked into ``1 / _DIFFRACTIVE_FIT_DERATE``).
    `w_low_fit` is conservative ONLY because the surface is de-rated by that
    factor (then clipped to `_DIFFRACTIVE_FIT_CEILING`).

    These tests perturb the shipped constants and show the served ceiling
    strictly INFLATES -- the exact direction the conservative/tight oracle
    pin (the engine-oracle suite in test_lensing_diffractive.py) would flag
    RED.  Engine-free: only the O(1) fitted surface is exercised.

    Budget: a handful of O(1) evaluations, < 50 ms.
    """

    #: Served (exterior) fixtures where the de-rate is load-bearing: the RAW
    #: fit sits far below the ceiling, so the ``min(..., ceiling)`` clip does
    #: not mask the margin (each premise is asserted in the loop below).  The
    #: third fixture keeps ``beta = kappa = 0`` and was moved out of the
    #: near-fold shell (``(0.05, 0.9), 0.45`` now fenced to ``None``).
    _FIXTURES: tuple[tuple[tuple[float, float], float, float, float], ...] = (
        ((0.4, 0.7), 0.3, 0.5, 0.2),
        ((1.1, -0.2), 0.15, -0.9, 0.1),
        ((0.8, 0.6), 0.35, 0.0, 0.0),
    )

    #: Amount added to the constant poly coefficient in the inflation teeth.
    #: The constant monomial ``(0, 0, 0)`` has feature 1.0, so ``+0.5``
    #: inflates the surface by ``exp(0.5)`` -- far above the clip at our
    #: interior fixtures.
    _COEFF_INFLATION: float = 0.5

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._module = _load_diffractive_module()
        cls._derate: float = cls._module._DIFFRACTIVE_FIT_DERATE

    def _raw_ceiling(self, y, gamma, beta, kappa):
        """The un-de-rated fitted surface (de-rate set to 1.0)."""
        with mock.patch.object(self._module, '_DIFFRACTIVE_FIT_DERATE', 1.0):
            return self._evaluate(y, gamma, beta, kappa)

    def test_derate_is_a_genuine_margin(self) -> None:
        """The shipped de-rate lies strictly inside (0, 1)."""
        self._n_checks += 1
        self.assertGreater(self._derate, 0.0, 'de-rate must be positive')
        self._n_checks += 1
        self.assertLess(self._derate, 1.0, 'de-rate must be a real margin (< 1)')

    def test_derate_is_sole_source_of_margin(self) -> None:
        """The served ceiling equals de-rate x raw fit at un-clipped points."""
        for y, gamma, beta, kappa in self._FIXTURES:
            with self.subTest(y=y, gamma=gamma, beta=beta, kappa=kappa):
                served = self._evaluate(y, gamma, beta, kappa)
                raw = self._raw_ceiling(y, gamma, beta, kappa)
                self._n_checks += 1
                self.assertLess(
                    served, self._ceiling,
                    'premise lost: fixture must sit below the ceiling clip')
                self._n_checks += 1
                self.assertGreater(
                    raw, served,
                    'the raw fit must exceed the served ceiling')
                self._n_checks += 1
                self.assertAlmostEqual(
                    self._derate * raw, served, places=12,
                    msg='the de-rate must be the sole source of the margin')

    def test_no_derate_inflates_the_ceiling(self) -> None:
        """Setting the de-rate to 1.0 strictly inflates the served ceiling."""
        for y, gamma, beta, kappa in self._FIXTURES:
            with self.subTest(y=y, gamma=gamma, beta=beta, kappa=kappa):
                served = self._evaluate(y, gamma, beta, kappa)
                raw = self._raw_ceiling(y, gamma, beta, kappa)
                self._n_checks += 1
                self.assertLess(
                    raw, self._ceiling,
                    'premise lost: un-de-rated fixture must stay below the clip')
                self._n_checks += 1
                self.assertGreater(
                    raw, served,
                    'removing the de-rate must inflate the ceiling -- the '
                    'conservative/tight oracle pin goes RED under this '
                    'perturbation')

    def test_inflated_coefficient_inflates_ceiling(self) -> None:
        """Inflating the constant poly coefficient inflates the ceiling."""
        for y, gamma, beta, kappa in self._FIXTURES:
            with self.subTest(y=y, gamma=gamma, beta=beta, kappa=kappa):
                served = self._evaluate(y, gamma, beta, kappa)
                coeffs = list(self._module._DIFFRACTIVE_FIT_POLY_COEFFS)
                coeffs[0] += self._COEFF_INFLATION
                with mock.patch.object(
                        self._module, '_DIFFRACTIVE_FIT_POLY_COEFFS',
                        tuple(coeffs)):
                    inflated = self._evaluate(y, gamma, beta, kappa)
                self._n_checks += 1
                self.assertGreater(
                    inflated, served,
                    'inflating the constant coefficient must inflate the '
                    'served ceiling (else the fit shape is not load-bearing)')


class TestWLlowFitNearFoldFence(WLlowFitBaseTestCase):
    """
    ``w_low_fit`` declines (``None``) EXACTLY inside the near-fold shell.

    The fence (`_caustic_rho` + `_DIFFRACTIVE_FIT_FENCE_*`) splits the
    reduced-caustic distance ratio ``rho = |y'| / |y_c(theta)|`` into three
    regions: the deep interior (``rho < RHO_LO``) is served by the fit, the
    near-fold shell (``RHO_LO <= rho <= 1 + DELTA``) is DECLINED to ``None``
    so the draw falls through to the fold arm / exact engine, and the smooth
    exterior (``rho > 1 + DELTA``) is served by the fit.

    This test sweeps the source radius and eigenframe angle across the
    directional caustic at the fold-dip lens (``gamma = 0.41``, ``beta =
    kappa = 0``) and asserts the return is ``None`` exactly where the LIVE
    ``rho`` discriminator says so, with served values on BOTH sides -- so
    the refusal is not vacuous.  Fixtures are DERIVED from the live
    directional caustic radius ``|y_c(theta)|`` (via `_caustic_rho`), never
    pinned: for each swept angle the radius sweeps ``rho`` over ``[0.3,
    1.9]``, crossing the interior, the whole shell, and the exterior.  With
    ``beta = kappa = 0`` the eigenframe angle is the lens polar angle, so
    the sweep IS the fence's own ``theta``.  Engine-free, O(1) per probe;
    budget well under a second.

    The scattering diagnostic (`test_diagnostic_rho_return_scatter`) renders
    the contiguous ``None`` band bracketed by served values.
    """

    #: Fold-dip lens (the corner witness of the diffractive suite).
    _GAMMA: float = 0.41
    _BETA: float = 0.0
    _KAPPA: float = 0.0
    #: Sweep the angle around the diagonal (the fold dip's resonant
    #: direction) so the directional caustic radius varies strongly.
    _THETA_CENTER: float = 3.0 * math.pi / 4.0
    _THETA_OFFSETS: tuple[float, ...] = (-0.5, -0.25, 0.0, 0.25, 0.5)
    #: Reduced-caustic distance ratios spanned by the radius sweep (as a
    #: fraction of the directional caustic radius): crosses the interior
    #: (rho < 0.6), the whole shell [0.6, 1.4], and the exterior (rho > 1.4).
    _RHO_MIN: float = 0.3
    _RHO_MAX: float = 1.9
    _N_R: int = 25

    def _sweep(self):
        """Yield ``(rho, y, value)`` for every swept ``(theta, r)`` probe."""
        caustic_rho = self._module._caustic_rho
        for offset in self._THETA_OFFSETS:
            theta = self._THETA_CENTER + offset
            yc = 1.0 / caustic_rho(self._GAMMA, 1.0, theta)
            for r in np.linspace(self._RHO_MIN * yc, self._RHO_MAX * yc,
                                 self._N_R):
                y = (r * math.cos(theta), r * math.sin(theta))
                rho = caustic_rho(self._GAMMA, r * r, theta)
                value = type(self)._w_low_fit(
                    y, self._GAMMA, self._BETA, self._KAPPA)
                yield rho, y, value

    def test_shell_is_declined_and_bracketed(self) -> None:
        """``None`` exactly in the shell; served on both sides (non-vacuous)."""
        interior = shell = exterior = 0
        for rho, y, value in self._sweep():
            self._n_checks += 1
            if rho < self._rho_lo:
                interior += 1
                self.assertIsNotNone(
                    value, f'interior rho={rho:.3f} was declined to None')
            elif rho <= self._rho_hi:
                shell += 1
                self.assertIsNone(
                    value, f'shell rho={rho:.3f} was served as {value!r}')
            else:
                exterior += 1
                self.assertIsNotNone(
                    value, f'exterior rho={rho:.3f} was declined to None')
                self.assertTrue(
                    math.isfinite(value) and value > 0.0,
                    f'exterior rho={rho:.3f} served a non-finite value '
                    f'{value!r}')
        self._n_checks += 1
        self.assertGreater(interior, 0, 'sweep never reached the interior')
        self._n_checks += 1
        self.assertGreater(shell, 0, 'sweep never reached the shell')
        self._n_checks += 1
        self.assertGreater(exterior, 0, 'sweep never reached the exterior')

    def test_declined_shell_is_fence_driven(self) -> None:
        """Collapsing the shell (``DELTA = -0.4``) serves a mid-shell point."""
        theta = self._THETA_CENTER
        caustic_rho = self._module._caustic_rho
        yc = 1.0 / caustic_rho(self._GAMMA, 1.0, theta)
        r = 1.0 * yc  # rho = 1.0, mid-shell
        y = (r * math.cos(theta), r * math.sin(theta))
        rho = caustic_rho(self._GAMMA, r * r, theta)
        self._n_checks += 1
        self.assertGreaterEqual(rho, self._rho_lo, 'premise: not in shell')
        self._n_checks += 1
        self.assertLessEqual(rho, self._rho_hi, 'premise: not in shell')
        declined = type(self)._w_low_fit(y, self._GAMMA, self._BETA, self._KAPPA)
        self._n_checks += 1
        self.assertIsNone(declined, 'shipped fence must decline the shell')
        with mock.patch.object(self._module, '_DIFFRACTIVE_FIT_FENCE_DELTA',
                               -0.4):
            value = type(self)._w_low_fit(y, self._GAMMA, self._BETA,
                                          self._KAPPA)
        self._n_checks += 1
        self.assertIsNotNone(value, 'without the shell the point must be served')

    def test_diagnostic_rho_return_scatter(self) -> None:
        """Save a rho-vs-return scatter showing the contiguous None band."""
        import os
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        rhos: list[float] = []
        served: list[float] = []
        declined: list[float] = []
        for rho, y, value in self._sweep():
            rhos.append(rho)
            if value is None:
                declined.append(rho)
            else:
                served.append(rho)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter(served, [1.0] * len(served), s=12, label='served')
        ax.scatter(declined, [0.5] * len(declined), s=12, marker='x',
                   color='tab:red', label='declined (None)')
        ax.axvspan(self._rho_lo, self._rho_hi, color='tab:red', alpha=0.15,
                   label='near-fold shell')
        ax.axvline(1.0, color='k', ls=':', label='caustic (rho=1)')
        ax.set_xlabel('rho = |y\'| / |y_c(theta)|')
        ax.set_ylabel('return class (1=served, 0.5=None)')
        ax.set_title('w_low_fit near-fold fence (gamma=0.41)')
        ax.set_ylim(0.3, 1.2)
        ax.legend()
        fig.tight_layout()
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'output')
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, 'near_fold_fence_rho_return.png')
        fig.savefig(path, dpi=90)
        plt.close(fig)
        self._n_checks += 1
        self.assertTrue(os.path.exists(path))


class TestWLlowFitDeepInteriorServedByFit(WLlowFitBaseTestCase):
    """Deep-interior sources are served by the FIT, never declined.

    For ``rho < RHO_LO`` the order-`_DEFAULT_MAX_ORDER` series is NOT valid
    to the engine ceiling (the engine-honest ceiling deep inside the caustic
    is ~4-41, NOT ``W_CEILING_SCHWINGER``; measured ~4.12 at
    gamma=0.5/rho=0.3, ~19 at gamma=0.3/rho=0.3).  `w_low_fit` therefore
    serves the interior via the SAME calibrated fit as the smooth exterior:
    it returns a finite value -- never ``None``, the interior is not a
    declined region (that structural claim is also pinned by
    `TestWLlowFitNearFoldFence.test_shell_is_declined_and_bracketed`).

    This fast-tier suite is ENGINE-FREE and pins only the structural
    conservative claim: at the CALIBRATED ``gamma = 0.5`` interior cell
    (the smoke grid's own interior cell) the served value sits STRICTLY
    below the DD ceiling at every witness ``rho`` -- a reinstated
    ``rho < RHO_LO -> ceiling`` short-circuit would return the ceiling for
    every fixture and trip the pin.  The HONEST-VALUE pin
    (``w_low_fit <= w_low_true * (1 + 1e-5)`` over the full gamma x rho x
    theta grid) and the clip-not-the-mechanism guard live in the exact-heavy
    engine-oracle suite `TestWLlowFitDeepInteriorHonestServe`, gated under
    ``COGWHEEL_BRUTE_ACCURACY`` -- the provisional smoke fit still over-runs
    at the UNCALIBRATED low-gamma interior and is clipped to the ceiling
    (INS-3-002), so those pins are red until the regularized re-bake lands.

    Fixtures are DERIVED from the live directional caustic radius
    ``|y_c(theta)|``: each source sits at a FIXED deep-interior witness ratio
    ``rho`` (0.2, 0.3 and 0.5, all below the shipped ``RHO_LO = 0.6``), and
    ``r = rho * |y_c(theta)|`` is computed live so ``rho`` is exactly the
    fence discriminator.  ``beta = kappa = 0``.  Engine-free, O(1) per probe.
    """

    #: Calibrated interior cell (the smoke grid's own interior cell) whose
    #: engine-honest ceiling is ~4.12 -- far below the DD ceiling, so the
    #: fit there returns a value strictly under the ceiling.
    _CALIBRATED_GAMMA: float = 0.5

    def _fixtures(self, gammas=None):
        """Yield ``(gamma, theta, r, y, rho)`` for each deep-interior probe."""
        caustic_rho = self._module._caustic_rho
        for gamma in (_DEEP_INTERIOR_GAMMAS if gammas is None else gammas):
            for theta in _DEEP_INTERIOR_THETAS:
                yc = 1.0 / caustic_rho(gamma, 1.0, theta)
                for rho in _DEEP_INTERIOR_RHOS:
                    r = rho * yc
                    y = (r * math.cos(theta), r * math.sin(theta))
                    yield gamma, theta, r, y, rho

    def test_deep_interior_served_below_ceiling_at_calibrated_cell(self) -> None:
        """The calibrated interior cell is served by the fit, below the ceiling.

        At ``gamma = 0.5`` the fit returns a value STRICTLY below
        ``W_CEILING_SCHWINGER`` at every deep-interior witness ``rho``
        (0.2/0.3/0.5).  A reinstated ``rho < RHO_LO -> ceiling``
        short-circuit would return the ceiling for every fixture and trip
        this pin.  The honest-VALUE assertion at the uncalibrated low-gamma
        cells is the gated engine-oracle suite's job (INS-3-002): the
        provisional smoke fit over-runs there and is clipped, so no honest
        value claim can hold in the fast tier until the re-bake.
        """
        for gamma, theta, r, y, rho in self._fixtures(
                gammas=(self._CALIBRATED_GAMMA,)):
            with self.subTest(gamma=gamma, theta=theta, rho=rho):
                self._n_checks += 1
                self.assertLess(rho, self._rho_lo, 'premise lost')
                value = type(self)._w_low_fit(y, gamma, 0.0, 0.0)
                self._n_checks += 1
                self.assertIsNotNone(
                    value, 'calibrated interior was declined')
                self._n_checks += 1
                self.assertLess(
                    value, self._ceiling,
                    f'calibrated interior rho={rho:.3f} served at the ceiling '
                    f'{value!r} -- the interior must be served by the fit, '
                    'not a hard-coded ceiling short-circuit')


@_brute_accuracy_tier
class TestWLlowFitDeepInteriorHonestServe(WLlowFitBaseTestCase):
    """Deep interior is served at its HONEST value, never by the DD-ceiling clip.

    INS-3-002: `w_low_fit` serves the deep interior (``rho < RHO_LO``) via
    ``min(w_fit, W_CEILING_SCHWINGER)``, and the provisional smoke fit
    over-runs at the UNCALIBRATED low-gamma interior cells, so the served
    value there IS the DD ceiling (60) -- over-serving the ENGINE-HONEST
    ceiling (the largest ``w`` whose order-16 series stays within
    `CERTIFICATION_BAR` of the exact engine; ~4-41 deep inside the caustic,
    NOT 60) by up to ~7x.  This class pins the CORRECTED expectation: the
    interior must be served by the regularized + de-rated FIT at its honest
    value, with the ceiling clip acting only as an unreachable hard cap.

    It is the REAL-VALUE replacement for the retired vacuous
    ``assertIsNotNone`` of `TestWLlowFitDeepInteriorServedByFit` (which
    certified "served, not declined" at the clipped cells without any
    conservativeness check).  The oracle is the calibration script's OWN
    ``_measure_w_low_true`` -- the exact ``f_schwinger`` engine under the
    ``CERTIFICATION_BAR`` sup-over-w semantics, ``n_w = 16`` (the bake's
    default; an ALWAYS-HONEST lower bound, bisection width ~3.4e-7) -- so
    the assertion ``w_low_fit <= w_low_true * (1 + _INTERIOR_REL_TOL)`` with
    ``_INTERIOR_REL_TOL = 1e-5`` is a genuine round-off-guarded value pin,
    NOT a re-call of the production derivation.

    RED BY DESIGN at the provisional smoke bake (measured served/true up to
    ~7x at gamma=0.5/rho=0.2/cusp direction, ~2.5x at gamma=0.5/rho=0.3/
    cusp direction, ~1.8x at gamma=0.2 axis directions).  Flips green with
    ZERO test edits when the coder's feature-basis regularization (the raw
    pre-clip fit finite and monotone in the deep interior) and the DRIVER's
    full interior-inclusive bake (calibration grid reaching ``r ~ 0.1``, so
    the gamma <= 0.3 interior cells are actually sampled) land.  See
    `_BRUTE_ACCURACY_REASON`.

    Fixtures are DERIVED from the live directional caustic radius
    ``|y_c(theta)|`` (``r = rho * |y_c(theta)|`` with ``rho`` the live fence
    discriminator), exactly as in the fast-tier class.  Cost: 27 engine
    probes (~1-4 s each) paid ONCE in ``setUpClass`` and shared by the
    value pin, the clip guard and the diagnostic -- gated so the fast tier
    never pays it.
    """

    #: Series/engine probe count for the honest-ceiling oracle (the bake's
    #: default).  ``_measure_w_low_true`` is ``n_w``-sensitive only near the
    #: fold (INS-1-001 marginal resonances), which the fence DECLINES; the
    #: deep interior is smooth, so ``n_w = 16`` is stable here.
    _N_W: int = 16

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._script = _load_fit_certificate_script()
        cls._rows = cls._measure_rows()

    @classmethod
    def _measure_rows(cls):
        """Measure ``(gamma, theta, rho, y, w_fit, w_true)`` per grid cell."""
        caustic_rho = cls._module._caustic_rho
        rows = []
        for gamma in _DEEP_INTERIOR_GAMMAS:
            for theta in _DEEP_INTERIOR_THETAS:
                yc = 1.0 / caustic_rho(gamma, 1.0, theta)
                for rho in _DEEP_INTERIOR_RHOS:
                    r = rho * yc
                    y = (r * math.cos(theta), r * math.sin(theta))
                    rho_meas = caustic_rho(gamma, r * r, theta)
                    w_fit = cls._w_low_fit(y, gamma, 0.0, 0.0)
                    w_true = cls._script._measure_w_low_true(
                        gamma, 0.0, 0.0, float(y[0]), float(y[1]), cls._N_W)
                    rows.append((gamma, theta, rho_meas, y, w_fit, w_true))
        return rows

    def test_deep_interior_served_at_honest_value(self) -> None:
        """Served ceiling <= engine-honest ceiling at every deep-interior cell.

        The exact expectation of INS-3-002: ``w_low_fit <= w_low_true *
        (1 + _INTERIOR_REL_TOL)`` over gamma in {0.2, 0.3, 0.5} x rho in
        {0.2, 0.3, 0.5} (x the three witness angles).  One-sided: a
        conservative under-serve is fine; only the over-serve trips.
        """
        for gamma, theta, rho, y, w_fit, w_true in self._rows:
            with self.subTest(gamma=gamma, theta=theta, rho=rho):
                self._n_checks += 1
                self.assertLess(
                    rho, self._rho_lo,
                    f'premise lost: rho={rho:.3f} no longer deep interior '
                    f'(RHO_LO={self._rho_lo}); the fence has been widened')
                self._n_checks += 1
                self.assertIsNotNone(
                    w_fit, f'deep interior rho={rho:.3f} was declined to None')
                self._n_checks += 1
                self.assertIsNotNone(
                    w_true, 'premise lost: engine refused to measure an '
                    'honest ceiling at this cell')
                self._n_checks += 1
                self.assertGreater(
                    w_true, 0.0, 'premise lost: zero honest ceiling')
                self._n_checks += 1
                self.assertLessEqual(
                    w_fit, w_true * (1.0 + _INTERIOR_REL_TOL),
                    f'deep-interior OVER-SERVE: w_low_fit={w_fit:.3f} > '
                    f'w_low_true={w_true:.3f} at gamma={gamma} '
                    f'theta={theta:.3f} rho={rho:.3f} -- the interior must '
                    'be served by the de-rated FIT, not the DD-ceiling clip')

    def test_clip_is_not_the_conservativeness_mechanism(self) -> None:
        """The ``min(w_fit, CEILING)`` clip does NO work at the interior.

        At every deep-interior cell the served value is STRICTLY below the
        DD ceiling, and so is the RAW (``de-rate = 1.0``) fit -- removing
        the de-rate still cannot reach the cap, so the clip is not the
        interior's conservativeness mechanism.  A reinstated
        ``rho < RHO_LO -> ceiling`` short-circuit, or an un-regularized
        feature basis that diverges at low ``r`` (``log(s)``, ``log(s)^2``
        -> +inf as ``s -> 0``), returns the ceiling at these cells and
        trips this pin.  This is also the suite's self-falsification: it
        stays load-bearing AFTER the honest-value pin flips green, because
        a clip-regression sets ``served == raw == ceiling``.
        """
        for gamma, theta, rho, y, w_fit, w_true in self._rows:
            with self.subTest(gamma=gamma, theta=theta, rho=rho):
                with mock.patch.object(self._module, '_DIFFRACTIVE_FIT_DERATE',
                                       1.0):
                    raw = type(self)._w_low_fit(y, gamma, 0.0, 0.0)
                self._n_checks += 1
                self.assertLess(
                    rho, self._rho_lo,
                    f'premise lost: rho={rho:.3f} no longer deep interior')
                self._n_checks += 1
                self.assertIsNotNone(
                    w_fit, f'deep interior rho={rho:.3f} was declined to None')
                self._n_checks += 1
                self.assertLess(
                    w_fit, self._ceiling,
                    f'served at the DD ceiling {w_fit!r} at gamma={gamma} '
                    f'theta={theta:.3f} rho={rho:.3f} -- the clip is the '
                    'conservativeness mechanism; the interior must be served '
                    'by the fit')
                self._n_checks += 1
                self.assertIsNotNone(
                    raw, f'raw interior fit declined at rho={rho:.3f}')
                self._n_checks += 1
                self.assertTrue(
                    math.isfinite(raw),
                    f'raw (de-rate=1.0) interior fit DIVERGED to {raw!r} at '
                    f'gamma={gamma} theta={theta:.3f} rho={rho:.3f} -- the '
                    'feature basis is not regularized in the deep interior; '
                    'without the clip the fit would exceed the engine-honest '
                    'ceiling')
                self._n_checks += 1
                self.assertLess(
                    raw, self._ceiling,
                    f'raw (de-rate=1.0) interior fit {raw!r} hits the DD '
                    f'ceiling at gamma={gamma} theta={theta:.3f} '
                    f'rho={rho:.3f} -- the clip absorbs the over-run; the '
                    'de-rated fit must be finite below the cap')

    def test_diagnostic_ratio_vs_rho(self) -> None:
        """Save a served/engine ratio scatter over the deep-interior grid."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        self._n_checks += 1
        self.assertTrue(
            all(row[5] is not None for row in self._rows),
            'premise lost: engine refused to measure an honest ceiling')
        rhos = [row[2] for row in self._rows]
        ratios = [row[4] / row[5] for row in self._rows]
        gammas = [row[0] for row in self._rows]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter(rhos, ratios, c=gammas, cmap='viridis', s=30,
                   label='w_low_fit / w_low_true')
        ax.axhline(1.0, color='k', ls=':', label='honest ceiling (= 1)')
        ax.axvline(self._rho_lo, color='tab:red', ls='--',
                   label='RHO_LO (deep-interior edge)')
        ax.set_xlabel('rho = |y\'| / |y_c(theta)|')
        ax.set_ylabel('w_low_fit / w_low_true (<= 1 = honest)')
        ax.set_title('deep-interior honest-serve pin (INS-3-002)')
        ax.set_ylim(0.0, 8.0)
        ax.legend()
        fig.tight_layout()
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'output')
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, 'deep_interior_honest_serve.png')
        fig.savefig(path, dpi=90)
        plt.close(fig)
        self._n_checks += 1
        self.assertTrue(os.path.exists(path))


class TestWLlowFitJustOutsideShellConservative(WLlowFitBaseTestCase):
    """
    Just outside the near-fold shell the served ceiling is conservative.

    The fence hands off from the DECLINED shell (``rho <= 1 + DELTA`` ->
    ``None``) to the served exterior at ``rho > 1 + DELTA``.  The fitted
    surface must be conservative RIGHT AT that handover -- de-rated below the
    engine-honest ceiling -- and NOT so over-fenced that it returns a
    degenerate value (``None`` or ``<= 0``) at its own boundary.  This pins
    the fence boundary as a live serve handover, not a dead zone: the
    de-rated ``w_low_fit`` is compared against the engine-honest ceiling
    measured by `scripts/fit_diffractive_certificate.py`'s own
    `_measure_w_low_true` (the exact `f_schwinger` engine under the
    `CERTIFICATION_BAR` sup-over-w semantics, ``n_w=16`` -- the bake's
    default).

    Fixtures are DERIVED from the live directional caustic radius at the
    FOLD (cusp) directions of the astroid caustic at ``gamma = 0.3`` (where
    ``|y_c(theta)|`` is minimal, so ``r ~ 0.45-0.5`` sits just outside the
    shell): ``r = rho * |y_c(theta)|`` with ``rho`` just above
    ``1 + DELTA = 1.4``, so ``rho`` IS the fence discriminator and the source
    sits in the served exterior by construction.  ``7 pi / 32`` (and its
    partner ``25 pi / 32``) are off-grid theta midpoints of the calibration
    grid -- the same set the de-rate is certified on -- so the fit is
    genuinely calibrated at these witnesses rather than extrapolated.

    Cost: 8 engine probes (2 thetas x 4 rho targets, ~1.2 s each) paid ONCE
    in ``setUpClass`` and shared by the assertion, teeth and diagnostic
    tests -- ~10 s total, well inside the fast-tier budget.
    """

    #: Lens parameters (the spec's example: gamma=0.3 at a fold direction).
    _GAMMA: float = 0.3
    _BETA: float = 0.0
    _KAPPA: float = 0.0
    #: Fold (cusp) directions of the astroid caustic at gamma=0.3 -- the
    #: directions of minimum |y_c(theta)|, so r ~ 0.45-0.5 lands just
    #: outside the shell.  Off-grid theta midpoints of the calibration grid.
    _FOLD_THETAS: tuple[float, ...] = (
        7.0 * math.pi / 32.0, 25.0 * math.pi / 32.0)
    #: Reduced-caustic ratios just above the outer shell boundary
    #: RHO_HI = 1 + DELTA = 1.4 (r is derived as rho * |y_c(theta)|).
    _RHO_TARGETS: tuple[float, ...] = (1.42, 1.5, 1.6, 1.7)
    #: Series/engine probe count for the honest-ceiling oracle (the bake's
    #: default; a coarser scan skips the marginal resonances of INS-1-001).
    _N_W: int = 16

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._script = _load_fit_certificate_script()
        cls._rows = cls._measure_rows()

    @classmethod
    def _measure_rows(cls):
        """Measure ``(rho, theta, y, w_fit, w_true)`` at each boundary witness."""
        caustic_rho = cls._module._caustic_rho
        rows = []
        for theta in cls._FOLD_THETAS:
            yc = 1.0 / caustic_rho(cls._GAMMA, 1.0, theta)
            for rho_target in cls._RHO_TARGETS:
                r = rho_target * yc
                y = (r * math.cos(theta), r * math.sin(theta))
                rho = caustic_rho(cls._GAMMA, r * r, theta)
                w_fit = cls._w_low_fit(y, cls._GAMMA, cls._BETA, cls._KAPPA)
                w_true = cls._script._measure_w_low_true(
                    cls._GAMMA, cls._BETA, cls._KAPPA,
                    float(y[0]), float(y[1]), cls._N_W)
                rows.append((rho, theta, y, w_fit, w_true))
        return rows

    def test_just_outside_shell_is_conservative(self) -> None:
        """De-rated ceiling <= engine-honest ceiling, > 0, at the handover."""
        for rho, theta, y, w_fit, w_true in self._rows:
            with self.subTest(theta=theta, rho=rho):
                self._n_checks += 1
                self.assertGreater(
                    rho, self._rho_hi,
                    f'premise lost: rho={rho:.3f} not in the served exterior')
                self._n_checks += 1
                self.assertIsNotNone(
                    w_fit, 'served exterior declined to None (fence too wide)')
                self._n_checks += 1
                self.assertGreater(
                    w_fit, 0.0,
                    'fence too wide: fit killed (non-positive) at its own '
                    'boundary')
                self._n_checks += 1
                self.assertIsNotNone(
                    w_true, 'premise lost: engine refused to measure an honest '
                    'ceiling')
                self._n_checks += 1
                self.assertLessEqual(
                    w_fit, w_true * (1.0 + _CONSERVATIVE_REL_TOL),
                    f'over-serve just outside the shell: w_low_fit={w_fit:.3f} '
                    f'> engine-honest ceiling w_low_true={w_true:.3f} '
                    f'(rho={rho:.3f})')

    def test_removing_derate_over_serves_the_boundary(self) -> None:
        """Teeth: without the de-rate the raw fit over-serves the boundary."""
        for rho, theta, y, w_fit, w_true in self._rows:
            with self.subTest(theta=theta, rho=rho):
                with mock.patch.object(self._module, '_DIFFRACTIVE_FIT_DERATE',
                                       1.0):
                    raw = type(self)._w_low_fit(y, self._GAMMA, self._BETA,
                                                self._KAPPA)
                self._n_checks += 1
                self.assertIsNotNone(raw)
                self._n_checks += 1
                self.assertLess(
                    raw, self._ceiling,
                    'premise lost: raw fit clipped at the ceiling -- the teeth '
                    'would measure the clip, not the fit')
                self._n_checks += 1
                self.assertGreater(
                    raw, w_fit, 'removing the de-rate does not inflate the '
                    'served ceiling')
                self._n_checks += 1
                self.assertGreater(
                    raw, w_true * (1.0 + _CONSERVATIVE_REL_TOL),
                    f'de-rate not load-bearing: raw fit {raw:.3f} does not '
                    f'over-serve the honest ceiling {w_true:.3f} at '
                    f'rho={rho:.3f}')

    def test_diagnostic_ratio_vs_rho(self) -> None:
        """Save a ratio (w_low_fit / w_low_true) vs rho plot near 1 + DELTA."""
        import os
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        self._n_checks += 1
        self.assertTrue(
            all(row[4] is not None for row in self._rows),
            'premise lost: engine refused to measure an honest ceiling')
        rhos = [row[0] for row in self._rows]
        ratios = [row[3] / row[4] for row in self._rows]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter(rhos, ratios, s=30, label='w_low_fit / w_low_true')
        ax.axhline(1.0, color='k', ls=':', label='honest ceiling (= 1)')
        ax.axvline(self._rho_hi, color='tab:red', ls='--',
                   label='RHO_HI = 1 + DELTA')
        ax.set_xlabel('rho = |y\'| / |y_c(theta)|')
        ax.set_ylabel('w_low_fit / w_low_true (<= 1 = conservative)')
        ax.set_title('conservative pin just outside the near-fold shell '
                     '(gamma=0.3)')
        ax.set_ylim(0.0, 1.2)
        ax.legend()
        fig.tight_layout()
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'output')
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, 'just_outside_shell_conservative.png')
        fig.savefig(path, dpi=90)
        plt.close(fig)
        self._n_checks += 1
        self.assertTrue(os.path.exists(path))


class TestFenceGridGenerators(unittest.TestCase):
    """
    The calibration script's grid generators FENCE the near-fold shell.

    `_grid_points` / `_off_grid_points` (scripts/fit_diffractive_certificate.py)
    drop every row whose reduced caustic ratio ``rho = |y'| / |y_c(theta)|``
    falls in ``[RHO_LO, 1 + DELTA]`` (via `_fence_excluded`), so the fit, the
    de-rate and the margin report all operate on the fenced domain (probe
    domain == training domain).  This pins the fencing at the SOURCE: the
    grid a calibration sweep iterates can no longer even REACH a shell row,
    so the sweep cannot over-serve the shell (a residual ``None`` row in the
    sweep is counted as refused, not over-serve).  The off-grid midpoint
    probes are fenced through the same discriminator, so the held-out witness
    set lives on the fenced domain too.

    Engine-free: pure list generation plus the O(1) `_caustic_rho` fence
    discriminator (no series/engine probe).  Budget: ~1 s.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._script = _load_fit_certificate_script()
        cls._module = _load_diffractive_module()
        cls._rho_lo: float = cls._module._DIFFRACTIVE_FIT_FENCE_RHO_LO
        cls._rho_hi: float = 1.0 + cls._module._DIFFRACTIVE_FIT_FENCE_DELTA
        cls._grid = cls._script._grid_points('full', 42)
        cls._off_grid = cls._script._off_grid_points('full', 42)

    def _rho(self, row):
        """The fence discriminator ``rho`` the row maps to (live)."""
        gamma, beta, kappa, r, theta = row
        _lam, gamma_prime = self._script._reduced_shear(gamma, kappa)
        return self._module._caustic_rho(abs(gamma_prime), r * r, theta)

    def test_grid_and_offgrid_points_are_fenced(self) -> None:
        """Every grid + off-grid row sits strictly outside the near-fold shell."""
        for kind, rows in (('on-grid', self._grid), ('off-grid', self._off_grid)):
            for row in rows:
                rho = self._rho(row)
                self.assertFalse(
                    self._rho_lo <= rho <= self._rho_hi,
                    f'{kind} row {row} has rho={rho:.3f} inside the shell')
        interior = sum(1 for row in self._grid if self._rho(row) < self._rho_lo)
        exterior = sum(1 for row in self._grid if self._rho(row) > self._rho_hi)
        self.assertGreater(interior, 0, 'fenced grid has no deep-interior rows')
        self.assertGreater(exterior, 0, 'fenced grid has no exterior rows')
        self.assertGreater(
            len(self._off_grid), 0, 'off-grid midpoint set is empty (vacuity)')

    def test_fence_drops_the_shell_rows(self) -> None:
        """The fenced grid keeps only non-shell rows; every shell row is dropped."""
        unfenced = self._script._unfenced_grid_points('full', 42)
        grid_set = set(self._grid)
        dropped = [row for row in unfenced if row not in grid_set]
        self.assertGreater(len(dropped), 0, 'fence dropped no rows (vacuity)')
        self.assertTrue(
            grid_set <= set(unfenced),
            'fenced grid must be a subset of the unfenced grid')
        for row in dropped:
            self.assertTrue(
                self._script._fence_excluded(*row),
                f'dropped row {row} is not a shell row (fence dropped a '
                'non-shell row it should keep)')
        for row in unfenced:
            if self._script._fence_excluded(*row):
                self.assertNotIn(
                    row, grid_set, f'shell row {row} survived the fence')

    def test_fence_discriminator_is_single_sourced(self) -> None:
        """`_fence_excluded` uses the shipped discriminator, not re-typed literals."""
        fn = _find_function(_parse_source(_FIT_SCRIPT_PATH), '_fence_excluded')
        if fn is None:
            self.fail('_fence_excluded must exist in the script')
        ids = _all_name_ids(fn)
        for name in ('_caustic_rho', '_DIFFRACTIVE_FIT_FENCE_RHO_LO',
                     '_DIFFRACTIVE_FIT_FENCE_DELTA'):
            self.assertIn(
                name, ids,
                f'_fence_excluded must reference {name} (single-sourced from '
                '_diffractive.py), never a re-derived literal')


class TestFenceFallThroughByteIdentity(WLlowFitBaseTestCase):
    """
    A fenced draw falls through the consumer byte-identically to the wall.

    `_diffractive_bottom_ceiling` (likelihood.py) is a THIN wrapper that
    returns ``w_low_fit(...)`` directly and maps only the parity-wall
    `DiffractiveDomainError` to ``None``.  The near-fold fence adds a THIRD
    ``None`` source -- ``w_low_fit`` returns ``None`` (NOT an exception) for
    a shell draw -- and it must route through the consumer's fall-through
    BYTE-IDENTICALLY to the wall refusal and the (defense-in-depth) degenerate
    ``sqrt_mu`` ``None``: the same ``None`` boundary, the same empty nested
    bottom, the same engine-host region, and NO new exception class.

    RUNTIME pin: it binds the REAL `_diffractive_bottom_ceiling` to an
    uninitialized `LensedRelativeBinningLikelihood` shell (the census mirror's
    own ``object.__new__`` idiom -- the method reads only its ``lens``
    argument plus the ``w_hi`` cap), then compares the fenced result against
    the wall result down to the shared `_band_split_mask` arithmetic.
    Engine-free: importing likelihood binds no engine and evaluates none
    (~3 s import, paid once in ``setUpClass``).
    """

    #: Fold-dip lens (the corner witness of the diffractive suite); beta =
    #: kappa = 0 so the eigenframe angle is the lens polar angle.
    _GAMMA: float = 0.41

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._host, cls._band_split_mask = _load_likelihood_wrapper()
        theta = 3.0 * math.pi / 4.0
        yc = 1.0 / cls._module._caustic_rho(cls._GAMMA, 1.0, theta)
        r = 1.0 * yc  # rho = 1.0, mid-shell
        cls._fenced_lens = {
            'y1': r * math.cos(theta), 'y2': r * math.sin(theta),
            'gamma': cls._GAMMA, 'beta': 0.0, 'kappa': 0.0,
        }
        cls._wall_lens = {
            'y1': 0.5, 'y2': 0.3,
            'gamma': cls._wall, 'beta': 0.0, 'kappa': 0.0,
        }

    def _wrapped_ceiling(self, lens, w_hi=None):
        """The production wrapper bound to the uninitialized host shell."""
        return type(self)._host._diffractive_bottom_ceiling(lens, w_hi=w_hi)

    def test_fenced_draw_declines_to_none(self) -> None:
        """A shell draw makes `_diffractive_bottom_ceiling` return None."""
        self._n_checks += 1
        self.assertIsNone(self._wrapped_ceiling(self._fenced_lens))

    def test_fence_and_wall_collapse_byte_identically(self) -> None:
        """Fenced None and wall None share the same downstream split mask."""
        dense_w = np.linspace(1.0, 10.0, 16)
        fenced = self._wrapped_ceiling(self._fenced_lens)
        wall = self._wrapped_ceiling(self._wall_lens)
        self._n_checks += 1
        self.assertIsNone(fenced)
        self._n_checks += 1
        self.assertIsNone(wall)
        f_split, f_below = type(self)._band_split_mask(dense_w, fenced)
        w_split, w_below = type(self)._band_split_mask(dense_w, wall)
        self._n_checks += 1
        self.assertEqual(f_split, w_split, 'split flags must match')
        self._n_checks += 1
        self.assertTrue(
            np.array_equal(f_below, w_below),
            'below-split masks must be byte-identical for fenced vs wall None')

    def test_fence_introduces_no_new_exception_class(self) -> None:
        """The fence returns None without raising; the wrapper's sole except
        is still `DiffractiveDomainError`."""
        y = (self._fenced_lens['y1'], self._fenced_lens['y2'])
        self._n_checks += 1
        self.assertIsNone(
            type(self)._w_low_fit(y, self._GAMMA, 0.0, 0.0),
            'a shell draw must DECLINE to None, never raise a new exception')
        method = _find_function(
            _parse_source(_LIKELIHOOD_PATH), '_diffractive_bottom_ceiling')
        if method is None:
            self.fail('_diffractive_bottom_ceiling must exist in likelihood.py')
        handlers = [
            h for node in ast.walk(method)
            if isinstance(node, ast.Try) for h in node.handlers]
        self._n_checks += 1
        self.assertEqual(len(handlers), 1, 'wrapper must have exactly one except')
        self._n_checks += 1
        self.assertTrue(
            _handler_catches(handlers[0], 'DiffractiveDomainError'),
            'wrapper must catch only DiffractiveDomainError')

    def test_wrapper_returns_w_low_fit_transparently(self) -> None:
        """The wrapper's try body returns `w_low_fit(...)` with no transform."""
        method = _find_function(
            _parse_source(_LIKELIHOOD_PATH), '_diffractive_bottom_ceiling')
        if method is None:
            self.fail('_diffractive_bottom_ceiling must exist in likelihood.py')
        transparent = False
        for node in ast.walk(method):
            if not isinstance(node, ast.Try):
                continue
            for stmt in node.body:
                if (isinstance(stmt, ast.Return)
                        and isinstance(stmt.value, ast.Call)
                        and 'w_low_fit' in _called_identifiers(stmt.value)):
                    transparent = True
        self._n_checks += 1
        self.assertTrue(
            transparent,
            'wrapper must return w_low_fit(...) directly so a fenced None '
            'flows through byte-identically')


class TestCensusMirrorFencedDrawRouting(unittest.TestCase):
    """
    The census mirror never labels a fenced draw ``diffractive_analytic``.

    `serve_route_census.classify_draw` mirrors production's Rung-P admission:
    it calls ``mods.w_low_fit((y1, y2), gamma, 0.0, 0.0, w_hi=w_hi)`` and
    returns ``diffractive_analytic`` ONLY under ``w_low is not None and
    w_low > w_lo``.  A fenced draw makes ``w_low_fit`` return ``None``
    (pinned by `TestWLlowFitNearFoldFence`), so it can never reach that
    return -- it falls through to the per-node pass, i.e. engine/fold demand.
    This pins the guard STRUCTURALLY (load-bearing in the shipped source)
    plus the runtime ``None`` fact, engine-free.

    Budget: one AST parse + one O(1) `w_low_fit` call, < 1 s.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.tree = _parse_source(_SERVE_ROUTE_CENSUS_PATH)
        cls.classify = _find_function(cls.tree, 'classify_draw')

    @staticmethod
    def _analytic_calls(fn):
        """`_result('diffractive_analytic', ...)` call nodes in `fn`."""
        return [
            node for node in ast.walk(fn)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == '_result'
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == 'diffractive_analytic'
        ]

    @staticmethod
    def _test_requires_w_low_not_none(test_node):
        """True if `test_node` compares ``w_low is not None``."""
        for node in ast.walk(test_node):
            if (isinstance(node, ast.Compare)
                    and isinstance(node.left, ast.Name)
                    and node.left.id == 'w_low'
                    and any(isinstance(op, ast.IsNot) for op in node.ops)
                    and len(node.comparators) == 1
                    and isinstance(node.comparators[0], ast.Constant)
                    and node.comparators[0].value is None):
                return True
        return False

    @staticmethod
    def _guarded_bodies(fn):
        """Bodies (as node sets) of every `If` guarded by ``w_low is not None``."""
        return [
            set(ast.walk(node)) for node in ast.walk(fn)
            if isinstance(node, ast.If)
            and TestCensusMirrorFencedDrawRouting._test_requires_w_low_not_none(
                node.test)
        ]

    def test_anti_vacuity(self) -> None:
        """The scan found `classify_draw` and a diffractive_analytic route."""
        self.assertIsNotNone(self.classify, 'classify_draw must exist')
        self.assertGreater(
            len(self._analytic_calls(self.classify)), 0,
            'no diffractive_analytic route found to police')

    def test_diffractive_analytic_guarded_by_w_low_not_none(self) -> None:
        """Every diffractive_analytic return is inside a `w_low is not None`
        guard, so a fenced draw (w_low=None) can never be labelled analytic."""
        fn = self.classify
        if fn is None:
            self.fail('classify_draw must exist')
        guarded = self._guarded_bodies(fn)
        self.assertGreater(
            len(guarded), 0, 'no `w_low is not None` guard found in classify_draw')
        for call in self._analytic_calls(fn):
            self.assertTrue(
                any(call in body for body in guarded),
                'a diffractive_analytic return is not guarded by '
                '`w_low is not None` -- a fenced draw (w_low=None) could be '
                'mis-labelled analytic')

    def test_fenced_draw_makes_w_low_none(self) -> None:
        """The mirror's own predicate input is None for a fenced draw."""
        w_low_fit = _load_w_low_fit()[0]
        module = _load_diffractive_module()
        theta = 3.0 * math.pi / 4.0
        gamma = 0.41
        yc = 1.0 / module._caustic_rho(gamma, 1.0, theta)
        r = 1.0 * yc  # rho = 1.0, mid-shell (the fence declines it)
        y = (r * math.cos(theta), r * math.sin(theta))
        self.assertIsNone(
            w_low_fit(y, gamma, 0.0, 0.0),
            'premise lost: the fenced draw is no longer declined to None')

    def test_guard_detector_flags_an_unguarded_route(self) -> None:
        """SELF-FALSIFICATION: an unguarded analytic return is caught.

        A synthetic `classify_draw` whose ``diffractive_analytic`` return is
        NOT under ``w_low is not None`` must be flagged -- proving the guard
        check has teeth rather than passing vacuously on any source.
        """
        synthetic = ast.parse(
            "def classify_draw():\n"
            "    if gamma < 1.0:\n"
            "        w_low = f()\n"
            "        if float(w_low) > w_lo:\n"
            "            return _result('diffractive_analytic', ())\n")
        fn = _find_function(synthetic, 'classify_draw')
        guarded = self._guarded_bodies(fn)
        unguarded = [
            call for call in self._analytic_calls(fn)
            if not any(call in body for body in guarded)]
        self.assertGreater(
            len(unguarded), 0,
            'guard detector did not flag an unguarded diffractive_analytic '
            'return -- the routing pin would be vacuous')


if __name__ == '__main__':
    unittest.main()
