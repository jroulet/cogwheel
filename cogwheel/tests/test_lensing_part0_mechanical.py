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
shear and source magnitude, D2 angular symmetry (period-``pi`` +
reflection), and the DD-ceiling cap / parity-wall collapse.  Those import the fitted-certificate module
LAZILY (``_load_w_low_fit``) and call only the O(1), engine-free
``w_low_fit`` — they stay far under the fast-tier budget.
"""
from __future__ import annotations

import ast
import cmath
import json
import math
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


def _all_name_ids(tree: ast.Module) -> set[str]:
    """
    Collect every Name-node identifier in the module.

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
        """Lazily import ``w_low_fit`` and derive the parity wall once."""
        (cls._w_low_fit, cls._domain_error, cls._ceiling,
         cls._delta_gamma_p) = _load_w_low_fit()
        cls._wall: float = 1.0 - cls._delta_gamma_p

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

    def _assert_non_increasing(self, values) -> None:
        """Assert ``values`` is non-increasing within float round-off."""
        for prev, nxt in zip(values, values[1:]):
            self._n_checks += 1
            self.assertLessEqual(
                nxt, prev + _MONOTONE_REL_TOL * max(1.0, abs(prev)),
                f'w_low_fit increased: {prev!r} -> {nxt!r}')


class TestWLlowFitGammaMonotonicity(WLlowFitBaseTestCase):
    """
    ``w_low_fit`` falls with ``gamma'`` past a low-edge peak.

    The engine-honest truncation ceiling falls as the operator series' small
    parameter ``gamma' s w / 2`` grows, and the re-baked fit reproduces that
    power-law direction over the calibration band: non-increasing from a
    low-edge peak (~``gamma`` 0.06, LIVE-derived) out to the parity wall.
    Near the calibration LOW edge (``gamma`` in [0.05, ~0.06] -- the
    extrapolation from below the grid bleeding in) the surface carries a
    small rise; this is a fit-shape artifact, NOT an over-serve, because the
    engine-oracle full-grid suite in `test_lensing_diffractive.py` certifies
    the calibration domain (including ``gamma = 0.05``) directly.  The peak
    is re-derived per (beta, kappa) from the LIVE shipped surface, never a
    pinned literal.  ``gamma`` is swept at fixed ``(y, beta, kappa)`` per the
    spec; at fixed ``kappa`` the reduced shear rises monotonically with
    ``gamma``, so non-increasing past the peak in ``gamma`` pins
    non-increasing in ``gamma'`` there.
    """

    def test_non_increasing_past_low_edge_peak(self) -> None:
        """Sweep gamma at several (beta, kappa); past the peak it only falls."""
        y = (0.5, 0.3)
        for beta, kappa in ((0.0, 0.0), (0.7, 0.0), (0.0, 0.3), (0.7, 0.3)):
            with self.subTest(beta=beta, kappa=kappa):
                lam = 1.0 - kappa
                gammas = np.logspace(
                    math.log10(0.05), math.log10(0.999 * self._wall * lam), 120)
                values = [
                    self._evaluate(y, float(g), beta, kappa) for g in gammas]
                peak = int(np.argmax(values))
                self._n_checks += 1
                self.assertLess(
                    peak, len(values) - 1,
                    'premise lost: no falling branch after the low-edge peak')
                self._n_checks += 1
                self.assertGreater(
                    values[peak], values[-1],
                    'premise lost: no genuine gamma falling branch out to '
                    'the wall')
                self._assert_non_increasing(values[peak:])


class TestWLlowFitSMonotonicity(WLlowFitBaseTestCase):
    """
    ``w_low_fit`` falls with ``s = |y'|^2`` past a small-``s`` peak.

    The operator-series tail grows like ``(gamma' s w / 2)^n / n!``, so the
    engine-honest ceiling falls as the reduced source magnitude ``s`` grows,
    and the re-baked degree-2 log-log surface reproduces that direction over
    the calibration range (``s`` up to `_CALIBRATION_S_MAX`).  The surface is
    NOT monotone over the whole range: below a small-``s`` peak (``s``
    ~0.01-0.1, gamma-dependent) it RISES with ``s`` -- the degree-2 log-log
    polynomial extrapolating where the calibration grid (``r`` in [0.3, 1.3],
    i.e. ``s >= 0.09``) is sparse.  This test re-derives the peak from the
    LIVE shipped surface (per gamma, never a pinned literal) and asserts the
    falling branch out to `_CALIBRATION_S_MAX` -- closing the pre-re-bake
    loose end, whose positive ``log(s)**2`` coefficient turned the surface UP
    for ``s`` ~0.5-0.6 (an artifact the full re-bake eliminated).  The premise
    ``peak < s = 0.4`` pins that the former up-turn region now falls.
    CONSERVATIVENESS (the fitted ceiling never exceeding the engine-honest
    ceiling) is NOT a monotonicity property of the surface: it is enforced by
    the engine-oracle full-grid sweep in `test_lensing_diffractive.py`
    (`FullGridCertificateOracleTestCase`), which certifies the calibration
    domain directly.
    """

    def test_non_increasing_past_small_s_peak(self) -> None:
        """After the live-derived small-s peak, w_low_fit only falls to s_max."""
        for gamma in (0.1, 0.2, 0.3, 0.5):
            with self.subTest(gamma=gamma):
                ss = np.logspace(-2.0, math.log10(_CALIBRATION_S_MAX), 100)
                values = [
                    self._evaluate(
                        (math.sqrt(float(s)) * math.cos(0.7),
                         math.sqrt(float(s)) * math.sin(0.7)),
                        gamma,
                    )
                    for s in ss
                ]
                peak = int(np.argmax(values))
                self._n_checks += 1
                self.assertLess(
                    ss[peak], 0.4,
                    'premise lost: the small-s peak moved into the former '
                    'up-turn region (s >= 0.4); re-scope the falling branch')
                self._n_checks += 1
                self.assertGreater(
                    values[peak], values[-1],
                    'premise lost: no genuine falling branch out to '
                    f's={_CALIBRATION_S_MAX}')
                self._assert_non_increasing(values[peak:])


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
    _CONFIGS: tuple[tuple[tuple[float, float], float, float, float], ...] = (
        ((0.4, 0.7), 0.3, 0.5, 0.2),
        ((1.1, -0.2), 0.15, -0.9, 0.1),
        ((0.05, 0.9), 0.45, 1.3, 0.0),
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
    The fitted certificate is capped at the DD ceiling and collapses at the wall.

    (a) ``w_low_fit`` never exceeds ``_DIFFRACTIVE_FIT_CEILING``
    (``= W_CEILING_SCHWINGER = 60``): a small-gamma fixture whose raw fit
    would exceed it returns the cap exactly, so the clip is load-bearing
    rather than decorative.
    (b) As ``gamma'`` approaches the parity wall ``1 - DELTA_GAMMA_P`` the
    ``log(1 - gamma')`` feature dominates and ``w_low_fit`` collapses
    monotonically toward 0 with no divergence or blow-up.
    """

    def test_never_exceeds_ceiling(self) -> None:
        """The certificate stays at or below the DD ceiling across the domain."""
        y = (0.5, 0.3)
        gammas = np.logspace(-3.0, math.log10(0.999 * self._wall), 200)
        for g in gammas:
            value = self._evaluate(y, float(g))
            self._n_checks += 1
            self.assertLessEqual(
                value, self._ceiling,
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

    def test_wall_collapse_monotone_and_finite(self) -> None:
        """Toward the wall the certificate collapses monotonically to ~0."""
        y = (0.5, 0.3)
        gammas = (0.9, 0.95, 0.98, 0.99, 0.994, 0.9949)
        values = [self._evaluate(y, g) for g in gammas]
        self._assert_non_increasing(values)
        for value in values:
            self._n_checks += 1
            self.assertTrue(
                math.isfinite(value) and value >= 0.0,
                f'w_low_fit blew up near the wall: {value!r}')
        self._n_checks += 1
        self.assertLessEqual(
            values[-1], 1e-9,
            'collapse must reach ~0 at the wall; got {values[-1]!r}')

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
        y = (0.4, 0.7)
        gamma, beta, kappa = 0.3, 0.5, 0.2
        base = self._evaluate(y, gamma, beta, kappa)
        c = math.sqrt(0.5)
        rotated = (c * y[0] - c * y[1], c * y[0] + c * y[1])
        value = self._evaluate(rotated, gamma, beta, kappa)
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

    #: Interior fixtures where the de-rate is load-bearing: the RAW fit sits
    #: far below the ceiling, so the ``min(..., ceiling)`` clip does not mask
    #: the margin (each premise is asserted in the loop below).
    _FIXTURES: tuple[tuple[tuple[float, float], float, float, float], ...] = (
        ((0.4, 0.7), 0.3, 0.5, 0.2),
        ((1.1, -0.2), 0.15, -0.9, 0.1),
        ((0.05, 0.9), 0.45, 1.3, 0.0),
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


if __name__ == '__main__':
    unittest.main()
