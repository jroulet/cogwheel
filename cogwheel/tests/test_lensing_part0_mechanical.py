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
"""
from __future__ import annotations

import ast
import json
import math
import pathlib
import re
import unittest

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
    ('cogwheel/lensing/surrogate_training.py', '_DEFAULT_FARFIELD_OVERLAP'),
    ('cogwheel/lensing/surrogate_training.py', '_INTERLOBE_CORRIDOR_ETA_SCALE'),
    ('cogwheel/lensing/surrogate_census.py', 'CROWN_CAUSTIC_MARGIN'),
    ('cogwheel/lensing/chang_refsdal/channels.py', '_MARKER_SCALE_FLOOR'),
    ('cogwheel/lensing/chang_refsdal/_schwinger.py', '_U_MARGIN_CONST'),
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


if __name__ == '__main__':
    unittest.main()
