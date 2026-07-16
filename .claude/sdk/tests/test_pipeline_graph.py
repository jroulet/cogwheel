"""Tests for the importable PipelineGraph API in scripts/pipeline_graph.py.

Deterministic — builds a temporary DATA_CONTRACTS.yaml fixture and exercises
the in-process query methods. Run:

    conda run -n cogwheel_310 python -m unittest \
        discover -s .claude/sdk/tests -p 'test_*.py'
"""
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

# Resolve pipeline_graph.py from EITHER layout. Installed (here), this file is
# at <repo>/.claude/sdk/tests/ and the script is at <repo>/scripts/. In the
# teja-force template the same file is at assets/sdk/tests/ and the script is
# still an asset at assets/infrastructure/. Hardcoding only the installed path
# made this module raise FileNotFoundError at IMPORT time in the template tree,
# which aborted discovery for the whole directory — so all 9 tests below never
# ran there, silently. Keep both candidates in sync with the template copy.
_HERE = Path(__file__).resolve()
_CANDIDATES = (
    _HERE.parents[3] / "scripts" / "pipeline_graph.py",            # installed repo
    _HERE.parents[2] / "infrastructure" / "pipeline_graph.py",     # skill template
)
_PG_PATH = next((p for p in _CANDIDATES if p.is_file()), None)
if _PG_PATH is None:  # pragma: no cover - guards a layout change
    raise unittest.SkipTest(
        "pipeline_graph.py not found in any known layout; looked in: "
        + ", ".join(str(p) for p in _CANDIDATES)
    )
_spec = importlib.util.spec_from_file_location("pipeline_graph", _PG_PATH)
pipeline_graph = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pipeline_graph)
PipelineGraph = pipeline_graph.PipelineGraph

FIXTURE = """
schema_version: "1.0.0"
artifacts:
  posterior_samples:
    description: "Posterior samples written by a Sampler run."
    format: "feather"
    fields: [mchirp, q, iota]
    producer:
      module: "cogwheel/sampling.py"
      function: "Sampler.run"
    consumers:
      - module: "cogwheel/postprocessing.py"
        function: "load_samples"
      - module: "cogwheel/gw_plotting.py"
        function: "corner_plot"
"""


class PipelineGraphTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False)
        self.tmp.write(FIXTURE)
        self.tmp.close()
        self.pg = PipelineGraph(contracts_path=self.tmp.name,
                                registry_path="/nonexistent/registry.yaml",
                                graph_path="/nonexistent/CONSUMER_GRAPH.json")

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_artifacts_listed(self):
        self.assertIn("posterior_samples", self.pg.artifacts)

    def test_trace_returns_info(self):
        info = self.pg.trace("posterior_samples")
        self.assertEqual(info["format"], "feather")
        self.assertEqual(info["producer"]["module"], "cogwheel/sampling.py")

    def test_trace_unknown_is_none(self):
        self.assertIsNone(self.pg.trace("nope"))

    def test_consumers_of(self):
        cons = self.pg.consumers_of("posterior_samples")
        mods = {c["module"] for c in cons}
        self.assertEqual(
            mods, {"cogwheel/postprocessing.py", "cogwheel/gw_plotting.py"})

    def test_consumers_of_unknown_empty(self):
        self.assertEqual(self.pg.consumers_of("nope"), [])

    def test_resolve(self):
        r = self.pg.resolve("posterior_samples")
        self.assertEqual(r["producer"]["function"], "Sampler.run")
        self.assertEqual(r["format"], "feather")
        self.assertIsNone(r["disk_path"])  # registry file absent

    def test_inputs_for_consumer(self):
        found = dict(self.pg.inputs_for("cogwheel/postprocessing.py"))
        self.assertIn("posterior_samples", found)

    def test_inputs_for_producer(self):
        roles = self.pg.inputs_for("cogwheel/sampling.py")
        self.assertTrue(any("PRODUCES" in role for _, role in roles))

    def test_missing_contracts_file_is_empty(self):
        pg = PipelineGraph(contracts_path="/nope/DATA_CONTRACTS.yaml")
        self.assertEqual(pg.artifacts, {})
        self.assertEqual(pg.consumers_of("x"), [])


if __name__ == "__main__":
    unittest.main()
