"""Tests for the plan-mode pipeline-graph pre-read injection (item 1.7).

Exercises `_pre_read_pipeline_graph`'s matching / shared-producer expansion /
formatting / ~4 KB cap against a stub PipelineGraph, so the ported logic is
verified without coupling to the live DATA_CONTRACTS.yaml.

Run: conda run -n cogwheel_310 python -m unittest \
    discover -s .claude/sdk/tests -p 'test_*.py'
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sdk.orchestrator import BuildOrchestrator


class _StubPG:
    """Mimics B's scripts.pipeline_graph.PipelineGraph surface used by the
    pre-read: `.artifacts` (dict), `.trace(name)`, `.consumers_of(name)`."""

    def __init__(self, artifacts, consumers=None):
        self._artifacts = artifacts
        self._consumers = consumers or {}

    @property
    def artifacts(self):
        return self._artifacts

    def trace(self, art):
        return self._artifacts.get(art)

    def consumers_of(self, art):
        return self._consumers.get(art, [])


def _orch(task):
    o = BuildOrchestrator.__new__(BuildOrchestrator)
    o._log = lambda *a, **k: None
    o.task = task
    o._task_files_text = ""
    return o


ARTIFACTS = {
    "posterior_samples": {
        "format": "feather",
        "fields": ["m1", "m2", "chieff"],
        "producer": {"module": "cogwheel.sampling", "function": "run"},
        "consumers": [{"module": "cogwheel.postprocessing", "function": "load"}],
    },
    "evidence": {
        "format": "json",
        "fields": ["ln_z"],
        # SAME producer module as posterior_samples → shared-producer sibling.
        "producer": {"module": "cogwheel.sampling", "function": "run"},
        "consumers": [],
    },
    "strain_data": {
        "format": "hdf5",
        "fields": ["asd"],
        "producer": {"module": "cogwheel.data", "function": "make"},
        "consumers": [{"module": "cogwheel.likelihood", "function": "init"}],
    },
}
CONSUMERS = {
    "posterior_samples": [
        {"module": "cogwheel.postprocessing", "function": "load", "source": "contracts"}],
    "strain_data": [
        {"module": "cogwheel.likelihood", "function": "init", "source": "contracts"}],
}


class PreReadPipelineGraphTest(unittest.TestCase):
    def _run(self, task, pg):
        o = _orch(task)
        o._load_pipeline_graph = lambda: pg
        return o._pre_read_pipeline_graph()

    def test_matches_artifact_named_in_task(self):
        out = self._run("Fix the posterior_samples writer",
                        _StubPG(ARTIFACTS, CONSUMERS))
        self.assertIn("# Pipeline graph", out)
        self.assertIn("## posterior_samples", out)
        self.assertIn("cogwheel.postprocessing::load", out)
        self.assertIn("Fields: m1, m2, chieff", out)
        self.assertNotIn("## strain_data", out)

    def test_matches_on_producer_module_path(self):
        out = self._run("touching cogwheel/data.py for the ASD",
                        _StubPG(ARTIFACTS, CONSUMERS))
        self.assertIn("## strain_data", out)

    def test_shared_producer_expansion_pulls_sibling(self):
        # Naming posterior_samples must pull in evidence (same producer module),
        # even though 'evidence' is nowhere in the task text.
        out = self._run("regenerate posterior_samples",
                        _StubPG(ARTIFACTS, CONSUMERS))
        self.assertIn("## posterior_samples", out)
        self.assertIn("## evidence", out)

    def test_no_consumers_renders_placeholder(self):
        out = self._run("regenerate posterior_samples",
                        _StubPG(ARTIFACTS, CONSUMERS))
        self.assertIn("Consumers: (none registered)", out)  # from 'evidence'

    def test_producer_module_stored_as_py_path(self):
        # B's real DATA_CONTRACTS stores producer.module as a file path
        # (e.g. "cogwheel/sampling.py"), not a dotted module.
        arts = {
            "samples": {
                "format": "feather", "fields": ["m1"],
                "producer": {"module": "cogwheel/sampling.py", "function": "run"},
                "consumers": [],
            },
        }
        self.assertIn("## samples",
                      self._run("edit cogwheel/sampling.py", _StubPG(arts)))
        self.assertIn("## samples",
                      self._run("rework the sampling module", _StubPG(arts)))
        # The bare ".py" extension must NOT become a spurious match token.
        self.assertEqual("", self._run("fix a typo in setup.py", _StubPG(arts)))

    def test_no_match_returns_empty(self):
        out = self._run("update the README wording",
                        _StubPG(ARTIFACTS, CONSUMERS))
        self.assertEqual(out, "")

    def test_missing_graph_returns_empty(self):
        o = _orch("posterior_samples")
        o._load_pipeline_graph = lambda: None
        self.assertEqual(o._pre_read_pipeline_graph(), "")

    def test_output_capped_at_4kb(self):
        big = {
            f"artifact_{i}": {
                "format": "f", "fields": ["x"],
                "producer": {"module": f"mod{i}", "function": "g"},
                "consumers": [],
            }
            for i in range(500)
        }
        task = " ".join(big.keys())
        out = self._run(task, _StubPG(big))
        self.assertLessEqual(len(out), 4096 + 60)
        self.assertIn("truncated", out)


if __name__ == "__main__":
    unittest.main()
