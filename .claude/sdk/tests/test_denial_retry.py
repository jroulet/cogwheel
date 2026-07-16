"""Regression tests for the bare-denial detector behind the single retry.

The auto-mode classifier fails closed on its own transient errors and headless
agents receive only the bare sentence "The user doesn't want to take this
action right now. STOP ...". The pipeline OWNER confirmed (2026-07-16) that
this exact signature, in this pipeline, is a transient infrastructure artifact,
authorizing a single bounded nudge-retry. These tests pin the detector's
contract: real denials trip it; quotations of the sentence inside file/grep
tool results (it appears in TODO.md and META_PLAN) do NOT.

Run (from the repo root, under the project's interpreter):
    python -m unittest discover -s .claude/sdk/tests -p 'test_*.py'
"""
import os
import sys
import unittest
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sdk.orchestrator import BuildOrchestrator

BARE = ("The user doesn't want to take this action right now. STOP what you "
        "are doing and wait for the user to tell you how to proceed.")


@dataclass
class ToolResultBlock:
    """Duck-typed stand-in; the detector matches on the class NAME."""
    content: object


@dataclass
class TextBlock:
    text: str


@dataclass
class FakeMessage:
    content: list


class BareDenialDetectorTest(unittest.TestCase):

    def test_real_denial_trips(self):
        msg = FakeMessage([ToolResultBlock(BARE)])
        self.assertTrue(BuildOrchestrator._stream_saw_bare_denial(msg))

    def test_denial_in_list_content_trips(self):
        msg = FakeMessage([ToolResultBlock([{"type": "text", "text": BARE}])])
        self.assertTrue(BuildOrchestrator._stream_saw_bare_denial(msg))

    def test_quotation_in_long_file_read_does_not_trip(self):
        quoted = ("# TODO\n" + "x" * 400 + "\nthe denial reads: '" + BARE
                  + "' and is root-caused\n" + "y" * 200)
        msg = FakeMessage([ToolResultBlock(quoted)])
        self.assertFalse(BuildOrchestrator._stream_saw_bare_denial(msg))

    def test_text_blocks_are_ignored(self):
        # An agent TALKING about the denial must not trip the detector.
        msg = FakeMessage([TextBlock(BARE)])
        self.assertFalse(BuildOrchestrator._stream_saw_bare_denial(msg))

    def test_clean_result_does_not_trip(self):
        msg = FakeMessage([ToolResultBlock("stdout: 42\n")])
        self.assertFalse(BuildOrchestrator._stream_saw_bare_denial(msg))

    def test_message_without_content_is_safe(self):
        class Bare:
            pass
        self.assertFalse(BuildOrchestrator._stream_saw_bare_denial(Bare()))


if __name__ == "__main__":
    unittest.main()
