#!/usr/bin/env python3
"""Offline tests for the pinned sampler patcher."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1]
    / "patches"
    / "vllm-0261-main-tbfix"
    / "apply-thinking-budget-fix.py"
)
SPEC = importlib.util.spec_from_file_location("tbfix_patcher", MODULE_PATH)
assert SPEC and SPEC.loader
tbfix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tbfix)


class TbfixPatcherTests(unittest.TestCase):
    def test_exact_pinned_block_is_patched_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "sampler.py"
            target.write_text("prefix\n" + tbfix.OLD + "suffix\n")
            old_argv = sys.argv
            try:
                sys.argv = [str(MODULE_PATH), str(target)]
                tbfix.main()
                self.assertIn(tbfix.NEW, target.read_text())
                with self.assertRaisesRegex(SystemExit, "already-patched"):
                    tbfix.main()
            finally:
                sys.argv = old_argv

    def test_unexpected_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "sampler.py"
            target.write_text("unexpected source\n")
            old_argv = sys.argv
            try:
                sys.argv = [str(MODULE_PATH), str(target)]
                with self.assertRaisesRegex(SystemExit, "expected exactly one"):
                    tbfix.main()
            finally:
                sys.argv = old_argv


if __name__ == "__main__":
    unittest.main()
