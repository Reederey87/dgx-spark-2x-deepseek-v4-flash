#!/usr/bin/env python3
"""Offline regression tests for the W6 async client."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "qualification" / "async-scheduling" / "w6-async.py"
sys.modules.setdefault("aiohttp", types.ModuleType("aiohttp"))
SPEC = importlib.util.spec_from_file_location("w6_async", MODULE_PATH)
assert SPEC and SPEC.loader
w6 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(w6)


class W6AsyncTests(unittest.TestCase):
    def test_poisson_trace_is_frozen_and_rescaled(self) -> None:
        first = w6.arrival_offsets(60, 0.15, 20260813)
        second = w6.arrival_offsets(60, 0.15, 20260813)
        self.assertEqual(first, second)
        self.assertEqual(first[0], 0)
        self.assertAlmostEqual(first[-1], 60 / 0.15)
        self.assertTrue(all(a <= b for a, b in zip(first, first[1:])))

    def test_prompt_recipe_is_cache_resistant(self) -> None:
        a = {"nonce": "a", "copies": 5}
        b = {"nonce": "b", "copies": 5}
        self.assertNotEqual(w6.prompt_for(a), w6.prompt_for(b))
        self.assertNotEqual(w6.prompt_hash(a), w6.prompt_hash(b))

    def test_metric_parser_sums_labeled_series(self) -> None:
        parsed = w6.parse_metrics(
            'vllm:num_requests_running{model_name="a"} 2\n'
            'vllm:num_requests_running{model_name="b"} 3\n'
        )
        self.assertEqual(parsed["vllm:num_requests_running"], 5)

    def test_latency_percentiles(self) -> None:
        values = w6.percentiles([1, 2, 3, 4])
        self.assertEqual(values["p50"], 2.5)
        self.assertGreater(values["p99"], values["p95"])


if __name__ == "__main__":
    unittest.main()
