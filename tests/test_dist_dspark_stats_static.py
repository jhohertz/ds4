#!/usr/bin/env python3
"""Source-contract checks for distributed DSpark worker proposal telemetry."""

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "ds4.c").read_text(encoding="utf-8")


def c_function(name: str) -> str:
    start = SOURCE.index(name + "(")
    brace = SOURCE.index("{", start)
    depth = 0
    for pos in range(brace, len(SOURCE)):
        if SOURCE[pos] == "{":
            depth += 1
        elif SOURCE[pos] == "}":
            depth -= 1
            if depth == 0:
                return SOURCE[start : pos + 1]
    raise AssertionError(f"unterminated function {name}")


class DistributedDSparkStatsStaticTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fn = c_function("ds4_session_dist_dspark_draft")

    def test_every_valid_call_counts_one_cycle_and_first_token(self) -> None:
        prepare = self.fn.index("ds4_session_prepare_dspark_draft")
        cycle = self.fn.index("s->dspark_stats.cycles++")
        no_draft = self.fn.index("if (!s->dspark_draft_valid")
        self.assertLess(prepare, cycle)
        self.assertLess(cycle, no_draft)
        self.assertIn("s->dspark_stats.first_tokens++", self.fn)
        marker = self.fn.index("DSpark distributed worker stats are proposal-only")
        self.assertLess(marker, cycle)
        self.assertIn("if (s->dspark_stats.cycles == 0)", self.fn)

    def test_empty_and_nonempty_block_lengths_are_recorded(self) -> None:
        no_draft = self.fn.index("if (!s->dspark_draft_valid")
        success = self.fn.index("int n = (int)s->dspark_draft_len", no_draft)
        empty = self.fn[no_draft:success]
        nonempty = self.fn[success:]
        self.assertIn("s->dspark_stats.no_draft++", empty)
        self.assertIn("draft_len_hist, 0", empty)
        self.assertIn("s->dspark_stats.proposed_tokens += (uint64_t)n", nonempty)
        self.assertIn("draft_len_hist,\n                                  (uint32_t)n", nonempty)

    def test_worker_does_not_invent_acceptance_feedback(self) -> None:
        self.assertNotIn("accepted_draft", self.fn)
        self.assertNotIn("accepted_len_hist", self.fn)
        self.assertIn("Distributed acceptance is", self.fn)
        self.assertIn("not fed back to this worker", self.fn)
        self.assertIn("acceptance feedback unavailable", self.fn)


if __name__ == "__main__":
    unittest.main()
