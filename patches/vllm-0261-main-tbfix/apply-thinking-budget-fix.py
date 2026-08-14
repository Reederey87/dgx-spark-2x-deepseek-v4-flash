#!/usr/bin/env python3
"""Apply the pinned thinking-budget fast-path fix to vLLM's sampler."""

from pathlib import Path
import sys


OLD = """        if np.any(self.bad_words_state.num_bad_words.np[idx_mapping_np] > 0):
            return True

        states = self.sampling_states
"""

NEW = """        if np.any(self.bad_words_state.num_bad_words.np[idx_mapping_np] > 0):
            return True

        # Thinking budgets force the reasoning-end marker inside
        # apply_sampling_params(). Default sampling (temperature/top-p 1.0) would
        # otherwise take this method's fast path and silently skip the budget.
        if self.thinking_budget_state.enabled and np.any(
            self.thinking_budget_state.use_thinking_budget[idx_mapping_np]
        ):
            return True

        states = self.sampling_states
"""


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: apply-thinking-budget-fix.py <sampler.py>")
    target = Path(sys.argv[1])
    source = target.read_text()
    if NEW in source:
        raise SystemExit(f"refusing already-patched sampler: {target}")
    if source.count(OLD) != 1:
        raise SystemExit(f"expected exactly one pinned sampler fast path in {target}")
    target.write_text(source.replace(OLD, NEW))


if __name__ == "__main__":
    main()
