import pytest
from typing import List, Optional, Tuple
from unittest.mock import MagicMock, Mock
import math

from astro import MCTSAstro, Node, Verifier, LMPolicy

# --- Mock Classes ---
class MockVerifier(Verifier):
    def score(self, predicted_answer: Optional[str], ground_truth: str) -> float:
        return 0.0

class MockPolicy(LMPolicy):
    def __init__(self):
        super().__init__(tokenizer=MagicMock(), model=MagicMock(), device=MagicMock())

    def sample_step_candidates(self, x: str, steps: List[str], k: int) -> List[str]:
        return []

    def step_logprob(self, x: str, old_steps: List[str], new_step: str) -> float:
        return 0.0

    def rollout_from(self, x: str, steps: List[str]) -> Tuple[str, Optional[str]]:
        return "", None

    def self_eval_is_high_quality(self, solution_text: str) -> bool:
        return True


# ----- Global determinism -----

@pytest.fixture(scope="session", autouse=True)
def _seed():
    import random
    try:
        import torch
        torch.manual_seed(0)
    except Exception:
        pass
    random.seed(0)

# ----- Reusable builders -----

@pytest.fixture
def mcts_base():
    return MCTSAstro(
        policy=MockPolicy(),
        verifier=MockVerifier(),
        cpuct=1.0,
        k_actions=3,
        m_rollouts=1,
    )