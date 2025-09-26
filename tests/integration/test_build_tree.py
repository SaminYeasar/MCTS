
import math
import pytest
from unittest.mock import MagicMock, Mock
from typing import List, Optional, Tuple

from astro import Node
pytestmark = pytest.mark.integration



def test_build_tree_integration(mcts_base, monkeypatch):
    """
    Tests the main build_tree method, verifying the full MCTS loop.
    After a single rollout (m_rollouts=1), a new leaf is created and backpropagated.
    """

    actions = ["A1", "A2", "A3"]
    priors = {a: 1.0 / len(actions) for a in actions}  # simple uniform priors

    def fake_sample_step_candidates(x: str, steps: List[str], k: int) -> List[str]:
        # ensure we return exactly 3 actions for expansion
        return actions

    def fake_step_logprob(x: str, old_steps: List[str], new_step: str) -> float:
        # Prior = exp(logprob). Use ln(prior) so Prior_edge matches priors[new_step]
        return math.log(priors[new_step])


    # 1) Create MCTS with one rollout per iteration, allow a few iterations
    mcts = mcts_base

    # 2) Mock policy: fixed rollout reward, and controlled expansion (3 actions)
    mcts.policy.rollout_from = Mock(return_value=("Solution Text", 1.0))  # a perfect reward

    monkeypatch.setattr(mcts.policy, "sample_step_candidates", fake_sample_step_candidates)
    monkeypatch.setattr(mcts.policy, "step_logprob", fake_step_logprob)


    mcts.cpuct=1.0
    mcts.k_actions=3
    mcts.m_rollouts=1     # only one rollout per iteration
    mcts.max_depth=10
    mcts.iterations=5

    # 3) Run build_tree
    root_node = mcts.build_tree(x="Problem: 1+1", ground_truth="2")

    # 4) Assertions

    # Root should have 3 children after expansion
    assert len(root_node.children) == 3, "Root node should have 3 children after expansion."

    # Find which child got rolled out (N_edge == 1) and check Q/N updates
    found_rolled_out_child = False
    for child in root_node.children:
        edge_key = (root_node.id, child.action_from_parent)

        # Identify the rolled-out child edge
        if mcts.N_edge.get(edge_key) == 1:
            found_rolled_out_child = True

            # Q_edge should be updated to the reward (1.0) after one visit
            assert mcts.Q_edge.get(edge_key) == pytest.approx(1.0), \
                f"Q_edge for {child.action_from_parent} was not updated to 1.0."

            # Root N_state should be 2 (started at 1, incremented once on backprop)
            assert mcts.N_state.get(root_node.id) == 2, \
                "Root N_state should be 2 after one rollout/backprop."

            # Child N_state should also be 2 by the same logic
            assert mcts.N_state.get(child.id) == 2, \
                f"Child {child.action_from_parent} N_state should be 2 after one backprop."

    assert found_rolled_out_child, "No child showed N_edge == 1; expected one rollout/backprop to occur."
