

# ----------------------------------
# Test 2: Expansion (_expand method)
# ----------------------------------
"""
Objective: Check if the _expand method correctly creates new 
 - child nodes [Done]
 - runs rollouts [not added yet]
 - and initializes MCTS stats. [Done]


"""
import pytest
import math
from astro import Node
from typing import List, Optional, Tuple

# pytest that all tests in this file are marked with unit
# Equivalent to putting @pytest.mark.unit on every test function/class inside that file.
pytestmark = pytest.mark.unit 

def test_expand_method(mcts_base, monkeypatch):
    """
    _expand 
    - should create children, 
    - initialize Q_edge, N_edge, Prior_edge correctly,
    - and use policy priors from step_logprob.
    """
    mcts = mcts_base


    # parent to expand (depth = 0)
    parent = Node(id=mcts._new_id(), x="Problem", steps=[], parent=None, action_from_parent=None)
    mcts.N_state[parent.id] = 1  # root/state visit count starts at 1 in this impl

    # desired actions + priors
    new_actions = ["Action X", "Action Y", "Action Z"]
    priors = {"Action X": 0.5, "Action Y": 0.3, "Action Z": 0.2}

    # monkeypatch policy to produce exactly our candidates + log-probs
    def fake_sample_step_candidates(x: str, steps: List[str], k: int) -> List[str]:
        # make sure k doesn't truncate our fixed list in this test
        return new_actions

    def fake_step_logprob(x: str, old_steps: List[str], new_step: str) -> float:
        # If your implementation stores Prior_edge as exp(logprob), this is consistent.
        return math.log(priors[new_step])

    monkeypatch.setattr(mcts.policy, "sample_step_candidates", fake_sample_step_candidates)
    monkeypatch.setattr(mcts.policy, "step_logprob", fake_step_logprob)

    # Some implementations require a ground_truth arg; provide a dummy string
    newly_added_children = mcts._expand(parent, ground_truth="42")

    # ---- Assertions ----

    # Children count (and returned list if your _expand returns it)
    assert len(parent.children) == len(new_actions), "Incorrect number of children created."
    if newly_added_children is not None:
        assert len(newly_added_children) == len(new_actions)

    seen_actions = {c.action_from_parent for c in parent.children}
    assert seen_actions == set(new_actions), "Child actions do not match expected new actions."

    for child in parent.children:
        # Parent link
        assert child.parent is parent, "Child's parent not set correctly."


        # Edge keys
        edge = (parent.id, child.action_from_parent)

        # Q/N initialization
        assert edge in mcts.Q_edge, "Q_edge missing for new edge."
        assert mcts.Q_edge[edge] == 0.0, "Q_edge should initialize to 0.0."

        assert edge in mcts.N_edge, "N_edge missing for new edge."
        assert mcts.N_edge[edge] == 0, "N_edge should initialize to 0."

        # Prior from policy
        assert edge in mcts.Prior_edge, "Prior_edge missing for new edge."
        # If Prior_edge stores exp(logprob) (i.e., the raw probability), compare to priors dict
        assert mcts.Prior_edge[edge] == pytest.approx(priors[child.action_from_parent])

