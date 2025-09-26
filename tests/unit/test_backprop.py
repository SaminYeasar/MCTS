
# ----------------------------------
# Test 3: Backprop (_backprop method)
# ----------------------------------


import pytest
from typing import List, Optional, Tuple
from unittest.mock import MagicMock, Mock
import math

from astro import Node

# pytest that all tests in this file are marked with unit
# Equivalent to putting @pytest.mark.unit on every test function/class inside that file.
pytestmark = pytest.mark.unit 

"""
Objective: Ensure that the _backprop method correctly updates 
 - the Q-values and 
 - visit counts (N-edge, N-state) up the tree.
"""
def test_backprop_method(mcts_base):
    """
    Tests that the _backprop method correctly updates Q and N values for
    all nodes and edges along the path from a leaf to the root.

    Tree: Root -> Node_A -> Leaf_B
    Q_new = old_Q + (R - old_Q) / n_sa

    """
    # 1. Setup the MCTS object with mock dependencies
    mcts = mcts_base


    # 2. Manually build a simple path from a root to a leaf
    root = Node(id=mcts._new_id(), x="Problem", steps=[], parent=None, action_from_parent=None)
    child = Node(id=mcts._new_id(), x="Problem", steps=["Step 1"], parent=root, action_from_parent="Step 1")
    leaf_node = Node(id=mcts._new_id(), x="Problem", steps=["Step 1", "Step 2"], parent=child, action_from_parent="Step 2")

    # Connect the nodes
    root.children.append(child)
    child.children.append(leaf_node)
    
    # 3. Manually initialize the MCTS stats for the path
    # Initial N_state and N_edge: this gets increament while backprop
    mcts.N_state = {root.id: 0, child.id: 0, leaf_node.id: 0}
    mcts.N_edge[(root.id, "Step 1")] = 0
    mcts.N_edge[(child.id, "Step 2")] = 0

    # Initial Q_edge (pre-backprop)
    mcts.Q_edge[(root.id, "Step 1")] = 0.5
    mcts.Q_edge[(child.id, "Step 2")] = 0.5

    # 4. Define a reward and call the _backprop method
    reward = 1.0  # Simulate a successful rollout (it's done in the expand step and inserted)
    leaf_node.reward_from_rollouts = reward
    mcts._backprop(leaf_node)

    # 5. Assertions
    # Verify N_state updates
    assert mcts.N_state[root.id] == 1, "Root N_state visited 1"
    assert mcts.N_state[child.id] == 1, "Child N_state visited 1."
    assert mcts.N_state[leaf_node.id] == 0, "Leaf N_state will be visited only after next expansion."

    # Verify N_edge updates
    assert mcts.N_edge[(root.id, "Step 1")] == 1, "Root-to-child N_edge visited: 1."
    assert mcts.N_edge[(child.id, "Step 2")] == 1, "Child-to-leaf N_edge visited: 1"
    
    # Verify Q_edge updates
    # Q_new = Q_old + (Reward - Q_old) / N_new
    # Q for root-to-child: 0.5 + (1.0 - 0.5) / 1 = 1.0
    # Q for child-to-leaf: 0.5 + (1.0 - 0.5) / 1 = 1.0
    assert mcts.Q_edge[(root.id, "Step 1")] == 1.0, "Root-to-child Q_edge was not updated correctly."
    assert mcts.Q_edge[(child.id, "Step 2")] == 1.0, "Child-to-leaf Q_edge was not updated correctly."


def test_backprop_multiple_leaves_different_rewards(mcts_base):
    """
    Build: root --(S1)--> child --(S2a)--> leaf_a
                                 └--(S2b)--> leaf_b

    Backprop twice with different rewards:
      - leaf_a gets reward r1
      - leaf_b gets reward r2

    Expect:
      - Edge (child, S2a): Q=r1, N=1
      - Edge (child, S2b): Q=r2, N=1
      - Edge (root, S1): Q = (r1 + r2)/2, N=2   (simple incremental averaging)
      - N_state for root and child incremented twice; leaves unchanged (per your design).
    """
    mcts = mcts_base
    mcts.cpuct=1.0
    mcts.max_depth=10


    # Construct the tree
    root = Node(id=mcts._new_id(), x="Problem", steps=[], parent=None, action_from_parent=None)
    child = Node(id=mcts._new_id(), x="Problem", steps=["S1"], parent=root, action_from_parent="S1")
    leaf_a = Node(id=mcts._new_id(), x="Problem", steps=["S1", "S2a"], parent=child, action_from_parent="S2a")
    leaf_b = Node(id=mcts._new_id(), x="Problem", steps=["S1", "S2b"], parent=child, action_from_parent="S2b")

    root.children.append(child)
    child.children.extend([leaf_a, leaf_b])

    # Initialize counts and Qs
    mcts.N_state = {root.id: 0, child.id: 0, leaf_a.id: 0, leaf_b.id: 0}

    mcts.N_edge[(root.id, "S1")]  = 0
    mcts.N_edge[(child.id, "S2a")] = 0
    mcts.N_edge[(child.id, "S2b")] = 0

    mcts.Q_edge[(root.id, "S1")]  = 0.0
    mcts.Q_edge[(child.id, "S2a")] = 0.0
    mcts.Q_edge[(child.id, "S2b")] = 0.0

    # Two different rewards for two different leaves
    r1, r2 = 1.0, 0.0

    # Backprop #1: via leaf_a
    leaf_a.reward_from_rollouts = r1
    mcts._backprop(leaf_a)

    # Backprop #2: via leaf_b
    leaf_b.reward_from_rollouts = r2
    mcts._backprop(leaf_b)

    # --- Assertions ---

    # State visit counts (per your earlier convention: leaves remain 0)
    assert mcts.N_state[root.id] == 2, "Root should be visited twice."
    assert mcts.N_state[child.id] == 2, "Child should be visited twice."
    assert mcts.N_state[leaf_a.id] == 0, "Leaf A N_state remains 0. will be visited only after next expansion"
    assert mcts.N_state[leaf_b.id] == 0, "Leaf B N_state remains 0. will be visited only after next expansion."

    # Edge visit counts
    assert mcts.N_edge[(root.id, "S1")] == 2, "Root->Child edge should have two visits."
    assert mcts.N_edge[(child.id, "S2a")] == 1, "Child->LeafA edge should have one visit."
    assert mcts.N_edge[(child.id, "S2b")] == 1, "Child->LeafB edge should have one visit."

    # Leaf-edge Qs reflect their respective rewards
    assert mcts.Q_edge[(child.id, "S2a")] == pytest.approx(r1), "Q(child,S2a) should match reward r1."
    assert mcts.Q_edge[(child.id, "S2b")] == pytest.approx(r2), "Q(child,S2b) should match reward r2."

    # Parent edge Q is the average of both rollouts (simple incremental average assumption)
    expected_parent_q = (r1 + r2) / 2.0
    assert mcts.Q_edge[(root.id, "S1")] == pytest.approx(expected_parent_q), \
        "Q(root,S1) should be the mean of rewards from both leaves."
    
