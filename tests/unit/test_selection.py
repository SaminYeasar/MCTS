# ---------------------------------
# Test 1: Selection (_select method)
# ----------------------------------
"""
Objective: Verify that the _select method correctly navigates the tree using the PUCT formula.

Steps:

1. Initialize MCTSAstro with a mock policy that provides specific priors and a mock verifier.
    - the return of the verifier is not important for "selection" test
2. Create a simple tree structure manually with a root node and a few children.
3. Manually set the Q_edge, N_edge, and Prior_edge values for the nodes and edges.
4. Set up one path with a high Q-value (simulating a good path) and another with a low Q-value.
5. Set one path with a low visit count (N_edge) to test the exploration term.
6. Call the _select method on the root node.

"Assert" that the method returns the expected node:
 - the selection should first favor high Q-values 
 - if Q-values are similar, favor paths with fewer visits (higher PUCT score due to the exploration term).



Example Scenario:

 - Path A: steps=["step A"], Q_edge=0.9, N_edge=10
 - Path B: steps=["step B"], Q_edge=0.5, N_edge=2
 - Path C: steps=["step C"], Q_edge=0.8, N_edge=2

Selection should first choose Path C over Path A due to its lower visit count, 
then over Path B due to its higher Q-value.
"""
import pytest
from astro import Node



# pytest that all tests in this file are marked with unit
# Equivalent to putting @pytest.mark.unit on every test function/class inside that file.
pytestmark = pytest.mark.unit 

def test_select_puct_balance(mcts_base):
    """
    _select should pick the child with the highest PUCT, balancing Q (exploitation) and N (exploration).
    """
    mcts = mcts_base

    # If your MCTSAstro has its own id allocator, use it; otherwise you can assign manually.
    root = Node(id=mcts._new_id(), x="Problem: 1+1", steps=[], parent=None, action_from_parent=None)

    child_a = Node(id=mcts._new_id(), x="Problem: 1+1", steps=["Step A"], parent=root, action_from_parent="Step A")
    child_b = Node(id=mcts._new_id(), x="Problem: 1+1", steps=["Step B"], parent=root, action_from_parent="Step B")
    child_c = Node(id=mcts._new_id(), x="Problem: 1+1", steps=["Step C"], parent=root, action_from_parent="Step C")
    root.children = [child_a, child_b, child_c]

    # Visit counts (state-level) for sqrt term
    mcts.N_state = {root.id: 1, child_a.id: 1, child_b.id: 1, child_c.id: 1}

    # Edge stats
    # A: High Q, high N
    mcts.Q_edge[(root.id, "Step A")] = 0.9
    mcts.N_edge[(root.id, "Step A")] = 10
    mcts.Prior_edge[(root.id, "Step A")] = 1.0

    # B: Lower Q, low N
    mcts.Q_edge[(root.id, "Step B")] = 0.5
    mcts.N_edge[(root.id, "Step B")] = 2
    mcts.Prior_edge[(root.id, "Step B")] = 1.0

    # C: Good Q, low N → should win due to exploration bonus
    mcts.Q_edge[(root.id, "Step C")] = 0.8
    mcts.N_edge[(root.id, "Step C")] = 2
    mcts.Prior_edge[(root.id, "Step C")] = 1.0

    selected = mcts._select(root)
    # PUCT = Q + cpuct * Prior * (sqrt(N_state) / (1 + N_edge))
    # A  = 0.9 + 1.0 * 1.0 * (sqrt(1) / (1 + 10)) = 0.9 + 0.09 = 0.99
    # B = 0.5 + 1.0 * 1.0 * (sqrt(1) / (1 + 2)) = 0.5 + 0.33 = 0.83
    # C = 0.8 + 1.0 * 1.0 * (sqrt(1) / (1 + 2)) = 0.8 + 0.33 = 1.13    (the pick)

    assert selected.id == child_c.id, "Expected Child C to have the highest PUCT score."
    assert selected.action_from_parent == "Step C"


def test_select_multi_level_descent(mcts_base):
    """
    _select should descend the tree following the highest PUCT path across multiple levels.
    """
    mcts = mcts_base
    mcts.cpuct=1.0
    mcts.max_depth=10

    root = Node(id=mcts._new_id(), x="Problem: 1+1", steps=[], parent=None, action_from_parent=None)

    # Layer 2
    child_a = Node(id=mcts._new_id(), x="Problem: 1+1", steps=["Step A"], parent=root, action_from_parent="Step A")
    child_b = Node(id=mcts._new_id(), x="Problem: 1+1", steps=["Step B"], parent=root, action_from_parent="Step B")
    root.children = [child_a, child_b]

    # Layer 3
    g_a_a = Node(id=mcts._new_id(), x="Problem: 1+1", steps=["Step A", "Step A_A"], parent=child_a, action_from_parent="Step A_A")
    g_a_b = Node(id=mcts._new_id(), x="Problem: 1+1", steps=["Step A", "Step A_B"], parent=child_a, action_from_parent="Step A_B")
    child_a.children = [g_a_a, g_a_b]

    g_b_a = Node(id=mcts._new_id(), x="Problem: 1+1", steps=["Step B", "Step B_A"], parent=child_b, action_from_parent="Step B_A")
    g_b_b = Node(id=mcts._new_id(), x="Problem: 1+1", steps=["Step B", "Step B_B"], parent=child_b, action_from_parent="Step B_B")
    child_b.children = [g_b_a, g_b_b]

    # State visit counts
    mcts.N_state = {
        root.id: 1,
        child_a.id: 1, child_b.id: 1,
        g_a_a.id: 1, g_a_b.id: 1, g_b_a.id: 1, g_b_b.id: 1,
    }

    # Root -> children
    mcts.Q_edge[(root.id, "Step A")] = 0.9
    mcts.N_edge[(root.id, "Step A")] = 10
    mcts.Prior_edge[(root.id, "Step A")] = 1.0

    mcts.Q_edge[(root.id, "Step B")] = 0.8
    mcts.N_edge[(root.id, "Step B")] = 2
    mcts.Prior_edge[(root.id, "Step B")] = 1.0

    # child_a -> grandchildren
    mcts.Q_edge[(child_a.id, "Step A_A")] = 0.7
    mcts.N_edge[(child_a.id, "Step A_A")] = 1
    mcts.Prior_edge[(child_a.id, "Step A_A")] = 1.0

    mcts.Q_edge[(child_a.id, "Step A_B")] = 0.2
    mcts.N_edge[(child_a.id, "Step A_B")] = 1
    mcts.Prior_edge[(child_a.id, "Step A_B")] = 1.0

    # child_b -> grandchildren (make B_A best at level 2)
    mcts.Q_edge[(child_b.id, "Step B_A")] = 0.95
    mcts.N_edge[(child_b.id, "Step B_A")] = 1
    mcts.Prior_edge[(child_b.id, "Step B_A")] = 1.0

    mcts.Q_edge[(child_b.id, "Step B_B")] = 0.1
    mcts.N_edge[(child_b.id, "Step B_B")] = 1
    mcts.Prior_edge[(child_b.id, "Step B_B")] = 1.0

    selected = mcts._select(root)

    assert selected.id == g_b_a.id, "Expected to descend Root -> B -> B_A."
    assert selected.action_from_parent == "Step B_A"
