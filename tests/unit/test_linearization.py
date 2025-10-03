# test_linearization.py
import pytest
from dataclasses import dataclass, field
from typing import List
from astro import (
    Node,
    Verifier,
    cot_from_linearized_sequence,
    cot_from_nodes_to_text, dfs_collect_terminals, path_from_root, merge_in
)

SELF_REF_LINE = "(Self-reflection) The previous branch led to an incorrect direction; I backtrack to an earlier state."

def _mk_node(id, x, steps, parent, action_from_parent, final_answer=None):
    n = Node(
        id=id,
        x=x,
        steps=list(steps),
        parent=parent,
        action_from_parent=action_from_parent,
    )
    n.final_answer = final_answer
    if parent is not None:
        parent.children.append(n)
    return n


def test_cot_linearization_no_backtracking_uses_final_answer():
    """L_nodes follows a single path; renderer should list steps and use final_answer."""
    x = "2 + 5"
    ver = Verifier()

    # Build a simple chain: root -> n1 -> n2
    root = _mk_node(1, x, [], None, None)
    n1   = _mk_node(2, x, ["Add 2 and 5"], root, "Add 2 and 5")
    n2   = _mk_node(3, x, ["Add 2 and 5", "Compute sum"], n1, "Compute sum", final_answer="7")

    L_nodes = [root, n1, n2]
    text = cot_from_linearized_sequence(x, L_nodes, ver, ground_truth="7")

    # Must contain the problem & solution header
    assert text.startswith(f"Problem: {x}")
    assert "Solution:" in text

    # Must list the two steps in order
    assert "- Step: Add 2 and 5" in text
    assert "- Step: Compute sum" in text

    # No self-reflection in a straight path
    assert SELF_REF_LINE not in text

    # Final answer comes from final_answer
    assert text.strip().endswith("FINAL ANSWER: 7")


def test_cot_linearization_with_backtracking_and_extracted_answer():
    """
    L_nodes includes a backtrack: root -> a -> ab -> a -> ac
    Expect the self-reflection line before re-visiting 'a', and final answer
    extracted from steps (since last node has no final_answer).
    """
    x = "some problem"
    ver = Verifier()

    # Nodes
    root = _mk_node(10, x, [], None, None)
    a    = _mk_node(11, x, ["Try approach A"], root, "Try approach A")
    ab   = _mk_node(12, x, ["Try approach A", "Compute wrong thing 13"], a, "Compute wrong thing 13")
    # Backtrack to 'a' (same node object 'a' appears again in L_nodes)
    # Then proceed to a correct child 'ac'
    ac   = _mk_node(13, x, ["Try approach A", "Correct compute 7"], a, "Correct compute 7", final_answer=None)

    # Linearized sequence with backtrack: include 'a' again between ab and ac
    L_nodes = [root, a, ab, a, ac]

    text = cot_from_linearized_sequence(x, L_nodes, ver, ground_truth="7")

    # Should include self-reflection exactly once
    assert text.count(SELF_REF_LINE) == 1

    # Steps should appear in this order (note 'Try approach A' appears twice due to the backtrack hop)
    lines = text.splitlines()
    assert any(line.strip() == "- Step: Try approach A" for line in lines)
    assert any(line.strip() == "- Step: Compute wrong thing 13" for line in lines)
    assert any(line.strip() == "- Step: Correct compute 7" for line in lines)

    # Final answer should be extracted from the last node's steps (contains the number 7)
    assert text.strip().endswith("FINAL ANSWER: 7")

    # Also validate cot_from_nodes_to_text reconstructs the last path's steps
    rendered_path = cot_from_nodes_to_text(ac)
    assert rendered_path == "- Step: Try approach A\n- Step: Correct compute 7"

# --- Test case: three terminal nodes (2 incorrect, 1 correct) ---

def test_merge_in():
    """
    Tree (all terminals at layer 2):
          root(0)
            |
           A(1)
         /     \
       B(2)    C(3)   and D(4) as a sibling of C for a third terminal
                \
                 D(4)
    Paths:
      P_incorrect1: root -> A -> B
      P_incorrect2: root -> A -> C
      P_correct:    root -> A -> D

    Merge order: incorrect1, incorrect2, correct
    Expected L (by action_from_parent):
      [None, "A", "B", "A", "C", "A", "D"]
      - 'A' duplicates twice (one per branch switch) as backtrack markers.
      - Final node is the correct leaf (D).
    """
    
    x="rand: foo(bar)?"

    # Build nodes
    root = Node(id=0, x=x, steps=[], parent=None, action_from_parent=None)

    A = Node(id=1, x=root.x, steps=["A"], parent=root, action_from_parent="A")
    root.children = [A]

    B = Node(id=2, x=root.x, steps=A.steps + ["B"], parent=A, action_from_parent="B")
    C = Node(id=3, x=root.x, steps=A.steps + ["C"], parent=A, action_from_parent="C")
    D = Node(id=4, x=root.x, steps=A.steps + ["D"], parent=A, action_from_parent="D")
    A.children = [B, C, D]

    # Mark terminals (two incorrect, one correct)
    B.final_answer, B.reward_from_rollouts = "40", 0.0   # incorrect
    C.final_answer, C.reward_from_rollouts = "41", 0.0   # incorrect
    D.final_answer, D.reward_from_rollouts = "42", 1.0   # correct

    # Build paths to terminals
    P_incorrect1 = path_from_root(B)  # [root, A, B]
    P_incorrect2 = path_from_root(C)  # [root, A, C]
    P_correct    = path_from_root(D)  # [root, A, D]

    # Merge in order: incorrect1, incorrect2, correct
    L: List[Node] = []
    L = merge_in(L, P_incorrect1)
    L = merge_in(L, P_incorrect2)
    L = merge_in(L, P_correct)  

    # Assert the linearized sequence of actions (None for root)
    actions = [n.action_from_parent for n in L]
    assert actions == [None, "A", "B", "A", "C", "A", "D"]

    # Backtrack markers: the duplicated LCA 'A' should appear exactly 3 times:
    #   - once as part of first path,
    #   - plus 2 duplicates (one per branch switch).
    count_A_nodes = sum(1 for n in L if n is A)
    assert count_A_nodes == 3

    # Final node is the correct terminal
    assert L[-1] is D
    assert D.final_answer == "42" and D.reward_from_rollouts == 1.0

    cot_sequence = cot_from_linearized_sequence(x=x, L_nodes=L)

    print(cot_sequence)


def test_dfs_collect_terminals():
    @dataclass
    class dummy_Node:
        id: int
        children: List["dummy_Node"] = field(default_factory=list)

        def add_child(self, child: "dummy_Node"):
            self.children.append(child)
    
    root = dummy_Node(0)
    left = dummy_Node(1)        # will be a leaf
    right = dummy_Node(2)
    deep = dummy_Node(3)        # will be a leaf

    root.add_child(left)
    root.add_child(right)
    right.add_child(deep)

    terminals = dfs_collect_terminals(root)
    term_ids = sorted(n.id for n in terminals)

    assert term_ids == [1, 3], f"Expected leaves [1,3], got {term_ids}"


# Optional: run directly (handy for debugger)
# if __name__ == "__main__":
#     pytest.main(["-q", __file__])
