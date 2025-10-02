# test_linearization.py
import pytest
from dataclasses import dataclass, field
from typing import List
from astro import (
    Node,
    Verifier,
    cot_from_linearized_sequence,
    cot_from_nodes_to_text, dfs_collect_terminals
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
if __name__ == "__main__":
    pytest.main(["-q", __file__])
