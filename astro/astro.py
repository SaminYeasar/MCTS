# astro_mcts.py
# -----------------------------------------------------------------------------
# Astro-style data generation with MCTS (PUCT), multi-rollout expansion,
# search tree linearization with backtracking, and CoT translation.
# -----------------------------------------------------------------------------
# Problem formulation (from the user spec):
# - We treat data generation as an MDP (Puterman, 1994).
# - The LM policy Π_LM explores stepwise solution space for a math input x.
# - State S_t = (x, s_0, ..., s_t) where s_i are minimal reasoning steps.
# - Action a_{t+1} = next step s_{t+1}.
# - Verifier V returns rewards at terminal states by checking final answers.
#
# Overview pipeline:
#   1) Build a search tree T per x using MCTS w/ PUCT selection and rollout-based rewards.
#   2) Linearize T into a sequence of nodes L that includes backtracking (Algorithm 1).
#   3) Translate L into a chain-of-thought y with self-reflection and backtracking, yielding (x, y).
#
# MCTS specifics (Section 2.2):
# - Selection uses PUCT (Silver et al., 2016):
#   argmax over actions i of  Q(S_t, a_i) + c_puct * Π_LM(a_i | S_t) * sqrt(N(S_t)) / (1 + N(S_t, a_i))
# - Expansion from S_t:
#     • Sample k candidate next-steps (actions) using Π_LM.
#     • For each action, run M rollouts to end and score with V; average to get R(S_{t+1}).
#     • Add node n_{t+1} for each S_{t+1}.
# - Backpropagation updates:
#     N(S_t) ← N(S_t) + 1
#     Q(S_t, a) ← (Σ_i Q(S_{t+1}, a_i)·N(S_{t+1}, a_i) + R(S_{t+1})) / (Σ_i N(S_{t+1}, a_i) + 1)
#
# Notes:
# - This file is written to be runnable on small HF models (e.g., gpt-neo-125M) for demo.
#   For production, swap to a stronger policy (e.g., llama-3.1/3.2/3.3-instruct) and tune params.
# - Defaults are lightweight; the spec’s k=8, M=16, iterations=32, depth=50 are compute-heavy.
# -----------------------------------------------------------------------------

from __future__ import annotations
import argparse
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# ----------------------------- Utilities -------------------------------------

def set_seed(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_numeric_answer(text: str) -> Optional[str]:
    """Heuristic numerical answer extractor.
    Tries \boxed{}, 'FINAL ANSWER:', trailing number, 'answer:'.
    """
    t = text.strip()
    # Boxed pattern
    m = re.search(r"\\boxed\{\s*([-+]?\d+(?:\.\d+)?)\s*\}", t)
    if m:
        return m.group(1)
    # FINAL ANSWER pattern
    m = re.search(r"final\s*answer\s*[:=]\s*([-+]?\d+(?:\.\d+)?)", t, flags=re.I)
    if m:
        return m.group(1)
    # 'answer:' pattern
    m = re.search(r"answer\s*[:=]\s*([-+]?\d+(?:\.\d+)?)", t, flags=re.I)
    if m:
        return m.group(1)
    # last number in text
    m_iter = list(re.finditer(r"([-+]?\d+(?:\.\d+)?)", t))
    if m_iter:
        return m_iter[-1].group(1)
    return None


# ----------------------------- Policy ----------------------------------------

STEP_PREFIX = "- Step: "
FINAL_PREFIX = "FINAL ANSWER:"

@dataclass
class LMPolicy:
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    device: torch.device
    step_max_new_tokens: int = 48
    rollout_max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.95

    def state_to_text(self, x: str, steps: List[str]) -> str:
        header = (
            "You are solving a math problem step-by-step. "
            "Each step must be a minimal, self-contained reasoning unit.\n"
            "Output format:\n"
            f"{STEP_PREFIX}<content>\n(repeat as needed)\n"
            f"{FINAL_PREFIX} <final_number>\n"
        )
        prompt = f"Problem: {x}\nSolution so far:\n"
        for s in steps:
            prompt += f"{STEP_PREFIX}{s}\n"
        return header + "\n" + prompt

    def _generate(self, encodings: dict, max_new_tokens: int, num_return_sequences: int=1) -> torch.LongTensor:
        with torch.no_grad():
            return self.model.generate(
                input_ids=encodings['input_ids'],  # Pass input_ids
                attention_mask=encodings['attention_mask'], # Pass attention_mask
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                num_return_sequences=num_return_sequences,  # Request k alternative generations
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

    def sample_step_candidates(self, x: str, steps: List[str], k: int) -> List[str]:
        """Sample up to k *distinct* next-step candidates as strings.
        """
        context = self.state_to_text(x, steps)
        enc = self.tokenizer(context, return_tensors="pt", padding=True).to(self.device)

        def parse_new_text(new_text: str, sink: List[str]) -> None:
            """Parse a generation continuation into step candidates (unique append)."""
            for line in new_text.splitlines():
                if line.startswith(STEP_PREFIX):
                    s = line[len(STEP_PREFIX):].strip()
                    if s and s not in sink:
                        sink.append(s)
                elif line.startswith(FINAL_PREFIX):
                    final = line[len(FINAL_PREFIX):].strip()
                    tag = f"(declare final) {final}"
                    if final and tag not in sink:
                        sink.append(tag)

        candidates: List[str] = []

        # First sample
        out = self._generate(enc, self.step_max_new_tokens, num_return_sequences=k)
        # take only the newly generated token ids
        # new_ids = out[:, enc.input_ids.size(1):]
        # candidates = self.tokenizer.decode(new_ids, skip_special_tokens=True)

        # Process each generated sequence individually.
        for generated_sequence in out:
            # Take only the newly generated token IDs for the current sequence.
            # The original input is at the beginning, so we slice from its length.
            new_ids = generated_sequence[enc.input_ids.size(1):]
            
            # Decode the new IDs into a single string.
            new_text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            
            # Use the helper function to parse the decoded text and extract candidates.
            # The parse_new_text function handles extracting specific steps and ensuring uniqueness.
            #parse_new_text(new_text, candidates)
            candidates.append(new_text)
            # Stop if we have collected enough unique candidates.
            if len(candidates) >= k:
                break

        # new_text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        #parse_new_text(new_text, candidates)

        # Retry a few times if not enough unique candidates
        # attempts, max_attempts = 0, 4
        # while len(candidates) < k and attempts < max_attempts:
        #     out = self._generate(enc, self.step_max_new_tokens)
        #     new_ids = out[0, enc.input_ids.size(1):]
        #     new_text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        #     parse_new_text(new_text, candidates)
        #     attempts += 1
        # return candidates
        return candidates

    def step_logprob(self, x: str, old_steps: List[str], new_step: str) -> float:
        """Compute a (length-normalized) log-prob of proposing this step under Π_LM.
        We condition on state and score the tokens for the one step line.
        """
        prefix = self.state_to_text(x, old_steps)
        target = f"{STEP_PREFIX}{new_step}\n"
        enc = self.tokenizer(prefix + target, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(enc.input_ids, labels=enc.input_ids)
            # labels shift inside HF; compute loss and scale by number of new tokens only
            # We recompute by masking the prefix positions
            logits = out.logits[:, :-1, :]
            labels = enc.input_ids[:, 1:]
            # Mask out the prefix positions
            prefix_len = self.tokenizer(prefix, return_tensors="pt").input_ids.size(1)
            # New tokens count (excluding prefix shift):
            new_mask = torch.zeros_like(labels, dtype=torch.bool)
            new_mask[:, prefix_len - 1 :] = True
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            picked = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            picked = picked.masked_select(new_mask)
            if picked.numel() == 0:
                return -1e9
            return float(picked.mean())  # length-normalized

    def rollout_from(self, x: str, steps: List[str]) -> Tuple[str, Optional[str]]:
        """Generate to completion (or budget) from current state and return (full_text, extracted_answer)."""
        context = self.state_to_text(x, steps)
        enc = self.tokenizer(context, return_tensors="pt").to(self.device)
        out = self._generate(enc, self.rollout_max_new_tokens)
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # alternatively extract answer only from new text:
        # new_ids = out[0, enc.input_ids.size(1):]
        # text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        ans = extract_numeric_answer(text)
        return text, ans

    def self_eval_is_high_quality(self, solution_text: str) -> bool:
        """Very lightweight self-eval stub. Returns True if it *looks* structured.
        You can replace by prompting the LM for a 0/1 score.
        """
        # Heuristics: has at least 2 steps and a final answer
        steps = len(re.findall(re.escape(STEP_PREFIX), solution_text))
        return steps >= 2 and (FINAL_PREFIX in solution_text)


# ----------------------------- Verifier --------------------------------------

@dataclass
class Verifier:
    tol: float = 1e-6

    def score(self, predicted_answer: Optional[str], ground_truth: str) -> float:
        try:
            if predicted_answer is None:
                return 0.0
            return 1.0 if abs(float(predicted_answer) - float(ground_truth)) < self.tol else 0.0
        except Exception:
            return 0.0


# ----------------------------- Tree / MCTS -----------------------------------

@dataclass
class Node:
    id: int  # unique identifier for the node, used for efficient data lookup in the MCTS dictionaries
    x: str   # The original problem statement, preserved at each node
    steps: List[str] # The sequence of steps taken from the root to reach this node
    parent: Optional[Node] # A reference to the parent node, essential for backpropagating rewards up the tree.
    action_from_parent: Optional[str]  # the step text chosen to get here
    children: List[Node] = field(default_factory=list) # A list of child nodes, representing the possible next steps (actions)
    reward_from_rollouts: Optional[float] = None  # R(S_{t+1}) The average reward from M rollouts performed starting from this node. This is the estimated value of this path.
    final_answer: Optional[str] = None            # extracted answer for terminal solutions (from rollouts)

    def depth(self) -> int:
        """Returns the depth of the node in the tree, equivalent to the number of steps taken."""
        return len(self.steps)


class MCTSAstro:
    """PUCT-based MCTS over *steps* (not tokens), with multi-rollout expansion.

    Implements the spec’s Selection, Expansion (k actions, M rollouts per action), and Backpropagation Eq. (3)(4).
    """

    def __init__(
        self,
        policy: LMPolicy,
        verifier: Verifier,
        *,
        cpuct: float = 1.0,
        k_actions: int = 2,           # demo-friendly default (spec suggests 8)
        m_rollouts: int = 4,          # demo-friendly default (spec suggests 16)
        iterations: int = 8,          # demo-friendly default (spec suggests 32)
        max_depth: int = 12,
    ):
        self.policy = policy
        self.verifier = verifier
        self.cpuct = cpuct
        self.k_actions = k_actions
        self.m_rollouts = m_rollouts
        self.iterations = iterations
        self.max_depth = max_depth

        # Stats
        self._next_id = 0
        self.N_state: Dict[int, int] = {}                   # N(S_t)
        self.N_edge: Dict[Tuple[int, str], int] = {}        # N(S_t, a)
        self.Q_edge: Dict[Tuple[int, str], float] = {}      # Q(S_t, a)
        self.Prior_edge: Dict[Tuple[int, str], float] = {}  # Π_LM(a|S_t) normalized over proposed k

    # ---- Helpers ----
    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _puct(self, s_id: int, a: str) -> float:
        q = self.Q_edge.get((s_id, a), 0.0)
        p = self.Prior_edge.get((s_id, a), 1e-8)
        n_s = self.N_state.get(s_id, 0)
        n_sa = self.N_edge.get((s_id, a), 0)
        return q + self.cpuct * p * math.sqrt(max(1, n_s)) / (1 + n_sa)

    def _select(self, root: Node) -> Node:
        """Descend by PUCT until reaching a node that can be expanded or is at max depth."""
        node = root
        while node.children and node.depth() < self.max_depth:
            s_id = node.id
            best_child = None
            best_score = -1e18
            # Check if priors exist for all children before selection
            if all((s_id, ch.action_from_parent) in self.Prior_edge for ch in node.children):
                for child in node.children:
                    a = child.action_from_parent
                    score = self._puct(s_id, a)
                    if score > best_score:
                        best_score = score
                        best_child = child
            else:
                # If some children haven't been expanded yet (priors not set), select them
                # This is a heuristic to handle non-fully expanded nodes
                unexpanded_children = [ch for ch in node.children if (s_id, ch.action_from_parent) not in self.Prior_edge]
                if unexpanded_children:
                    node = unexpanded_children[0]
                    break
                else:
                    best_child = node.children[0] # fallback
            
            node = best_child if best_child is not None else node
            if node is None:
                break
        return node


    def _expand(self, node: Node, ground_truth: str):
        """
        Expand by sampling up to k actions;
        for each action compute average score R via M rollouts, add a child, and set priors.
        Also initializes N(S_t, a) and Q(S_t, a) if unseen.
        """
        if node.depth() >= self.max_depth:
            return

        candidates = self.policy.sample_step_candidates(node.x, node.steps, self.k_actions)
        if not candidates:
            return

        logps = [self.policy.step_logprob(node.x, node.steps, c) for c in candidates]
        maxl = max(logps)
        exps = [math.exp(lp - maxl) for lp in logps]
        z = sum(exps) if sum(exps) > 0 else 1.0
        priors = [e / z for e in exps]
        
        # New children to backprop from
        newly_added_children = []

        for c, p in zip(candidates, priors):
            # 1. EXPANSION: Create a new child node for the candidate step.
            # This node represents a new, un-explored state in the search tree.
            child = Node(
                id=self._new_id(),
                x=node.x,
                steps=node.steps + [c],
                parent=node,
                action_from_parent=c,
            )

            # 2. ROLLOUTS: Simulate M solutions to estimate the value of this new path.
            scores = []
            final_ans = None
            for _ in range(self.m_rollouts):
                # Use the LM to generate a full solution from the child node's state.
                # child.x is the prompt
                # child.steps contain actions/steps taken until now
                # rollout will generate sequence for max "rollout_max_new_tokens" assigned in config
                full_text, ans = self.policy.rollout_from(child.x, child.steps)
                if final_ans is None:
                    final_ans = ans

                 # Get a reward (1.0 for correct, 0.0 for incorrect) from the verifier.
                scores.append(self.verifier.score(ans, ground_truth))
            
            # Calculate the average reward from all rollouts and store it in the child node.
            child.reward_from_rollouts = float(sum(scores) / max(1, len(scores)))
            child.final_answer = final_ans

            # 3. TREE MANAGEMENT: Add the new child and initialize its stats.
            node.children.append(child)
            newly_added_children.append(child)


            # Initialize the MCTS statistics for the new edge (S_t, a).
            s_id = node.id
            a = c
            self.Prior_edge[(s_id, a)] = p # Store the LM's prior probability (P(a|S_t)).
            self.N_edge.setdefault((s_id, a), 0) # Initialize the visit count for the edge (N(S_t, a)).
            self.Q_edge.setdefault((s_id, a), 0.0) # Initialize the Q-value (expected reward) for the edge (Q(S_t, a)).

        return newly_added_children

    def _backprop(self, leaf: Node):
        """Backpropagate a single rollout reward from leaf to root."""
        R = leaf.reward_from_rollouts if leaf.reward_from_rollouts is not None else 0.0
        # Initialize leaf count 
        self.N_state[leaf.id] = self.N_state.get(leaf.id, 0)
        node = leaf

        # for each leaf-nodes (S_t+1, a) average-rollout, we back-prop and update the Q(S_t) values starting leaf-->root
        while node.parent is not None:
            parent = node.parent
            action_a = node.action_from_parent
            parent_a_id = (parent.id, action_a)
            
            # The update Q value (Eq. 4): update Q based on the average reward
            # of all rollouts that have passed through this edge
            self.N_edge[parent_a_id] += 1
            self.N_state[parent.id] += 1
            old_Q = self.Q_edge.get(parent_a_id, 0.0)
            n_sa = self.N_edge[parent_a_id]
            updated_Q = old_Q + (R - old_Q) / n_sa
            self.Q_edge[parent_a_id] = updated_Q

            node = parent

        # Helper function to validate the backpropagation is working
        def validate_backprop_invariants(self):
            """
            Check that MCTS invariants hold after backpropagation.
            Call this periodically during development to catch bugs.
            """
            for (s_id, action), n_sa in self.N_edge.items():
                # Visit count should be positive
                assert n_sa > 0, f"Non-positive visit count for ({s_id}, {action})"
                
                # Q-value should exist
                assert (s_id, action) in self.Q_edge, f"Missing Q-value for ({s_id}, {action})"
                
                # Q-value should be reasonable (between 0 and 1 for this problem)
                q_val = self.Q_edge[(s_id, action)]
                assert 0 <= q_val <= 1, f"Q-value {q_val} out of range for ({s_id}, {action})"
            
            # State visit counts should be sum of edge visit counts
            for s_id, n_s in self.N_state.items():
                edge_sum = sum(self.N_edge.get((s_id, a), 0) 
                            for (sid, a) in self.N_edge.keys() if sid == s_id)
                if edge_sum > 0:  # Only check if there are outgoing edges
                    assert n_s >= edge_sum, f"State visit count {n_s} < edge sum {edge_sum} for state {s_id}"

    # -----------------------
    # build tree using Astro:
    # -----------------------
    def build_tree(self, x: str, ground_truth: str) -> Node:
        """
        The main MCTS loop to build the search tree.
        x : input
        """
        root = Node(id=self._new_id(), x=x, steps=[], parent=None, action_from_parent=None)
        self.N_state[root.id] = 1 # Initialize root visit count
        
        for i in range(self.iterations):
            # 1) Selection
            sel_node = self._select(root)

            # 2) Expansion
            newly_added_children = self._expand(sel_node, ground_truth)
            
            if not newly_added_children:
                # If no new children were added (e.g., max depth reached)
                # still need to backpropagate from the selected node itself
                # if it has a reward.
                if sel_node.reward_from_rollouts is not None:
                    self._backprop(sel_node)
                continue

            
            # 3) Backpropagation
            # We treat each of the k expansions as a separate simulation result.
            for child in newly_added_children:
                self._backprop(child)
        return root


# ----------------------------- Linearization ---------------------------------

@dataclass
class LinearizationResult:
    L_nodes: List[Node]
    correct_terminal: Node
    incorrect_terminals: List[Node]


def is_terminal_correct(node: Node, verifier: Verifier, ground_truth: str) -> bool:
    return verifier.score(node.final_answer, ground_truth) > 0.5


def dfs_collect_terminals(root: Node) -> List[Node]:
    stack = [root]
    visited = {root.id}
    terminals: List[Node] = []

    while stack:
        n = stack.pop()

        # terminal if it has no children
        if not n.children:
            terminals.append(n)

        for ch in n.children:
            if ch.id not in visited:
                visited.add(ch.id)
                stack.append(ch)

    return terminals


def path_from_root(node: Node) -> List[Node]:
    path = []
    cur = node
    while cur is not None:
        path.append(cur)
        cur = cur.parent
    return list(reversed(path))


def lca_prefix_length(a: List[Node], b: List[Node]) -> int:
    """
    Length of longest common prefix between two root-to-node paths.
    Finding the Backtrack Point: would tell you exactly where the two sequences diverged.
    """

    # Sets the maximum possible length to check (L) as the length of the shorter path. This prevents index errors.
    L = min(len(a), len(b))
    i = 0
    # The loop continues as long as: 
    # 1) the index i is within the bounds of the shorter list, AND 
    # 2) the node at the current index in path a has the same unique ID as the node at the same index in path b
    while i < L and a[i].id == b[i].id:
        # Increments the counter i every time a common node is found, extending the common prefix length.
        i += 1
    return i


def merge_in(L: List[Node],path_nodes: List[Node]) -> List[Node]:
    """
    Merge a root->leaf path into the running linearization L without
    deleting prior content, and explicitly insert a backtrack hop to the LCA
    so downstream CoT can detect self-reflection.
    """
    if not L:
        L = path_nodes[:]                 # take the first path as-is
        return L

    # Current "position" is the path to the last node we just wrote into L.
    curr_path = path_from_root(L[-1])

    # Find LCA prefix length between the current position and the new path.
    prefix = lca_prefix_length(curr_path, path_nodes)

    # 1) Insert an explicit hop back to the LCA (even if already in L)
    if prefix > 0:
        lca_node = curr_path[prefix - 1]
        L.append(lca_node)                # duplicate on purpose → signals backtrack

    # 2) Append only the novel suffix of the new path (l − L by id, in order)
    L_ids = {n.id for n in L}
    novel_tail = [n for n in path_nodes[prefix:] if n.id not in L_ids]
    L.extend(novel_tail)
    return L

def linearize_with_backtracking(
    root: Node,
    verifier: Verifier,
    ground_truth: str,
    policy: LMPolicy,
    k_backtracks: int = 1,
) -> LinearizationResult:
    """
    Implements Algorithm 1 at a high level.
    - Collect terminal nodes (correct & incorrect).
    - Filter to a subset of correct terminals with "high-quality" steps via policy self-eval.
    - Sample 1 correct (ψ*) and up to k incorrect with unique answers.
    - Merge paths to create L with backtracking.
    """
    terminals = dfs_collect_terminals(root)
    correct = [n for n in terminals if is_terminal_correct(n, verifier, ground_truth)]
    incorrect = [n for n in terminals if not is_terminal_correct(n, verifier, ground_truth)]

    # Self-eval filter Ψ' (simple heuristic stub)
    correct_hq = []
    for n in correct:
        full_text, _ = policy.rollout_from(n.x, n.steps)  # reconstruct plausible solution text
        if policy.self_eval_is_high_quality(full_text):
            correct_hq.append(n)
    if not correct_hq and correct:
        correct_hq = correct  # fallback
    if not correct_hq:
        if terminals:
            best = max(terminals, key=lambda t: (t.reward_from_rollouts or 0.0))
            correct_hq = [best]
        else:
            return LinearizationResult(L_nodes=[root], correct_terminal=root, incorrect_terminals=[])

    correct_star = random.choice(correct_hq)

    # choose up to k incorrect with unique answers
    uniq_incorrect = []
    seen_answers = set()
    for n in incorrect:
        ans = n.final_answer
        if ans is None:
            continue
        if ans not in seen_answers:
            seen_answers.add(ans)
            uniq_incorrect.append(n)
        if len(uniq_incorrect) >= k_backtracks:
            break

    # correct_star = incorrect[0]
    # uniq_incorrect = incorrect

    # Merge sequences
    L: List[Node] = []
    # def merge_in(path_nodes: List[Node]):
    #     nonlocal L
    #     if not L:
    #         L = path_nodes[:]
    #         return
    #     L_ids = {n.id for n in L}
    #     L.extend([n for n in path_nodes if n.id not in L_ids])

        
    for n in uniq_incorrect + [correct_star]:
        pn = path_from_root(n)
        L = merge_in(L, pn)


    return LinearizationResult(L_nodes=L, correct_terminal=correct_star, incorrect_terminals=uniq_incorrect)


# ------------------------- CoT Translation -----------------------------------

def cot_from_linearized_sequence(x: str, L_nodes: List[Node]) -> str:
    """Translate L into a natural-language CoT that includes self-reflection/backtracking cues.
    This is a simple deterministic renderer based on node steps and path relations.
    """
    lines = [f"Problem: {x}", "Solution:"]
    for i in range(1, len(L_nodes)):
        prev = L_nodes[i - 1]
        cur = L_nodes[i]
        path_prev_ids = {n.id for n in path_from_root(prev)}
        path_cur_ids = {n.id for n in path_from_root(cur)}
        
        # Check for backtracking by seeing if the current node is an ancestor of the previous one.
        # This occurs when we jump back up the tree.
        if cur.id in path_prev_ids and len(path_cur_ids) < len(path_prev_ids):
            lines.append("(Self-reflection) The previous branch led to an incorrect direction; I backtrack to an earlier state.")
        
        if cur.action_from_parent:
            lines.append(f"{STEP_PREFIX}{cur.action_from_parent}")
    
    last_node = L_nodes[-1]
    ans = last_node.final_answer or extract_numeric_answer(cot_from_nodes_to_text(last_node))
    
    if ans is not None:
        lines.append(f"{FINAL_PREFIX} {ans}")
    return "\n".join(lines)

def cot_from_nodes_to_text(node: Node) -> str:
    """Reconstruct a full solution string from a node's path for rendering."""
    path = path_from_root(node)
    steps = [n.action_from_parent for n in path if n.action_from_parent]
    return '\n'.join([f"{STEP_PREFIX}{s}" for s in steps])


# ----------------------------- Runner / CLI ----------------------------------

@dataclass
class RunConfig:
    model_name: str = "EleutherAI/gpt-neo-125M"
    split: str = "train[:6]"  # keep tiny for demo
    subset: str = "algebra__linear_1d"
    k_actions: int = 2
    m_rollouts: int = 4
    iterations: int = 3
    max_depth: int = 2
    cpuct: float = 1.0
    seed: int = 0
    out_jsonl: str = "astro_mcts_dataset.jsonl"


class LMContextHolder:
    def __init__(self):
        self.policy: Optional[LMPolicy] = None

LMCTX = LMContextHolder()


def main(cfg: RunConfig):
    set_seed(cfg.seed)


    print(f"Loading model: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    policy = LMPolicy(tokenizer=tokenizer, model=model, device=device)
    verifier = Verifier()
    LMCTX.policy = policy

    print("Loading dataset…")
    ds = load_dataset("deepmind/math_dataset", cfg.subset, split=cfg.split, trust_remote_code=True)

    mcts = MCTSAstro(
        policy=policy,
        verifier=verifier,
        cpuct=cfg.cpuct,
        k_actions=cfg.k_actions,
        m_rollouts=cfg.m_rollouts,
        iterations=cfg.iterations,
        max_depth=cfg.max_depth,
    )

    out_path = cfg.out_jsonl
    with open(out_path, "w", encoding="utf-8") as f:
        pass

    total = len(ds)
    correct_any = 0
    for i, ex in enumerate(ds):
        x = ex["question"]
        gt = ex["answer"]
        print(f"\n=== {i+1}/{total} ===")
        print("Problem:", x)
        print("Ground truth:", gt)

        root = mcts.build_tree(x, gt)

        lin = linearize_with_backtracking(root, verifier, gt, policy, k_backtracks=1)
        y = cot_from_linearized_sequence(x, lin.L_nodes, verifier, gt)

        terminals = dfs_collect_terminals(root)
        is_correct_tree = any(is_terminal_correct(n, verifier, gt) for n in terminals)
        correct_any += int(is_correct_tree)

        record = {
            "x": x,
            "y": y,
            "correct_any": bool(is_correct_tree),
            "n_nodes": int(len(terminals)),
        }
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        print("Linearized CoT preview:\n", y.splitlines()[:8], "…")
        print("Any correct terminal in tree:", is_correct_tree)

    print(f"\nWrote dataset to {out_path}")
    print(f"Trees with any correct terminal: {correct_any}/{total}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=RunConfig.model_name)
    p.add_argument("--split", type=str, default=RunConfig.split)
    p.add_argument("--subset", type=str, default=RunConfig.subset)
    p.add_argument("--k", type=int, default=RunConfig.k_actions)
    p.add_argument("--M", type=int, default=RunConfig.m_rollouts)
    p.add_argument("--iters", type=int, default=RunConfig.iterations)
    p.add_argument("--depth", type=int, default=RunConfig.max_depth)
    p.add_argument("--cpuct", type=float, default=RunConfig.cpuct)
    p.add_argument("--seed", type=int, default=RunConfig.seed)
    p.add_argument("--out", type=str, default=RunConfig.out_jsonl)
    args = p.parse_args()

    cfg = RunConfig(
        model_name=args.model,
        split=args.split,
        subset=args.subset,
        k_actions=args.k,
        m_rollouts=args.M,
        iterations=args.iters,
        max_depth=args.depth,
        cpuct=args.cpuct,
        seed=args.seed,
        out_jsonl=args.out,
    )
    main(cfg)