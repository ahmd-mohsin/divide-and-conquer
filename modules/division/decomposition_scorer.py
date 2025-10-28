# decomposition_scorer.py
"""Evaluation metrics for decomposition quality assessment."""
from __future__ import annotations

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from typing import Dict, List, Set, Tuple
from collections import defaultdict
import re

from schemas import Decomposition


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_keywords(text: str) -> Set[str]:
    """Extract mathematical/symbolic keywords from text."""
    words = re.findall(r'[a-zA-Z_]\w*|\d+|[+\-*/=<>]', text.lower())
    return set(words) - {'the', 'a', 'an', 'to', 'for', 'from', 'in', 'of', 'with'}


def extract_variables(text: str) -> Set[str]:
    """Extract variable names from text."""
    return set(re.findall(r'\b([a-zA-Z])\b', text))


# ============================================================================
# 1. STRUCTURAL VALIDITY
# ============================================================================

def structural_validity_score(decomp: Decomposition) -> float:
    """Check DAG validity and edge justification."""
    is_valid, _ = decomp.validate_hierarchy()
    if not is_valid:
        return 0.0
    
    # Check edge justification
    justified = total = 0
    for node in decomp.nodes:
        for dep_id in node.depends_on:
            dep_node = decomp.get_node(dep_id)
            if not dep_node:
                continue
            total += 1
            node_kw = extract_keywords(node.goal + " " + node.plan)
            dep_kw = extract_keywords(dep_node.goal + " " + dep_node.plan)
            if len(node_kw & dep_kw) > 0:
                justified += 1
    
    justification_rate = justified / total if total > 0 else 1.0
    return (1.0 + justification_rate) / 2  # Average of validity and justification


# ============================================================================
# 2. COVERAGE
# ============================================================================

def coverage_score(decomp: Decomposition, problem_text: str) -> float:
    """Measure if effects cover the goal."""
    goal_vars = extract_variables(problem_text)
    if not goal_vars:
        return 0.5
    
    leaf_nodes = [n for n in decomp.nodes if len(decomp.get_children(n.id)) == 0]
    effect_vars = set()
    for node in leaf_nodes:
        effect_vars.update(extract_variables(node.goal + node.plan))
    
    covered = len(goal_vars & effect_vars)
    return covered / len(goal_vars) if goal_vars else 1.0


def dead_end_penalty(decomp: Decomposition) -> float:
    """Penalize nodes that don't contribute to goal."""
    if not decomp.nodes:
        return 0.0
    
    all_deps = set()
    for node in decomp.nodes:
        all_deps.update(node.depends_on)
    
    leaf_ids = {n.id for n in decomp.nodes if len(decomp.get_children(n.id)) == 0}
    
    dead_ends = sum(1 for n in decomp.nodes
                    if n.id not in leaf_ids and n.id not in all_deps)
    
    return dead_ends / len(decomp.nodes)


# ============================================================================
# 3. CHECKABILITY (PVSR Proxy)
# ============================================================================

def checkability_score(decomp: Decomposition) -> float:
    """Fraction of nodes with verification checks."""
    if not decomp.nodes:
        return 0.0
    checkable = sum(1 for n in decomp.nodes if n.suggested_check.value != "none")
    return checkable / len(decomp.nodes)


def step_sanity_score(decomp: Decomposition) -> float:
    """Quick sanity checks on steps."""
    if not decomp.nodes:
        return 0.0
    
    passed = 0
    for node in decomp.nodes:
        text = (node.goal + " " + node.plan).lower()
        if "divide by zero" in text or "undefined" in text:
            continue
        if "=" in text or any(kw in text for kw in ["solve", "find", "calculate"]):
            passed += 1
    
    return passed / len(decomp.nodes)


# ============================================================================
# 4. ATOMICITY
# ============================================================================

def atomicity_score(decomp: Decomposition) -> float:
    """Measure if sub-problems are atomic (5-30 tokens ideal)."""
    if not decomp.nodes:
        return 0.0
    
    scores = []
    for node in decomp.nodes:
        token_count = len((node.goal + " " + node.plan).split())
        if 5 <= token_count <= 30:
            scores.append(1.0)
        elif token_count < 5:
            scores.append(token_count / 5.0)
        else:
            scores.append(max(0.0, 1.0 - (token_count - 30) / 50.0))
    
    return sum(scores) / len(scores)


# ============================================================================
# 5. PARALLELISM
# ============================================================================

def parallelism_score(decomp: Decomposition) -> float:
    """Compute work/span ratio for parallelism."""
    if not decomp.nodes:
        return 0.0
    
    work = sum(len((n.goal + n.plan).split()) for n in decomp.nodes)
    span = compute_critical_path_length(decomp)
    
    if span == 0:
        return 1.0
    
    max_parallelism = len(decomp.nodes)
    actual = work / span
    return min(1.0, actual / max_parallelism)


def compute_critical_path_length(decomp: Decomposition) -> int:
    """Compute longest dependency chain."""
    memo = {}
    
    def dfs(node_id: str) -> int:
        if node_id in memo:
            return memo[node_id]
        node = decomp.get_node(node_id)
        if not node or not node.depends_on:
            memo[node_id] = 1
            return 1
        max_dep = max(dfs(dep_id) for dep_id in node.depends_on)
        memo[node_id] = max_dep + 1
        return memo[node_id]
    
    return max(dfs(n.id) for n in decomp.nodes) if decomp.nodes else 0


def width_profile_score(decomp: Decomposition) -> float:
    """Measure average width of dependency layers."""
    if not decomp.nodes:
        return 0.0
    
    # Compute layers via topological sort
    in_degree = {n.id: sum(1 for _ in n.depends_on) for n in decomp.nodes}
    layers = []
    current = [n.id for n in decomp.nodes if in_degree[n.id] == 0]
    
    while current:
        layers.append(current)
        next_layer = []
        for node_id in current:
            for node in decomp.nodes:
                if node_id in node.depends_on:
                    in_degree[node.id] -= 1
                    if in_degree[node.id] == 0:
                        next_layer.append(node.id)
        current = next_layer
    
    if not layers:
        return 0.0
    
    widths = [len(layer) for layer in layers]
    avg_width = sum(widths) / len(widths)
    ideal = len(decomp.nodes) ** 0.5
    return min(1.0, avg_width / ideal)


# ============================================================================
# 6. INDEPENDENCE
# ============================================================================

def independence_score(decomp: Decomposition) -> float:
    """Measure independence of sibling nodes."""
    if len(decomp.nodes) <= 1:
        return 1.0
    
    parent_to_children = defaultdict(list)
    for node in decomp.nodes:
        parent_to_children[node.parent_id].append(node)
    
    scores = []
    for children in parent_to_children.values():
        if len(children) <= 1:
            continue
        for i in range(len(children)):
            for j in range(i + 1, len(children)):
                kw_i = extract_keywords(children[i].goal + " " + children[i].plan)
                kw_j = extract_keywords(children[j].goal + " " + children[j].plan)
                if kw_i and kw_j:
                    jaccard = len(kw_i & kw_j) / len(kw_i | kw_j)
                    scores.append(1.0 - jaccard)
    
    return sum(scores) / len(scores) if scores else 1.0


# ============================================================================
# 7. PENALTIES
# ============================================================================

def redundancy_penalty(decomp: Decomposition) -> float:
    """Detect duplicate nodes."""
    if len(decomp.nodes) <= 1:
        return 0.0
    
    duplicates = comparisons = 0
    for i in range(len(decomp.nodes)):
        for j in range(i + 1, len(decomp.nodes)):
            comparisons += 1
            tokens_i = set((decomp.nodes[i].goal + decomp.nodes[i].plan).lower().split())
            tokens_j = set((decomp.nodes[j].goal + decomp.nodes[j].plan).lower().split())
            if tokens_i and tokens_j:
                jaccard = len(tokens_i & tokens_j) / len(tokens_i | tokens_j)
                if jaccard > 0.8:
                    duplicates += 1
    
    return duplicates / comparisons if comparisons > 0 else 0.0


def overhead_penalty(decomp: Decomposition, problem_text: str) -> float:
    """Penalize excessive work vs baseline."""
    baseline = len(problem_text.split())
    actual = sum(len((n.goal + n.plan).split()) for n in decomp.nodes)
    ratio = actual / max(1, baseline)
    return min(1.0, max(0.0, (ratio - 3.0) / 5.0)) if ratio > 3.0 else 0.0


# ============================================================================
# MAIN SCORING FUNCTION
# ============================================================================

def score_decomposition(
    decomp: Decomposition,
    problem_text: str,
    domain: str = "algebra"
) -> Dict[str, float]:
    """
    Comprehensive scoring of decomposition quality.
    
    Returns dict with individual metrics and Q_total aggregate score.
    """
    m = {}
    
    # 1. Structural validity (gate)
    m["struct_valid"] = structural_validity_score(decomp)
    if m["struct_valid"] < 0.5:
        m["Q_total"] = 0.0
        return m
    
    # 2. Coverage
    m["coverage"] = coverage_score(decomp, problem_text)
    
    # 3. Checkability
    m["check_rate"] = checkability_score(decomp)
    m["step_sanity"] = step_sanity_score(decomp)
    
    # 4. Atomicity
    m["atomicity"] = atomicity_score(decomp)
    
    # 5. Parallelism
    m["parallelism"] = parallelism_score(decomp)
    m["width_profile"] = width_profile_score(decomp)
    
    # 6. Independence
    m["independence"] = independence_score(decomp)
    
    # 7. Penalties
    m["dead_end_penalty"] = dead_end_penalty(decomp)
    m["redundancy_penalty"] = redundancy_penalty(decomp)
    m["overhead_penalty"] = overhead_penalty(decomp, problem_text)
    
    # Aggregate score (normalized with stronger penalties)
    Q_total = (
        0.50 * m["struct_valid"] +
        0.20 * m["coverage"] +
        0.20 * m["check_rate"] +
        0.15 * m["parallelism"] +
        0.15 * m["independence"] +
        0.15 * m["atomicity"] +
        0.15 * m["step_sanity"] +
        0.10 * m["width_profile"] -
        0.30 * m["redundancy_penalty"] -
        0.20 * m["overhead_penalty"] -
        0.10 * m["dead_end_penalty"]
    )
    
    m["Q_total"] = max(0.0, min(1.0, Q_total))
    
    return m


# ============================================================================
# BATCH EVALUATION
# ============================================================================

def evaluate_dataset(dataset_dir: str) -> Dict:
    """Evaluate all decompositions in a dataset."""
    from pathlib import Path
    import json
    from utils import load_decomposition
    
    index_file = Path(dataset_dir) / "dataset_index.json"
    if not index_file.exists():
        raise FileNotFoundError(f"Index file not found: {index_file}")
    
    with open(index_file, 'r') as f:
        index = json.load(f)
    
    all_scores = []
    for entry in index:
        decomp_file = Path(dataset_dir) / entry["file"]
        decomp = load_decomposition(decomp_file)
        scores = score_decomposition(decomp, entry["problem"])
        scores["decomp_id"] = entry["id"]
        all_scores.append(scores)
    
    # Summary statistics
    q_totals = [s["Q_total"] for s in all_scores]
    return {
        "total_decompositions": len(all_scores),
        "avg_Q_total": sum(q_totals) / len(q_totals) if q_totals else 0.0,
        "high_quality_count": sum(1 for q in q_totals if q >= 0.65),
        "low_quality_count": sum(1 for q in q_totals if q < 0.5),
        "individual_scores": all_scores
    }