# utils.py
"""Utility functions for the HCOT system."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys
import os

# Add current directory to path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from schemas import Decomposition, SubProblem


def save_decomposition(decomp: Decomposition, filepath: str | Path) -> None:
    """Save a decomposition to a JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(decomp.model_dump(), f, indent=2, ensure_ascii=False)


def load_decomposition(filepath: str | Path) -> Decomposition:
    """Load a decomposition from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Decomposition(**data)


def get_execution_order(decomp: Decomposition) -> List[List[str]]:
    """Compute a valid execution order respecting dependencies."""
    node_ids = {node.id for node in decomp.nodes}
    in_degree = {node.id: 0 for node in decomp.nodes}
    
    for node in decomp.nodes:
        for dep_id in node.depends_on:
            if dep_id in node_ids:
                in_degree[node.id] += 1
    
    waves = []
    remaining = set(node_ids)
    
    while remaining:
        current_wave = [
            node_id for node_id in remaining
            if in_degree[node_id] == 0
        ]
        
        if not current_wave:
            raise ValueError(f"Cannot compute execution order: possible cycle. Remaining: {remaining}")
        
        waves.append(current_wave)
        
        for node_id in current_wave:
            remaining.remove(node_id)
            for node in decomp.nodes:
                if node_id in node.depends_on:
                    in_degree[node.id] -= 1
    
    return waves


def print_execution_plan(decomp: Decomposition) -> None:
    """Print a human-readable execution plan showing parallel waves."""
    try:
        waves = get_execution_order(decomp)
        
        print(f"\n{'='*70}")
        print(f"EXECUTION PLAN: {len(waves)} wave(s)")
        print(f"{'='*70}\n")
        
        for i, wave in enumerate(waves, 1):
            print(f"Wave {i} (can be executed in parallel):")
            for node_id in wave:
                node = decomp.get_node(node_id)
                if node:
                    deps_str = f" (depends on: {', '.join(node.depends_on)})" if node.depends_on else ""
                    print(f"  [{node_id}] {node.goal}{deps_str}")
            print()
        
        print(f"{'='*70}\n")
        
    except ValueError as e:
        print(f"Error computing execution plan: {e}")


def export_to_graphviz(decomp: Decomposition, filepath: str | Path) -> None:
    """Export decomposition as Graphviz DOT file."""
    lines = ["digraph HCOT {"]
    lines.append('  rankdir=TB;')
    lines.append('  node [shape=box, style=rounded];')
    lines.append('')
    
    for node in decomp.nodes:
        label = f"{node.id}\\n{node.goal[:40]}"
        if len(node.goal) > 40:
            label += "..."
        lines.append(f'  "{node.id}" [label="{label}"];')
    
    lines.append('')
    
    for node in decomp.nodes:
        if node.parent_id:
            lines.append(f'  "{node.parent_id}" -> "{node.id}" [color=black];')
    
    for node in decomp.nodes:
        for dep_id in node.depends_on:
            lines.append(f'  "{dep_id}" -> "{node.id}" [color=red, style=dashed];')
    
    lines.append('}')
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"Graphviz DOT file saved to: {filepath}")


def get_statistics(decomp: Decomposition) -> Dict[str, Any]:
    """Compute statistics about the decomposition."""
    stats = {
        "total_nodes": len(decomp.nodes),
        "root_nodes": len(decomp.get_root_nodes()),
        "leaf_nodes": 0,
        "max_depth": 0,
        "avg_children": 0,
        "nodes_with_dependencies": 0,
        "total_dependencies": 0,
        "check_types": {},
    }
    
    def get_depth(node_id: str) -> int:
        node = decomp.get_node(node_id)
        if not node or not node.parent_id:
            return 0
        return 1 + get_depth(node.parent_id)
    
    depths = []
    child_counts = []
    
    for node in decomp.nodes:
        depth = get_depth(node.id)
        depths.append(depth)
        
        children = len(decomp.get_children(node.id))
        child_counts.append(children)
        
        if children == 0:
            stats["leaf_nodes"] += 1
        
        if node.depends_on:
            stats["nodes_with_dependencies"] += 1
            stats["total_dependencies"] += len(node.depends_on)
        
        check = node.suggested_check.value
        stats["check_types"][check] = stats["check_types"].get(check, 0) + 1
    
    stats["max_depth"] = max(depths) if depths else 0
    stats["avg_children"] = sum(child_counts) / len(child_counts) if child_counts else 0
    
    return stats


def print_statistics(decomp: Decomposition) -> None:
    """Print statistics about the decomposition."""
    stats = get_statistics(decomp)
    
    print(f"\n{'='*50}")
    print("DECOMPOSITION STATISTICS")
    print(f"{'='*50}")
    print(f"Total nodes:              {stats['total_nodes']}")
    print(f"Root nodes:               {stats['root_nodes']}")
    print(f"Leaf nodes:               {stats['leaf_nodes']}")
    print(f"Max depth:                {stats['max_depth']}")
    print(f"Avg children per node:    {stats['avg_children']:.2f}")
    print(f"Nodes with dependencies:  {stats['nodes_with_dependencies']}")
    print(f"Total dependencies:       {stats['total_dependencies']}")
    print(f"\nCheck types:")
    for check_type, count in stats['check_types'].items():
        print(f"  {check_type:25s} {count}")
    print(f"{'='*50}\n")


def validate_node_id_uniqueness(decomp: Decomposition) -> tuple[bool, List[str]]:
    """Check if all node IDs are unique."""
    seen = set()
    duplicates = []
    
    for node in decomp.nodes:
        if node.id in seen:
            duplicates.append(node.id)
        seen.add(node.id)
    
    return len(duplicates) == 0, duplicates