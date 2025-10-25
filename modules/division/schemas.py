# schemas.py
"""Data models for the HCOT system."""
from __future__ import annotations
import sys
import os

# Add current directory to path for direct imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field, conlist, validator


class CheckType(str, Enum):
    """Types of verification checks for sub-problems."""
    numeric_equal = "numeric_equal"
    algebra_equiv = "algebra_equiv"
    unit_consistency = "unit_consistency"
    justification_entailment = "justification_entailment"
    none = "none"


class SubProblem(BaseModel):
    """A node in the problem decomposition hierarchy."""
    id: str = Field(..., description="Unique identifier for this sub-problem")
    parent_id: Optional[str] = Field(None, description="ID of parent problem (None for root)")
    goal: str = Field(..., description="What this sub-problem aims to achieve")
    plan: str = Field(..., description="High-level approach to solve this sub-problem")
    depends_on: List[str] = Field(
        default_factory=list,
        description="IDs of other sub-problems this depends on"
    )
    hint: Optional[str] = Field(None, description="Helpful hint for solving this")
    suggested_check: CheckType = Field(
        CheckType.none,
        description="Type of verification check to apply"
    )
    expected_form: Optional[str] = Field(
        None,
        description="Expected format of the solution"
    )
    
    @validator("depends_on", pre=True)
    def ensure_list(cls, v):
        if v is None:
            return []
        return v


class Decomposition(BaseModel):
    """Complete hierarchical decomposition of a problem."""
    problem: str = Field(..., description="Original problem statement")
    depth_limit: int = Field(..., description="Maximum hierarchy depth")
    branching_limit: int = Field(..., description="Maximum children per node")
    nodes: conlist(SubProblem, min_length=1) = Field(
        ...,
        description="List of all sub-problems in the hierarchy"
    )
    
    def get_root_nodes(self) -> List[SubProblem]:
        """Get all root-level nodes (no parent)."""
        return [node for node in self.nodes if node.parent_id is None]
    
    def get_children(self, node_id: str) -> List[SubProblem]:
        """Get all direct children of a node."""
        return [node for node in self.nodes if node.parent_id == node_id]
    
    def get_dependencies(self, node_id: str) -> List[SubProblem]:
        """Get all nodes that this node depends on."""
        node = next((n for n in self.nodes if n.id == node_id), None)
        if not node:
            return []
        return [n for n in self.nodes if n.id in node.depends_on]
    
    def get_node(self, node_id: str) -> Optional[SubProblem]:
        """Get a specific node by ID."""
        return next((n for n in self.nodes if n.id == node_id), None)
    
    def validate_hierarchy(self) -> tuple[bool, List[str]]:
        """
        Validate the hierarchy structure.
        
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        node_ids = {node.id for node in self.nodes}
        
        # Check all parent_ids exist
        for node in self.nodes:
            if node.parent_id and node.parent_id not in node_ids:
                errors.append(f"Node {node.id} has invalid parent_id: {node.parent_id}")
        
        # Check all depends_on IDs exist
        for node in self.nodes:
            for dep_id in node.depends_on:
                if dep_id not in node_ids:
                    errors.append(f"Node {node.id} depends on non-existent node: {dep_id}")
        
        # Check for cycles in dependencies
        def has_cycle(node_id: str, visited: set, rec_stack: set) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            
            node = self.get_node(node_id)
            if node:
                for dep_id in node.depends_on:
                    if dep_id not in visited:
                        if has_cycle(dep_id, visited, rec_stack):
                            return True
                    elif dep_id in rec_stack:
                        return True
            
            rec_stack.remove(node_id)
            return False
        
        visited = set()
        for node in self.nodes:
            if node.id not in visited:
                if has_cycle(node.id, visited, set()):
                    errors.append(f"Cycle detected in dependency graph involving node {node.id}")
        
        return len(errors) == 0, errors


class CoTStep(BaseModel):
    """A single step in a chain of thought."""
    step_id: str
    subproblem_id: str
    content: str
    reasoning: str
    confidence: Optional[float] = None
    prm_score: Optional[float] = None  # For future PRM module


class CoTChain(BaseModel):
    """A complete chain of thought for a sub-problem."""
    subproblem_id: str
    steps: List[CoTStep]
    final_answer: str
    total_score: Optional[float] = None
    is_pruned: bool = False


class Solution(BaseModel):
    """Final merged solution."""
    problem: str
    answer: str
    decomposition: Decomposition
    cot_chains: List[CoTChain]
    metadata: dict = Field(default_factory=dict)


# JSON Schema for structured output from LLMs
DECOMPOSITION_SCHEMA = {
    "type": "object",
    "properties": {
        "problem": {"type": "string"},
        "depth_limit": {"type": "integer"},
        "branching_limit": {"type": "integer"},
        "nodes": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "parent_id": {"type": ["string", "null"]},
                    "goal": {"type": "string"},
                    "plan": {"type": "string"},
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": []
                    },
                    "hint": {"type": ["string", "null"]},
                    "suggested_check": {
                        "type": "string",
                        "enum": [
                            "numeric_equal",
                            "algebra_equiv",
                            "unit_consistency",
                            "justification_entailment",
                            "none"
                        ]
                    },
                    "expected_form": {"type": ["string", "null"]}
                },
                "required": ["id", "parent_id", "goal", "plan", "suggested_check"]
            }
        }
    },
    "required": ["problem", "depth_limit", "branching_limit", "nodes"]
}