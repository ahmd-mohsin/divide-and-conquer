# expansion_schemas.py
"""Data models for CoT expansion."""
from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class CoTStep(BaseModel):
    """A single step in a reasoning chain."""
    step_number: int
    content: str = Field(..., description="The reasoning content")
    strategy: str = Field(..., description="Strategy used (e.g., 'step_by_step')")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CoTChain(BaseModel):
    """A complete chain of thought for a sub-problem."""
    chain_id: str
    subproblem_id: str
    steps: List[CoTStep]
    final_answer: Optional[str] = None
    strategy: str = Field(..., description="Overall strategy")
    
    # Quality metrics
    coherence_score: Optional[float] = None
    correctness_score: Optional[float] = None
    
    # Diversity metrics
    diversity_score: Optional[float] = None
    novelty_score: Optional[float] = None
    
    # RL metadata
    rl_reward: Optional[float] = None
    exploration_bonus: Optional[float] = None
    
    created_at: datetime = Field(default_factory=datetime.now)
    
    def get_text(self) -> str:
        """Get full text of the chain."""
        return "\n".join([f"Step {s.step_number}: {s.content}" for s in self.steps])


class SubProblemExpansion(BaseModel):
    """All chains for a single sub-problem."""
    subproblem_id: str
    subproblem_goal: str
    chains: List[CoTChain]
    best_chain_id: Optional[str] = None
    diversity_matrix: Optional[List[List[float]]] = None  # Pairwise diversity
    
    def get_best_chain(self) -> Optional[CoTChain]:
        """Get the highest-scoring chain."""
        if not self.chains:
            return None
        return max(self.chains, key=lambda c: c.rl_reward or 0.0)


class ExpansionResult(BaseModel):
    """Complete expansion result for a decomposition."""
    decomposition_id: str
    problem: str
    subproblem_expansions: List[SubProblemExpansion]
    
    # Aggregate statistics
    total_chains: int = 0
    avg_diversity: float = 0.0
    avg_quality: float = 0.0
    
    training_metadata: Dict[str, Any] = Field(default_factory=dict)


class RLState(BaseModel):
    """State representation for RL agent."""
    subproblem_text: str
    current_chains: List[str]  # Existing chains as text
    num_existing_chains: int
    avg_chain_length: float
    diversity_so_far: float
    
    def to_feature_vector(self) -> List[float]:
        """Convert to feature vector for RL agent."""
        # Simple feature extraction
        features = [
            len(self.subproblem_text) / 100.0,  # Normalized length
            self.num_existing_chains / 10.0,
            self.avg_chain_length / 100.0,
            self.diversity_so_far,
        ]
        return features


class RLAction(BaseModel):
    """Action taken by RL agent."""
    strategy_index: int
    strategy_name: str
    temperature: float = 0.7
    metadata: Dict[str, Any] = Field(default_factory=dict)