# reward_model.py
"""Reward model for RL agent."""
from typing import List, Tuple
import numpy as np
from expansion_schemas import CoTChain
from expansion_config import RewardConfig
from diversity_metrics import DiversityMetrics

class RewardModel:
    """Compute rewards for generated chains."""
    
    def __init__(self, config: RewardConfig):
        self.config = config
        self.diversity_metrics = DiversityMetrics(
            use_embeddings=config.use_embedding_similarity
        )
    
    def compute_reward(
        self,
        new_chain: CoTChain,
        existing_chains: List[CoTChain],
        subproblem_text: str
    ) -> float:
        """
        Compute total reward for a new chain.
        
        Args:
            new_chain: Newly generated chain
            existing_chains: Previously generated chains
            subproblem_text: Original sub-problem
            
        Returns:
            Total reward (higher is better)
        """
        # Diversity reward
        diversity_reward = self._compute_diversity_reward(new_chain, existing_chains)
        
        # Quality reward
        quality_reward = self._compute_quality_reward(new_chain, subproblem_text)
        
        # Progress reward
        progress_reward = self._compute_progress_reward(new_chain)
        
        # Penalties
        penalties = self._compute_penalties(new_chain, existing_chains)
        
        # Weighted sum
        total_reward = (
            self.config.diversity_weight * diversity_reward +
            self.config.quality_weight * quality_reward +
            self.config.progress_weight * progress_reward +
            penalties
        )
        
        return total_reward
    
    def _compute_diversity_reward(
        self,
        new_chain: CoTChain,
        existing_chains: List[CoTChain]
    ) -> float:
        """Reward for being different from existing chains."""
        if not existing_chains:
            return 1.0  # First chain gets full diversity reward
        
        new_text = new_chain.get_text()
        existing_texts = [c.get_text() for c in existing_chains]
        
        novelty = self.diversity_metrics.compute_novelty(new_text, existing_texts)
        
        return novelty
    
    def _compute_quality_reward(
        self,
        chain: CoTChain,
        subproblem_text: str
    ) -> float:
        """Reward for quality of reasoning."""
        score = 0.0
        
        # Length check (not too short, not too long)
        total_length = sum(len(step.content) for step in chain.steps)
        if 100 < total_length < 2000:
            score += 0.3
        
        # Number of steps (prefer moderate depth)
        num_steps = len(chain.steps)
        if 3 <= num_steps <= 8:
            score += 0.3
        
        # Check if final answer exists
        if chain.final_answer:
            score += 0.2
        
        # Check for mathematical content
        text = chain.get_text()
        if any(char in text for char in "=+-*/()<>[]"):
            score += 0.2
        
        return min(score, 1.0)
    
    def _compute_progress_reward(self, chain: CoTChain) -> float:
        """Reward for making progress toward solution."""
        # Simple heuristic: more steps with final answer = progress
        score = 0.0
        
        if len(chain.steps) > 0:
            score += 0.5
        
        if chain.final_answer and len(chain.final_answer) > 10:
            score += 0.5
        
        return score
    
    def _compute_penalties(
        self,
        new_chain: CoTChain,
        existing_chains: List[CoTChain]
    ) -> float:
        """Compute penalties for undesired behaviors."""
        penalty = 0.0
        
        # Repetition penalty - check if too similar to any existing chain
        if existing_chains:
            new_text = new_chain.get_text()
            for existing in existing_chains:
                existing_text = existing.get_text()
                similarity = 1.0 - self.diversity_metrics.compute_diversity(
                    new_text, existing_text
                )
                if similarity > 0.9:  # Very similar
                    penalty += self.config.repetition_penalty
        
        # Length penalty - penalize extremely long chains
        total_length = sum(len(step.content) for step in chain.steps)
        if total_length > 3000:
            penalty += self.config.length_penalty_coef * (total_length - 3000) / 1000
        
        return penalty
    
    def compute_batch_rewards(
        self,
        chains: List[CoTChain],
        subproblem_text: str
    ) -> List[float]:
        """Compute rewards for a batch of chains."""
        rewards = []
        
        for i, chain in enumerate(chains):
            existing = chains[:i]  # Chains generated before this one
            reward = self.compute_reward(chain, existing, subproblem_text)
            rewards.append(reward)
        
        return rewards