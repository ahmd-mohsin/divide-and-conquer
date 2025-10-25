# rl_environment.py
"""Gymnasium environment for CoT expansion with RL."""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'division'))

from expansion_schemas import CoTChain, RLState, RLAction
from expansion_config import ExpansionConfig
from cot_generator import CoTGenerator
from reward_model import RewardModel
from schemas import SubProblem  # From division module


class CoTExpansionEnv(gym.Env):
    """
    Gymnasium environment for Chain-of-Thought expansion.
    
    STATE: Current sub-problem + existing chains + diversity metrics
    ACTION: Choose which strategy to use for next chain
    REWARD: Diversity + quality + progress (from RewardModel)
    
    The agent learns which strategies to pick to maximize diverse, high-quality chains.
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, config: ExpansionConfig, subproblems: List[SubProblem]):
        super().__init__()
        
        self.config = config
        self.subproblems = subproblems
        self.current_subproblem_idx = 0
        self.current_subproblem: Optional[SubProblem] = None
        self.generated_chains: List[CoTChain] = []
        
        # Initialize components
        self.cot_generator = CoTGenerator(config.cot)
        self.reward_model = RewardModel(config.reward)
        
        # Action space: discrete choice of strategy
        self.num_strategies = len(config.rl.exploration_strategies)
        self.action_space = spaces.Discrete(self.num_strategies)
        
        # Observation space: encoded state
        # [subproblem_length, num_existing_chains, avg_chain_length, diversity_so_far]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Episode tracking
        self.episode_rewards = []
        self.episode_chains = []
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to a new sub-problem."""
        super().reset(seed=seed)
        
        # Select next sub-problem (cycle through them)
        if not self.subproblems:
            raise ValueError("No sub-problems provided!")
        
        self.current_subproblem_idx = (self.current_subproblem_idx + 1) % len(self.subproblems)
        self.current_subproblem = self.subproblems[self.current_subproblem_idx]
        
        # Reset chains
        self.generated_chains = []
        self.episode_rewards = []
        self.episode_chains = []
        
        # Get initial observation
        obs = self._get_observation()
        info = {"subproblem_id": self.current_subproblem.id}
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step: generate a chain using the chosen strategy.
        
        Args:
            action: Strategy index (0 to num_strategies-1)
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Map action to strategy
        strategy = self.config.rl.exploration_strategies[action]
        
        # Generate chain using this strategy
        chain = self.cot_generator.generate_chain(
            subproblem_text=self.current_subproblem.goal,
            strategy=strategy,
            temperature=self.config.cot.temperature,
            existing_chains=self.generated_chains
        )
        chain.subproblem_id = self.current_subproblem.id
        
        # Compute reward
        reward = self.reward_model.compute_reward(
            new_chain=chain,
            existing_chains=self.generated_chains,
            subproblem_text=self.current_subproblem.goal
        )
        
        # Store reward in chain
        chain.rl_reward = reward
        
        # Add to generated chains
        self.generated_chains.append(chain)
        self.episode_rewards.append(reward)
        self.episode_chains.append(chain)
        
        # Get new observation
        obs = self._get_observation()
        
        # Episode terminates when we've generated enough chains
        terminated = len(self.generated_chains) >= self.config.cot.num_chains
        truncated = False
        
        # Info
        info = {
            "strategy": strategy,
            "reward": reward,
            "num_chains": len(self.generated_chains),
            "avg_reward": np.mean(self.episode_rewards),
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Encode current state as observation vector."""
        if not self.current_subproblem:
            return np.zeros(4, dtype=np.float32)
        
        # Feature 1: Normalized sub-problem length
        subproblem_len = len(self.current_subproblem.goal) / 200.0
        subproblem_len = min(subproblem_len, 1.0)
        
        # Feature 2: Number of existing chains (normalized)
        num_chains = len(self.generated_chains) / self.config.cot.num_chains
        
        # Feature 3: Average chain length (normalized)
        if self.generated_chains:
            avg_length = np.mean([
                sum(len(step.content) for step in chain.steps)
                for chain in self.generated_chains
            ]) / 1000.0
            avg_length = min(avg_length, 1.0)
        else:
            avg_length = 0.0
        
        # Feature 4: Average diversity so far
        if len(self.generated_chains) >= 2:
            diversities = []
            for i in range(len(self.generated_chains)):
                for j in range(i+1, len(self.generated_chains)):
                    div = self.reward_model.diversity_metrics.compute_diversity(
                        self.generated_chains[i].get_text(),
                        self.generated_chains[j].get_text()
                    )
                    diversities.append(div)
            avg_diversity = np.mean(diversities)
        else:
            avg_diversity = 0.0
        
        obs = np.array([
            subproblem_len,
            num_chains,
            avg_length,
            avg_diversity
        ], dtype=np.float32)
        
        return obs
    
    def render(self):
        """Render current state (for debugging)."""
        if self.current_subproblem:
            print(f"\n{'='*60}")
            print(f"Sub-problem: {self.current_subproblem.goal[:80]}")
            print(f"Generated chains: {len(self.generated_chains)}/{self.config.cot.num_chains}")
            if self.episode_rewards:
                print(f"Avg reward: {np.mean(self.episode_rewards):.3f}")
            print(f"{'='*60}\n")
    
    def get_episode_chains(self) -> List[CoTChain]:
        """Get all chains generated in current episode."""
        return self.episode_chains.copy()