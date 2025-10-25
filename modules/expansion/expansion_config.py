# expansion_config.py
"""Configuration for CoT expansion and RL training."""
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class CoTConfig:
    """Configuration for Chain-of-Thought expansion."""
    num_chains: int = 5                    # Number of chains per sub-problem
    max_steps: int = 10                    # Max steps per chain
    temperature: float = 0.7               # Higher for diversity
    model: str = "deepseek-r1:7b"         # Reasoning model
    min_step_length: int = 20              # Min chars per step
    max_step_length: int = 500             # Max chars per step


@dataclass
class RLConfig:
    """Configuration for RL agent."""
    # PPO hyperparameters
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01                # Entropy coefficient for exploration
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Training
    total_timesteps: int = 100000
    save_freq: int = 10000
    
    # Exploration strategies (action space)
    exploration_strategies: List[str] = field(default_factory=lambda: [
        "step_by_step",           # Standard step-by-step reasoning
        "backwards",              # Work backwards from goal
        "analogical",             # Use analogies
        "case_analysis",          # Break into cases
        "contradiction",          # Proof by contradiction
        "visual",                 # Visual/spatial reasoning
        "algebraic",              # Pure algebraic manipulation
        "numerical",              # Numerical/computational approach
    ])


@dataclass
class RewardConfig:
    """Configuration for reward computation."""
    diversity_weight: float = 0.5         # Weight for diversity
    quality_weight: float = 0.3           # Weight for quality
    progress_weight: float = 0.2          # Weight for progress
    
    # Diversity metrics
    use_embedding_similarity: bool = True  # Use sentence embeddings
    use_ngram_diversity: bool = True       # Use n-gram overlap
    use_structural_diversity: bool = True  # Use step structure
    
    # Quality metrics
    check_coherence: bool = True
    check_correctness: bool = False        # Requires ground truth
    
    # Penalties
    repetition_penalty: float = -0.1
    length_penalty_coef: float = 0.01


@dataclass
class ExpansionConfig:
    """Main configuration for expansion module."""
    cot: CoTConfig = field(default_factory=CoTConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    
    # Integration with division module
    division_module_path: str = "../division"
    
    # Paths
    model_save_path: str = "models/rl_agent"
    results_path: str = "results"
    
    verbose: bool = True