# modules/expansion/__init__.py
"""HCOT Module 2: Chain-of-Thought Expansion with Hierarchical RL."""

# Import from current package
from .expansion_config import ExpansionConfig, CoTConfig, RLConfig, RewardConfig
from .expansion_schemas import CoTChain, CoTStep, SubProblemExpansion, ExpansionResult
from .cot_generator import CoTGenerator
from .diversity_metrics import DiversityMetrics
from .reward_model import RewardModel
from .rl_environment import CoTExpansionEnv
from .rl_agent import HierarchicalRLAgent
from .dataset_loader import DecompositionLoader

__all__ = [
    'ExpansionConfig',
    'CoTConfig',
    'RLConfig',
    'RewardConfig',
    'CoTChain',
    'CoTStep',
    'SubProblemExpansion',
    'ExpansionResult',
    'CoTGenerator',
    'DiversityMetrics',
    'RewardModel',
    'CoTExpansionEnv',
    'HierarchicalRLAgent',
    'DecompositionLoader',
]