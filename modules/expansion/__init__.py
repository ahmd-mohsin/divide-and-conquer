# modules/expansion/__init__.py
"""HCOT Module 2: Chain-of-Thought Expansion with Hierarchical RL.

Public API:
- Configs:     ExpansionConfig, CoTConfig, RLConfig, RewardConfig
- Schemas:     CoTChain, CoTStep, RLState, RLAction, SubProblemExpansion, ExpansionResult
- Components:  CoTGenerator, DiversityMetrics, RewardModel
- RL:          CoTExpansionEnv, HierarchicalRLAgent
- Data I/O:    DecompositionLoader
- Cross-types: SubProblem, Decomposition (from modules.division.schemas)
- Utils:       utils (module namespace)
"""

from __future__ import annotations

__version__ = "0.1.0"

# --- Configs ---
from .expansion_config import (
    ExpansionConfig,
    CoTConfig,
    RLConfig,
    RewardConfig,
)

# --- Schemas / types ---
from .expansion_schemas import (
    CoTChain,
    CoTStep,
    RLState,
    RLAction,
    SubProblemExpansion,
    ExpansionResult,
)

# --- Components ---
from .cot_generator import CoTGenerator
from .diversity_metrics import DiversityMetrics
from .reward_model import RewardModel

# --- RL ---
from .rl_environment import CoTExpansionEnv
from .rl_agent import HierarchicalRLAgent

# --- Data I/O ---
from .dataset_loader import DecompositionLoader

# --- Cross-package types (imported for convenience) ---
from ..division.schemas import SubProblem, Decom
