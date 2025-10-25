# modules/division/__init__.py
"""HCOT Module 1: Hierarchical Problem Decomposition.

Public API:
- Schemas:     Decomposition, SubProblem, CheckType
- Config:      ModelConfig, DecompositionConfig, HCOTConfig, get_model_config
- Decomposer:  HierarchicalDecomposer, quick_decompose
- I/O Utils:   save_decomposition, load_decomposition
- Batch:       batch_decompose (module namespace)
- LLM Clients: llm_clients (module namespace)
- Assets:      prompts_path() -> str (path to hcot_prompts.json)
"""

from __future__ import annotations

__version__ = "0.1.0"

# --- Schemas ---
from .schemas import Decomposition, SubProblem, CheckType

# --- Configs ---
from .config import (
    ModelConfig,
    DecompositionConfig,
    HCOTConfig,
    get_model_config,
)

# --- Core Decomposer ---
from .hcot_decomposer import HierarchicalDecomposer, quick_decompose

# --- I/O Utils ---
from .utils import save_decomposition, load_decomposition

# --- Namespaced helpers (keep as modules to avoid brittle re-exports) ---
from . import batch_decompose
from . import llm_clients

# --- Asset helper ---
def prompts_path() -> str:
    """Return filesystem path to the bundled 'hcot_prompts.json' file."""
    try:
        # Python 3.9+: use importlib.resources.files
        from importlib.resources import files
        return str(files(__package__) / "hcot_prompts.json")
    except Exception:
        # Fallback: derive relative to this file
        import os
        return os.path.join(os.path.dirname(__file__), "hcot_prompts.json")


__all__ = [
    # Schemas
    "Decomposition",
    "SubProblem",
    "CheckType",
    # Configs
    "ModelConfig",
    "DecompositionConfig",
    "HCOTConfig",
    "get_model_config",
    # Core
    "HierarchicalDecomposer",
    "quick_decompose",
    # I/O
    "save_decomposition",
    "load_decomposition",
    # Namespaced modules
    "batch_decompose",
    "llm_clients",
    # Assets
    "prompts_path",
]
