# modules/division/__init__.py
"""HCOT Module 1: Hierarchical Problem Decomposition"""

# Import key classes for easier access
from .schemas import Decomposition, SubProblem, CheckType
from .config import ModelConfig, DecomposerConfig
from .hcot_decomposer import HierarchicalDecomposer, quick_decompose
from .utils import save_decomposition, load_decomposition

__all__ = [
    'Decomposition',
    'SubProblem',
    'CheckType',
    'ModelConfig',
    'DecomposerConfig',
    'HierarchicalDecomposer',
    'quick_decompose',
    'save_decomposition',
    'load_decomposition',
]