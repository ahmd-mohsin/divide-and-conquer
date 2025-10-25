# dataset_loader.py
"""Load decompositions from Module 1 for expansion in Module 2."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'division'))

import json
from pathlib import Path
from typing import List, Optional, Dict, Any
import random

from expansion_schemas import Decomposition, SubProblem
from utils import load_decomposition


class DecompositionLoader:
    """Load decompositions from Module 1 dataset."""
    
    def __init__(self, dataset_dir: str = "../division/data/decompositions"):
        self.dataset_dir = Path(dataset_dir)
        
        # Load index
        self.index_file = self.dataset_dir / "dataset_index.json"
        if not self.index_file.exists():
            raise FileNotFoundError(
                f"Dataset index not found: {self.index_file}\n"
                f"Run batch_decompose.py first to generate dataset."
            )
        
        with open(self.index_file, 'r') as f:
            self.index = json.load(f)
        
        # Load metadata
        self.metadata_file = self.dataset_dir / "dataset_metadata.json"
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        
        print(f"✓ Loaded dataset with {len(self.index)} decompositions")
        print(f"  Categories: {', '.join(self.metadata.get('categories', []))}")
        print(f"  Models: {', '.join(self.metadata.get('models_used', []))}")
    
    def get_decomposition(self, decomp_id: str) -> tuple[Decomposition, Dict[str, Any]]:
        """
        Get a single decomposition by ID.
        
        Returns:
            (decomposition, metadata_entry)
        """
        # Find in index
        entry = next((e for e in self.index if e["id"] == decomp_id), None)
        if not entry:
            raise ValueError(f"Decomposition {decomp_id} not found")
        
        # Load decomposition
        decomp_file = self.dataset_dir / entry["file"]
        decomp = load_decomposition(decomp_file)
        
        return decomp, entry
    
    def get_all_decompositions(self) -> List[tuple[Decomposition, Dict[str, Any]]]:
        """Get all decompositions."""
        results = []
        for entry in self.index:
            try:
                decomp_file = self.dataset_dir / entry["file"]
                decomp = load_decomposition(decomp_file)
                results.append((decomp, entry))
            except Exception as e:
                print(f"Warning: Could not load {entry['id']}: {e}")
        return results
    
    def get_by_category(self, category: str) -> List[tuple[Decomposition, Dict[str, Any]]]:
        """Get decompositions for a specific category."""
        filtered_entries = [e for e in self.index if e["category"] == category]
        
        results = []
        for entry in filtered_entries:
            try:
                decomp_file = self.dataset_dir / entry["file"]
                decomp = load_decomposition(decomp_file)
                results.append((decomp, entry))
            except Exception as e:
                print(f"Warning: Could not load {entry['id']}: {e}")
        
        return results
    
    def get_by_model(self, model: str) -> List[tuple[Decomposition, Dict[str, Any]]]:
        """Get decompositions from a specific model."""
        filtered_entries = [e for e in self.index if e["model"] == model]
        
        results = []
        for entry in filtered_entries:
            try:
                decomp_file = self.dataset_dir / entry["file"]
                decomp = load_decomposition(decomp_file)
                results.append((decomp, entry))
            except Exception as e:
                print(f"Warning: Could not load {entry['id']}: {e}")
        
        return results
    
    def get_sample(
        self,
        n: int = 10,
        category: Optional[str] = None,
        model: Optional[str] = None,
        seed: Optional[int] = None
    ) -> List[tuple[Decomposition, Dict[str, Any]]]:
        """
        Get a random sample of decompositions.
        
        Args:
            n: Number of samples
            category: Filter by category
            model: Filter by model
            seed: Random seed for reproducibility
            
        Returns:
            List of (decomposition, metadata) tuples
        """
        if seed is not None:
            random.seed(seed)
        
        # Filter entries
        entries = self.index
        if category:
            entries = [e for e in entries if e["category"] == category]
        if model:
            entries = [e for e in entries if e["model"] == model]
        
        # Sample
        n = min(n, len(entries))
        sampled_entries = random.sample(entries, n)
        
        # Load decompositions
        results = []
        for entry in sampled_entries:
            try:
                decomp_file = self.dataset_dir / entry["file"]
                decomp = load_decomposition(decomp_file)
                results.append((decomp, entry))
            except Exception as e:
                print(f"Warning: Could not load {entry['id']}: {e}")
        
        return results
    
    def get_all_subproblems(
        self,
        category: Optional[str] = None,
        max_decomps: Optional[int] = None
    ) -> List[SubProblem]:
        """
        Extract all sub-problems from decompositions.
        Useful for training RL agent.
        
        Args:
            category: Filter by category
            max_decomps: Maximum decompositions to process
            
        Returns:
            List of all sub-problems
        """
        entries = self.index
        if category:
            entries = [e for e in entries if e["category"] == category]
        if max_decomps:
            entries = entries[:max_decomps]
        
        all_subproblems = []
        
        for entry in entries:
            try:
                decomp_file = self.dataset_dir / entry["file"]
                decomp = load_decomposition(decomp_file)
                all_subproblems.extend(decomp.nodes)
            except Exception as e:
                print(f"Warning: Could not load {entry['id']}: {e}")
        
        return all_subproblems
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            "total_decompositions": len(self.index),
            "total_subproblems": sum(e["num_subproblems"] for e in self.index),
            "avg_subproblems": sum(e["num_subproblems"] for e in self.index) / len(self.index),
            "categories": {},
            "models": {},
        }
        
        for entry in self.index:
            # By category
            cat = entry["category"]
            if cat not in stats["categories"]:
                stats["categories"][cat] = 0
            stats["categories"][cat] += 1
            
            # By model
            model = entry["model"]
            if model not in stats["models"]:
                stats["models"][model] = 0
            stats["models"][model] += 1
        
        return stats