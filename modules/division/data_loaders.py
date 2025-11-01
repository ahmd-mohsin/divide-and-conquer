#!/usr/bin/env python3
"""Data loaders for various MATH datasets."""
import sys
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


class MATHDatasetLoader:
    """Base class for loading MATH datasets."""
    
    def __init__(self, dataset_name: str, split: str = "train"):
        self.dataset_name = dataset_name
        self.split = split
        self._dataset = None
    
    def load(self, max_problems: Optional[int] = None, category: Optional[str] = None) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Load problems from the dataset.
        
        Returns:
            List of (problem, answer, metadata) tuples
        """
        raise NotImplementedError
    
    def get_categories(self) -> List[str]:
        """Get list of available categories/types in the dataset."""
        raise NotImplementedError


class HendrycksCompetitionMathLoader(MATHDatasetLoader):
    """Loader for qwedsacf/competition_math (Hendrycks MATH dataset)."""
    
    def __init__(self, split: str = "train"):
        super().__init__("qwedsacf/competition_math", split)
        self._load_dataset()
    
    def _load_dataset(self):
        """Load the dataset from Hugging Face."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets package not found. Install with: pip install datasets --break-system-packages"
            )
        
        print(f"Loading {self.dataset_name} dataset...")
        self._dataset = load_dataset(self.dataset_name, split=self.split)
        print(f"✓ Loaded {len(self._dataset)} problems")
    
    def load(self, max_problems: Optional[int] = None, category: Optional[str] = None) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Load problems from Hendrycks MATH dataset.
        
        Args:
            max_problems: Maximum number of problems to load
            category: Filter by problem type (e.g., 'Algebra', 'Geometry')
        
        Returns:
            List of (problem, answer, metadata) tuples
        """
        if self._dataset is None:
            self._load_dataset()
        
        problems = []
        
        for i, item in enumerate(self._dataset):
            if category and item.get('type') != category:
                continue
            
            problem = item['problem']
            solution = item['solution']
            
            answer = self._extract_answer(solution)
            
            metadata = {
                'type': item.get('type', 'Unknown'),
                'level': item.get('level', 'Unknown'),
                'full_solution': solution,
                'dataset': self.dataset_name,
                'index': i
            }
            
            problems.append((problem, answer, metadata))
            
            if max_problems and len(problems) >= max_problems:
                break
        
        return problems
    
    def _extract_answer(self, solution: str) -> str:
        """Extract the final answer from a solution string."""
        # Look for \boxed{answer}
        import re
        match = re.search(r'\\boxed\{(.+?)\}', solution)
        if match:
            return match.group(1)
        
        # If no boxed answer found, return a placeholder
        return "[Answer not found in solution]"
    
    def get_categories(self) -> List[str]:
        """Get list of problem types in the dataset."""
        if self._dataset is None:
            self._load_dataset()
        
        types = set()
        for item in self._dataset:
            if 'type' in item:
                types.add(item['type'])
        
        return sorted(list(types))


class DeepMindMathLoader(MATHDatasetLoader):
    """Loader for DeepMind Mathematics dataset (backward compatibility)."""
    
    def __init__(self, split: str = "train"):
        super().__init__("deepmind/mathematics", split)
    
    def load(self, max_problems: Optional[int] = None, category: Optional[str] = None) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Load problems from local file or hardcoded examples."""
        # This maintains backward compatibility with the original batch_decompose.py
        examples_by_category = {
            "algebra__linear_1d": [
                ("Solve -42*r + 27*c = -1167 and 130*r + 4*c = 372 for r.", "4"),
                ("Solve 3*x + 7 = 22 for x.", "5"),
                ("What is the value of y in 2*y - 5 = 11?", "8"),
                ("Solve -5*a + 3 = -17 for a.", "4"),
                ("Find x when 4*x + 6 = 2*x + 14.", "4"),
            ],
        }
        
        category = category or "algebra__linear_1d"
        category_examples = examples_by_category.get(category, examples_by_category["algebra__linear_1d"])
        
        result = []
        idx = 0
        while len(result) < (max_problems or len(category_examples)):
            problem, answer = category_examples[idx % len(category_examples)]
            metadata = {
                'type': category,
                'level': 'Unknown',
                'dataset': 'deepmind/mathematics',
                'index': idx
            }
            result.append((problem, answer, metadata))
            idx += 1
            
            if idx >= (max_problems or len(category_examples)):
                break
        
        return result
    
    def get_categories(self) -> List[str]:
        """Get list of available categories."""
        return ["algebra__linear_1d"]


def create_loader(dataset_name: str, split: str = "train") -> MATHDatasetLoader:
    """
    Factory function to create appropriate data loader.
    
    Args:
        dataset_name: Name of the dataset ('hendrycks_math', 'deepmind', etc.)
        split: Dataset split (usually 'train' or 'test')
    
    Returns:
        MATHDatasetLoader instance
    """
    dataset_name_lower = dataset_name.lower()
    
    if 'hendrycks' in dataset_name_lower or 'competition_math' in dataset_name_lower:
        return HendrycksCompetitionMathLoader(split)
    elif 'deepmind' in dataset_name_lower:
        return DeepMindMathLoader(split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: 'hendrycks_math', 'deepmind'")


def get_available_datasets() -> List[str]:
    """Get list of available dataset loaders."""
    return [
        "hendrycks_math (qwedsacf/competition_math)",
        "deepmind (local/hardcoded)",
    ]