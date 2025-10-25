#!/usr/bin/env python3
"""
Batch decomposition of math problems to create training dataset.
Processes DeepMind Math Dataset and stores decompositions.
"""
from hcot_decomposer import quick_decompose
from utils import save_decomposition, print_statistics
from schemas import Decomposition
from pathlib import Path
import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import random


class DecompositionDataset:
    """Manages a dataset of decomposed problems."""
    
    def __init__(self, output_dir: str = "data/decompositions"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.decomp_dir = self.output_dir / "decompositions"
        self.decomp_dir.mkdir(exist_ok=True)
        
        self.metadata_file = self.output_dir / "dataset_metadata.json"
        self.index_file = self.output_dir / "dataset_index.json"
        
        # Load or initialize metadata
        self.metadata = self._load_or_create_metadata()
        self.index = self._load_or_create_index()
    
    def _load_or_create_metadata(self) -> Dict[str, Any]:
        """Load existing metadata or create new."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        else:
            return {
                "created_at": datetime.now().isoformat(),
                "version": "1.0",
                "total_problems": 0,
                "total_decompositions": 0,
                "models_used": [],
                "categories": [],
                "statistics": {}
            }
    
    def _load_or_create_index(self) -> List[Dict[str, Any]]:
        """Load existing index or create new."""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                return json.load(f)
        else:
            return []
    
    def add_decomposition(
        self,
        problem: str,
        answer: str,
        decomposition: Decomposition,
        model: str,
        category: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a decomposition to the dataset.
        
        Returns:
            decomposition_id
        """
        # Generate unique ID
        decomp_id = f"decomp_{len(self.index):06d}"
        
        # Save decomposition
        decomp_file = self.decomp_dir / f"{decomp_id}.json"
        save_decomposition(decomposition, decomp_file)
        
        # Create index entry
        entry = {
            "id": decomp_id,
            "problem": problem,
            "answer": answer,
            "model": model,
            "category": category,
            "file": str(decomp_file.relative_to(self.output_dir)),
            "num_subproblems": len(decomposition.nodes),
            "depth": decomposition.depth_limit,
            "branching": decomposition.branching_limit,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.index.append(entry)
        
        # Update metadata
        self.metadata["total_decompositions"] += 1
        if model not in self.metadata["models_used"]:
            self.metadata["models_used"].append(model)
        if category not in self.metadata["categories"]:
            self.metadata["categories"].append(category)
        
        # Save periodically
        if len(self.index) % 10 == 0:
            self.save()
        
        return decomp_id
    
    def save(self):
        """Save metadata and index."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self.index:
            return {}
        
        stats = {
            "total": len(self.index),
            "by_model": {},
            "by_category": {},
            "avg_subproblems": sum(e["num_subproblems"] for e in self.index) / len(self.index),
            "avg_depth": sum(e["depth"] for e in self.index) / len(self.index),
        }
        
        for entry in self.index:
            # By model
            model = entry["model"]
            stats["by_model"][model] = stats["by_model"].get(model, 0) + 1
            
            # By category
            cat = entry["category"]
            stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1
        
        return stats


def load_math_problems(
    category: str = "algebra__linear_1d",
    max_problems: int = 100,
    local_path: Optional[str] = None
) -> List[tuple[str, str]]:
    """
    Load math problems from DeepMind dataset.
    
    Args:
        category: Problem category
        max_problems: Maximum number of problems to load
        local_path: Path to local dataset file
        
    Returns:
        List of (question, answer) tuples
    """
    # Try local file first
    if local_path and Path(local_path).exists():
        print(f"Loading from local file: {local_path}")
        examples = []
        with open(local_path, 'r') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 2):
                if len(examples) >= max_problems:
                    break
                if i+1 < len(lines):
                    question = lines[i].strip()
                    answer = lines[i+1].strip()
                    examples.append((question, answer))
        return examples
    
    # Hardcoded examples for different categories
    print(f"Using hardcoded examples for {category}")
    
    examples_by_category = {
        "algebra__linear_1d": [
            ("Solve -42*r + 27*c = -1167 and 130*r + 4*c = 372 for r.", "4"),
            ("Solve 3*x + 7 = 22 for x.", "5"),
            ("What is the value of y in 2*y - 5 = 11?", "8"),
            ("Solve -5*a + 3 = -17 for a.", "4"),
            ("Find x when 4*x + 6 = 2*x + 14.", "4"),
        ],
        "algebra__polynomial_roots": [
            ("Find the roots of x^2 - 5*x + 6 = 0.", "2, 3"),
            ("Solve x^2 + 4*x + 4 = 0.", "-2"),
            ("What are the solutions to x^2 - x - 6 = 0?", "-2, 3"),
        ],
        "arithmetic__add_sub": [
            ("Calculate 145 + 267.", "412"),
            ("What is 523 - 178?", "345"),
            ("Compute -45 + 73.", "28"),
        ],
        "calculus__differentiate": [
            ("What is the derivative of x^2 + 3*x + 5?", "2*x + 3"),
            ("Differentiate 3*x^3 - 2*x + 1.", "9*x^2 - 2"),
        ],
    }
    
    category_examples = examples_by_category.get(
        category,
        examples_by_category["algebra__linear_1d"]  # Default
    )
    
    # Repeat to reach max_problems
    result = []
    while len(result) < max_problems:
        result.extend(category_examples)
    
    return result[:max_problems]


def batch_decompose(
    problems: List[tuple[str, str]],
    models: List[str],
    category: str,
    dataset: DecompositionDataset,
    depth: int = 3,
    branching: int = 3,
    verbose: bool = True,
    delay: float = 1.0
) -> Dict[str, Any]:
    """
    Batch decompose problems using multiple models.
    
    Args:
        problems: List of (question, answer) tuples
        models: List of model names to use
        category: Problem category
        dataset: DecompositionDataset to store results
        depth: Maximum hierarchy depth
        branching: Maximum branching factor
        verbose: Print progress
        delay: Delay between API calls (seconds)
        
    Returns:
        Statistics dictionary
    """
    stats = {
        "total_attempted": 0,
        "total_successful": 0,
        "total_failed": 0,
        "by_model": {},
        "failures": []
    }
    
    total = len(problems) * len(models)
    current = 0
    
    for model in models:
        print(f"\n{'='*70}")
        print(f"Processing with model: {model}")
        print(f"{'='*70}\n")
        
        stats["by_model"][model] = {
            "attempted": 0,
            "successful": 0,
            "failed": 0
        }
        
        for i, (question, answer) in enumerate(problems, 1):
            current += 1
            stats["total_attempted"] += 1
            stats["by_model"][model]["attempted"] += 1
            
            if verbose:
                print(f"\n[{current}/{total}] Model: {model} | Problem {i}/{len(problems)}")
                print(f"Question: {question[:80]}...")
            
            try:
                # Decompose
                decomp = quick_decompose(
                    problem=question,
                    model=model,
                    prompts_path="hcot_prompts.json",
                    depth=depth,
                    branching=branching,
                    verbose=False
                )
                
                # Add to dataset
                decomp_id = dataset.add_decomposition(
                    problem=question,
                    answer=answer,
                    decomposition=decomp,
                    model=model,
                    category=category,
                    metadata={
                        "depth": depth,
                        "branching": branching
                    }
                )
                
                stats["total_successful"] += 1
                stats["by_model"][model]["successful"] += 1
                
                if verbose:
                    print(f"✓ Success: {decomp_id} ({len(decomp.nodes)} sub-problems)")
                
            except Exception as e:
                stats["total_failed"] += 1
                stats["by_model"][model]["failed"] += 1
                stats["failures"].append({
                    "model": model,
                    "problem": question,
                    "error": str(e)
                })
                
                if verbose:
                    print(f"✗ Failed: {e}")
            
            # Delay to avoid rate limiting
            time.sleep(delay)
    
    return stats


def main():
    """Main entry point for batch decomposition."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Batch decompose math problems to create dataset"
    )
    parser.add_argument(
        "--category",
        default="algebra__linear_1d",
        help="Problem category"
    )
    parser.add_argument(
        "--num-problems",
        type=int,
        default=50,
        help="Number of problems to process"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["llama3.1:latest", "qwen2.5:7b", "deepseek-r1:7b"],
        help="Models to use for decomposition"
    )
    parser.add_argument(
        "--output-dir",
        default="data/decompositions",
        help="Output directory for dataset"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Maximum decomposition depth"
    )
    parser.add_argument(
        "--branching",
        type=int,
        default=3,
        help="Maximum branching factor"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between API calls (seconds)"
    )
    parser.add_argument(
        "--local-path",
        help="Path to local DeepMind dataset file"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("HCOT MODULE 1: Batch Decomposition Dataset Generator")
    print("="*70)
    print(f"Category: {args.category}")
    print(f"Problems: {args.num_problems}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Output: {args.output_dir}")
    print(f"Total decompositions: {args.num_problems * len(args.models)}")
    print("="*70 + "\n")
    
    # Check Ollama
    try:
        import ollama
        ollama.list()
        print("✓ Ollama is running\n")
    except:
        print("✗ Ollama not running. Start with: ollama serve")
        return
    
    # Load problems
    print("Loading problems...")
    problems = load_math_problems(
        category=args.category,
        max_problems=args.num_problems,
        local_path=args.local_path
    )
    print(f"✓ Loaded {len(problems)} problems\n")
    
    # Create dataset
    dataset = DecompositionDataset(output_dir=args.output_dir)
    
    # Batch decompose
    start_time = time.time()
    
    stats = batch_decompose(
        problems=problems,
        models=args.models,
        category=args.category,
        dataset=dataset,
        depth=args.depth,
        branching=args.branching,
        delay=args.delay
    )
    
    # Save final dataset
    dataset.save()
    
    elapsed = time.time() - start_time
    
    # Print summary
    print("\n" + "="*70)
    print("BATCH DECOMPOSITION COMPLETE")
    print("="*70)
    print(f"Time elapsed: {elapsed:.1f}s")
    print(f"Total attempted: {stats['total_attempted']}")
    print(f"Total successful: {stats['total_successful']}")
    print(f"Total failed: {stats['total_failed']}")
    print(f"Success rate: {stats['total_successful']/stats['total_attempted']*100:.1f}%")
    
    print("\nBy model:")
    for model, model_stats in stats['by_model'].items():
        success_rate = model_stats['successful'] / model_stats['attempted'] * 100
        print(f"  {model}: {model_stats['successful']}/{model_stats['attempted']} ({success_rate:.1f}%)")
    
    # Dataset statistics
    dataset_stats = dataset.get_statistics()
    print(f"\nDataset statistics:")
    print(f"  Total decompositions: {dataset_stats['total']}")
    print(f"  Avg sub-problems: {dataset_stats['avg_subproblems']:.1f}")
    print(f"  Avg depth: {dataset_stats['avg_depth']:.1f}")
    
    print(f"\n✓ Dataset saved to: {args.output_dir}")
    print(f"  - Decompositions: {dataset.decomp_dir}")
    print(f"  - Index: {dataset.index_file}")
    print(f"  - Metadata: {dataset.metadata_file}")
    print("="*70 + "\n")
    
    # Save stats
    stats_file = Path(args.output_dir) / "batch_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Statistics saved to: {stats_file}\n")


if __name__ == "__main__":
    main()