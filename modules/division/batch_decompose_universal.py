#!/usr/bin/env python3
"""Universal batch decomposition for multiple MATH datasets."""
import sys
import os

# Add current directory to path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from hcot_decomposer import quick_decompose
from utils import save_decomposition
from schemas import Decomposition
from data_loaders import create_loader, get_available_datasets
from pathlib import Path
import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime


class DecompositionDataset:
    """Manages a dataset of decomposed problems with enhanced metadata tracking."""
    
    def __init__(self, output_dir: str = "data/decompositions"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.decomp_dir = self.output_dir / "decompositions"
        self.decomp_dir.mkdir(exist_ok=True)
        
        self.metadata_file = self.output_dir / "dataset_metadata.json"
        self.index_file = self.output_dir / "dataset_index.json"
        
        self.metadata = self._load_or_create_metadata()
        self.index = self._load_or_create_index()
    
    def _load_or_create_metadata(self) -> Dict[str, Any]:
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        else:
            return {
                "created_at": datetime.now().isoformat(),
                "version": "2.0",  # Updated version for multi-dataset support
                "total_problems": 0,
                "total_decompositions": 0,
                "models_used": [],
                "categories": [],
                "datasets": [],
                "statistics": {}
            }
    
    def _load_or_create_index(self) -> List[Dict[str, Any]]:
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
        problem_metadata: Dict[str, Any],
        decomp_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a decomposition to the dataset with full metadata tracking."""
        decomp_id = f"decomp_{len(self.index):06d}"
        
        decomp_file = self.decomp_dir / f"{decomp_id}.json"
        save_decomposition(decomposition, decomp_file)
        
        # Merge problem metadata with decomposition metadata
        full_metadata = problem_metadata.copy()
        if decomp_metadata:
            full_metadata.update(decomp_metadata)
        
        entry = {
            "id": decomp_id,
            "problem": problem,
            "answer": answer,
            "model": model,
            "category": problem_metadata.get('type', 'Unknown'),
            "level": problem_metadata.get('level', 'Unknown'),
            "dataset": problem_metadata.get('dataset', 'Unknown'),
            "file": str(decomp_file.relative_to(self.output_dir)),
            "num_subproblems": len(decomposition.nodes),
            "depth": decomposition.depth_limit,
            "branching": decomposition.branching_limit,
            "created_at": datetime.now().isoformat(),
            "metadata": full_metadata
        }
        
        self.index.append(entry)
        
        # Update aggregate metadata
        self.metadata["total_decompositions"] += 1
        if model not in self.metadata["models_used"]:
            self.metadata["models_used"].append(model)
        
        category = problem_metadata.get('type', 'Unknown')
        if category not in self.metadata["categories"]:
            self.metadata["categories"].append(category)
        
        dataset = problem_metadata.get('dataset', 'Unknown')
        if dataset not in self.metadata["datasets"]:
            self.metadata["datasets"].append(dataset)
        
        # Save periodically
        if len(self.index) % 10 == 0:
            self.save()
        
        return decomp_id
    
    def save(self):
        """Save metadata and index to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Compute comprehensive statistics about the dataset."""
        if not self.index:
            return {}
        
        stats = {
            "total": len(self.index),
            "by_model": {},
            "by_category": {},
            "by_dataset": {},
            "by_level": {},
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
            
            # By dataset
            dataset = entry["dataset"]
            stats["by_dataset"][dataset] = stats["by_dataset"].get(dataset, 0) + 1
            
            # By level
            level = entry["level"]
            stats["by_level"][level] = stats["by_level"].get(level, 0) + 1
        
        return stats


def batch_decompose(
    problems: List[tuple],
    models: List[str],
    dataset: DecompositionDataset,
    depth: int = 3,
    branching: int = 3,
    verbose: bool = True,
    delay: float = 1.0,
    prompts_path: str = "hcot_prompts.json"
) -> Dict[str, Any]:
    """
    Decompose a batch of problems using specified models.
    
    Args:
        problems: List of (problem, answer, metadata) tuples
        models: List of model names to use
        dataset: DecompositionDataset to store results
        depth: Maximum decomposition depth
        branching: Maximum branching factor
        verbose: Print progress
        delay: Delay between requests (seconds)
        prompts_path: Path to prompts file
    
    Returns:
        Dictionary with statistics
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
        
        stats["by_model"][model] = {"attempted": 0, "successful": 0, "failed": 0}
        
        for i, (question, answer, metadata) in enumerate(problems, 1):
            current += 1
            stats["total_attempted"] += 1
            stats["by_model"][model]["attempted"] += 1
            
            if verbose:
                print(f"\n[{current}/{total}] Model: {model} | Problem {i}/{len(problems)}")
                print(f"Dataset: {metadata.get('dataset', 'Unknown')}")
                print(f"Category: {metadata.get('type', 'Unknown')} | Level: {metadata.get('level', 'Unknown')}")
                print(f"Question: {question[:80]}...")
            
            try:
                decomp = quick_decompose(
                    problem=question,
                    model=model,
                    prompts_path=prompts_path,
                    depth=depth,
                    branching=branching,
                    verbose=False
                )
                
                decomp_id = dataset.add_decomposition(
                    problem=question,
                    answer=answer,
                    decomposition=decomp,
                    model=model,
                    problem_metadata=metadata,
                    decomp_metadata={"depth": depth, "branching": branching}
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
                    "problem": question[:200],  # Truncate for logging
                    "metadata": metadata,
                    "error": str(e)
                })
                
                if verbose:
                    print(f"✗ Failed: {e}")
            
            time.sleep(delay)
    
    return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Universal batch decomposition for MATH datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Decompose 20 problems from Hendrycks MATH dataset
  python batch_decompose.py --dataset hendrycks_math --num-problems 20
  
  # Filter by category (Algebra only)
  python batch_decompose.py --dataset hendrycks_math --category Algebra --num-problems 10
  
  # Use multiple models
  python batch_decompose.py --dataset hendrycks_math --models llama3.1:latest qwen2.5:7b --num-problems 5
  
  # List available categories in a dataset
  python batch_decompose.py --dataset hendrycks_math --list-categories
        """
    )
    
    parser.add_argument("--dataset", default="hendrycks_math", 
                       help="Dataset to use (hendrycks_math, deepmind)")
    parser.add_argument("--split", default="train", 
                       help="Dataset split (train/test)")
    parser.add_argument("--category", default=None, 
                       help="Filter by category (dataset-specific)")
    parser.add_argument("--num-problems", type=int, default=10, 
                       help="Number of problems to process")
    parser.add_argument("--models", nargs="+", default=["llama3.1:latest"],
                       help="Models to use for decomposition")
    parser.add_argument("--output-dir", default="data/decompositions",
                       help="Output directory for decompositions")
    parser.add_argument("--prompts-path", default="hcot_prompts.json",
                       help="Path to prompts file")
    parser.add_argument("--depth", type=int, default=3,
                       help="Maximum decomposition depth")
    parser.add_argument("--branching", type=int, default=3,
                       help="Maximum branching factor")
    parser.add_argument("--delay", type=float, default=1.0,
                       help="Delay between API calls (seconds)")
    parser.add_argument("--list-categories", action="store_true",
                       help="List available categories and exit")
    parser.add_argument("--list-datasets", action="store_true",
                       help="List available datasets and exit")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Print detailed progress")
    
    args = parser.parse_args()
    
    # List datasets if requested
    if args.list_datasets:
        print("\nAvailable datasets:")
        for ds in get_available_datasets():
            print(f"  - {ds}")
        return
    
    print("\n" + "="*70)
    print("HCOT MODULE 1: Universal Batch Decomposition")
    print("="*70 + "\n")
    
    # Check Ollama availability (if using local models)
    try:
        import ollama
        ollama.list()
        print("✓ Ollama is running\n")
    except:
        print("⚠ Ollama not detected. Make sure it's running if using local models.")
        print("  Start with: ollama serve\n")
    
    # Load dataset
    print(f"Loading dataset: {args.dataset} (split: {args.split})")
    try:
        loader = create_loader(args.dataset, args.split)
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        print(f"\nAvailable datasets:")
        for ds in get_available_datasets():
            print(f"  - {ds}")
        return
    
    # List categories if requested
    if args.list_categories:
        try:
            categories = loader.get_categories()
            print(f"\nAvailable categories in {args.dataset}:")
            for cat in categories:
                print(f"  - {cat}")
        except Exception as e:
            print(f"✗ Error getting categories: {e}")
        return
    
    # Load problems
    print(f"Loading problems...")
    try:
        problems = loader.load(max_problems=args.num_problems, category=args.category)
        print(f"✓ Loaded {len(problems)} problems")
        
        if args.category:
            print(f"  Filtered by category: {args.category}")
    except Exception as e:
        print(f"✗ Error loading problems: {e}")
        return
    
    # Show sample
    if problems:
        print(f"\nSample problem:")
        sample_problem, sample_answer, sample_meta = problems[0]
        print(f"  Dataset: {sample_meta.get('dataset', 'Unknown')}")
        print(f"  Type: {sample_meta.get('type', 'Unknown')}")
        print(f"  Level: {sample_meta.get('level', 'Unknown')}")
        print(f"  Problem: {sample_problem[:100]}...")
        print(f"  Answer: {sample_answer[:50]}...\n")
    
    # Initialize dataset
    dataset = DecompositionDataset(output_dir=args.output_dir)
    
    # Run batch decomposition
    print(f"Starting batch decomposition...")
    print(f"  Models: {', '.join(args.models)}")
    print(f"  Depth: {args.depth}, Branching: {args.branching}")
    print(f"  Delay: {args.delay}s\n")
    
    start_time = time.time()
    stats = batch_decompose(
        problems, 
        args.models, 
        dataset, 
        args.depth, 
        args.branching,
        verbose=args.verbose,
        delay=args.delay,
        prompts_path=args.prompts_path
    )
    dataset.save()
    
    elapsed = time.time() - start_time
    
    # Print summary
    print("\n" + "="*70)
    print("BATCH DECOMPOSITION COMPLETE")
    print("="*70)
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"Success: {stats['total_successful']}/{stats['total_attempted']}")
    print(f"Failed: {stats['total_failed']}")
    
    print(f"\nBy model:")
    for model, model_stats in stats['by_model'].items():
        success_rate = (model_stats['successful'] / model_stats['attempted'] * 100) if model_stats['attempted'] > 0 else 0
        print(f"  {model}: {model_stats['successful']}/{model_stats['attempted']} ({success_rate:.1f}%)")
    
    # Show dataset statistics
    dataset_stats = dataset.get_statistics()
    if dataset_stats:
        print(f"\nDataset statistics:")
        print(f"  Total decompositions: {dataset_stats['total']}")
        print(f"  Average subproblems: {dataset_stats['avg_subproblems']:.2f}")
        
        if dataset_stats['by_dataset']:
            print(f"  Datasets: {', '.join(dataset_stats['by_dataset'].keys())}")
        
        if dataset_stats['by_category']:
            print(f"  Categories: {len(dataset_stats['by_category'])}")
            for cat, count in sorted(dataset_stats['by_category'].items(), key=lambda x: -x[1])[:5]:
                print(f"    - {cat}: {count}")
        
        if dataset_stats['by_level']:
            print(f"  Levels: {', '.join(sorted(dataset_stats['by_level'].keys()))}")
    
    # Save failure log if there were failures
    if stats['failures']:
        failure_log = args.output_dir / "failures.json"
        with open(failure_log, 'w') as f:
            json.dump(stats['failures'], f, indent=2)
        print(f"\n⚠ {len(stats['failures'])} failures logged to: {failure_log}")
    
    print(f"\n✓ Dataset saved to: {args.output_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()