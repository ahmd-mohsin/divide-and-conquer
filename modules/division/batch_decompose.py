#!/usr/bin/env python3
"""Batch decomposition of math problems to create training dataset."""
import sys
import os

# Add current directory to path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from hcot_decomposer import quick_decompose
from utils import save_decomposition
from schemas import Decomposition
from decomposition_scorer import score_decomposition
from pathlib import Path
import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime


class DecompositionDataset:
    """Manages a dataset of decomposed problems."""
    
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
                "version": "1.0",
                "total_problems": 0,
                "total_decompositions": 0,
                "models_used": [],
                "categories": [],
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
        category: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        decomp_id = f"decomp_{len(self.index):06d}"
        
        decomp_file = self.decomp_dir / f"{decomp_id}.json"
        save_decomposition(decomposition, decomp_file)
        
        # Score the decomposition
        scores = score_decomposition(decomposition, problem, category)
        
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
            "metadata": metadata or {},
            "quality_scores": scores  # Add quality metrics
        }
        
        self.index.append(entry)
        
        self.metadata["total_decompositions"] += 1
        if model not in self.metadata["models_used"]:
            self.metadata["models_used"].append(model)
        if category not in self.metadata["categories"]:
            self.metadata["categories"].append(category)
        
        if len(self.index) % 10 == 0:
            self.save()
        
        return decomp_id
    
    def save(self):
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def get_statistics(self) -> Dict[str, Any]:
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
            model = entry["model"]
            stats["by_model"][model] = stats["by_model"].get(model, 0) + 1
            
            cat = entry["category"]
            stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1
        
        return stats


def load_math_problems(
    category: str = "algebra__linear_1d",
    max_problems: int = 100,
    local_path: Optional[str] = None
) -> List[tuple[str, str]]:
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
    
    print(f"Using hardcoded examples for {category}")
    
    examples_by_category = {
        "algebra__linear_1d": [
            ("Solve -42*r + 27*c = -1167 and 130*r + 4*c = 372 for r.", "4"),
            ("Solve 3*x + 7 = 22 for x.", "5"),
            ("What is the value of y in 2*y - 5 = 11?", "8"),
            ("Solve -5*a + 3 = -17 for a.", "4"),
            ("Find x when 4*x + 6 = 2*x + 14.", "4"),
        ],
    }
    
    category_examples = examples_by_category.get(category, examples_by_category["algebra__linear_1d"])
    
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
        
        for i, (question, answer) in enumerate(problems, 1):
            current += 1
            stats["total_attempted"] += 1
            stats["by_model"][model]["attempted"] += 1
            
            if verbose:
                print(f"\n[{current}/{total}] Model: {model} | Problem {i}/{len(problems)}")
                print(f"Question: {question[:80]}...")
            
            try:
                decomp = quick_decompose(
                    problem=question,
                    model=model,
                    prompts_path="hcot_prompts.json",
                    depth=depth,
                    branching=branching,
                    verbose=False
                )
                
                decomp_id = dataset.add_decomposition(
                    problem=question,
                    answer=answer,
                    decomposition=decomp,
                    model=model,
                    category=category,
                    metadata={"depth": depth, "branching": branching}
                )
                
                stats["total_successful"] += 1
                stats["by_model"][model]["successful"] += 1
                
                if verbose:
                    print(f"✓ Success: {decomp_id} ({len(decomp.nodes)} sub-problems)")
                
            except Exception as e:
                stats["total_failed"] += 1
                stats["by_model"][model]["failed"] += 1
                stats["failures"].append({"model": model, "problem": question, "error": str(e)})
                
                if verbose:
                    print(f"✗ Failed: {e}")
            
            time.sleep(delay)
    
    return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch decompose math problems")
    parser.add_argument("--category", default="algebra__linear_1d")
    parser.add_argument("--num-problems", type=int, default=10)
    parser.add_argument("--models", nargs="+", default=["llama3.1:latest"])
    parser.add_argument("--output-dir", default="data/decompositions")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--branching", type=int, default=3)
    parser.add_argument("--delay", type=float, default=1.0)
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("HCOT MODULE 1: Batch Decomposition")
    print("="*70 + "\n")
    
    try:
        import ollama
        # Try to list models - this will fail if Ollama isn't running
        models = ollama.list()
        print("✓ Ollama is running\n")
    except Exception as e:
        # Check if it's a connection error specifically
        error_str = str(e).lower()
        if "connection" in error_str or "refused" in error_str:
            print("✗ Ollama not running. Start with: ollama serve")
            return
        else:
            # Ollama is running but returned an error - continue anyway
            print(f"⚠ Ollama check warning: {e}")
            print("Continuing anyway...\n")
    
    problems = load_math_problems(args.category, args.num_problems)
    print(f"✓ Loaded {len(problems)} problems\n")
    
    dataset = DecompositionDataset(output_dir=args.output_dir)
    
    start_time = time.time()
    stats = batch_decompose(problems, args.models, args.category, dataset, args.depth, args.branching, delay=args.delay)
    dataset.save()
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print("BATCH DECOMPOSITION COMPLETE")
    print("="*70)
    print(f"Time: {elapsed:.1f}s")
    print(f"Success: {stats['total_successful']}/{stats['total_attempted']}")
    print(f"✓ Dataset saved to: {args.output_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()