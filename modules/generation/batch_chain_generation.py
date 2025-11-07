#!/usr/bin/env python3
"""
Batch Chain Generation
Processes math datasets through decomposition and chain expansion pipeline with proper ground truth handling.
"""
import sys
import os
from pathlib import Path
import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import from division module for decomposition
division_path = os.path.join(parent_dir, 'division')
if division_path not in sys.path:
    sys.path.insert(0, division_path)

from division.hcot_decomposer import quick_decompose

# Import math data loaders from generation/math module
from math_data_loaders import create_math_loader

# Import from generation module (current directory)
from chain_expander import HierarchicalChainExecutor, expand_decomposition_hierarchical


class ChainDataset:
    """Manages dataset of problems with decompositions and chains."""
    
    def __init__(self, output_dir: str = "data/chains"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.chains_dir = self.output_dir / "chains"
        self.chains_dir.mkdir(exist_ok=True)
        
        self.metadata_file = self.output_dir / "chain_dataset_metadata.json"
        self.index_file = self.output_dir / "chain_dataset_index.json"
        
        self.metadata = self._load_or_create_metadata()
        self.index = self._load_or_create_index()
    
    def _load_or_create_metadata(self) -> Dict[str, Any]:
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            "created_at": datetime.now().isoformat(),
            "version": "1.0",
            "total_problems": 0,
            "total_subproblems": 0,
            "total_chains": 0,
            "datasets": [],
            "models": [],
            "statistics": {}
        }
    
    def _load_or_create_index(self) -> List[Dict[str, Any]]:
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                return json.load(f)
        return []
    
    def add_problem_chains(
        self,
        problem: str,
        ground_truth: str,
        chains_data: Dict[str, Any],
        dataset_name: str,
        model: str,
        problem_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add chains for a problem to the dataset."""
        
        problem_id = f"problem_{len(self.index):06d}"
        
        # Save chains to file
        chains_file = self.chains_dir / f"{problem_id}.json"
        with open(chains_file, 'w') as f:
            json.dump(chains_data, f, indent=2)
        
        # Create index entry
        entry = {
            "id": problem_id,
            "problem": problem[:500],  # Truncate long problems
            "ground_truth": ground_truth,
            "dataset": dataset_name,
            "model": model,
            "file": str(chains_file.relative_to(self.output_dir)),
            "num_subproblems": chains_data.get('num_subproblems', 0),
            "num_chains": chains_data.get('statistics', {}).get('total_chains', 0),
            "avg_reward": chains_data.get('statistics', {}).get('avg_reward', 0.0),
            "success_rate": chains_data.get('statistics', {}).get('success_rate', 0.0),
            "created_at": datetime.now().isoformat(),
            "metadata": problem_metadata or {}
        }
        
        self.index.append(entry)
        
        # Update metadata
        self.metadata["total_problems"] += 1
        self.metadata["total_subproblems"] += entry["num_subproblems"]
        self.metadata["total_chains"] += entry["num_chains"]
        
        if dataset_name not in self.metadata["datasets"]:
            self.metadata["datasets"].append(dataset_name)
        if model not in self.metadata["models"]:
            self.metadata["models"].append(model)
        
        # Save periodically
        if len(self.index) % 10 == 0:
            self.save()
        
        return problem_id
    
    def save(self):
        """Save metadata and index."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Compute dataset statistics."""
        if not self.index:
            return {}
        
        stats = {
            "total_problems": len(self.index),
            "total_subproblems": self.metadata["total_subproblems"],
            "total_chains": self.metadata["total_chains"],
            "by_dataset": {},
            "by_model": {},
            "avg_chains_per_problem": self.metadata["total_chains"] / len(self.index) if self.index else 0,
            "avg_reward": sum(e["avg_reward"] for e in self.index) / len(self.index) if self.index else 0,
            "avg_success_rate": sum(e["success_rate"] for e in self.index) / len(self.index) if self.index else 0
        }
        
        for entry in self.index:
            dataset = entry["dataset"]
            stats["by_dataset"][dataset] = stats["by_dataset"].get(dataset, 0) + 1
            
            model = entry["model"]
            stats["by_model"][model] = stats["by_model"].get(model, 0) + 1
        
        return stats


def batch_generate_chains(
    dataset_name: str,
    num_problems: int = 10,
    decompose_model: str = "llama3.1:latest",
    generation_model: str = "llama3.1:latest",
    num_chains: int = 5,
    temperature: float = 0.8,
    max_tokens: int = 4096,
    output_dir: str = "data/chains",
    depth: int = 3,
    branching: int = 3,
    verbose: bool = True,
    delay: float = 1.0
) -> Dict[str, Any]:
    """
    Main pipeline: Load dataset -> Decompose -> Generate chains -> Store.
    
    Args:
        dataset_name: Dataset to use ('gsm8k', 'calc-svamp', 'hendrycks')
        num_problems: Number of problems to process
        decompose_model: Model for problem decomposition
        generation_model: Model for chain generation
        num_chains: Number of chains per problem
        temperature: Sampling temperature
        max_tokens: Maximum tokens per generation step
        output_dir: Output directory
        depth: Decomposition depth
        branching: Decomposition branching factor
        verbose: Print progress
        delay: Delay between problems (seconds)
    
    Returns:
        Statistics dictionary
    """
    
    stats = {
        "total_attempted": 0,
        "total_successful": 0,
        "total_failed": 0,
        "total_chains_generated": 0,
        "failures": []
    }
    
    # Load dataset with proper ground truth extraction
    if verbose:
        print(f"\n{'='*70}")
        print(f"LOADING DATASET: {dataset_name}")
        print(f"{'='*70}\n")
    
    loader = create_math_loader(dataset_name)
    problems = loader.load(max_problems=num_problems)
    
    if verbose:
        print(f"✓ Loaded {len(problems)} problems")
        # Verify ground truths are loaded
        empty_gt_count = sum(1 for p, gt, m in problems if not gt or gt == "")
        if empty_gt_count > 0:
            print(f"⚠ WARNING: {empty_gt_count} problems have empty ground truth!")
        else:
            print(f"✓ All problems have valid ground truth")
    
    # Initialize hierarchical chain executor
    if verbose:
        print(f"\n{'='*70}")
        print(f"INITIALIZING HIERARCHICAL CHAIN EXECUTOR")
        print(f"{'='*70}\n")
    
    executor = HierarchicalChainExecutor(
        model=generation_model,
        num_chains=num_chains,
        temperature=temperature,
        max_tokens=max_tokens,
        verbose=verbose
    )
    
    # Initialize dataset
    chain_dataset = ChainDataset(output_dir=output_dir)
    
    # Process each problem
    if verbose:
        print(f"\n{'='*70}")
        print(f"PROCESSING PROBLEMS")
        print(f"{'='*70}\n")
    
    for i, (problem, ground_truth, metadata) in enumerate(problems, 1):
        stats["total_attempted"] += 1
        
        if verbose:
            print(f"\n[{i}/{len(problems)}] Processing problem")
            print(f"  Dataset: {metadata.get('dataset', 'Unknown')}")
            print(f"  Type: {metadata.get('type', 'Unknown')}")
            print(f"  Difficulty: {metadata.get('difficulty', 'Unknown')}")
            print(f"  Problem: {problem[:80]}...")
            print(f"  Ground Truth: {ground_truth}")
        
        # Verify ground truth is present
        if not ground_truth or ground_truth == "":
            print(f"  ⚠ WARNING: Empty ground truth - skipping problem")
            stats["total_failed"] += 1
            stats["failures"].append({
                "problem": problem[:100],
                "error": "Empty ground truth",
                "metadata": metadata
            })
            continue
        
        try:
            # Step 1: Decompose problem
            if verbose:
                print(f"\n  Step 1: Decomposing problem...")
            
            # Find prompts file
            prompts_file = "hcot_prompts.json"
            if not os.path.exists(prompts_file):
                # Try division directory
                division_prompts = os.path.join(division_path, "hcot_prompts.json")
                if os.path.exists(division_prompts):
                    prompts_file = division_prompts
            
            decomposition = quick_decompose(
                problem=problem,
                model=decompose_model,
                prompts_path=prompts_file,
                depth=depth,
                branching=branching,
                verbose=False
            )
            
            if verbose:
                print(f"    ✓ Created {len(decomposition.nodes)} subproblems")
            
            # Step 2: Generate chains with hierarchical execution
            if verbose:
                print(f"\n  Step 2: Generating chains with hierarchical execution...")
            
            chains_data = expand_decomposition_hierarchical(
                decomposition=decomposition,
                problem=problem,
                ground_truth=ground_truth,
                executor=executor,
                verbose=False
            )
            
            num_chains_generated = chains_data['statistics']['total_chains']
            success_rate = chains_data['statistics']['success_rate']
            avg_reward = chains_data['statistics']['avg_reward']
            
            if verbose:
                print(f"    ✓ Generated {num_chains_generated} chains")
                print(f"    Success rate: {success_rate:.1%}")
                print(f"    Avg reward: {avg_reward:.3f}")
            
            # Step 3: Store in dataset
            problem_id = chain_dataset.add_problem_chains(
                problem=problem,
                ground_truth=ground_truth,
                chains_data=chains_data,
                dataset_name=dataset_name,
                model=generation_model,
                problem_metadata=metadata
            )
            
            stats["total_successful"] += 1
            stats["total_chains_generated"] += num_chains_generated
            
            if verbose:
                print(f"\n  ✓ Saved as {problem_id}")
        
        except Exception as e:
            stats["total_failed"] += 1
            stats["failures"].append({
                "problem": problem[:100],
                "error": str(e),
                "metadata": metadata
            })
            
            if verbose:
                print(f"\n  ✗ Failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Delay between problems
        if i < len(problems):
            time.sleep(delay)
    
    # Save final dataset
    chain_dataset.save()
    
    return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate chains for math problems with hierarchical decomposition"
    )
    
    parser.add_argument("--dataset", default="gsm8k",
                       help="Dataset: gsm8k, calc-svamp, hendrycks")
    parser.add_argument("--num-problems", type=int, default=10,
                       help="Number of problems")
    parser.add_argument("--decompose-model", default="llama3.1:latest",
                       help="Model for decomposition")
    parser.add_argument("--generation-model", default="llama3.1:latest",
                       help="Model for generation")
    parser.add_argument("--num-chains", type=int, default=5,
                       help="Chains per problem")
    parser.add_argument("--temperature", type=float, default=0.6,
                       help="Temperature")
    parser.add_argument("--max-tokens", type=int, default=4096,
                       help="Max tokens")
    parser.add_argument("--output-dir", default="data/chains",
                       help="Output directory")
    parser.add_argument("--depth", type=int, default=3,
                       help="Decomposition depth")
    parser.add_argument("--branching", type=int, default=3,
                       help="Decomposition branching")
    parser.add_argument("--delay", type=float, default=1.0,
                       help="Delay between problems")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress output")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("HIERARCHICAL CHAIN GENERATION WITH PROPER GROUND TRUTH")
    print("="*70 + "\n")
    
    # Check Ollama
    try:
        import ollama
        ollama.list()
        print("✓ Ollama is running\n")
    except:
        print("✗ Ollama not running. Start with: ollama serve")
        return 1
    
    start_time = time.time()
    
    stats = batch_generate_chains(
        dataset_name=args.dataset,
        num_problems=args.num_problems,
        decompose_model=args.decompose_model,
        generation_model=args.generation_model,
        num_chains=args.num_chains,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        output_dir=args.output_dir,
        depth=args.depth,
        branching=args.branching,
        verbose=not args.quiet,
        delay=args.delay
    )
    
    elapsed = time.time() - start_time
    
    # Print summary
    print("\n" + "="*70)
    print("CHAIN GENERATION COMPLETE")
    print("="*70)
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"Success: {stats['total_successful']}/{stats['total_attempted']}")
    print(f"Failed: {stats['total_failed']}")
    print(f"Total chains: {stats['total_chains_generated']}")
    
    if stats['failures']:
        print(f"\n⚠ {len(stats['failures'])} problems failed")
        failure_file = Path(args.output_dir) / "failures.json"
        with open(failure_file, 'w') as f:
            json.dump(stats['failures'], f, indent=2)
        print(f"  Failures logged to: {failure_file}")
    
    print(f"\n✓ Dataset saved to: {args.output_dir}")
    print("="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())