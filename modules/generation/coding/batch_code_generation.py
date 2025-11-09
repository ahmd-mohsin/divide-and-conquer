#!/usr/bin/env python3
"""
Batch Code Generation
Processes coding datasets through decomposition and chain expansion with code rewards.
"""
import sys
import os
from pathlib import Path
import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
generation_dir = os.path.dirname(current_dir)  # modules/generation
modules_dir = os.path.dirname(generation_dir)  # modules
division_dir = os.path.join(modules_dir, "division")  # modules/division

# Add to path
for path in [generation_dir, modules_dir, division_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Import division modules
from hcot_decomposer import quick_decompose

# Import local modules
from code_data_loaders import create_code_loader
from code_chain_expander import CodeChainExecutor, expand_code_decomposition
from code_reward_metrics import CodeRewardCalculator


class CodeChainDataset:
    """Manages dataset of code problems with decompositions and chains."""
    
    def __init__(self, output_dir: str = "data/code_chains"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.chains_dir = self.output_dir / "chains"
        self.chains_dir.mkdir(exist_ok=True)
        
        self.metadata_file = self.output_dir / "code_chain_metadata.json"
        self.index_file = self.output_dir / "code_chain_index.json"
        
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
        """Add chains for a code problem to the dataset."""
        
        problem_id = f"code_problem_{len(self.index):06d}"
        
        # Save chains to file
        chains_file = self.chains_dir / f"{problem_id}.json"
        with open(chains_file, 'w') as f:
            json.dump(chains_data, f, indent=2)
        
        # Create index entry
        entry = {
            "id": problem_id,
            "problem": problem[:500],  # Truncate long problems
            "ground_truth": ground_truth[:500],  # Truncate
            "dataset": dataset_name,
            "model": model,
            "file": str(chains_file.relative_to(self.output_dir)),
            "num_subproblems": chains_data['decomposition']['num_subproblems'],
            "num_chains": chains_data['statistics']['total_chains'],
            "avg_reward": chains_data['statistics']['avg_reward'],
            "success_rate": chains_data['statistics']['success_rate'],
            "avg_codebleu": chains_data['statistics']['avg_codebleu'],
            "avg_ast": chains_data['statistics']['avg_ast_similarity'],
            "avg_codebert": chains_data['statistics']['avg_codebert_score'],
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
        if len(self.index) % 5 == 0:
            self.save()
        
        return problem_id
    
    def save(self):
        """Save metadata and index."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)


# ---------------------- NEW: resume helper ----------------------
def compute_resume_offset(output_dir: str, dataset_name: str, generation_model: str) -> int:
    """
    Read the existing index file and count how many entries were saved
    for the given dataset+model. That count is the start_from offset.
    """
    idx_path = Path(output_dir) / "code_chain_index.json"
    if not idx_path.exists():
        return 0
    try:
        with open(idx_path, "r") as f:
            idx = json.load(f)
    except Exception:
        return 0

    count = 0
    for e in idx:
        if e.get("dataset") == dataset_name and e.get("model") == generation_model:
            count += 1
    return count
# ---------------------------------------------------------------


def batch_generate_code_chains(
    dataset_name: str,
    num_problems: int = 10,
    decompose_model: str = "qwen2.5-coder:7b",
    generation_model: str = "qwen2.5-coder:7b",
    num_chains: int = 5,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    reward_threshold: float = 0.7,
    output_dir: str = "data/code_chains",
    depth: int = 3,
    branching: int = 3,
    verbose: bool = True,
    delay: float = 1.0,
    start_from: int = 0,
    resume: bool = False   # NEW
) -> Dict[str, Any]:
    """
    Main pipeline for code problems: Load -> Decompose -> Generate -> Reward.
    
    Args:
        dataset_name: Dataset ('apps', 'ds1000', 'codeforces')
        num_problems: Number of problems
        decompose_model: Model for decomposition
        generation_model: Model for generation
        num_chains: Chains per problem
        temperature: Sampling temperature
        max_tokens: Max tokens per step
        reward_threshold: Success threshold
        output_dir: Output directory
        depth: Decomposition depth
        branching: Decomposition branching
        verbose: Print progress
        delay: Delay between problems (seconds)
        start_from: Start from problem N (skip first N problems)
        resume: If True, compute start_from from index for this dataset+model
    
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
    
    # Load dataset
    if verbose:
        print(f"\n{'='*70}")
        print(f"LOADING DATASET: {dataset_name}")
        print(f"{'='*70}\n")
    
    loader = create_code_loader(dataset_name)

    # NEW: auto-resume logic
    if resume:
        auto_offset = compute_resume_offset(output_dir, dataset_name, generation_model)
        if auto_offset > 0:
            if verbose:
                print(f"↻ Resume enabled: found {auto_offset} previously saved problems "
                      f"for dataset='{dataset_name}', model='{generation_model}'")
            start_from = max(start_from, auto_offset)

    # Load problems with offset support
    all_problems = loader.load(max_problems=num_problems + start_from)
    problems = all_problems[start_from:] if start_from > 0 else all_problems
    
    if verbose:
        if start_from > 0:
            print(f"✓ Loaded {len(all_problems)} problems (starting from problem {start_from + 1})")
            print(f"✓ Processing {len(problems)} problems")
        else:
            print(f"✓ Loaded {len(problems)} problems")
    
    # Initialize reward calculator
    reward_calculator = CodeRewardCalculator(
        use_codebleu=True,
        use_ast_similarity=True,
        use_codebert=True,
        success_threshold=reward_threshold,
        verbose=verbose
    )
    
    # Initialize chain executor
    if verbose:
        print(f"\n{'='*70}")
        print(f"INITIALIZING CODE CHAIN EXECUTOR")
        print(f"{'='*70}\n")
    
    executor = CodeChainExecutor(
        model=generation_model,
        num_chains=num_chains,
        temperature=temperature,
        max_tokens=max_tokens,
        reward_calculator=reward_calculator,
        verbose=verbose
    )
    
    # Initialize dataset
    code_dataset = CodeChainDataset(output_dir=output_dir)
    
    # Process each problem
    if verbose:
        print(f"\n{'='*70}")
        print(f"PROCESSING CODE PROBLEMS")
        print(f"{'='*70}\n")
    
    for i, (problem, solution, metadata) in enumerate(problems, 1):
        stats["total_attempted"] += 1
        
        actual_problem_num = i + start_from
        
        if verbose:
            print(f"\n[{actual_problem_num}/{num_problems + start_from}] Processing problem")
            print(f"  Dataset: {metadata.get('dataset', 'Unknown')}")
            print(f"  Type: {metadata.get('type', 'Unknown')}")
            print(f"  Difficulty: {metadata.get('difficulty', 'Unknown')}")
            print(f"  Problem: {problem[:80]}...")
        
        try:
            # Step 1: Decompose problem
            if verbose:
                print(f"\n  Step 1: Decomposing problem...")
            
            prompts_file = "hcot_prompts.json"
            if not os.path.exists(prompts_file):
                division_prompts = os.path.join(division_dir, "hcot_prompts.json")  
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
            
            # Step 2: Generate chains with code rewards
            if verbose:
                print(f"\n  Step 2: Generating chains with code rewards...")
            
            chains_data = expand_code_decomposition(
                decomposition=decomposition,
                problem=problem,
                ground_truth=solution,
                executor=executor,
                language=metadata.get('language', 'python'),
                verbose=False
            )
            
            num_chains_generated = chains_data['statistics']['total_chains']
            success_rate = chains_data['statistics']['success_rate']
            avg_reward = chains_data['statistics']['avg_reward']
            
            if verbose:
                print(f"    ✓ Generated {num_chains_generated} chains")
                print(f"    Success rate: {success_rate:.1%}")
                print(f"    Avg reward: {avg_reward:.3f}")
                print(f"    CodeBLEU: {chains_data['statistics']['avg_codebleu']:.3f}")
                print(f"    AST: {chains_data['statistics']['avg_ast_similarity']:.3f}")
                print(f"    CodeBERT: {chains_data['statistics']['avg_codebert_score']:.3f}")
            
            # Step 3: Store in dataset
            problem_id = code_dataset.add_problem_chains(
                problem=problem,
                ground_truth=solution,
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
        
        # Delay
        if i < len(problems):
            time.sleep(delay)
    
    # Save final dataset
    code_dataset.save()
    
    return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate chains for coding problems with code-specific rewards"
    )
    
    parser.add_argument("--dataset", default="apps",
                       help="Dataset: apps, ds1000, codeforces")
    parser.add_argument("--num-problems", type=int, default=10,
                       help="Number of problems")
    parser.add_argument("--decompose-model", default="qwen2.5-coder:7b",
                       help="Model for decomposition")
    parser.add_argument("--generation-model", default="qwen2.5-coder:7b",
                       help="Model for generation")
    parser.add_argument("--num-chains", type=int, default=5,
                       help="Chains per problem")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature")
    parser.add_argument("--max-tokens", type=int, default=4096,
                       help="Max tokens")
    parser.add_argument("--reward-threshold", type=float, default=0.7,
                       help="Success threshold")
    parser.add_argument("--output-dir", default="data/code_chains",
                       help="Output directory")
    parser.add_argument("--depth", type=int, default=3,
                       help="Decomposition depth")
    parser.add_argument("--branching", type=int, default=3,
                       help="Decomposition branching")
    parser.add_argument("--start-from", type=int, default=0,
                       help="Start from problem N (skip first N problems)")
    parser.add_argument("--delay", type=float, default=1.0,
                       help="Delay between problems")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress output")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from last saved problem for this dataset+model in output_dir")  # NEW
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("CODE CHAIN GENERATION WITH REWARDS")
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
    
    stats = batch_generate_code_chains(
        dataset_name=args.dataset,
        num_problems=args.num_problems,
        decompose_model=args.decompose_model,
        generation_model=args.generation_model,
        num_chains=args.num_chains,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        reward_threshold=args.reward_threshold,
        output_dir=args.output_dir,
        depth=args.depth,
        branching=args.branching,
        verbose=not args.quiet,
        delay=args.delay,
        start_from=args.start_from,
        resume=args.resume,  # NEW
    )
    
    elapsed = time.time() - start_time
    
    # Print summary
    print("\n" + "="*70)
    print("CODE CHAIN GENERATION COMPLETE")
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
