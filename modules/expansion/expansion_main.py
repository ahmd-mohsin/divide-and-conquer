# expansion_main.py
"""Main entry point for Module 2: CoT Expansion with Hierarchical RL."""
import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


from pathlib import Path
import json
from typing import Optional

from expansion_config import ExpansionConfig, CoTConfig, RLConfig, RewardConfig
from rl_agent import HierarchicalRLAgent
from dataset_loader import DecompositionLoader
from expansion_schemas import ExpansionResult
import time


def train_rl_agent():
    """Train the RL agent on decomposed sub-problems."""
    print("\n" + "="*70)
    print("HCOT MODULE 2: Training RL Agent for CoT Expansion")
    print("="*70 + "\n")
    
    # Load dataset
    try:
        loader = DecompositionLoader(dataset_dir="../division/data/decompositions")
    except FileNotFoundError as e:
        print(f"✗ {e}")
        print("\nPlease run batch_decompose.py first:")
        print("  cd ../division")
        print("  python batch_decompose.py --num-problems 50")
        return
    
    # Get all sub-problems for training
    print("\nExtracting sub-problems from dataset...")
    subproblems = loader.get_all_subproblems(max_decomps=100)
    print(f"✓ Loaded {len(subproblems)} sub-problems for training")
    
    # Show statistics
    stats = loader.get_statistics()
    print(f"\nDataset statistics:")
    print(f"  Total decompositions: {stats['total_decompositions']}")
    print(f"  Total sub-problems: {stats['total_subproblems']}")
    print(f"  Categories: {', '.join(stats['categories'].keys())}")
    
    # Create config
    config = ExpansionConfig(
        cot=CoTConfig(
            num_chains=5,
            max_steps=8,
            temperature=0.8,  # Higher for more diversity
            model="deepseek-r1:7b"
        ),
        rl=RLConfig(
            learning_rate=3e-4,
            total_timesteps=50000,  # Increase for better results
            ent_coef=0.05,  # High entropy = more exploration
            save_freq=5000,
        ),
        reward=RewardConfig(
            diversity_weight=0.6,  # Prioritize diversity
            quality_weight=0.3,
            progress_weight=0.1
        )
    )
    
    # Create agent
    agent = HierarchicalRLAgent(config)
    
    # Train
    print(f"\nStarting training...")
    start_time = time.time()
    
    train_stats = agent.train(
        subproblems=subproblems,
        n_envs=1  # Increase to 4-8 for faster training
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"{'='*70}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"Model saved to: {train_stats['final_model_path']}")
    print(f"{'='*70}\n")


def expand_dataset(use_trained_model: bool = False, max_problems: int = 10):
    """Expand decompositions from dataset into CoT chains."""
    print("\n" + "="*70)
    print("HCOT MODULE 2: Expanding Decompositions into CoT Chains")
    print("="*70 + "\n")
    
    # Load dataset
    try:
        loader = DecompositionLoader(dataset_dir="../division/data/decompositions")
    except FileNotFoundError as e:
        print(f"✗ {e}")
        return
    
    # Get sample decompositions
    print(f"Loading {max_problems} decompositions...")
    decomps_with_meta = loader.get_sample(n=max_problems, seed=42)
    print(f"✓ Loaded {len(decomps_with_meta)} decompositions\n")
    
    # Create config
    config = ExpansionConfig(
        cot=CoTConfig(
            num_chains=5,
            model="deepseek-r1:7b"
        )
    )
    
    # Create agent
    agent = HierarchicalRLAgent(config)
    
    # Load trained model if requested
    if use_trained_model:
        model_path = f"{config.model_save_path}/ppo_cot_agent_final"
        if Path(f"{model_path}.zip").exists():
            agent.load_model(model_path)
            print(f"✓ Loaded trained model from {model_path}\n")
        else:
            print(f"⚠ No trained model found at {model_path}")
            print("Using random policy instead.\n")
            use_trained_model = False
    
    # Expand each decomposition
    all_results = []
    
    for i, (decomp, meta) in enumerate(decomps_with_meta, 1):
        print(f"\n{'='*70}")
        print(f"Expanding {i}/{len(decomps_with_meta)}")
        print(f"{'='*70}")
        print(f"Problem: {meta['problem'][:80]}...")
        print(f"Category: {meta['category']}")
        print(f"Model: {meta['model']}")
        
        # Expand
        result = agent.expand_decomposition(
            decomposition=decomp,
            use_learned_policy=use_trained_model
        )
        
        # Add metadata
        result.training_metadata = {
            "original_problem": meta["problem"],
            "expected_answer": meta["answer"],
            "category": meta["category"],
            "decomposition_model": meta["model"],
            "decomposition_id": meta["id"],
        }
        
        all_results.append(result)
    
    # Save results
    results_dir = Path(config.results_path) / "expansions"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for i, result in enumerate(all_results):
        result_file = results_dir / f"expansion_{i:03d}.json"
        with open(result_file, 'w') as f:
            json.dump(result.model_dump(), f, indent=2)
    
    # Save summary
    summary = {
        "total_decompositions": len(all_results),
        "total_subproblems": sum(len(r.subproblem_expansions) for r in all_results),
        "total_chains": sum(r.total_chains for r in all_results),
        "avg_diversity": sum(r.avg_diversity for r in all_results) / len(all_results),
        "avg_quality": sum(r.avg_quality for r in all_results) / len(all_results),
        "used_trained_model": use_trained_model,
    }
    
    summary_file = results_dir / "expansion_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Expansion Complete")
    print(f"{'='*70}")
    print(f"Total decompositions: {summary['total_decompositions']}")
    print(f"Total sub-problems: {summary['total_subproblems']}")
    print(f"Total chains: {summary['total_chains']}")
    print(f"Avg diversity: {summary['avg_diversity']:.3f}")
    print(f"Avg quality: {summary['avg_quality']:.3f}")
    print(f"\n✓ Results saved to: {results_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="HCOT Module 2: CoT Expansion with RL")
    parser.add_argument("--train", action="store_true", help="Train the RL agent")
    parser.add_argument("--expand", action="store_true", help="Expand decompositions")
    parser.add_argument("--use-trained", action="store_true", help="Use trained model")
    parser.add_argument("--max-problems", type=int, default=10, help="Max problems to expand")
    
    args = parser.parse_args()
    
    if args.train:
        train_rl_agent()
    elif args.expand:
        expand_dataset(use_trained_model=args.use_trained, max_problems=args.max_problems)
    else:
        print("Usage:")
        print("  python expansion_main.py --train                    # Train RL agent")
        print("  python expansion_main.py --expand                   # Expand with random policy")
        print("  python expansion_main.py --expand --use-trained     # Expand with trained agent")
        print("  python expansion_main.py --expand --max-problems 20 # Expand 20 problems")