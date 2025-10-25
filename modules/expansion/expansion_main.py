# expansion_main.py
"""Main entry point for Module 2: CoT Expansion with Hierarchical RL."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'division'))

from pathlib import Path
import json
from typing import Optional

from expansion_config import ExpansionConfig, CoTConfig, RLConfig, RewardConfig
from rl_agent import HierarchicalRLAgent
from schemas import Decomposition
from utils import load_decomposition, save_decomposition


def train_rl_agent_example():
    """Example: Train the RL agent on sub-problems."""
    print("\n" + "="*70)
    print("HCOT MODULE 2: Training RL Agent for CoT Expansion")
    print("="*70 + "\n")
    
    # Load example decomposition from Module 1
    decomp_path = "../division/example_decomposition.json"
    if Path(decomp_path).exists():
        decomp = load_decomposition(decomp_path)
        subproblems = decomp.nodes
    else:
        print(f"No decomposition found at {decomp_path}")
        print("Please run Module 1 first to generate a decomposition.")
        return
    
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
            total_timesteps=10000,  # Start small for testing
            ent_coef=0.05,  # High entropy = more exploration
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
    stats = agent.train(
        subproblems=subproblems,
        n_envs=1  # Start with 1, increase to 4-8 for faster training
    )
    
    print("\n✓ Training complete!")
    print(f"Model saved to: {stats['final_model_path']}")


def expand_problem_example(use_trained_model: bool = False):
    """Example: Expand a decomposition into CoT chains."""
    print("\n" + "="*70)
    print("HCOT MODULE 2: Expanding Sub-Problems into CoT Chains")
    print("="*70 + "\n")
    
    # Load decomposition
    decomp_path = "../division/example_decomposition.json"
    if not Path(decomp_path).exists():
        print(f"No decomposition found. Please run Module 1 first.")
        return
    
    decomp = load_decomposition(decomp_path)
    
    # Create config
    config = ExpansionConfig(
        cot=CoTConfig(
            num_chains=3,  # Fewer for quick demo
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
        else:
            print(f"No trained model found at {model_path}")
            print("Using random policy instead.")
            use_trained_model = False
    
    # Expand
    result = agent.expand_decomposition(
        decomposition=decomp,
        use_learned_policy=use_trained_model
    )
    
    # Save result
    result_path = f"{config.results_path}/expansion_result.json"
    Path(config.results_path).mkdir(exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump(result.model_dump(), f, indent=2)
    
    print(f"\n✓ Results saved to: {result_path}")
    
    # Show example chains
    print("\n" + "="*70)
    print("EXAMPLE CHAINS (First Sub-Problem)")
    print("="*70)
    if result.subproblem_expansions:
        expansion = result.subproblem_expansions[0]
        print(f"\nSub-problem: {expansion.subproblem_goal}")
        for i, chain in enumerate(expansion.chains[:2], 1):  # Show first 2
            print(f"\n--- Chain {i} (Strategy: {chain.strategy}) ---")
            for step in chain.steps[:3]:  # Show first 3 steps
                print(f"Step {step.step_number}: {step.content[:100]}...")
            if chain.final_answer:
                print(f"Final Answer: {chain.final_answer[:100]}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="HCOT Module 2: CoT Expansion with RL")
    parser.add_argument("--train", action="store_true", help="Train the RL agent")
    parser.add_argument("--expand", action="store_true", help="Expand a decomposition")
    parser.add_argument("--use-trained", action="store_true", help="Use trained model for expansion")
    
    args = parser.parse_args()
    
    if args.train:
        train_rl_agent_example()
    elif args.expand:
        expand_problem_example(use_trained_model=args.use_trained)
    else:
        print("Usage:")
        print("  python expansion_main.py --train              # Train RL agent")
        print("  python expansion_main.py --expand             # Expand with random policy")
        print("  python expansion_main.py --expand --use-trained  # Expand with trained agent")