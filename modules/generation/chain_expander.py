#!/usr/bin/env python3
"""
Hierarchical Chain Expander with Strategy-Based Diversity
Generates multiple chains using different reasoning strategies for diversity.
"""
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import random

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import from division module
division_path = os.path.join(parent_dir, 'division')
if division_path not in sys.path:
    sys.path.insert(0, division_path)

from division.schemas import Decomposition, SubProblem
from division.llm_clients import create_client


# Strategy prompts for diverse reasoning
STRATEGY_PROMPTS = {
    "step_by_step": "Solve this problem step by step. Show all your work clearly.",
    "backwards": "Work backwards from the desired result. What would we need to get there?",
    "analogical": "Think of a similar problem you know how to solve, and use that approach as a guide.",
    "case_analysis": "Break this into separate cases and analyze each one systematically.",
    "first_principles": "Reason from first principles. What are the fundamental concepts here?",
    "visual": "Visualize this problem spatially or graphically. Describe what you see and use that to reason.",
    "algebraic": "Use algebraic manipulation and symbolic reasoning to solve this.",
    "numerical": "Take a computational/numerical approach with concrete examples and calculations.",
}


class HierarchicalChainExecutor:
    """
    Execute decomposition hierarchically with strategy-based chain diversity.
    
    Each chain uses a different reasoning strategy for diversity beyond just temperature.
    """
    
    def __init__(
        self,
        model: str = "deepseek-r1:32b",
        num_chains: int = 8,
        temperature: float = 0.6,
        max_tokens: int = 4096,
        use_strategies: bool = True,
        verbose: bool = False
    ):
        self.model = model
        self.num_chains = num_chains
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_strategies = use_strategies
        self.verbose = verbose
        self.client = create_client(model)
        
        # Get list of strategies (cycle through them if num_chains > len(strategies))
        self.strategies = list(STRATEGY_PROMPTS.keys())
    
    def _get_strategy_for_chain(self, chain_idx: int) -> tuple[str, str]:
        """
        Get reasoning strategy for this chain.
        
        Returns:
            (strategy_name, strategy_prompt)
        """
        if not self.use_strategies or chain_idx >= self.num_chains:
            return "standard", "Solve this problem carefully."
        
        # Cycle through strategies if we have more chains than strategies
        strategy_name = self.strategies[chain_idx % len(self.strategies)]
        strategy_prompt = STRATEGY_PROMPTS[strategy_name]
        
        return strategy_name, strategy_prompt
    
    def solve_subproblem(
        self,
        subproblem: SubProblem,
        problem_context: str,
        previous_results: Dict[str, str],
        strategy_name: str,
        strategy_prompt: str
    ) -> str:
        """
        Solve a single subproblem using the specified reasoning strategy.
        
        Args:
            subproblem: The subproblem to solve
            problem_context: Original problem statement
            previous_results: Results from dependency subproblems
            strategy_name: Name of reasoning strategy
            strategy_prompt: Strategy instruction
        
        Returns:
            Solution string
        """
        # Build prompt with strategy
        system_prompt = f"""You are a mathematical reasoning expert. Your task is to solve the given sub-problem.

REASONING STRATEGY: {strategy_name.upper()}
{strategy_prompt}

Provide your reasoning and then give your final answer clearly marked as "ANSWER: [your answer]"."""
        
        # Build user prompt with context
        user_prompt = f"""Original Problem: {problem_context}

Current Sub-Problem:
Goal: {subproblem.goal}
{f"Plan: {subproblem.plan}" if subproblem.plan else ""}
{f"Hint: {subproblem.hint}" if subproblem.hint else ""}
"""
        
        # Add dependency results if available
        if previous_results:
            user_prompt += "\nPrevious Results:\n"
            for dep_id, result in previous_results.items():
                user_prompt += f"  Sub-problem {dep_id}: {result}\n"
        
        user_prompt += f"\nUsing the {strategy_name} approach, solve this sub-problem."
        
        # Generate solution
        try:
            response = self.client.chat(
                system=system_prompt,
                user=user_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.strip()
        
        except Exception as e:
            if self.verbose:
                print(f"    Error solving subproblem: {e}")
            return f"Error: {str(e)}"
    
    def execute_chain(
        self,
        decomposition: Decomposition,
        problem: str,
        chain_idx: int
    ) -> Dict[str, Any]:
        """
        Execute one complete reasoning chain using a specific strategy.
        
        Args:
            decomposition: Problem decomposition
            problem: Original problem statement
            chain_idx: Index of this chain (determines strategy)
        
        Returns:
            Chain execution data
        """
        # Get strategy for this chain
        strategy_name, strategy_prompt = self._get_strategy_for_chain(chain_idx)
        
        if self.verbose:
            print(f"  Chain {chain_idx + 1}: Using '{strategy_name}' strategy")
        
        # Get execution waves (topological order)
        waves = decomposition.get_execution_waves()
        
        # Track results
        results = {}
        steps = []
        
        # Execute each wave
        for wave_idx, wave in enumerate(waves):
            if self.verbose:
                print(f"    Wave {wave_idx + 1}: {len(wave)} subproblems")
            
            for node_id in wave:
                node = decomposition.get_node(node_id)
                
                # Get dependency results
                dep_results = {
                    dep_id: results.get(dep_id, "Not available")
                    for dep_id in (node.depends_on or [])
                }
                
                # Solve subproblem with strategy
                solution = self.solve_subproblem(
                    subproblem=node,
                    problem_context=problem,
                    previous_results=dep_results,
                    strategy_name=strategy_name,
                    strategy_prompt=strategy_prompt
                )
                
                # Extract answer (look for "ANSWER: ..." pattern)
                answer = solution
                if "ANSWER:" in solution.upper():
                    parts = solution.split("ANSWER:", 1)
                    if len(parts) > 1:
                        answer = parts[1].strip()
                
                # Store result
                results[node_id] = answer
                
                # Record step
                steps.append({
                    "subproblem_id": node_id,
                    "goal": node.goal,
                    "reasoning": solution,
                    "answer": answer,
                    "wave": wave_idx,
                    "dependencies_used": list(dep_results.keys()),
                    "strategy": strategy_name
                })
        
        # Combine all reasoning into full chain
        full_reasoning = f"Strategy: {strategy_name.upper()}\n\n"
        for i, step in enumerate(steps, 1):
            full_reasoning += f"Step {i}: {step['goal']}\n"
            full_reasoning += f"Reasoning: {step['reasoning']}\n"
            full_reasoning += f"Result: {step['answer']}\n\n"
        
        # Final answer is the last step's answer
        final_answer = steps[-1]['answer'] if steps else ""
        
        return {
            "chain_id": f"chain_{chain_idx}",
            "strategy": strategy_name,
            "steps": steps,
            "final_answer": final_answer,
            "full_reasoning": full_reasoning,
            "num_steps": len(steps),
            "temperature": self.temperature
        }


def expand_decomposition_hierarchical(
    decomposition: Decomposition,
    problem: str,
    ground_truth: str,
    executor: HierarchicalChainExecutor,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Expand decomposition into multiple diverse reasoning chains.
    
    Args:
        decomposition: Problem decomposition
        problem: Original problem
        ground_truth: Ground truth answer
        executor: Chain executor
        verbose: Print progress
    
    Returns:
        Dictionary with chains and statistics
    """
    if verbose:
        print(f"\nGenerating {executor.num_chains} chains with diverse strategies...")
    
    chains = []
    
    # Generate chains with different strategies
    for i in range(executor.num_chains):
        if verbose:
            print(f"\n  Generating chain {i + 1}/{executor.num_chains}...")
        
        try:
            chain_data = executor.execute_chain(
                decomposition=decomposition,
                problem=problem,
                chain_idx=i
            )
            
            # Calculate reward (simple similarity for now)
            reward = calculate_reward(chain_data['final_answer'], ground_truth)
            chain_data['final_reward'] = reward
            
            chains.append(chain_data)
            
            if verbose:
                print(f"    ✓ Strategy: {chain_data['strategy']}")
                print(f"    Reward: {reward:.3f}")
        
        except Exception as e:
            if verbose:
                print(f"    ✗ Failed: {e}")
    
    # Calculate statistics
    rewards = [c['final_reward'] for c in chains]
    success_threshold = 0.9
    
    stats = {
        "total_chains": len(chains),
        "avg_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "best_reward": max(rewards) if rewards else 0.0,
        "success_rate": sum(1 for r in rewards if r >= success_threshold) / len(rewards) if rewards else 0.0,
        "chains_with_correct_answer": sum(1 for c in chains if is_correct(c['final_answer'], ground_truth)),
        "strategies_used": list(set(c['strategy'] for c in chains))
    }
    
    return {
        "problem": problem,
        "ground_truth": ground_truth,
        "decomposition_id": f"problem_{id(decomposition)}",
        "num_subproblems": len(decomposition.nodes),
        "chains": chains,
        "statistics": stats
    }


def calculate_reward(predicted: str, ground_truth: str) -> float:
    """
    Calculate reward based on answer similarity.
    
    For math problems, this is typically numeric equality or string match.
    """
    if not predicted or not ground_truth:
        return 0.0
    
    # Normalize strings
    pred_clean = str(predicted).strip().lower()
    gt_clean = str(ground_truth).strip().lower()
    
    # Exact match
    if pred_clean == gt_clean:
        return 1.0
    
    # Try numeric comparison
    try:
        pred_num = float(pred_clean.replace(',', ''))
        gt_num = float(gt_clean.replace(',', ''))
        
        # Close enough (within 0.1%)
        if abs(pred_num - gt_num) / max(abs(gt_num), 1.0) < 0.001:
            return 1.0
        
        # Partial credit for being close
        diff = abs(pred_num - gt_num) / max(abs(gt_num), 1.0)
        return max(0.0, 1.0 - diff)
    
    except (ValueError, ZeroDivisionError):
        pass
    
    # String similarity fallback
    from difflib import SequenceMatcher
    similarity = SequenceMatcher(None, pred_clean, gt_clean).ratio()
    return similarity


def is_correct(predicted: str, ground_truth: str) -> bool:
    """Check if prediction is correct (reward >= 0.95)"""
    return calculate_reward(predicted, ground_truth) >= 0.95


if __name__ == "__main__":
    # Test the strategies
    print("Available Reasoning Strategies:")
    for i, (name, prompt) in enumerate(STRATEGY_PROMPTS.items(), 1):
        print(f"  {i}. {name}: {prompt}")