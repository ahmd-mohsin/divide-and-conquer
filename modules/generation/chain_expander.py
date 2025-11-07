#!/usr/bin/env python3
"""
Chain-of-Thought Expansion Module - Hierarchical Execution
Generates multiple diverse reasoning chains through hierarchical subproblems.
Executes subproblems in dependency order and chains results forward.
"""
from __future__ import annotations
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import random  # NEW

# ------------------- NEW: Hard strategy prompts for diversity -------------------
STRATEGY_PROMPTS = {
    "step_by_step": "Solve this problem step by step. Show all your work.",
    "backwards": "Work backwards from the desired result. What would we need to get there?",
    "analogical": "Think of a similar problem you know how to solve, and use that approach.",
    "case_analysis": "Break this into separate cases and analyze each one.",
    "contradiction": "Assume the opposite and show why it leads to a contradiction.",
    "visual": "Visualize this problem spatially or graphically. Describe what you see.",
    "algebraic": "Use pure algebraic manipulation to solve this.",
    "numerical": "Take a computational/numerical approach with concrete examples.",
}
# -------------------------------------------------------------------------------

# Global model instance (load once)
_similarity_model = None

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import from division module
division_path = os.path.join(parent_dir, 'division')
if division_path not in sys.path:
    sys.path.insert(division_path)

try:
    from division.schemas import SubProblem, Decomposition
    from division.config import ModelConfig, get_model_config
    from division.llm_clients import create_client, LLMClient
    from division.utils import get_execution_order
except ImportError:
    # Fallback for when running from same directory as division
    from schemas import SubProblem, Decomposition
    from config import ModelConfig, get_model_config
    from llm_clients import create_client, LLMClient
    from utils import get_execution_order


def get_similarity_model():
    """Load sentence transformer model once."""
    global _similarity_model
    if _similarity_model is None:
        print("Loading sentence-transformers model for semantic similarity...")
        _similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _similarity_model

class HierarchicalChainExecutor:
    """Executes hierarchical reasoning chains through subproblems in dependency order."""
    
    def __init__(
        self,
        model: str = "llama3.1:latest",
        num_chains: int = 10,
        temperature: float = 0.8,
        max_tokens: int = 4096,  # Increased from 1024 to 4096 for longer reasoning
        verbose: bool = False,
        strategy_prompts: Optional[Dict[str, str]] = None  # NEW
    ):
        """
        Initialize the hierarchical chain executor.
        
        Args:
            model: Model identifier (e.g., "llama3.1:latest")
            num_chains: Number of diverse execution paths to generate
            temperature: Sampling temperature for diversity
            max_tokens: Maximum tokens per generation step
            verbose: Print progress
            strategy_prompts: Optional override of strategy prompts dict
        """
        self.num_chains = num_chains
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose

        # NEW: setup strategy prompts
        self.strategy_prompts: Dict[str, str] = strategy_prompts or STRATEGY_PROMPTS
        self._strategy_keys: List[str] = list(self.strategy_prompts.keys())
        
        # Create model client with higher temperature for diversity
        model_config = get_model_config(
            model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        self.client = create_client(model_config)
        
        if self.verbose:
            print(f"HierarchicalChainExecutor initialized with {model}")
            print(f"  Chains per problem: {num_chains}")
            print(f"  Temperature: {temperature}")
            print(f"  Strategies available: {', '.join(self._strategy_keys)}")
    
    # NEW: deterministic-but-diverse strategy assignment (round-robin across chains)
    def _pick_strategy(self, chain_index: int) -> Tuple[str, str]:
        key = self._strategy_keys[chain_index % len(self._strategy_keys)]
        return key, self.strategy_prompts[key]
    
    def execute_hierarchical_chains(
        self,
        decomposition: Decomposition,
        problem: str,
        ground_truth: str
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple execution chains through the hierarchical decomposition.
        
        Each chain:
        1. Executes subproblems in dependency order
        2. Passes results forward to dependent subproblems
        3. Continues until final answer is reached
        4. Gets reward based on final answer correctness
        
        Args:
            decomposition: The problem decomposition
            problem: Original problem statement
            ground_truth: Correct answer
        
        Returns:
            List of complete execution chain dictionaries
        """
        chains = []
        
        if self.verbose:
            print(f"\nGenerating {self.num_chains} hierarchical execution chains")
        
        # Get execution order (waves of subproblems that can be executed in parallel)
        try:
            execution_waves = get_execution_order(decomposition)
        except ValueError as e:
            if self.verbose:
                print(f"  Error computing execution order: {e}")
            return []
        
        if self.verbose:
            print(f"  Execution plan: {len(execution_waves)} waves")
            for i, wave in enumerate(execution_waves, 1):
                print(f"    Wave {i}: {len(wave)} subproblems")
        
        # Generate N diverse execution chains
        for chain_idx in range(self.num_chains):
            # NEW: select a strategy for this chain
            strategy_key, strategy_text = self._pick_strategy(chain_idx)
            try:
                chain = self._execute_single_chain(
                    decomposition=decomposition,
                    execution_waves=execution_waves,
                    problem=problem,
                    ground_truth=ground_truth,
                    chain_index=chain_idx,
                    strategy_key=strategy_key,       # NEW
                    strategy_text=strategy_text      # NEW
                )
                chains.append(chain)
                
                if self.verbose:
                    reward_str = f"{chain['final_reward']:.2f}" if chain['final_reward'] is not None else "N/A"
                    print(f"  Chain {chain_idx + 1}/{self.num_chains}: final_reward={reward_str} | strategy={strategy_key}")
                
            except Exception as e:
                if self.verbose:
                    print(f"  Chain {chain_idx + 1} failed: {e}")
                # Add failed chain placeholder
                chains.append({
                    'chain_id': f"chain_{chain_idx}",
                    'steps': [],
                    'final_answer': None,
                    'final_reward': 0.0,
                    'error': str(e),
                    'strategy': strategy_key  # NEW
                })
        
        return chains
    
    def _execute_single_chain(
        self,
        decomposition: Decomposition,
        execution_waves: List[List[str]],
        problem: str,
        ground_truth: str,
        chain_index: int,
        strategy_key: str,        # NEW
        strategy_text: str        # NEW
    ) -> Dict[str, Any]:
        """Execute a single reasoning chain through all subproblems."""
        
        # Storage for intermediate results
        subproblem_results = {}
        execution_steps = []
        
        # Execute each wave in order
        for wave_idx, wave in enumerate(execution_waves):
            for subproblem_id in wave:
                subproblem = decomposition.get_node(subproblem_id)
                
                if subproblem is None:
                    continue
                
                # Get results from dependencies
                dependency_results = {}
                for dep_id in subproblem.depends_on:
                    if dep_id in subproblem_results:
                        dependency_results[dep_id] = subproblem_results[dep_id]
                
                # Execute this subproblem
                step_result = self._execute_subproblem_step(
                    subproblem=subproblem,
                    problem_context=problem,
                    dependency_results=dependency_results,
                    wave_idx=wave_idx,
                    strategy_key=strategy_key,     # NEW
                    strategy_text=strategy_text    # NEW
                )
                
                # Store result for future dependencies
                subproblem_results[subproblem_id] = step_result['answer']
                
                # Add to execution trace
                execution_steps.append({
                    'subproblem_id': subproblem_id,
                    'goal': subproblem.goal,
                    'reasoning': step_result['reasoning'],
                    'answer': step_result['answer'],
                    'wave': wave_idx,
                    'dependencies_used': list(dependency_results.keys()),
                    'strategy': strategy_key  # NEW (record which strategy guided this step)
                })
        
        # Extract final answer
        final_answer = self._extract_final_answer(
            execution_steps=execution_steps,
            problem=problem,
            ground_truth=ground_truth
        )
        
        # BUILD FULL REASONING TEXT FOR SEMANTIC SIMILARITY
        full_reasoning = self._build_full_reasoning_text(execution_steps, final_answer)
        
        # Compute reward with semantic similarity
        final_reward = self._compute_reward(final_answer, ground_truth, full_reasoning)
        
        return {
            'chain_id': f"chain_{chain_index}",
            'steps': execution_steps,
            'final_answer': final_answer,
            'final_reward': final_reward,
            'num_steps': len(execution_steps),
            'temperature': self.temperature,
            'full_reasoning': full_reasoning,  # Store for analysis
            'strategy': strategy_key,          # NEW (chain-level)
            'strategy_text': strategy_text     # NEW (for full reproducibility)
        }
    
    def _build_full_reasoning_text(self, execution_steps: List[Dict[str, Any]], final_answer: str) -> str:
        """Build complete reasoning chain as single text for semantic similarity."""
        
        parts = []
        
        for i, step in enumerate(execution_steps, 1):
            parts.append(f"Step {i}: {step['goal']}")
            parts.append(f"Reasoning: {step['reasoning']}")
            parts.append(f"Result: {step['answer']}")
            parts.append("")  # Blank line
        
        parts.append(f"Final Answer: {final_answer}")
        
        return "\n".join(parts)
    
    def _execute_subproblem_step(
        self,
        subproblem: SubProblem,
        problem_context: str,
        dependency_results: Dict[str, str],
        wave_idx: int,
        strategy_key: str,      # NEW
        strategy_text: str      # NEW
    ) -> Dict[str, Any]:
        """Execute a single subproblem step, using results from dependencies."""
        
        # Build prompt with context and dependencies
        prompt = self._create_step_prompt(
            subproblem=subproblem,
            problem_context=problem_context,
            dependency_results=dependency_results,
            strategy_key=strategy_key,     # NEW
            strategy_text=strategy_text    # NEW
        )
        
        # Generate reasoning for this step
        reasoning = self._generate_reasoning(prompt, strategy_key=strategy_key)
        
        # Extract answer from reasoning
        answer = self._extract_step_answer(reasoning)
        
        return {
            'reasoning': reasoning,
            'answer': answer
        }
    
    def _create_step_prompt(
        self,
        subproblem: SubProblem,
        problem_context: str,
        dependency_results: Dict[str, str],
        strategy_key: str,     # NEW
        strategy_text: str     # NEW
    ) -> str:
        """Create prompt for a subproblem step, including dependency results."""
        
        # NEW: Hard constraint header (so the model must follow the strategy)
        hard_constraint = [
            f"STRATEGY ({strategy_key}): {strategy_text}",
            "HARD CONSTRAINT: Strictly follow the strategy above. Do not switch strategies.",
            "If the strategy suggests a style (e.g., backwards, case analysis, contradiction), keep that style throughout this step.",
            ""
        ]
        
        prompt_parts = hard_constraint + [
            f"Original Problem: {problem_context}",
            "",
            f"Current Subproblem: {subproblem.goal}",
            f"Approach: {subproblem.plan}"
        ]
        
        if subproblem.hint:
            prompt_parts.append(f"Hint: {subproblem.hint}")
        
        # Add results from dependencies
        if dependency_results:
            prompt_parts.append("")
            prompt_parts.append("Results from previous steps:")
            for dep_id, result in dependency_results.items():
                prompt_parts.append(f"  - {dep_id}: {result}")
        
        if subproblem.expected_form:
            prompt_parts.append(f"\nExpected answer format: {subproblem.expected_form}")
        
        prompt_parts.append("")
        prompt_parts.append("Solve this subproblem in the specified strategy style. Provide your reasoning and final answer for THIS step.")
        prompt_parts.append("End with: ANSWER: <your answer for this step>")
        
        return "\n".join(prompt_parts)
    
    def _generate_reasoning(self, prompt: str, strategy_key: Optional[str] = None) -> str:
        """Generate reasoning from the model."""
        
        # NEW: Slightly remind in system prompt that strategy must be followed
        system_prompt = f"""You are a mathematics expert solving problems step by step.
Follow the specified STRATEGY strictly ({strategy_key if strategy_key else 'given in the prompt'}).
Provide clear, detailed reasoning for each step.
Always conclude with your answer in the format: ANSWER: <your answer>
Be precise and show all your work."""
        
        try:
            response = self.client.ollama.chat(
                model=self.client.config.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                stream=False,
                options={
                    "temperature": self.temperature,
                    "num_ctx": self.max_tokens
                }
            )
            
            return response["message"]["content"]
            
        except AttributeError:
            # Fallback for non-Ollama clients
            return self.client.chat_json(
                system=system_prompt,
                user=prompt,
                schema=None
            )
    
    def _extract_step_answer(self, reasoning: str) -> str:
        """Extract answer from a step's reasoning."""
        import re
        
        # Look for "ANSWER: ..." pattern
        answer_match = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', reasoning, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).strip()
        
        # Look for boxed answer
        boxed_match = re.search(r'\\boxed\{(.+?)\}', reasoning)
        if boxed_match:
            return boxed_match.group(1).strip()
        
        # Look for last line that might be an answer
        lines = reasoning.strip().split('\n')
        if lines:
            last_line = lines[-1].strip()
            if len(last_line) < 100:
                return last_line
        
        return "[No answer extracted]"
    
    def _extract_final_answer(
        self,
        execution_steps: List[Dict[str, Any]],
        problem: str,
        ground_truth: str
    ) -> str:
        """
        Extract or synthesize the final answer from all execution steps.
        
        Strategy:
        1. Check if last step's answer looks like the final answer
        2. If not, synthesize from all steps
        """
        
        if not execution_steps:
            return "[No steps executed]"
        
        # Get the last step's answer
        last_step_answer = execution_steps[-1]['answer']
        
        # If it looks reasonable, use it
        if last_step_answer and last_step_answer != "[No answer extracted]":
            # Check if it's similar to ground truth format
            return last_step_answer
        
        # Otherwise, synthesize final answer from all steps
        return self._synthesize_final_answer(execution_steps, problem)
    
    def _synthesize_final_answer(
        self,
        execution_steps: List[Dict[str, Any]],
        problem: str
    ) -> str:
        """Synthesize final answer from all execution steps."""
        
        # Build summary of all steps
        steps_summary = "\n\n".join([
            f"Step {i+1} ({step['goal']}):\n{step['reasoning']}\nResult: {step['answer']}"
            for i, step in enumerate(execution_steps)
        ])
        
        synthesis_prompt = f"""Original Problem: {problem}

Execution steps completed:
{steps_summary}

Based on all the steps above, what is the final answer to the original problem?
Provide ONLY the final answer, nothing else.

FINAL ANSWER:"""
        
        system_prompt = "You are a mathematics expert. Synthesize the final answer from the steps provided. Give only the answer, no explanation."
        
        try:
            response = self.client.ollama.chat(
                model=self.client.config.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": synthesis_prompt}
                ],
                stream=False,
                options={
                    "temperature": 0.3,  # Lower temp for synthesis
                    "num_ctx": 2048
                }
            )
            
            return response["message"]["content"].strip()
            
        except Exception as e:
            # Fallback: return last step's answer
            if execution_steps:
                return execution_steps[-1]['answer']
            return "[Synthesis failed]"
    
    def _compute_reward(self, predicted: Optional[str], ground_truth: str, full_reasoning: str = "") -> float:
        """
        Compute reward with semantic similarity fallback.
        
        Returns:
            1.0 if exact match
            0.0-0.99 based on semantic similarity if no match
        """
        if predicted is None:
            return 0.0
        
        # Normalize both answers
        pred_normalized = self._normalize_answer(predicted)
        gt_normalized = self._normalize_answer(ground_truth)
        
        # Exact match after normalization
        if pred_normalized == gt_normalized:
            return 1.0
        
        # Try numeric comparison (small error tolerance)
        try:
            pred_num = float(pred_normalized)
            gt_num = float(gt_normalized)
            
            # Within 0.1% error
            if abs(pred_num - gt_num) < abs(gt_num * 0.001):
                return 1.0
        except (ValueError, TypeError):
            pass
        
        # SEMANTIC SIMILARITY FALLBACK
        if full_reasoning:
            try:
                model = get_similarity_model()
                
                # Embed the full reasoning chain
                reasoning_embedding = model.encode(full_reasoning, convert_to_numpy=True)
                
                # Embed the ground truth (use as simple string if no full solution available)
                gt_text = f"The answer is {ground_truth}"
                gt_embedding = model.encode(gt_text, convert_to_numpy=True)
                
                # Cosine similarity
                similarity = np.dot(reasoning_embedding, gt_embedding) / (
                    np.linalg.norm(reasoning_embedding) * np.linalg.norm(gt_embedding)
                )
                
                # Convert similarity (-1 to 1) to reward (0 to 1)
                semantic_reward = max(0.0, min(1, (similarity + 1) / 2 * 1))
                
                return semantic_reward
                
            except Exception as e:
                print(f"Warning: Semantic similarity computation failed: {e}")
                return 0.0
        
        return 0.0
        
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        if answer is None:
            return ""
        
        # Convert to string and lowercase
        answer = str(answer).lower().strip()
        
        # Remove common prefixes
        for prefix in ['answer:', 'final answer:', 'the answer is:', '=', 'answer is:']:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()
        
        # Remove whitespace
        answer = ''.join(answer.split())
        
        # Remove latex formatting
        answer = answer.replace('\\boxed{', '').replace('}', '')
        answer = answer.replace('$', '')
        
        return answer


def expand_decomposition_hierarchical(
    decomposition: Decomposition,
    problem: str,
    ground_truth: str,
    executor: HierarchicalChainExecutor,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Execute multiple hierarchical chains through the decomposition.
    
    Args:
        decomposition: The problem decomposition
        problem: Original problem statement
        ground_truth: Correct answer
        executor: HierarchicalChainExecutor instance
        verbose: Print progress
    
    Returns:
        Dictionary with all execution chains and statistics
    """
    result = {
        'problem': problem,
        'ground_truth': ground_truth,
        'decomposition_id': f"problem_{hash(problem) % 1000000}",
        'num_subproblems': len(decomposition.nodes),
        'chains': []
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Executing hierarchical chains through {len(decomposition.nodes)} subproblems")
        print(f"{'='*70}")
    
    # Execute multiple chains
    chains = executor.execute_hierarchical_chains(
        decomposition=decomposition,
        problem=problem,
        ground_truth=ground_truth
    )
    
    result['chains'] = chains
    
    # Compute statistics
    rewards = [c['final_reward'] for c in chains if c['final_reward'] is not None]
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    best_reward = max(rewards) if rewards else 0.0
    success_rate = sum(1 for r in rewards if r >= 1.0) / len(rewards) if rewards else 0.0
    
    result['statistics'] = {
        'total_chains': len(chains),
        'avg_reward': avg_reward,
        'best_reward': best_reward,
        'success_rate': success_rate,
        'chains_with_correct_answer': sum(1 for r in rewards if r >= 1.0)
    }
    
    if verbose:
        print(f"\n  Generated {len(chains)} chains")
        print(f"    Avg reward: {avg_reward:.2f}")
        print(f"    Best reward: {best_reward:.2f}")
        print(f"    Success rate: {success_rate:.1%}")
    
    return result
