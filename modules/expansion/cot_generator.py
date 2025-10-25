# cot_generator.py
"""Chain-of-Thought generation using reasoning models."""
from __future__ import annotations
import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from division.schemas import SubProblem

from typing import List, Optional
import ollama
from expansion_config import CoTConfig
from expansion_schemas import CoTStep, CoTChain
import json
import uuid

class CoTGenerator:
    """Generates chain-of-thought reasoning for sub-problems."""
    
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
    
    def __init__(self, config: CoTConfig):
        self.config = config
        self.client = ollama
        
        # Verify model is available
        self._ensure_model()
    
    def _ensure_model(self):
        """Ensure the reasoning model is available."""
        try:
            models = self.client.list()
            available = {m["model"] for m in models.get("models", [])}
            
            # Check for exact match or base model
            model_base = self.config.model.split(':')[0]
            for model in available:
                if model.startswith(model_base):
                    self.config.model = model
                    return
            
            print(f"Model {self.config.model} not found. Pulling...")
            self.client.pull(self.config.model)
            
        except Exception as e:
            print(f"Warning: Could not verify model: {e}")
    
    def generate_chain(
        self,
        subproblem_text: str,
        strategy: str = "step_by_step",
        temperature: float = None,
        existing_chains: Optional[List[CoTChain]] = None
    ) -> CoTChain:
        """
        Generate a single chain of thought.
        
        Args:
            subproblem_text: The sub-problem to solve
            strategy: Reasoning strategy to use
            temperature: Sampling temperature (higher = more diverse)
            existing_chains: Already generated chains (for diversity)
            
        Returns:
            CoTChain object
        """
        temp = temperature or self.config.temperature
        
        # Build prompt based on strategy
        system_prompt = self._build_system_prompt(strategy)
        user_prompt = self._build_user_prompt(subproblem_text, strategy, existing_chains)
        
        # Generate reasoning
        try:
            response = self.client.chat(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                options={
                    "temperature": temp,
                    "num_ctx": 8192,
                }
            )
            
            reasoning_text = response["message"]["content"]
            
            # Parse into steps
            steps = self._parse_into_steps(reasoning_text, strategy)
            
            # Create chain
            chain = CoTChain(
                chain_id=str(uuid.uuid4()),
                subproblem_id="",  # Will be set by caller
                steps=steps,
                strategy=strategy,
                final_answer=self._extract_final_answer(reasoning_text)
            )
            
            return chain
            
        except Exception as e:
            print(f"Error generating chain: {e}")
            # Return minimal chain
            return CoTChain(
                chain_id=str(uuid.uuid4()),
                subproblem_id="",
                steps=[CoTStep(step_number=1, content="Error generating reasoning", strategy=strategy)],
                strategy=strategy
            )
    
    def _build_system_prompt(self, strategy: str) -> str:
        """Build system prompt based on strategy."""
        base = "You are an expert mathematical reasoner. "
        strategy_instruction = self.STRATEGY_PROMPTS.get(
            strategy,
            "Solve this problem step by step."
        )
        return base + strategy_instruction
    
    def _build_user_prompt(
        self,
        subproblem: str,
        strategy: str,
        existing_chains: Optional[List[CoTChain]]
    ) -> str:
        """Build user prompt, encouraging diversity."""
        prompt = f"Problem: {subproblem}\n\n"
        
        if existing_chains:
            prompt += "Note: Other solution approaches have been tried. "
            prompt += "Please explore a DIFFERENT reasoning path.\n\n"
        
        prompt += f"Approach: Use {strategy.replace('_', ' ')} reasoning.\n"
        prompt += "Show your complete reasoning process."
        
        return prompt
    
    def _parse_into_steps(self, text: str, strategy: str) -> List[CoTStep]:
        """Parse reasoning text into discrete steps."""
        steps = []
        
        # Try to split by numbered steps or newlines
        lines = text.split('\n')
        current_step = []
        step_num = 1
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_step:
                    content = ' '.join(current_step)
                    if len(content) >= self.config.min_step_length:
                        steps.append(CoTStep(
                            step_number=step_num,
                            content=content,
                            strategy=strategy
                        ))
                        step_num += 1
                        current_step = []
            else:
                current_step.append(line)
        
        # Add final step
        if current_step:
            content = ' '.join(current_step)
            if len(content) >= self.config.min_step_length:
                steps.append(CoTStep(
                    step_number=step_num,
                    content=content,
                    strategy=strategy
                ))
        
        # If no steps parsed, create one from full text
        if not steps:
            steps.append(CoTStep(
                step_number=1,
                content=text[:self.config.max_step_length],
                strategy=strategy
            ))
        
        return steps[:self.config.max_steps]
    
    def _extract_final_answer(self, text: str) -> Optional[str]:
        """Try to extract final answer from reasoning."""
        # Look for common patterns
        patterns = [
            "final answer:",
            "therefore,",
            "answer:",
            "solution:",
            "result:",
        ]
        
        text_lower = text.lower()
        for pattern in patterns:
            if pattern in text_lower:
                idx = text_lower.rfind(pattern)
                answer = text[idx:].split('\n')[0]
                return answer.strip()
        
        # Return last sentence as fallback
        sentences = text.split('.')
        if sentences:
            return sentences[-1].strip()
        
        return None
    
    def generate_multiple_chains(
        self,
        subproblem_text: str,
        strategies: List[str],
        temperatures: Optional[List[float]] = None
    ) -> List[CoTChain]:
        """
        Generate multiple chains with different strategies.
        
        Args:
            subproblem_text: The sub-problem
            strategies: List of strategies to try
            temperatures: Optional list of temperatures
            
        Returns:
            List of CoTChain objects
        """
        if temperatures is None:
            temperatures = [self.config.temperature] * len(strategies)
        
        chains = []
        for i, strategy in enumerate(strategies):
            temp = temperatures[i] if i < len(temperatures) else self.config.temperature
            
            chain = self.generate_chain(
                subproblem_text=subproblem_text,
                strategy=strategy,
                temperature=temp,
                existing_chains=chains
            )
            chains.append(chain)
        
        return chains