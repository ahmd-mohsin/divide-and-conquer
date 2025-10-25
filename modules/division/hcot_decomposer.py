# hcot_decomposer.py
"""Hierarchical Chain-of-Thought decomposition system."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from pydantic import ValidationError
from config import HCOTConfig, ModelConfig, DecompositionConfig
from schemas import Decomposition, DECOMPOSITION_SCHEMA
from llm_clients import create_client, LLMClient


class HierarchicalDecomposer:
    """Decomposes complex math problems into hierarchical sub-problems."""
    
    def __init__(self, config: HCOTConfig, client: Optional[LLMClient] = None):
        self.config = config
        self.client = client or create_client(config.model)
        self._load_prompts()
    
    def _load_prompts(self):
        """Load system and user prompt templates."""
        prompts_path = Path(self.config.prompts_path)
        if not prompts_path.exists():
            raise FileNotFoundError(f"Prompts file not found: {prompts_path}")
        
        with open(prompts_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.system_prompt = data.get("system_prompt", "")
        self.user_template = data.get("user_template", "")
        
        if not self.system_prompt or not self.user_template:
            raise ValueError("Prompts file must contain 'system_prompt' and 'user_template'")
    
    def decompose(
        self,
        problem: str,
        depth: Optional[int] = None,
        branching: Optional[int] = None,
        retry_on_failure: bool = True
    ) -> Decomposition:
        """Decompose a problem into hierarchical sub-problems."""
        depth = depth or self.config.decomposition.depth_limit
        branching = branching or self.config.decomposition.branching_limit
        
        for attempt in range(self.config.decomposition.max_retries):
            try:
                if self.config.verbose:
                    print(f"Decomposition attempt {attempt + 1}/{self.config.decomposition.max_retries}")
                
                decomp = self._decompose_once(problem, depth, branching)
                
                is_valid, errors = decomp.validate_hierarchy()
                if not is_valid:
                    if self.config.verbose:
                        print(f"Validation errors: {errors}")
                    if retry_on_failure and attempt < self.config.decomposition.max_retries - 1:
                        continue
                    raise RuntimeError(f"Invalid hierarchy structure: {errors}")
                
                if self.config.verbose:
                    print(f"Successfully decomposed into {len(decomp.nodes)} sub-problems")
                
                return decomp
                
            except (json.JSONDecodeError, ValidationError) as e:
                if self.config.verbose:
                    print(f"Attempt {attempt + 1} failed: {e}")
                
                if not retry_on_failure or attempt >= self.config.decomposition.max_retries - 1:
                    raise RuntimeError(f"Decomposition failed after {attempt + 1} attempts: {e}")
        
        raise RuntimeError("Decomposition failed: max retries exceeded")
    
    def _decompose_once(self, problem: str, depth: int, branching: int) -> Decomposition:
        """Perform a single decomposition attempt."""
        user_prompt = self.user_template.format(
            problem=problem.strip(),
            depth=depth,
            branching=branching
        )
        
        raw_json = self.client.chat_json(
            system=self.system_prompt,
            user=user_prompt,
            schema=DECOMPOSITION_SCHEMA
        )
        
        data = self._parse_json_response(raw_json)
        decomp = Decomposition(**data)
        
        return decomp
    
    def _parse_json_response(self, raw: str) -> dict:
        """Parse JSON from LLM response, handling markdown code blocks."""
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                cleaned = "\n".join(lines)
            
            return json.loads(cleaned)
    
    def decompose_batch(self, problems: list[str], **kwargs) -> list[Decomposition]:
        """Decompose multiple problems."""
        results = []
        for i, problem in enumerate(problems):
            if self.config.verbose:
                print(f"\n=== Problem {i+1}/{len(problems)} ===")
            
            try:
                decomp = self.decompose(problem, **kwargs)
                results.append(decomp)
            except Exception as e:
                if self.config.verbose:
                    print(f"Failed to decompose problem {i+1}: {e}")
                results.append(None)
        
        return results


def quick_decompose(
    problem: str,
    model: str = "llama3.1:8b",
    prompts_path: str = "hcot_prompts.json",
    depth: int = 3,
    branching: int = 3,
    verbose: bool = False,
    **model_kwargs
) -> Decomposition:
    """Quick helper function to decompose a single problem."""
    from config import get_model_config
    
    model_config = get_model_config(model, **model_kwargs)
    decomp_config = DecompositionConfig(
        depth_limit=depth,
        branching_limit=branching
    )
    
    config = HCOTConfig(
        model=model_config,
        decomposition=decomp_config,
        prompts_path=prompts_path,
        verbose=verbose
    )
    
    decomposer = HierarchicalDecomposer(config)
    return decomposer.decompose(problem)