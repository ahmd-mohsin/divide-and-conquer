# config.py
"""Configuration management for HCOT system."""
import sys
import os

# Add current directory to path for direct imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Optional, Literal

@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    provider: Literal["ollama", "openai", "anthropic"]
    model_name: str
    temperature: float = 0.2
    max_tokens: int = 8192
    seed: Optional[int] = None
    
    # Ollama-specific
    keep_alive: Optional[str | int] = None  # e.g., "10m" or 0
    
    # API-specific
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 60


@dataclass
class DecompositionConfig:
    """Configuration for problem decomposition."""
    depth_limit: int = 3
    branching_limit: int = 3
    min_nodes: int = 1
    max_retries: int = 3


@dataclass
class HCOTConfig:
    """Main configuration for the HCOT system."""
    model: ModelConfig
    decomposition: DecompositionConfig = field(default_factory=DecompositionConfig)
    prompts_path: str = "hcot.json"
    verbose: bool = False


# Predefined model configurations
OLLAMA_MODELS = {
    "llama3.1:8b": ModelConfig(
        provider="ollama",
        model_name="llama3.1:8b-instruct",
        temperature=0.2,
        max_tokens=8192
    ),
    "llama3.2": ModelConfig(
        provider="ollama",
        model_name="llama3.2",
        temperature=0.2,
        max_tokens=8192
    ),
    "qwen2.5:7b": ModelConfig(
        provider="ollama",
        model_name="qwen2.5:7b-instruct",
        temperature=0.2,
        max_tokens=8192
    ),
    "deepseek-r1:7b": ModelConfig(
        provider="ollama",
        model_name="deepseek-r1:7b",
        temperature=0.2,
        max_tokens=8192
    ),
    "gemma2:9b": ModelConfig(
        provider="ollama",
        model_name="gemma2:9b-instruct",
        temperature=0.2,
        max_tokens=8192
    ),
}

API_MODELS = {
    "gpt-4": ModelConfig(
        provider="openai",
        model_name="gpt-4-turbo-preview",
        temperature=0.2,
        max_tokens=4096,
        api_key=os.getenv("OPENAI_API_KEY")
    ),
    "gpt-3.5": ModelConfig(
        provider="openai",
        model_name="gpt-3.5-turbo",
        temperature=0.2,
        max_tokens=4096,
        api_key=os.getenv("OPENAI_API_KEY")
    ),
    "claude-3.5": ModelConfig(
        provider="anthropic",
        model_name="claude-3-5-sonnet-20241022",
        temperature=0.2,
        max_tokens=8192,
        api_key=os.getenv("ANTHROPIC_API_KEY")
    ),
    "claude-3": ModelConfig(
        provider="anthropic",
        model_name="claude-3-opus-20240229",
        temperature=0.2,
        max_tokens=4096,
        api_key=os.getenv("ANTHROPIC_API_KEY")
    ),
}


def get_model_config(model_id: str, **kwargs) -> ModelConfig:
    """
    Get a model configuration by ID.
    
    Args:
        model_id: Model identifier (e.g., "llama3.1:8b", "gpt-4")
        **kwargs: Override any config parameters
    
    Returns:
        ModelConfig instance
    """
    # Check predefined configs
    if model_id in OLLAMA_MODELS:
        config = OLLAMA_MODELS[model_id]
    elif model_id in API_MODELS:
        config = API_MODELS[model_id]
    else:
        # Assume it's an Ollama model if not found
        config = ModelConfig(provider="ollama", model_name=model_id)
    
    # Apply overrides
    if kwargs:
        config = ModelConfig(
            provider=kwargs.get("provider", config.provider),
            model_name=kwargs.get("model_name", config.model_name),
            temperature=kwargs.get("temperature", config.temperature),
            max_tokens=kwargs.get("max_tokens", config.max_tokens),
            seed=kwargs.get("seed", config.seed),
            keep_alive=kwargs.get("keep_alive", config.keep_alive),
            api_key=kwargs.get("api_key", config.api_key),
            base_url=kwargs.get("base_url", config.base_url),
            timeout=kwargs.get("timeout", config.timeout),
        )
    
    return config