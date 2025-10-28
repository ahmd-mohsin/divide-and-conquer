# llm_clients.py
"""LLM client implementations for various providers."""
from __future__ import annotations
import sys
import os

# Add current directory to path for direct imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
import json
import socket
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from config import ModelConfig


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    @abstractmethod
    def chat_json(
        self,
        system: str,
        user: str,
        schema: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Send a chat request and get JSON response.
        
        Args:
            system: System prompt
            user: User message
            schema: Optional JSON schema for structured output
            
        Returns:
            JSON string response
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM service is available."""
        pass


class OllamaClient(LLMClient):
    """Client for local Ollama models."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            import ollama
            self.ollama = ollama
        except ImportError:
            raise ImportError(
                "ollama package not found. Install with: pip install ollama --break-system-packages"
            )
        
        # Check if Ollama is available using multiple methods
        if not self.is_available():
            print("\n" + "="*70)
            print("WARNING: Ollama daemon not reachable!")
            print("="*70)
            print("Please start Ollama in another terminal:")
            print("  ollama serve")
            print("\nOr if already running, try:")
            print("  killall ollama && ollama serve &")
            print("="*70 + "\n")
            raise RuntimeError(
                "Ollama daemon not reachable. Start it with 'ollama serve' or install from https://ollama.ai"
            )
        
        # Ensure model is available
        self._ensure_model()
    
    def is_available(self) -> bool:
        """Check if Ollama daemon is running using multiple methods."""
        # Method 1: Socket check (most reliable)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(('127.0.0.1', 11434))
            sock.close()
            if result == 0:
                return True
        except Exception:
            pass
        
        # Method 2: Try listing models
        try:
            self.ollama.list()
            return True
        except Exception:
            pass
        
        # Method 3: Try version check
        try:
            self.ollama.version()
            return True
        except Exception:
            pass
        
        return False
    
    def _ensure_model(self):
        """Pull model if not already available."""
        try:
            models = self.ollama.list()
            available = {m["model"] for m in models.get("models", [])}
            
            # Check if exact model name exists
            if self.config.model_name in available:
                return
            
            # Check if model exists with :latest suffix
            model_base = self.config.model_name.split(':')[0]
            for model in available:
                if model.startswith(model_base):
                    print(f"Using existing model: {model}")
                    self.config.model_name = model  # Use the available variant
                    return
            
            # Model not found, try to pull it
            print(f"\nModel {self.config.model_name} not found locally.")
            print(f"Pulling model {self.config.model_name}...")
            print("This may take a few minutes...")
            for _ in self.ollama.pull(self.config.model_name, stream=True):
                pass
            print(f"✓ Model {self.config.model_name} ready.\n")
            
        except Exception as e:
            print(f"Warning: Could not verify model availability: {e}")
            print(f"Will attempt to use model anyway: {self.config.model_name}")
    
    def _build_options(self) -> Dict[str, Any]:
        """Build options dict for Ollama."""
        opts = {
            "temperature": self.config.temperature,
            "num_ctx": self.config.max_tokens
        }
        if self.config.seed is not None:
            opts["seed"] = self.config.seed
        return opts
    
    def chat_json(
        self,
        system: str,
        user: str,
        schema: Optional[Dict[str, Any]] = None
    ) -> str:
        """Send chat request to Ollama with JSON format."""
        format_spec = schema if schema else "json"
        
        response = self.ollama.chat(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            stream=False,
            format=format_spec,
            options=self._build_options(),
            keep_alive=self.config.keep_alive,
        )
        
        return response["message"]["content"]


class OpenAIClient(LLMClient):
    """Client for OpenAI API models."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            from openai import OpenAI
            self.OpenAI = OpenAI
        except ImportError:
            raise ImportError(
                "openai package not found. Install with: pip install openai --break-system-packages"
            )
        
        if not config.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        
        self.client = self.OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout
        )
    
    def is_available(self) -> bool:
        """Check if OpenAI API is accessible."""
        try:
            self.client.models.list()
            return True
        except Exception:
            return False
    
    def chat_json(
        self,
        system: str,
        user: str,
        schema: Optional[Dict[str, Any]] = None
    ) -> str:
        """Send chat request to OpenAI API."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
        
        kwargs = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        
        if self.config.seed is not None:
            kwargs["seed"] = self.config.seed
        
        # Use response_format for JSON mode if no schema
        if schema is None:
            kwargs["response_format"] = {"type": "json_object"}
        
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
    

class AnthropicClient(LLMClient):
    """Client for Anthropic API models."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            from anthropic import Anthropic
            self.Anthropic = Anthropic
        except ImportError:
            raise ImportError(
                "anthropic package not found. Install with: pip install anthropic --break-system-packages"
            )
        
        if not config.api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY environment variable.")
        
        self.client = self.Anthropic(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout
        )
    
    def is_available(self) -> bool:
        """Check if Anthropic API is accessible."""
        try:
            # Simple check - the client validates on init
            return True
        except Exception:
            return False
    
    def chat_json(
        self,
        system: str,
        user: str,
        schema: Optional[Dict[str, Any]] = None
    ) -> str:
        """Send chat request to Anthropic API."""
        # Anthropic needs JSON instruction in the prompt
        if schema:
            user = f"{user}\n\nYou must respond with valid JSON only."
        
        response = self.client.messages.create(
            model=self.config.model_name,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=system,
            messages=[{"role": "user", "content": user}]
        )
        
        return response.content[0].text


class HuggingFaceClient(LLMClient):
    """Client for HuggingFace Inference API."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            from huggingface_hub import InferenceClient
            self.InferenceClient = InferenceClient
        except ImportError:
            raise ImportError(
                "huggingface_hub package not found. Install with: pip install huggingface_hub"
            )
        
        if not config.api_key:
            raise ValueError("HuggingFace API token required. Set HF_TOKEN environment variable.")
        
        self.client = self.InferenceClient(token=config.api_key)
    
    def is_available(self) -> bool:
        """Check if HuggingFace API is accessible."""
        try:
            # Simple check - if we got here, token is set
            return True
        except Exception:
            return False
    
    def chat_json(
        self,
        system: str,
        user: str,
        schema: Optional[Dict[str, Any]] = None
    ) -> str:
        """Send chat request to HuggingFace Inference API."""
        # Combine system and user prompts
        full_prompt = f"{system}\n\n{user}\n\nYou must respond with valid JSON only."
        
        try:
            response = self.client.text_generation(
                model=self.config.model_name,
                prompt=full_prompt,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                return_full_text=False,
            )
            return response
        except Exception as e:
            print(f"HuggingFace API error: {e}")
            raise


def create_client(config: ModelConfig) -> LLMClient:
    """
    Factory function to create the appropriate LLM client.
    
    Args:
        config: Model configuration
        
    Returns:
        LLMClient instance
    """
    if config.provider == "ollama":
        return OllamaClient(config)
    elif config.provider == "openai":
        return OpenAIClient(config)
    elif config.provider == "anthropic":
        return AnthropicClient(config)
    elif config.provider == "huggingface":
        return HuggingFaceClient(config)
    else:
        raise ValueError(f"Unknown provider: {config.provider}")