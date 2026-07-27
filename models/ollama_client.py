# models/ollama_client.py
# HTTP client for interacting with a local Ollama instance.
# Ollama runs LLMs locally (in this case, qwen3.5:2b-q4_K_M).
# This client sends prompts to Ollama's /api/generate endpoint
# and returns the generated text response.
#
# The agent controller uses this client for ReAct-style reasoning:
# the LLM decides which MCP tool to call next based on the current
# state (previous observations and available data).
#
# If Ollama is unavailable, the agent falls back to a deterministic
# plan, so this client is optional — the agent works without it too.

import json
import httpx
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class GenerationConfig(BaseModel):
    """Pydantic model for Ollama generation request parameters."""
    model: str
    prompt: str
    system: Optional[str] = None
    temperature: float = 0.5
    max_tokens: int = 2048
    stream: bool = False


class OllamaResponse(BaseModel):
    """Pydantic model for Ollama generation response."""
    model: str
    response: str
    done: bool


class OllamaClient:
    """HTTP client for Ollama's local API.
    
    Connects to a local Ollama server (default: http://localhost:11434)
    and sends prompts for text generation. Supports configurable model,
    temperature, system prompt, and timeout.
    """

    def __init__(
        self,
        model: str = "qwen3.5:2b-q4_K_M",
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
    ):
        """Initialize the Ollama client.
        
        Args:
            model: Ollama model name to use for generation.
            base_url: Base URL of the Ollama server.
            timeout: HTTP request timeout in seconds.
        """
        self.model = model
        self.base_url = base_url
        # Reuse a single httpx.Client for all requests (connection pooling)
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.5,
    ) -> str:
        """Send a prompt to Ollama and return the generated text.
        
        Args:
            prompt: The input prompt for the LLM.
            system: Optional system prompt to set the LLM's behavior.
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).
            
        Returns:
            The generated text response from the LLM.
            
        Raises:
            ConnectionError: If Ollama server is not reachable.
            ValueError: If the requested model is not found.
            RuntimeError: For other HTTP errors.
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "temperature": temperature,
            "options": {
                "num_predict": 2048,  # Maximum tokens to generate
            },
        }
        if system:
            payload["system"] = system

        try:
            logger.debug(f"Ollama generate: model={self.model}")
            response = self._client.post("/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except httpx.ConnectError:
            # Ollama not running — provide clear error message
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running (ollama serve)."
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                # Model not pulled yet
                raise ValueError(
                    f"Model '{self.model}' not found. "
                    f"Run: ollama pull {self.model}"
                )
            raise RuntimeError(f"Ollama HTTP error: {e}")
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise

    def close(self) -> None:
        """Close the underlying HTTP client and release resources."""
        self._client.close()
