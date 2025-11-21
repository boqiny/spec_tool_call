"""Configuration for speculative tool calling framework."""
import os
from dataclasses import dataclass


@dataclass
class SpecConfig:
    """Configuration for speculative execution."""
    # Execution mode
    enable_speculation: bool = True  # Set to False for baseline actor-only mode
    
    # Model provider: "openai" or "vllm"
    model_provider: str = "openai"
    
    # Models
    actor_model: str = "gpt-5"
    spec_model: str = "gpt-5-mini"
    
    # vLLM settings (only used when model_provider="vllm")
    vllm_actor_url: str = "http://localhost:8003/v1"  # Actor model endpoint
    vllm_spec_url: str = "http://localhost:8004/v1"   # Spec model endpoint
    vllm_api_key: str = "EMPTY"  # vLLM doesn't need real API key

    # Speculation parameters
    top_k_spec: int = 3
    verification_strategy: str = "exact"  # Options: "exact", "tool_name_only", "none"

    # Execution limits
    max_steps: int = 15

    # LLM parameters
    llm_max_tokens: int = 2048

    @classmethod
    def from_env(cls) -> "SpecConfig":
        """Load configuration from environment variables."""
        return cls(
            enable_speculation=(os.getenv("DISABLE_SPECULATION", "0") != "1"),
            model_provider=os.getenv("MODEL_PROVIDER", "openai"),
            actor_model=os.getenv("GAIA_ACTOR_MODEL", "gpt-5"),
            spec_model=os.getenv("GAIA_SPEC_MODEL", "gpt-5-mini"),
            vllm_actor_url=os.getenv("VLLM_ACTOR_URL", "http://localhost:8003/v1"),
            vllm_spec_url=os.getenv("VLLM_SPEC_URL", "http://localhost:8004/v1"),
            vllm_api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
            top_k_spec=int(os.getenv("GAIA_TOPK", "3")),
            verification_strategy=os.getenv("VERIFICATION_STRATEGY", "exact"),
            max_steps=int(os.getenv("GAIA_MAX_STEPS", "15")),
        )


# Global config instance
config = SpecConfig.from_env()
