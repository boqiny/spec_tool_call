"""Configuration for speculative tool calling framework."""
import os
from dataclasses import dataclass


@dataclass
class SpecConfig:
    """Configuration for speculative execution."""
    # Execution mode
    enable_speculation: bool = True  # Set to False for baseline actor-only mode
    
    # Models
    actor_model: str = "gpt-5"
    spec_model: str = "gpt-5-mini"

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
            actor_model=os.getenv("GAIA_ACTOR_MODEL", "gpt-5"),
            spec_model=os.getenv("GAIA_SPEC_MODEL", "gpt-5-mini"),
            top_k_spec=int(os.getenv("GAIA_TOPK", "3")),
            verification_strategy=os.getenv("VERIFICATION_STRATEGY", "exact"),
            max_steps=int(os.getenv("GAIA_MAX_STEPS", "15")),
        )


# Global config instance
config = SpecConfig.from_env()
