"""
Tool call verification strategies for speculative execution.

This module defines different strategies for matching actor tool calls
with speculative predictions to determine if cached results can be reused.
"""

from typing import Any, Dict, Optional
from abc import ABC, abstractmethod
import json


class ToolVerifier(ABC):
    """Base class for tool verification strategies."""
    
    @abstractmethod
    def verify(
        self,
        actor_tool_name: str,
        actor_tool_args: Dict[str, Any],
        spec_tool_name: str,
        spec_tool_args: Dict[str, Any]
    ) -> tuple[bool, float, str]:
        """
        Verify if a speculative prediction matches the actor's request.
        
        Args:
            actor_tool_name: Tool name requested by actor
            actor_tool_args: Arguments requested by actor
            spec_tool_name: Tool name in speculative prediction
            spec_tool_args: Arguments in speculative prediction
            
        Returns:
            Tuple of (is_match, similarity_score, reason)
            - is_match: Whether the prediction can be used
            - similarity_score: Confidence score (0.0 to 1.0)
            - reason: Human-readable explanation
        """
        pass


class ExactMatchVerifier(ToolVerifier):
    """
    Exact match verifier - requires perfect match of tool name and all arguments.
    
    This is the most conservative strategy:
    - Tool names must match exactly
    - All argument keys must match
    - All argument values must match exactly (after normalization)
    
    Pros: No false positives, always correct results
    Cons: Low cache hit rate, may miss semantically equivalent queries
    """
    
    def verify(
        self,
        actor_tool_name: str,
        actor_tool_args: Dict[str, Any],
        spec_tool_name: str,
        spec_tool_args: Dict[str, Any]
    ) -> tuple[bool, float, str]:
        # Tool names must match
        if actor_tool_name != spec_tool_name:
            return False, 0.0, f"Tool mismatch: {spec_tool_name} != {actor_tool_name}"
        
        # Normalize and compare arguments
        actor_normalized = self._normalize_args(actor_tool_args)
        spec_normalized = self._normalize_args(spec_tool_args)
        
        # Check if arguments match exactly
        if actor_normalized == spec_normalized:
            return True, 1.0, "Exact match"
        
        # Find differences
        diff_keys = set(actor_normalized.keys()) ^ set(spec_normalized.keys())
        if diff_keys:
            return False, 0.0, f"Different argument keys: {diff_keys}"
        
        # Find mismatched values
        mismatches = []
        for key in actor_normalized:
            if actor_normalized[key] != spec_normalized[key]:
                mismatches.append(key)
        
        if mismatches:
            return False, 0.0, f"Mismatched arguments: {', '.join(mismatches)}"
        
        return True, 1.0, "Exact match"
    
    def _normalize_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize arguments for comparison."""
        normalized = {}
        for key, value in args.items():
            if isinstance(value, str):
                # Strip whitespace, lowercase for case-insensitive comparison
                normalized[key] = value.strip()
            elif isinstance(value, (list, dict)):
                # Convert to JSON string for comparison
                normalized[key] = json.dumps(value, sort_keys=True)
            else:
                normalized[key] = value
        return normalized


class ToolNameOnlyVerifier(ToolVerifier):
    """
    Tool name only verifier - only checks if tool names match.
    
    This is the current implementation (broken):
    - Only verifies tool name matches
    - Ignores all arguments
    
    Pros: Highest cache hit rate
    Cons: Returns wrong results, confuses actor, degrades performance
    
    ⚠️ WARNING: This strategy is known to degrade performance!
    Use only for comparison/debugging purposes.
    """
    
    def verify(
        self,
        actor_tool_name: str,
        actor_tool_args: Dict[str, Any],
        spec_tool_name: str,
        spec_tool_args: Dict[str, Any]
    ) -> tuple[bool, float, str]:
        if actor_tool_name == spec_tool_name:
            return True, 1.0, "Tool name match (args ignored)"
        return False, 0.0, f"Tool mismatch: {spec_tool_name} != {actor_tool_name}"


class NoSpeculationVerifier(ToolVerifier):
    """
    No speculation verifier - always returns False to disable speculation.
    
    Use this to run baseline comparisons without speculation overhead.
    """
    
    def verify(
        self,
        actor_tool_name: str,
        actor_tool_args: Dict[str, Any],
        spec_tool_name: str,
        spec_tool_args: Dict[str, Any]
    ) -> tuple[bool, float, str]:
        return False, 0.0, "Speculation disabled"


# Factory for creating verifiers
def create_verifier(strategy: str = "exact") -> ToolVerifier:
    """
    Create a verifier instance based on strategy name.
    
    Args:
        strategy: One of "exact", "tool_name_only", "none"
        
    Returns:
        ToolVerifier instance
    """
    strategies = {
        "exact": ExactMatchVerifier,
        "tool_name_only": ToolNameOnlyVerifier,
        "none": NoSpeculationVerifier,
    }
    
    if strategy not in strategies:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Choose from: {', '.join(strategies.keys())}"
        )
    
    return strategies[strategy]()

