# agent/memory.py
# Implements the agent's episodic memory — tracks every step the agent takes
# during a run. Each step records:
#   - The agent's internal thought/reasoning
#   - The action taken (MCP tool called)
#   - The input parameters passed to the tool
#   - The observation/result returned
#
# This memory is used to:
# 1. Build the ReAct prompt for the LLM (providing context of previous steps).
# 2. Generate the workflow history shown in the Streamlit UI.
# 3. Provide traceability for debugging.

import json
from typing import List, Dict, Any, Optional
from datetime import datetime


class AgentStep:
    """A single step in the agent's workflow.
    
    Captures the ReAct cycle: Thought → Action → Observation.
    Each step is timestamped for traceability.
    """

    def __init__(
        self,
        thought: str,
        action: str,
        action_input: Dict[str, Any],
        observation: Any,
    ):
        self.timestamp = datetime.now().isoformat()
        self.thought = thought          # Agent's reasoning at this step
        self.action = action            # MCP tool name that was called
        self.action_input = action_input  # Parameters passed to the tool
        self.observation = observation    # Result returned by the tool

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the step to a dictionary for storage/debugging."""
        return {
            "timestamp": self.timestamp,
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
        }

    def summary(self, max_obs_length: int = 300) -> str:
        """Generate a concise text summary of this step.
        
        Used in the ReAct prompt to show the LLM what happened.
        Observations are truncated to max_obs_length to avoid
        exceeding token limits.
        
        Args:
            max_obs_length: Maximum characters for the observation string.
            
        Returns:
            Formatted summary string.
        """
        obs_str = json.dumps(self.observation, indent=2)
        if len(obs_str) > max_obs_length:
            obs_str = obs_str[:max_obs_length] + "..."
        return (
            f"Step: {self.action}\n"
            f"  Thought: {self.thought}\n"
            f"  Observation: {obs_str}\n"
        )


class AgentMemory:
    """Stores the sequence of steps taken by the agent during a run.
    
    Also holds a context dictionary for arbitrary key-value storage
    that the agent can use to pass data between steps.
    """

    def __init__(self):
        self.steps: List[AgentStep] = []    # Ordered list of all steps
        self.context: Dict[str, Any] = {}   # Arbitrary key-value storage

    def add_step(self, step: AgentStep) -> None:
        """Append a step to the agent's workflow history."""
        self.steps.append(step)

    def get_history(self) -> List[Dict[str, Any]]:
        """Return all steps as a list of dictionaries."""
        return [s.to_dict() for s in self.steps]

    def get_last_observation(self) -> Any:
        """Get the observation from the most recent step."""
        if self.steps:
            return self.steps[-1].observation
        return None

    def get_summary(self, max_obs_length: int = 300) -> str:
        """Generate a full workflow summary in Markdown format.
        
        Used for the "Workflow Log" tab in the Streamlit UI.
        
        Args:
            max_obs_length: Maximum characters per observation.
            
        Returns:
            Markdown-formatted string of all steps.
        """
        lines = ["## Agent Workflow History\n"]
        for i, step in enumerate(self.steps, 1):
            lines.append(f"### Step {i}: {step.action}")
            lines.append(f"**Thought:** {step.thought}")
            obs_str = json.dumps(step.observation, indent=2)
            if len(obs_str) > max_obs_length:
                obs_str = obs_str[:max_obs_length] + "..."
            lines.append(f"**Observation:** {obs_str}")
            lines.append("")
        return "\n".join(lines)

    def clear(self) -> None:
        """Reset the memory — clears all steps and context."""
        self.steps.clear()
        self.context.clear()
