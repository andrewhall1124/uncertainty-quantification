"""
Base agent class for SAUP system.
"""

from typing import List, Dict, Optional
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """
    Abstract base class for agents in SAUP system.

    Agents follow a multi-step reasoning process and maintain
    a trajectory of thoughts, actions, and observations.
    """

    def __init__(self, max_steps: int = 7):
        """
        Initialize base agent.

        Parameters:
            max_steps: Maximum number of reasoning steps
        """
        self.max_steps = max_steps
        self.trajectory: List[Dict] = []

    @abstractmethod
    def step(self, state: Dict) -> Dict:
        """
        Execute a single reasoning step.

        Parameters:
            state: Current state information

        Returns:
            Dictionary with 'thought', 'action', 'observation', 'uncertainty'
        """
        pass

    @abstractmethod
    def run(self, question: str) -> Dict:
        """
        Run the agent on a question.

        Parameters:
            question: Question to answer

        Returns:
            Dictionary with 'answer', 'trajectory', 'success'
        """
        pass

    def reset(self):
        """Reset agent state."""
        self.trajectory = []

    def get_trajectory(self) -> List[Dict]:
        """
        Get the agent's trajectory.

        Returns:
            List of trajectory steps
        """
        return self.trajectory
