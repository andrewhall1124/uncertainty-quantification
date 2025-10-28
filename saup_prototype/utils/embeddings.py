"""
Semantic distance calculation using sentence embeddings.
"""

from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from rich.table import Table
from rich.panel import Panel

from utils import console


class SemanticDistanceCalculator:
    """
    Calculates semantic distances for SAUP uncertainty propagation.

    Uses sentence-transformers to compute embeddings and cosine distance
    for measuring inquiry drift and inference gaps.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the semantic distance calculator.

        Parameters:
            model_name: Name of the sentence-transformers model to use
        """
        with console.status(f"[info]Loading semantic distance model: {model_name}...[/info]"):
            self.model = SentenceTransformer(model_name)
        console.print(f"[success]✓ Semantic distance calculator loaded[/success]")

    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a text string.

        Parameters:
            text: Input text

        Returns:
            Embedding vector
        """
        return self.model.encode(text, convert_to_numpy=True)

    def calculate_inquiry_drift(self, question: str, trajectory: List[Dict]) -> float:
        """
        Calculate inquiry drift (D_a): semantic shift from question to trajectory.

        Measures how far the agent's reasoning has drifted from the original question.
        Based on Section 3.2 of the SAUP paper.

        Parameters:
            question: Original question posed to the agent
            trajectory: List of trajectory steps with 'thought', 'action', 'observation'

        Returns:
            Inquiry drift value (0-1, where higher means more drift)

        Example:
            >>> calculator = SemanticDistanceCalculator()
            >>> drift = calculator.calculate_inquiry_drift(
            ...     "Can coffee help with weight loss?",
            ...     [{'thought': '...', 'action': '...', 'observation': '...'}]
            ... )
        """
        if not trajectory:
            return 0.0

        # Concatenate all trajectory elements
        trajectory_text_parts = []
        for step in trajectory:
            if 'thought' in step and step['thought']:
                trajectory_text_parts.append(step['thought'])
            if 'action' in step and step['action']:
                trajectory_text_parts.append(step['action'])
            if 'observation' in step and step['observation']:
                trajectory_text_parts.append(step['observation'])

        trajectory_text = " ".join(trajectory_text_parts)

        # Calculate embeddings
        question_embedding = self._get_embedding(question)
        trajectory_embedding = self._get_embedding(trajectory_text)

        # Calculate cosine distance
        distance = cosine(question_embedding, trajectory_embedding)

        # Ensure valid range [0, 1]
        return float(np.clip(distance, 0.0, 1.0))

    def calculate_inference_gap(self, thought: str, observation: str) -> float:
        """
        Calculate inference gap (D_o): discrepancy between thought and observation.

        Measures the semantic distance between what the agent expected (thought)
        and what it actually observed. Based on Section 3.2 of the SAUP paper.

        Parameters:
            thought: Agent's thought/reasoning
            observation: Actual observation received

        Returns:
            Inference gap value (0-1, where higher means larger gap)

        Example:
            >>> calculator = SemanticDistanceCalculator()
            >>> gap = calculator.calculate_inference_gap(
            ...     "Coffee should increase metabolism",
            ...     "No significant effect found on metabolism"
            ... )
        """
        if not thought or not observation:
            return 0.0

        # Calculate embeddings
        thought_embedding = self._get_embedding(thought)
        observation_embedding = self._get_embedding(observation)

        # Calculate cosine distance
        distance = cosine(thought_embedding, observation_embedding)

        # Ensure valid range [0, 1]
        return float(np.clip(distance, 0.0, 1.0))

    def display_distances_table(self,
                               step_number: int,
                               inquiry_drift: float,
                               inference_gap: float):
        """
        Display semantic distances in a formatted table.

        Parameters:
            step_number: Current step number
            inquiry_drift: Inquiry drift value
            inference_gap: Inference gap value
        """
        table = Table(title=f"Semantic Distances - Step {step_number}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        table.add_column("Status", justify="center")

        def get_status(value: float) -> str:
            if value < 0.3:
                return "[low_unc]Low[/low_unc]"
            elif value < 0.6:
                return "[med_unc]Moderate[/med_unc]"
            else:
                return "[high_unc]High[/high_unc]"

        table.add_row("Inquiry Drift", f"{inquiry_drift:.3f}", get_status(inquiry_drift))
        table.add_row("Inference Gap", f"{inference_gap:.3f}", get_status(inference_gap))

        console.print(table)
