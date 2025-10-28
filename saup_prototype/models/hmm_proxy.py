"""
HMM Proxy for situational weight estimation.

Simplified rule-based proxy for the Continuous HMM described in the SAUP paper.
"""

from typing import Tuple
from rich.panel import Panel

from utils import console


class HMMProxy:
    """
    Simple surrogate for CHMM-based situational weight estimation.

    Instead of a full Continuous Hidden Markov Model, this uses rule-based
    heuristics to estimate situational weights based on semantic distances
    and step position.

    Based on Section 3.2 of the SAUP paper.
    """

    def __init__(self):
        """
        Initialize the HMM proxy with three hidden states.

        States:
        - correct: Agent is on track
        - moderate: Agent shows moderate deviation
        - high_deviation: Agent is significantly off track
        """
        self.states = ['correct', 'moderate', 'high_deviation']
        console.print("[success]✓ HMM Proxy initialized[/success]")

    def estimate_situational_weight(self,
                                    inquiry_drift: float,
                                    inference_gap: float,
                                    step_number: int,
                                    total_steps: int) -> Tuple[float, str]:
        """
        Estimate situational weight for a trajectory step.

        The weight reflects how much this step should contribute to overall
        uncertainty. Higher weights indicate steps where the agent may be
        deviating from the correct reasoning path.

        Parameters:
            inquiry_drift: Semantic drift from original question (0-1)
            inference_gap: Gap between thought and observation (0-1)
            step_number: Current step number (1-indexed)
            total_steps: Total number of steps in trajectory

        Returns:
            Tuple of (weight, inferred_state)
            where weight is the situational weight and inferred_state is one of:
            'correct', 'moderate', 'high_deviation'

        Example:
            >>> hmm = HMMProxy()
            >>> weight, state = hmm.estimate_situational_weight(0.4, 0.3, 2, 5)
            >>> print(f"Weight: {weight:.2f}, State: {state}")
        """
        # Combine semantic distances
        combined_distance = (inquiry_drift + inference_gap) / 2

        # Position weight: later steps slightly more important
        # This reflects that errors compound over time
        position_weight = 1.0 + (step_number / total_steps) * 0.2

        # Distance-based state inference and weight
        if combined_distance < 0.3:
            # Low deviation - agent is on track
            inferred_state = 'correct'
            distance_weight = 0.8
        elif combined_distance < 0.6:
            # Moderate deviation - some uncertainty
            inferred_state = 'moderate'
            distance_weight = 1.2
        else:
            # High deviation - significant uncertainty
            inferred_state = 'high_deviation'
            distance_weight = 1.8

        # Additional weighting based on specific patterns
        # High inquiry drift but low inference gap: agent drifting from question
        if inquiry_drift > 0.6 and inference_gap < 0.3:
            distance_weight *= 1.3

        # Low inquiry drift but high inference gap: observations contradict expectations
        elif inquiry_drift < 0.3 and inference_gap > 0.6:
            distance_weight *= 1.4

        # Both high: compounded issues
        elif inquiry_drift > 0.6 and inference_gap > 0.6:
            distance_weight *= 1.5

        # Calculate final situational weight
        situational_weight = position_weight * distance_weight

        return situational_weight, inferred_state

    def display_state_transition(self,
                                 step_number: int,
                                 state: str,
                                 weight: float,
                                 inquiry_drift: float,
                                 inference_gap: float):
        """
        Display state information in a Rich panel.

        Parameters:
            step_number: Current step number
            state: Inferred hidden state
            weight: Situational weight
            inquiry_drift: Inquiry drift value
            inference_gap: Inference gap value
        """
        # Color code by state
        state_colors = {
            'correct': 'low_unc',
            'moderate': 'med_unc',
            'high_deviation': 'high_unc'
        }
        color = state_colors.get(state, 'info')

        content = (
            f"[{color}]State: {state.upper()}[/{color}]\n"
            f"Situational Weight: {weight:.2f}\n"
            f"Inquiry Drift: {inquiry_drift:.2f} | Inference Gap: {inference_gap:.2f}"
        )

        panel = Panel(
            content,
            title=f"Step {step_number} - HMM State",
            border_style=color
        )

        console.print(panel)
