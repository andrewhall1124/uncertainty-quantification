"""
SAUP (Situation-Aware Uncertainty Propagation) algorithm.

Implements Algorithm 1 from the SAUP paper.
"""

from typing import List, Dict, Optional
import numpy as np
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel

from utils import console, format_uncertainty
from utils.embeddings import SemanticDistanceCalculator
from models.hmm_proxy import HMMProxy


class SAUPPropagator:
    """
    SAUP uncertainty propagation system.

    Propagates single-step uncertainties through a trajectory using
    situational awareness (semantic distances and HMM-based weighting).
    """

    def __init__(self, use_hmm_proxy: bool = True):
        """
        Initialize SAUP propagator.

        Parameters:
            use_hmm_proxy: Whether to use HMM proxy for situational weighting
        """
        self.distance_calculator = SemanticDistanceCalculator()
        self.hmm_proxy = HMMProxy() if use_hmm_proxy else None
        self.use_hmm = use_hmm_proxy

    def propagate_uncertainty(self,
                             question: str,
                             trajectory: List[Dict]) -> Dict:
        """
        Main SAUP algorithm (Algorithm 1 from paper).

        Propagates uncertainty through a reasoning trajectory using
        situation-aware weighting.

        Parameters:
            question: Original question posed to the agent
            trajectory: List of trajectory steps, each containing:
                - 'thought': Agent's reasoning
                - 'action': Agent's action
                - 'observation': Observed result
                - 'uncertainty': Single-step uncertainty U_n
                - 'thought_probs': Token probabilities (optional)
                - 'action_probs': Token probabilities (optional)

        Returns:
            Dictionary containing:
                - 'agent_uncertainty': Overall agent uncertainty (RMS with weights)
                - 'step_uncertainties': List of step uncertainties
                - 'situational_weights': List of situational weights
                - 'inquiry_drifts': List of inquiry drift values
                - 'inference_gaps': List of inference gap values
                - 'hmm_states': List of HMM states (if using HMM proxy)

        Example:
            >>> propagator = SAUPPropagator()
            >>> result = propagator.propagate_uncertainty(question, trajectory)
            >>> print(f"Agent uncertainty: {result['agent_uncertainty']:.3f}")
        """
        if not trajectory:
            return {
                'agent_uncertainty': 0.0,
                'step_uncertainties': [],
                'situational_weights': [],
                'inquiry_drifts': [],
                'inference_gaps': [],
                'hmm_states': []
            }

        n_steps = len(trajectory)

        # Storage for computed values
        step_uncertainties = []
        situational_weights = []
        inquiry_drifts = []
        inference_gaps = []
        hmm_states = []

        console.print("\n[info]Starting SAUP uncertainty propagation...[/info]\n")

        # Progress bar for trajectory processing
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[info]Processing trajectory steps...[/info]", total=n_steps)

            # Algorithm 1: For each step in trajectory
            for i, step in enumerate(trajectory):
                step_num = i + 1

                # Extract step information
                thought = step.get('thought', '')
                action = step.get('action', '')
                observation = step.get('observation', '')
                step_uncertainty = step.get('uncertainty', 0.0)

                # Calculate semantic distances
                # D_a: Inquiry drift - drift from original question
                trajectory_so_far = trajectory[:step_num]
                inquiry_drift = self.distance_calculator.calculate_inquiry_drift(
                    question, trajectory_so_far
                )

                # D_o: Inference gap - gap between thought and observation
                inference_gap = self.distance_calculator.calculate_inference_gap(
                    thought, observation
                )

                # Estimate situational weight W_n using HMM proxy
                if self.use_hmm and self.hmm_proxy:
                    weight, state = self.hmm_proxy.estimate_situational_weight(
                        inquiry_drift=inquiry_drift,
                        inference_gap=inference_gap,
                        step_number=step_num,
                        total_steps=n_steps
                    )
                    hmm_states.append(state)
                else:
                    # Fallback: uniform weighting
                    weight = 1.0
                    state = 'unknown'
                    hmm_states.append(state)

                # Store computed values
                step_uncertainties.append(step_uncertainty)
                situational_weights.append(weight)
                inquiry_drifts.append(inquiry_drift)
                inference_gaps.append(inference_gap)

                progress.advance(task)

        # Calculate agent-level uncertainty using Equation 1 (weighted RMS)
        # U_agent = sqrt(1/N * sum((W_i * U_i)^2))
        weighted_squared_uncertainties = [
            (w * u) ** 2
            for w, u in zip(situational_weights, step_uncertainties)
        ]
        agent_uncertainty = np.sqrt(np.mean(weighted_squared_uncertainties))

        result = {
            'agent_uncertainty': float(agent_uncertainty),
            'step_uncertainties': step_uncertainties,
            'situational_weights': situational_weights,
            'inquiry_drifts': inquiry_drifts,
            'inference_gaps': inference_gaps,
            'hmm_states': hmm_states
        }

        # Display results
        self.display_results(result, question)

        return result

    def display_results(self, result: Dict, question: str):
        """
        Display SAUP results in formatted tables.

        Parameters:
            result: SAUP result dictionary
            question: Original question
        """
        console.print()

        # Summary panel
        agent_unc = result['agent_uncertainty']
        confidence = (1 - agent_unc) * 100

        summary_text = (
            f"Agent Uncertainty: {format_uncertainty(agent_unc)}\n"
            f"Confidence: {confidence:.1f}%\n"
            f"Total Steps: {len(result['step_uncertainties'])}"
        )

        panel = Panel(
            summary_text,
            title="SAUP Results Summary",
            border_style="info"
        )
        console.print(panel)

        # Detailed step-by-step table
        table = Table(title="Step-by-Step Analysis")
        table.add_column("Step", justify="center", style="cyan")
        table.add_column("Uncertainty", justify="right")
        table.add_column("Inq. Drift", justify="right")
        table.add_column("Inf. Gap", justify="right")
        table.add_column("Weight", justify="right")
        if result['hmm_states']:
            table.add_column("State", justify="center")

        for i in range(len(result['step_uncertainties'])):
            step_num = i + 1
            uncertainty = result['step_uncertainties'][i]
            inquiry_drift = result['inquiry_drifts'][i]
            inference_gap = result['inference_gaps'][i]
            weight = result['situational_weights'][i]

            row = [
                f"Step {step_num}",
                f"{uncertainty:.3f}",
                f"{inquiry_drift:.3f}",
                f"{inference_gap:.3f}",
                f"{weight:.2f}"
            ]

            if result['hmm_states']:
                state = result['hmm_states'][i]
                row.append(state)

            table.add_row(*row)

        console.print(table)
        console.print()

    def compare_methods(self,
                       question: str,
                       trajectory: List[Dict]) -> Dict:
        """
        Compare SAUP with simpler uncertainty quantification methods.

        Parameters:
            question: Original question
            trajectory: Trajectory with uncertainties

        Returns:
            Dictionary with uncertainty values from different methods:
                - 'saup': SAUP uncertainty (situation-aware)
                - 'simple_average': Simple average of step uncertainties
                - 'simple_rms': Simple RMS without weighting
                - 'max_uncertainty': Maximum step uncertainty
        """
        if not trajectory:
            return {
                'saup': 0.0,
                'simple_average': 0.0,
                'simple_rms': 0.0,
                'max_uncertainty': 0.0
            }

        # Get SAUP uncertainty
        saup_result = self.propagate_uncertainty(question, trajectory)
        saup_uncertainty = saup_result['agent_uncertainty']

        # Extract step uncertainties
        step_uncertainties = [step.get('uncertainty', 0.0) for step in trajectory]

        # Simple average
        simple_average = np.mean(step_uncertainties)

        # Simple RMS (no weighting)
        simple_rms = np.sqrt(np.mean(np.array(step_uncertainties) ** 2))

        # Maximum uncertainty
        max_uncertainty = np.max(step_uncertainties)

        result = {
            'saup': saup_uncertainty,
            'simple_average': float(simple_average),
            'simple_rms': float(simple_rms),
            'max_uncertainty': float(max_uncertainty)
        }

        # Display comparison
        self.display_method_comparison(result)

        return result

    def display_method_comparison(self, comparison: Dict):
        """
        Display comparison of different uncertainty methods.

        Parameters:
            comparison: Dictionary with method names and uncertainty values
        """
        table = Table(title="Method Comparison")
        table.add_column("Method", style="cyan")
        table.add_column("Uncertainty", justify="right")
        table.add_column("Description", style="dim")

        method_descriptions = {
            'saup': 'Situation-Aware propagation with HMM',
            'simple_rms': 'RMS without situational weighting',
            'simple_average': 'Simple arithmetic average',
            'max_uncertainty': 'Maximum single-step uncertainty'
        }

        for method, uncertainty in comparison.items():
            description = method_descriptions.get(method, '')
            table.add_row(
                method.upper(),
                f"{uncertainty:.3f}",
                description
            )

        console.print(table)
