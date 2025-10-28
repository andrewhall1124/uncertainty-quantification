"""
Example scripts demonstrating individual SAUP components.

Run these examples to understand how each component works.
"""

import os
import sys
from dotenv import load_dotenv

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import console
from rich.panel import Panel


def example_1_uncertainty_calculation():
    """Example 1: Calculate uncertainty from token probabilities."""
    console.print("\n" + "="*70)
    console.print("[cyan]Example 1: Single-Step Uncertainty Calculation[/cyan]")
    console.print("="*70 + "\n")

    from uncertainty.single_step import (
        calculate_predictive_entropy,
        calculate_step_uncertainty
    )

    # Simulate token probabilities
    # High confidence tokens (low uncertainty)
    high_conf_probs = [0.95, 0.92, 0.88, 0.90]
    # Low confidence tokens (high uncertainty)
    low_conf_probs = [0.45, 0.52, 0.48, 0.55]

    console.print("High confidence tokens: [0.95, 0.92, 0.88, 0.90]")
    high_unc = calculate_predictive_entropy(high_conf_probs)
    console.print(f"Uncertainty: [low_unc]{high_unc:.3f}[/low_unc] (Low)\n")

    console.print("Low confidence tokens: [0.45, 0.52, 0.48, 0.55]")
    low_unc = calculate_predictive_entropy(low_conf_probs)
    console.print(f"Uncertainty: [high_unc]{low_unc:.3f}[/high_unc] (High)\n")

    # Combined step uncertainty
    thinking_unc = 0.3
    action_unc = 0.4
    step_unc = calculate_step_uncertainty(thinking_unc, action_unc)
    console.print(f"Combined step uncertainty (α=0.5):")
    console.print(f"  Thinking: {thinking_unc:.2f} + Action: {action_unc:.2f}")
    console.print(f"  = Step: [med_unc]{step_unc:.2f}[/med_unc]\n")


def example_2_semantic_distance():
    """Example 2: Calculate semantic distances."""
    console.print("\n" + "="*70)
    console.print("[cyan]Example 2: Semantic Distance Calculation[/cyan]")
    console.print("="*70 + "\n")

    from utils.embeddings import SemanticDistanceCalculator

    console.print("[info]Loading semantic distance model...[/info]")
    calc = SemanticDistanceCalculator()

    # Test inquiry drift
    question = "Can drinking coffee help with weight loss?"
    trajectory = [
        {
            'thought': "Coffee contains caffeine which affects metabolism",
            'action': "search[caffeine metabolism]",
            'observation': "Caffeine slightly increases metabolic rate"
        },
        {
            'thought': "I should check if this leads to weight loss",
            'action': "search[caffeine weight loss studies]",
            'observation': "Studies show minimal weight loss effect"
        }
    ]

    drift = calc.calculate_inquiry_drift(question, trajectory)
    console.print(f"\nInquiry drift: [low_unc]{drift:.3f}[/low_unc]")
    console.print("(Agent stayed on topic)\n")

    # Test inference gap
    thought = "I expect to find that coffee helps significantly with weight loss"
    observation = "Studies show coffee has minimal effect on weight loss"

    gap = calc.calculate_inference_gap(thought, observation)
    console.print(f"Inference gap: [high_unc]{gap:.3f}[/high_unc]")
    console.print("(Large gap between expectation and reality)\n")


def example_3_hmm_proxy():
    """Example 3: HMM proxy for situational weighting."""
    console.print("\n" + "="*70)
    console.print("[cyan]Example 3: HMM Proxy Situational Weighting[/cyan]")
    console.print("="*70 + "\n")

    from models.hmm_proxy import HMMProxy

    hmm = HMMProxy()

    console.print("Scenario 1: Low drift, low gap (agent on track)")
    w1, s1 = hmm.estimate_situational_weight(
        inquiry_drift=0.2, inference_gap=0.15,
        step_number=1, total_steps=5
    )
    console.print(f"  Weight: [low_unc]{w1:.2f}[/low_unc], State: {s1}\n")

    console.print("Scenario 2: Moderate drift and gap")
    w2, s2 = hmm.estimate_situational_weight(
        inquiry_drift=0.5, inference_gap=0.45,
        step_number=3, total_steps=5
    )
    console.print(f"  Weight: [med_unc]{w2:.2f}[/med_unc], State: {s2}\n")

    console.print("Scenario 3: High drift, high gap (agent lost)")
    w3, s3 = hmm.estimate_situational_weight(
        inquiry_drift=0.8, inference_gap=0.75,
        step_number=5, total_steps=5
    )
    console.print(f"  Weight: [high_unc]{w3:.2f}[/high_unc], State: {s3}\n")


def example_4_saup_propagation():
    """Example 4: SAUP uncertainty propagation."""
    console.print("\n" + "="*70)
    console.print("[cyan]Example 4: SAUP Uncertainty Propagation[/cyan]")
    console.print("="*70 + "\n")

    from uncertainty.propagation import SAUPPropagator
    import numpy as np

    # Create mock trajectory
    question = "Is the Eiffel Tower taller than Big Ben?"

    trajectory = [
        {
            'thought': "I need to find the height of the Eiffel Tower",
            'action': "search[Eiffel Tower height]",
            'observation': "The Eiffel Tower is 300 meters tall",
            'uncertainty': 0.15  # Low uncertainty
        },
        {
            'thought': "Now I need Big Ben's height",
            'action': "search[Big Ben height]",
            'observation': "Big Ben (Elizabeth Tower) is 96 meters tall",
            'uncertainty': 0.12  # Low uncertainty
        },
        {
            'thought': "I can compare: 300m vs 96m, Eiffel Tower is taller",
            'action': "finish[Yes, the Eiffel Tower is taller]",
            'observation': "Task finished",
            'uncertainty': 0.08  # Very low uncertainty
        }
    ]

    propagator = SAUPPropagator(use_hmm_proxy=True)
    result = propagator.propagate_uncertainty(question, trajectory)

    console.print(f"\n[success]Agent uncertainty: {result['agent_uncertainty']:.3f}[/success]")
    console.print(f"Confidence: {(1-result['agent_uncertainty'])*100:.1f}%\n")


def example_5_evaluation():
    """Example 5: Evaluation metrics."""
    console.print("\n" + "="*70)
    console.print("[cyan]Example 5: Evaluation Metrics (AUROC)[/cyan]")
    console.print("="*70 + "\n")

    from evaluation.metrics import calculate_auroc, visualize_results

    # Mock results: uncertainty should be higher for incorrect predictions
    predictions = [True, True, False, True, False, True, False, True]
    uncertainties = [0.2, 0.15, 0.75, 0.18, 0.82, 0.25, 0.68, 0.12]

    console.print("Predictions: ", predictions)
    console.print("Uncertainties:", [f"{u:.2f}" for u in uncertainties])
    console.print()

    auroc = calculate_auroc(predictions, uncertainties)
    console.print(f"AUROC: [success]{auroc:.3f}[/success]")
    console.print(f"(Higher is better, 0.5 = random, 1.0 = perfect)\n")

    if auroc > 0.7:
        console.print("[success]Good calibration! Uncertainty separates correct/incorrect.[/success]")
    else:
        console.print("[warning]Poor calibration. Uncertainty doesn't predict correctness well.[/warning]")


def example_6_wikipedia_tool():
    """Example 6: Wikipedia search tool."""
    console.print("\n" + "="*70)
    console.print("[cyan]Example 6: Wikipedia Search Tool[/cyan]")
    console.print("="*70 + "\n")

    from agents.react_agent import ReactAgent
    import wikipedia

    console.print("Searching Wikipedia for 'Python programming'...")
    try:
        summary = wikipedia.summary("Python programming", sentences=2)
        console.print(f"\n[observation]{summary}[/observation]\n")
    except Exception as e:
        console.print(f"[error]Error: {e}[/error]\n")


def main():
    """Run all examples."""
    load_dotenv()

    header = Panel(
        "[cyan]SAUP System - Component Examples[/cyan]\n\n"
        "These examples demonstrate individual components.\n"
        "No API key needed (except for full LLM wrapper demo).",
        title="Examples",
        border_style="cyan"
    )
    console.print(header)

    try:
        # Run examples that don't need API key
        example_1_uncertainty_calculation()
        input("\nPress Enter to continue...")

        example_2_semantic_distance()
        input("\nPress Enter to continue...")

        example_3_hmm_proxy()
        input("\nPress Enter to continue...")

        example_4_saup_propagation()
        input("\nPress Enter to continue...")

        example_5_evaluation()
        input("\nPress Enter to continue...")

        example_6_wikipedia_tool()

        console.print("\n[success]All examples completed![/success]\n")
        console.print("To run the full demo with LLM: python main.py")

    except KeyboardInterrupt:
        console.print("\n[warning]Examples interrupted by user[/warning]")
    except Exception as e:
        console.print(f"\n[error]Error: {e}[/error]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")


if __name__ == "__main__":
    main()
