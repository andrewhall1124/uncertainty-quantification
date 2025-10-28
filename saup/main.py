"""
Main demo for SAUP (Situation-Aware Uncertainty Propagation) system.

This demonstrates the complete pipeline:
1. ReAct agent generates reasoning trajectory
2. SAUP propagates uncertainty through trajectory
3. Evaluation metrics compare with baseline methods
"""

import os
import sys
from typing import List, Dict
from dotenv import load_dotenv
from rich.panel import Panel
from rich.console import Group
from rich.text import Text

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import console
from models.llm_wrapper import LLMWrapper
from agents.react_agent import ReactAgent
from uncertainty.propagation import SAUPPropagator
from evaluation.metrics import visualize_results, compare_methods, create_uncertainty_histogram


def print_header():
    """Print ASCII art header for SAUP system."""
    header_text = """
╔══════════════════════════════════════════════════════════════╗
║           SAUP Uncertainty Quantification System             ║
║        Situation-Aware Uncertainty Propagation v1.0          ║
╚══════════════════════════════════════════════════════════════╝
    """
    console.print(f"[cyan]{header_text}[/cyan]")


def initialize_system(model_name: str = None) -> tuple:
    """
    Initialize all system components.

    Parameters:
        model_name: LLM model name (defaults to env variable or gpt-4)

    Returns:
        Tuple of (llm, agent, propagator)
    """
    console.print("\n[info]Initializing system components...[/info]\n")

    # Load environment variables
    load_dotenv()

    # Get model name
    if model_name is None:
        model_name = os.getenv("MODEL_NAME", "gpt-4")

    # Initialize LLM wrapper
    with console.status("[info]Loading LLM wrapper...[/info]"):
        llm = LLMWrapper(model_name=model_name)

    # Initialize ReAct agent
    agent = ReactAgent(llm=llm, max_steps=7, verbose=True)
    console.print("[success]✓ ReAct agent initialized[/success]")

    # Initialize SAUP propagator
    propagator = SAUPPropagator(use_hmm_proxy=True)
    console.print("[success]✓ SAUP propagator initialized[/success]")

    console.print("\n[success]All components ready![/success]\n")

    return llm, agent, propagator


def run_question(question: str,
                agent: ReactAgent,
                propagator: SAUPPropagator,
                ground_truth: bool = None) -> Dict:
    """
    Run a single question through the system.

    Parameters:
        question: Question to answer
        agent: ReAct agent
        propagator: SAUP propagator
        ground_truth: Whether the answer is correct (for evaluation)

    Returns:
        Result dictionary with answer, uncertainty, trajectory
    """
    console.print("\n" + "="*70 + "\n")

    # Run agent
    agent_result = agent.run(question)

    # Get trajectory
    trajectory = agent_result['trajectory']
    answer = agent_result['answer']

    # Propagate uncertainty with SAUP
    saup_result = propagator.propagate_uncertainty(question, trajectory)

    # Also calculate baseline methods for comparison
    comparison = propagator.compare_methods(question, trajectory)

    result = {
        'question': question,
        'answer': answer,
        'trajectory': trajectory,
        'saup_uncertainty': saup_result['agent_uncertainty'],
        'comparison': comparison,
        'correct': ground_truth,
        'steps': len(trajectory),
        'uncertainty': saup_result['agent_uncertainty']  # For evaluation
    }

    # Display final answer
    confidence = (1 - saup_result['agent_uncertainty']) * 100
    answer_text = (
        f"[cyan]Question:[/cyan] {question}\n\n"
        f"[success]Final Answer:[/success] {answer}\n\n"
        f"[uncertainty]SAUP Uncertainty:[/uncertainty] {saup_result['agent_uncertainty']:.3f}\n"
        f"[info]Confidence:[/info] {confidence:.1f}%\n"
        f"[dim]Steps taken:[/dim] {len(trajectory)}"
    )

    panel = Panel(
        answer_text,
        title="Final Result",
        border_style="success"
    )
    console.print("\n")
    console.print(panel)

    return result


def main():
    """Main demonstration function."""
    print_header()

    # Initialize system
    try:
        llm, agent, propagator = initialize_system()
    except Exception as e:
        console.print(f"[error]Initialization failed: {e}[/error]")
        console.print("\n[warning]Make sure to set OPENAI_API_KEY in .env file[/warning]")
        return

    # Test questions from different domains
    # Format: (question, ground_truth_correct)
    test_questions = [
        (
            "Can drinking coffee help with weight loss?",
            False  # Generally not without diet/exercise
        ),
        (
            "Can drinking water help improve concentration?",
            True  # Yes, hydration improves cognitive function
        ),
        (
            "Is the Eiffel Tower taller than the Statue of Liberty?",
            True  # Eiffel Tower is 300m vs Statue of Liberty 93m
        ),
    ]

    results = []

    # Process each question
    for i, (question, ground_truth) in enumerate(test_questions):
        console.print(f"\n[cyan]{'='*70}[/cyan]")
        console.print(f"[cyan]Question {i+1}/{len(test_questions)}[/cyan]")
        console.print(f"[cyan]{'='*70}[/cyan]\n")

        try:
            result = run_question(question, agent, propagator, ground_truth)
            results.append(result)
        except Exception as e:
            console.print(f"[error]Error processing question: {e}[/error]")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
            continue

    # Final evaluation
    if results:
        console.print("\n\n")
        console.print("[cyan]" + "="*70 + "[/cyan]")
        console.print("[cyan]Final Evaluation[/cyan]")
        console.print("[cyan]" + "="*70 + "[/cyan]\n")

        # Visualize results
        visualize_results(results, method_name="SAUP", show_per_question=True)

        # Show uncertainty distribution
        predictions = [r['correct'] for r in results if r['correct'] is not None]
        uncertainties = [r['uncertainty'] for r in results if r['correct'] is not None]

        if predictions and uncertainties:
            create_uncertainty_histogram(uncertainties, predictions)

        # Display LLM usage statistics
        console.print()
        llm.display_usage_stats()

    console.print("\n[success]Demo completed![/success]\n")


if __name__ == "__main__":
    main()
