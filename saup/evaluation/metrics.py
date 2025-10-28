"""
Evaluation metrics for SAUP system.

Includes AUROC calculation and visualization utilities.
"""

from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from rich.table import Table
from rich.panel import Panel
from rich.console import Group
from rich.text import Text

from utils import console, get_uncertainty_color


def calculate_auroc(predictions: List[bool],
                   uncertainties: List[float]) -> float:
    """
    Calculate AUROC (Area Under ROC Curve) score.

    AUROC measures how well uncertainty scores separate correct from
    incorrect predictions. Higher AUROC indicates better calibration.

    Parameters:
        predictions: List of correctness labels (True = correct, False = incorrect)
        uncertainties: List of uncertainty scores (0-1)

    Returns:
        AUROC score (0-1, where 0.5 is random and 1.0 is perfect)

    Example:
        >>> predictions = [True, True, False, True, False]
        >>> uncertainties = [0.1, 0.2, 0.8, 0.15, 0.75]
        >>> auroc = calculate_auroc(predictions, uncertainties)
        >>> print(f"AUROC: {auroc:.3f}")
    """
    if len(predictions) != len(uncertainties):
        raise ValueError("predictions and uncertainties must have the same length")

    if len(predictions) < 2:
        raise ValueError("Need at least 2 predictions to calculate AUROC")

    # Convert predictions to binary (correct=0, incorrect=1)
    # We want high uncertainty for incorrect predictions
    labels = [0 if pred else 1 for pred in predictions]

    # Calculate AUROC
    try:
        auroc = roc_auc_score(labels, uncertainties)
        return float(auroc)
    except ValueError as e:
        console.print(f"[warning]Warning: Could not calculate AUROC: {e}[/warning]")
        return 0.5  # Return random baseline


def calculate_accuracy(predictions: List[bool]) -> float:
    """
    Calculate accuracy from predictions.

    Parameters:
        predictions: List of correctness labels

    Returns:
        Accuracy (0-1)
    """
    if not predictions:
        return 0.0

    return sum(predictions) / len(predictions)


def calculate_selective_accuracy(predictions: List[bool],
                                 uncertainties: List[float],
                                 threshold: float) -> Tuple[float, int, int]:
    """
    Calculate accuracy on predictions below an uncertainty threshold.

    This measures performance when the system abstains from high-uncertainty
    predictions.

    Parameters:
        predictions: List of correctness labels
        uncertainties: List of uncertainty scores
        threshold: Uncertainty threshold (predictions above this are excluded)

    Returns:
        Tuple of (selective_accuracy, num_selected, total_predictions)

    Example:
        >>> predictions = [True, True, False, True, False]
        >>> uncertainties = [0.1, 0.2, 0.8, 0.15, 0.75]
        >>> acc, selected, total = calculate_selective_accuracy(
        ...     predictions, uncertainties, threshold=0.5
        ... )
    """
    selected_preds = [
        pred for pred, unc in zip(predictions, uncertainties)
        if unc < threshold
    ]

    if not selected_preds:
        return 0.0, 0, len(predictions)

    accuracy = sum(selected_preds) / len(selected_preds)
    return accuracy, len(selected_preds), len(predictions)


def visualize_results(results: List[Dict],
                     method_name: str = "SAUP",
                     show_per_question: bool = True):
    """
    Visualize evaluation results with rich formatting.

    Parameters:
        results: List of result dictionaries, each containing:
            - 'question': Question text
            - 'answer': Agent's answer
            - 'correct': Whether answer is correct
            - 'uncertainty': Uncertainty score
            - 'steps': Number of steps taken
        method_name: Name of the method being evaluated
        show_per_question: Whether to show per-question breakdown
    """
    if not results:
        console.print("[warning]No results to visualize[/warning]")
        return

    # Extract data
    predictions = [r['correct'] for r in results]
    uncertainties = [r['uncertainty'] for r in results]

    # Calculate metrics
    accuracy = calculate_accuracy(predictions)

    try:
        auroc = calculate_auroc(predictions, uncertainties)
    except Exception as e:
        auroc = 0.5
        console.print(f"[warning]Could not calculate AUROC: {e}[/warning]")

    # Calculate selective accuracy at different thresholds
    thresholds = [0.3, 0.5, 0.7]
    selective_accs = []
    for threshold in thresholds:
        sel_acc, num_sel, total = calculate_selective_accuracy(
            predictions, uncertainties, threshold
        )
        selective_accs.append((threshold, sel_acc, num_sel, total))

    # Overall summary
    console.print()
    summary_text = (
        f"[cyan]Method:[/cyan] {method_name}\n"
        f"[cyan]Questions:[/cyan] {len(results)}\n"
        f"[cyan]Accuracy:[/cyan] {accuracy:.1%}\n"
        f"[cyan]AUROC:[/cyan] {auroc:.3f}"
    )

    panel = Panel(
        summary_text,
        title="Evaluation Summary",
        border_style="info"
    )
    console.print(panel)
    console.print()

    # Per-question breakdown
    if show_per_question:
        table = Table(title=f"{method_name} - Per-Question Results")
        table.add_column("#", justify="center", style="cyan")
        table.add_column("Question", style="white", max_width=40)
        table.add_column("Answer", max_width=30)
        table.add_column("Correct", justify="center")
        table.add_column("Uncertainty", justify="right")
        table.add_column("Steps", justify="center")

        for i, result in enumerate(results):
            question_preview = result['question'][:40]
            if len(result['question']) > 40:
                question_preview += "..."

            answer_preview = str(result['answer'])[:30]
            if len(str(result['answer'])) > 30:
                answer_preview += "..."

            correct_str = "[success]✓[/success]" if result['correct'] else "[error]✗[/error]"

            uncertainty = result['uncertainty']
            unc_color = get_uncertainty_color(uncertainty)
            unc_str = f"[{unc_color}]{uncertainty:.3f}[/{unc_color}]"

            table.add_row(
                str(i + 1),
                question_preview,
                answer_preview,
                correct_str,
                unc_str,
                str(result.get('steps', '?'))
            )

        console.print(table)
        console.print()

    # Selective accuracy table
    sel_table = Table(title="Selective Accuracy (Abstaining on High Uncertainty)")
    sel_table.add_column("Threshold", justify="center", style="cyan")
    sel_table.add_column("Accuracy", justify="right")
    sel_table.add_column("Coverage", justify="right")
    sel_table.add_column("Selected", justify="center", style="dim")

    for threshold, sel_acc, num_sel, total in selective_accs:
        coverage = num_sel / total if total > 0 else 0
        sel_table.add_row(
            f"< {threshold:.1f}",
            f"{sel_acc:.1%}",
            f"{coverage:.1%}",
            f"{num_sel}/{total}"
        )

    console.print(sel_table)
    console.print()


def compare_methods(results_dict: Dict[str, List[Dict]]):
    """
    Compare multiple uncertainty quantification methods.

    Parameters:
        results_dict: Dictionary mapping method names to result lists
                     Each result list should follow the format in visualize_results

    Example:
        >>> results = {
        ...     'SAUP': saup_results,
        ...     'Simple RMS': rms_results,
        ...     'Average': avg_results
        ... }
        >>> compare_methods(results)
    """
    if not results_dict:
        console.print("[warning]No results to compare[/warning]")
        return

    table = Table(title="Method Comparison")
    table.add_column("Method", style="cyan")
    table.add_column("Accuracy", justify="right")
    table.add_column("AUROC", justify="right")
    table.add_column("Avg Uncertainty", justify="right")
    table.add_column("# Questions", justify="center", style="dim")

    for method_name, results in results_dict.items():
        if not results:
            continue

        predictions = [r['correct'] for r in results]
        uncertainties = [r['uncertainty'] for r in results]

        accuracy = calculate_accuracy(predictions)

        try:
            auroc = calculate_auroc(predictions, uncertainties)
        except:
            auroc = 0.5

        avg_uncertainty = np.mean(uncertainties)

        table.add_row(
            method_name,
            f"{accuracy:.1%}",
            f"{auroc:.3f}",
            f"{avg_uncertainty:.3f}",
            str(len(results))
        )

    console.print(table)
    console.print()


def create_uncertainty_histogram(uncertainties: List[float],
                                predictions: List[bool],
                                bins: int = 10):
    """
    Create a simple text-based histogram of uncertainties.

    Parameters:
        uncertainties: List of uncertainty scores
        predictions: List of correctness labels
        bins: Number of histogram bins
    """
    if not uncertainties:
        return

    # Separate correct and incorrect
    correct_unc = [u for u, p in zip(uncertainties, predictions) if p]
    incorrect_unc = [u for u, p in zip(uncertainties, predictions) if not p]

    console.print("\n[info]Uncertainty Distribution[/info]\n")

    if correct_unc:
        avg_correct = np.mean(correct_unc)
        console.print(f"[success]Correct predictions:[/success] Avg uncertainty = {avg_correct:.3f}")

    if incorrect_unc:
        avg_incorrect = np.mean(incorrect_unc)
        console.print(f"[error]Incorrect predictions:[/error] Avg uncertainty = {avg_incorrect:.3f}")

    console.print()
