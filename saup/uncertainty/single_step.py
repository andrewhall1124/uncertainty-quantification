"""
Single-step uncertainty calculation.

Implements entropy-based uncertainty measures from SAUP paper.
"""

from typing import List, Tuple
import numpy as np
from rich.progress import Progress, SpinnerColumn, TextColumn

from utils import console


def calculate_predictive_entropy(token_probs: List[float]) -> float:
    """
    Calculate normalized predictive entropy.

    Implements Equation 3 from SAUP paper: H(R_n | Q, Z_{n-1})
    with length normalization.

    Parameters:
        token_probs: List of token probabilities (0-1)

    Returns:
        Normalized entropy value (0-1)

    Example:
        >>> probs = [0.9, 0.85, 0.7, 0.8]
        >>> entropy = calculate_predictive_entropy(probs)
        >>> print(f"Entropy: {entropy:.3f}")
    """
    if not token_probs or len(token_probs) == 0:
        return 0.0

    # Convert to numpy array and ensure valid probabilities
    probs = np.array(token_probs)
    probs = np.clip(probs, 1e-10, 1.0)  # Avoid log(0)

    # Calculate entropy: -sum(p * log(p))
    entropy = -np.sum(probs * np.log(probs))

    # Length normalization: divide by number of tokens
    normalized_entropy = entropy / len(probs)

    # Normalize to [0, 1] range
    # Maximum entropy for a token is -log(p) where p is very small
    # For practical purposes, normalize by log(2) to get roughly [0, 1]
    normalized_entropy = normalized_entropy / np.log(2)

    return float(np.clip(normalized_entropy, 0.0, 1.0))


def calculate_thinking_uncertainty(thought_text: str, token_probs: List[float]) -> float:
    """
    Calculate thinking uncertainty (U^T_n).

    Measures uncertainty in the agent's reasoning/thought process.
    This is the predictive entropy over the thought tokens.

    Parameters:
        thought_text: The thought text (used for length info)
        token_probs: Token probabilities for the thought

    Returns:
        Thinking uncertainty value (0-1)

    Example:
        >>> thought = "I should search for information about coffee"
        >>> probs = [0.9, 0.85, 0.8, 0.88, 0.92]
        >>> unc = calculate_thinking_uncertainty(thought, probs)
    """
    if not token_probs:
        return 0.0

    return calculate_predictive_entropy(token_probs)


def calculate_action_uncertainty(action_text: str, token_probs: List[float]) -> float:
    """
    Calculate action uncertainty (U^A_n).

    Measures uncertainty in the agent's action selection.
    This is the predictive entropy over the action tokens.

    Parameters:
        action_text: The action text (used for length info)
        token_probs: Token probabilities for the action

    Returns:
        Action uncertainty value (0-1)

    Example:
        >>> action = "search[coffee metabolism weight loss]"
        >>> probs = [0.85, 0.82, 0.78]
        >>> unc = calculate_action_uncertainty(action, probs)
    """
    if not token_probs:
        return 0.0

    return calculate_predictive_entropy(token_probs)


def calculate_step_uncertainty(thought_uncertainty: float,
                               action_uncertainty: float,
                               alpha: float = 0.5) -> float:
    """
    Calculate combined step uncertainty (U_n).

    Combines thinking and action uncertainties into a single step uncertainty.
    Based on: U_n = α * U^T_n + (1-α) * U^A_n

    Parameters:
        thought_uncertainty: Thinking uncertainty (0-1)
        action_uncertainty: Action uncertainty (0-1)
        alpha: Weight for thinking uncertainty (default 0.5 for equal weighting)

    Returns:
        Step uncertainty value (0-1)

    Example:
        >>> step_unc = calculate_step_uncertainty(0.3, 0.4)
        >>> print(f"Step uncertainty: {step_unc:.3f}")
    """
    # Weighted combination
    step_uncertainty = alpha * thought_uncertainty + (1 - alpha) * action_uncertainty

    return float(np.clip(step_uncertainty, 0.0, 1.0))


def calculate_step_uncertainty_from_probs(thought_text: str,
                                         thought_probs: List[float],
                                         action_text: str,
                                         action_probs: List[float],
                                         show_progress: bool = False) -> Tuple[float, float, float]:
    """
    Calculate all uncertainty components for a step from token probabilities.

    Convenience function that calculates thinking, action, and combined
    step uncertainties.

    Parameters:
        thought_text: The thought text
        thought_probs: Token probabilities for thought
        action_text: The action text
        action_probs: Token probabilities for action
        show_progress: Whether to show progress bar

    Returns:
        Tuple of (thinking_uncertainty, action_uncertainty, step_uncertainty)

    Example:
        >>> t_unc, a_unc, s_unc = calculate_step_uncertainty_from_probs(
        ...     "I need to search", [0.9, 0.85],
        ...     "search[query]", [0.8, 0.75],
        ...     show_progress=True
        ... )
    """
    if show_progress:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("[info]Calculating step uncertainty...[/info]", total=3)

            # Calculate thinking uncertainty
            thinking_unc = calculate_thinking_uncertainty(thought_text, thought_probs)
            progress.advance(task)

            # Calculate action uncertainty
            action_unc = calculate_action_uncertainty(action_text, action_probs)
            progress.advance(task)

            # Calculate step uncertainty
            step_unc = calculate_step_uncertainty(thinking_unc, action_unc)
            progress.advance(task)
    else:
        thinking_unc = calculate_thinking_uncertainty(thought_text, thought_probs)
        action_unc = calculate_action_uncertainty(action_text, action_probs)
        step_unc = calculate_step_uncertainty(thinking_unc, action_unc)

    return thinking_unc, action_unc, step_unc
