"""
Utility functions and shared Rich console setup for SAUP system.
"""

from rich.console import Console
from rich.theme import Theme

# Custom theme for SAUP system
custom_theme = Theme({
    "thinking": "cyan",
    "action": "yellow",
    "observation": "green",
    "uncertainty": "red",
    "success": "bright_green",
    "error": "bright_red",
    "low_unc": "green",
    "med_unc": "yellow",
    "high_unc": "red",
    "info": "blue",
    "warning": "yellow bold",
})

# Shared console instance
console = Console(theme=custom_theme)

def get_uncertainty_color(uncertainty: float) -> str:
    """
    Get the color style for an uncertainty value.

    Parameters:
        uncertainty: Uncertainty value (0-1)

    Returns:
        Color style name for Rich formatting
    """
    if uncertainty < 0.3:
        return "low_unc"
    elif uncertainty < 0.7:
        return "med_unc"
    else:
        return "high_unc"

def get_uncertainty_emoji(uncertainty: float) -> str:
    """
    Get an emoji indicator for uncertainty level.

    Parameters:
        uncertainty: Uncertainty value (0-1)

    Returns:
        Emoji string
    """
    if uncertainty < 0.3:
        return "✓"
    elif uncertainty < 0.7:
        return "⚠️"
    else:
        return "⚠"

def format_uncertainty(uncertainty: float) -> str:
    """
    Format uncertainty value with color and emoji.

    Parameters:
        uncertainty: Uncertainty value (0-1)

    Returns:
        Formatted string with color tags
    """
    color = get_uncertainty_color(uncertainty)
    emoji = get_uncertainty_emoji(uncertainty)

    if uncertainty < 0.3:
        level = "Low"
    elif uncertainty < 0.7:
        level = "Moderate"
    else:
        level = "High"

    return f"[{color}]{uncertainty:.2f} ({level}) {emoji}[/{color}]"
