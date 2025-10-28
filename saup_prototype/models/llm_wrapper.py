"""
LLM Wrapper for interfacing with language models and extracting uncertainties.
"""

import time
import os
from typing import Tuple, List, Dict, Optional
import numpy as np
from openai import OpenAI
import tiktoken
from rich.panel import Panel
from rich.table import Table

from utils import console


class LLMWrapper:
    """
    Wrapper for language models supporting OpenAI API.

    Provides methods to generate text with uncertainty estimates
    based on token probabilities.
    """

    def __init__(self,
                 model_name: str = "gpt-4",
                 api_key: Optional[str] = None,
                 max_retries: int = 3):
        """
        Initialize the LLM wrapper.

        Parameters:
            model_name: Name of the model to use (e.g., 'gpt-4', 'gpt-3.5-turbo')
            api_key: OpenAI API key (if None, reads from environment)
            max_retries: Maximum number of retries for API calls
        """
        self.model_name = model_name
        self.max_retries = max_retries
        self.total_tokens = 0
        self.total_cost = 0.0

        # Initialize OpenAI client
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

        self.client = OpenAI(api_key=api_key)

        # Initialize tokenizer for counting tokens
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        console.print(f"[success]✓ LLM Wrapper initialized ({model_name})[/success]")

    def _count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        return len(self.tokenizer.encode(text))

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Calculate cost based on token usage.

        Prices as of 2024 (adjust as needed):
        - GPT-4: $0.03/1K prompt tokens, $0.06/1K completion tokens
        - GPT-3.5-turbo: $0.0015/1K prompt tokens, $0.002/1K completion tokens
        """
        if "gpt-4" in self.model_name:
            prompt_cost = (prompt_tokens / 1000) * 0.03
            completion_cost = (completion_tokens / 1000) * 0.06
        else:  # gpt-3.5-turbo
            prompt_cost = (prompt_tokens / 1000) * 0.0015
            completion_cost = (completion_tokens / 1000) * 0.002

        return prompt_cost + completion_cost

    def generate(self,
                 prompt: str,
                 max_tokens: int = 500,
                 temperature: float = 0.7,
                 stop: Optional[List[str]] = None) -> str:
        """
        Generate text from a prompt.

        Parameters:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop: Stop sequences

        Returns:
            Generated text
        """
        response, _ = self.generate_with_uncertainty(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop
        )
        return response

    def generate_with_uncertainty(self,
                                  prompt: str,
                                  max_tokens: int = 500,
                                  temperature: float = 0.7,
                                  stop: Optional[List[str]] = None,
                                  logprobs: int = 5) -> Tuple[str, List[float]]:
        """
        Generate text with token-level uncertainty estimates.

        Uses logprobs from OpenAI API to estimate uncertainty at each token.

        Parameters:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop: Stop sequences
            logprobs: Number of top logprobs to return (max 5 for OpenAI)

        Returns:
            Tuple of (generated_text, token_probabilities)
            where token_probabilities is a list of probabilities for each token

        Example:
            >>> llm = LLMWrapper()
            >>> text, probs = llm.generate_with_uncertainty("What is 2+2?")
            >>> print(f"Text: {text}, Avg prob: {np.mean(probs):.3f}")
        """
        for attempt in range(self.max_retries):
            try:
                # Display API call
                with console.status(f"[info]Calling {self.model_name}...[/info]"):
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stop=stop,
                        logprobs=True,
                        top_logprobs=logprobs
                    )

                # Extract response text
                text = response.choices[0].message.content or ""

                # Extract token probabilities
                token_probs = []
                if response.choices[0].logprobs and response.choices[0].logprobs.content:
                    for token_info in response.choices[0].logprobs.content:
                        # Convert log probability to probability
                        prob = np.exp(token_info.logprob)
                        token_probs.append(float(prob))

                # Track usage
                if response.usage:
                    prompt_tokens = response.usage.prompt_tokens
                    completion_tokens = response.usage.completion_tokens
                    total_tokens = response.usage.total_tokens

                    self.total_tokens += total_tokens
                    cost = self._calculate_cost(prompt_tokens, completion_tokens)
                    self.total_cost += cost

                    console.print(
                        f"[dim]API call: {prompt_tokens} + {completion_tokens} = "
                        f"{total_tokens} tokens (${cost:.4f})[/dim]"
                    )

                return text, token_probs

            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    console.print(
                        f"[warning]API error: {e}. Retrying in {wait_time}s...[/warning]"
                    )
                    time.sleep(wait_time)
                else:
                    console.print(f"[error]API error after {self.max_retries} attempts: {e}[/error]")
                    raise

        return "", []

    def display_usage_stats(self):
        """Display token usage and cost statistics."""
        table = Table(title="LLM Usage Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="yellow")

        table.add_row("Total Tokens", f"{self.total_tokens:,}")
        table.add_row("Total Cost", f"${self.total_cost:.4f}")
        table.add_row("Model", self.model_name)

        console.print(table)
