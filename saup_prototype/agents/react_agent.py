"""
ReAct agent implementation with Wikipedia search tool.

Implements the Thought-Action-Observation reasoning loop.
"""

import re
from typing import Dict, List, Optional, Tuple
import wikipedia
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from agents.base_agent import BaseAgent
from models.llm_wrapper import LLMWrapper
from uncertainty.single_step import calculate_step_uncertainty_from_probs
from utils import console, format_uncertainty


class ReactAgent(BaseAgent):
    """
    ReAct (Reasoning + Acting) agent.

    Follows a Thought-Action-Observation loop to answer questions,
    using Wikipedia as its knowledge source.
    """

    def __init__(self,
                 llm: LLMWrapper,
                 max_steps: int = 7,
                 verbose: bool = True):
        """
        Initialize ReAct agent.

        Parameters:
            llm: LLM wrapper for generating thoughts and actions
            max_steps: Maximum number of reasoning steps
            verbose: Whether to display step-by-step output
        """
        super().__init__(max_steps=max_steps)
        self.llm = llm
        self.verbose = verbose

    def _parse_action(self, action_text: str) -> Tuple[str, str]:
        """
        Parse action from LLM output.

        Expected formats:
        - search[query]
        - finish[answer]

        Parameters:
            action_text: Raw action text from LLM

        Returns:
            Tuple of (action_type, action_argument)
        """
        # Try to match search[...]
        search_match = re.search(r'search\[(.*?)\]', action_text, re.IGNORECASE)
        if search_match:
            return 'search', search_match.group(1).strip()

        # Try to match finish[...]
        finish_match = re.search(r'finish\[(.*?)\]', action_text, re.IGNORECASE)
        if finish_match:
            return 'finish', finish_match.group(1).strip()

        # Fallback: treat entire text as search
        return 'search', action_text.strip()

    def _search_wikipedia(self, query: str) -> str:
        """
        Search Wikipedia and return summary.

        Parameters:
            query: Search query

        Returns:
            Wikipedia summary or error message
        """
        try:
            # Search for the query
            results = wikipedia.search(query)

            if not results:
                return f"No Wikipedia results found for '{query}'"

            # Get the first result's summary
            page = wikipedia.page(results[0], auto_suggest=False)
            summary = wikipedia.summary(results[0], sentences=3, auto_suggest=False)

            return summary

        except wikipedia.exceptions.DisambiguationError as e:
            # If disambiguation, use the first option
            try:
                summary = wikipedia.summary(e.options[0], sentences=3, auto_suggest=False)
                return summary
            except:
                return f"Disambiguation error: {', '.join(e.options[:5])}"

        except wikipedia.exceptions.PageError:
            return f"Wikipedia page not found for '{query}'"

        except Exception as e:
            return f"Wikipedia search error: {str(e)}"

    def _generate_thought(self,
                          question: str,
                          history: List[Dict]) -> Tuple[str, List[float]]:
        """
        Generate thought/reasoning for the current step.

        Parameters:
            question: Original question
            history: Previous steps

        Returns:
            Tuple of (thought_text, token_probabilities)
        """
        # Build prompt with history
        prompt_parts = [f"Question: {question}\n"]

        for i, step in enumerate(history):
            prompt_parts.append(f"Thought {i+1}: {step['thought']}")
            prompt_parts.append(f"Action {i+1}: {step['action']}")
            prompt_parts.append(f"Observation {i+1}: {step['observation']}\n")

        prompt_parts.append(
            f"Thought {len(history)+1}: Let me think step by step about what to do next."
        )

        prompt = "\n".join(prompt_parts)

        # Generate thought with uncertainty
        thought, probs = self.llm.generate_with_uncertainty(
            prompt=prompt,
            max_tokens=150,
            temperature=0.7,
            stop=["\n"]
        )

        return thought.strip(), probs

    def _generate_action(self,
                        question: str,
                        thought: str,
                        history: List[Dict]) -> Tuple[str, List[float]]:
        """
        Generate action based on thought.

        Parameters:
            question: Original question
            thought: Current thought
            history: Previous steps

        Returns:
            Tuple of (action_text, token_probabilities)
        """
        prompt = f"Question: {question}\nThought: {thought}\nAction: "

        # Generate action with uncertainty
        action, probs = self.llm.generate_with_uncertainty(
            prompt=prompt,
            max_tokens=50,
            temperature=0.5,
            stop=["\n", "Observation"]
        )

        return action.strip(), probs

    def step(self, state: Dict) -> Dict:
        """
        Execute a single reasoning step.

        Parameters:
            state: Dictionary with 'question' and 'history'

        Returns:
            Dictionary with 'thought', 'action', 'observation', 'uncertainty'
        """
        question = state['question']
        history = state.get('history', [])

        # Generate thought
        thought, thought_probs = self._generate_thought(question, history)

        # Generate action
        action, action_probs = self._generate_action(question, thought, history)

        # Execute action
        action_type, action_arg = self._parse_action(action)

        if action_type == 'search':
            observation = self._search_wikipedia(action_arg)
        elif action_type == 'finish':
            observation = "Task finished"
        else:
            observation = "Invalid action"

        # Calculate step uncertainty
        _, _, step_uncertainty = calculate_step_uncertainty_from_probs(
            thought_text=thought,
            thought_probs=thought_probs,
            action_text=action,
            action_probs=action_probs,
            show_progress=False
        )

        return {
            'thought': thought,
            'action': action,
            'observation': observation,
            'uncertainty': step_uncertainty,
            'thought_probs': thought_probs,
            'action_probs': action_probs,
            'action_type': action_type,
            'action_arg': action_arg
        }

    def run(self, question: str) -> Dict:
        """
        Run the agent on a question.

        Parameters:
            question: Question to answer

        Returns:
            Dictionary with 'answer', 'trajectory', 'success'
        """
        self.reset()

        if self.verbose:
            console.print()
            panel = Panel(
                f"[info]{question}[/info]",
                title="Question",
                border_style="info"
            )
            console.print(panel)
            console.print()

        answer = None
        success = False

        for step_num in range(1, self.max_steps + 1):
            if self.verbose:
                console.print(f"[cyan]━━━ Step {step_num}/{self.max_steps} ━━━[/cyan]\n")

            # Execute step
            state = {
                'question': question,
                'history': self.trajectory
            }

            step_result = self.step(state)

            # Add to trajectory
            self.trajectory.append(step_result)

            # Display step
            if self.verbose:
                self._display_step(step_num, step_result)

            # Check if finished
            if step_result['action_type'] == 'finish':
                answer = step_result['action_arg']
                success = True
                break

        # If didn't finish, use last observation as answer
        if not answer and self.trajectory:
            answer = self.trajectory[-1]['observation']

        return {
            'answer': answer,
            'trajectory': self.trajectory,
            'success': success,
            'steps_taken': len(self.trajectory)
        }

    def _display_step(self, step_num: int, step: Dict):
        """
        Display a step with rich formatting.

        Parameters:
            step_num: Step number
            step: Step dictionary
        """
        # Thought
        console.print(f"[thinking]💭 Thought:[/thinking] {step['thought']}\n")

        # Action
        action_display = step['action']
        if step['action_type'] == 'search':
            action_display = f"search[{step['action_arg']}]"
        elif step['action_type'] == 'finish':
            action_display = f"finish[{step['action_arg']}]"

        console.print(f"[action]🎯 Action:[/action] {action_display}\n")

        # Observation
        obs_preview = step['observation'][:200]
        if len(step['observation']) > 200:
            obs_preview += "..."
        console.print(f"[observation]📊 Observation:[/observation] {obs_preview}\n")

        # Uncertainty
        uncertainty = step['uncertainty']
        console.print(f"[uncertainty]📈 Uncertainty:[/uncertainty] {format_uncertainty(uncertainty)}\n")

        console.print()
