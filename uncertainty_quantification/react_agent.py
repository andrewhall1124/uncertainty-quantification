"""
Simple ReAct Agent using Hugging Face models.
Follows: Thought -> Action -> Observation loop.
"""

import re
from typing import List, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rich.console import Console


class Tool:
    """A tool the agent can use."""
    def __init__(self, name: str, func: callable, description: str):
        self.name = name
        self.func = func
        self.description = description

    def __call__(self, input_str: str) -> str:
        return self.func(input_str)


class ReActAgent:
    """ReAct Agent: Thinks, Acts, Observes, Repeats."""

    def __init__(self, model_name: str, tools: List[Tool] = None, max_iterations: int = 10):
        self.max_iterations = max_iterations
        self.tools = {tool.name: tool for tool in (tools or [])}
        self.console = Console()

        # Load model
        self.console.print(f"Loading {model_name}...", style="dim")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = "mps" if torch.mps.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.console.print(f"Loaded on {self.device}", style="dim")

    def _make_prompt(self) -> str:
        """Create system prompt with tools."""
        tools_text = "\n".join(f"- {name}: {tool.description}"
                               for name, tool in self.tools.items())

        if not self.tools:
            tools_text = "No tools available"

        return f"""You are a ReAct (Reasoning and Acting) agent that solves tasks step-by-step.

Follow this exact format for each step:

Thought: [Your reasoning about what to do next]

If you need to use a tool:
Action: [tool_name]
Action Input: [input for the tool]

If you can answer directly without tools:
Final Answer: [Your complete answer to the task]

After each action, you will receive an Observation with the result. Use this to inform your next thought.

Available Tools:
{tools_text}

CRITICAL RULES:
- ONLY use tools from the Available Tools list above
- If you don't have a tool for something, answer from your own knowledge
- If you cannot answer without a tool you don't have, say so in your Final Answer
- NEVER invent or use tools that are not in the Available Tools list
- Think step-by-step and be explicit about your reasoning
"""

    def _generate(self, prompt: str) -> str:
        """Generate response from model."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,  # Reduced to force conciseness
                temperature=0.3,  # Lower temperature for more focused output
                do_sample=True,
                top_p=0.9,  # Nucleus sampling for better quality
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()

        # Stop at double newline or "Observation:" to prevent hallucinated obs
        for stop in ['\n\nObservation:', '\nObservation:', 'Observation:']:
            if stop in response:
                response = response.split(stop)[0].strip()
                break

        return response

    def _parse_action(self, text: str) -> Optional[Tuple[str, str]]:
        """Extract action and input from response."""
        action = re.search(r'Action:\s*(.+?)(?:\n|$)', text, re.I)
        action_input = re.search(r'Action Input:\s*(.+?)(?:\n|$)', text, re.I)
        if action and action_input:
            return action.group(1).strip(), action_input.group(1).strip()
        return None

    def _parse_final(self, text: str) -> Optional[str]:
        """Extract final answer from response."""
        match = re.search(r'Final Answer:\s*(.+)', text, re.I | re.DOTALL)
        return match.group(1).strip() if match else None

    def run(self, task: str) -> str:
        """Run agent on task."""
        prompt = f"{self._make_prompt()}\n\nTask: {task}\n\n"
        self.console.print(f"\n[bold cyan]Task:[/bold cyan] {task}\n")

        for i in range(self.max_iterations):
            self.console.print(f"[bold]--- Step {i+1} ---[/bold]")

            # Generate
            response = self._generate(prompt)
            self.console.print(f"[yellow]{response}[/yellow]\n")
            prompt += response + "\n"

            # Check if done
            final = self._parse_final(response)
            if final:
                self.console.print(f"[bold green]Final Answer:[/bold green] {final}\n")
                return final

            # Execute action
            action = self._parse_action(response)
            if action:
                tool_name, tool_input = action
                if tool_name in self.tools:
                    try:
                        result = self.tools[tool_name](tool_input)
                        obs = f"Observation: {result}\n"
                    except Exception as e:
                        obs = f"Observation: Error - {e}\n"
                else:
                    obs = f"Observation: Unknown tool '{tool_name}'\n"

                self.console.print(f"[blue]{obs}[/blue]")
                prompt += obs

        return f"Max iterations ({self.max_iterations}) reached"


# Simple tool builders
def calculator_tool() -> Tool:
    """Math calculator."""
    def calc(expr: str) -> str:
        try:
            return str(eval(expr, {"__builtins__": {}}, {}))
        except Exception as e:
            return f"Error: {e}"

    return Tool("Calculator", calc, "Evaluates math like '2+2' or '10*5'")


def wikipedia_tool() -> Tool:
    """Wikipedia search and summary retrieval."""
    import urllib.parse
    import urllib.request
    import json

    def search_wikipedia(query: str) -> str:
        """Search Wikipedia and return a summary."""
        try:
            # Clean the query
            query = query.strip()

            # Wikipedia API endpoint for search
            search_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={urllib.parse.quote(query)}&limit=1&format=json"

            with urllib.request.urlopen(search_url, timeout=10) as response:
                search_results = json.loads(response.read().decode())

            if not search_results[1]:  # No results found
                return f"No Wikipedia results found for: {query}"

            # Get the title of the first result
            title = search_results[1][0]

            # Get page summary using TextExtracts API
            summary_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro=true&explaintext=true&titles={urllib.parse.quote(title)}&format=json"

            with urllib.request.urlopen(summary_url, timeout=10) as response:
                summary_data = json.loads(response.read().decode())

            # Extract the page content
            pages = summary_data['query']['pages']
            page_id = list(pages.keys())[0]

            if page_id == '-1':
                return f"No Wikipedia page found for: {query}"

            extract = pages[page_id].get('extract', 'No summary available')

            # Limit to first 500 characters for conciseness
            if len(extract) > 500:
                extract = extract[:500] + "..."

            return f"Wikipedia: {title}\n{extract}"

        except urllib.error.URLError as e:
            return f"Network error accessing Wikipedia: {str(e)}"
        except Exception as e:
            return f"Error searching Wikipedia: {str(e)}"

    return Tool(
        "Wikipedia",
        search_wikipedia,
        "Search Wikipedia for information about a topic. Input should be a search query."
    )
