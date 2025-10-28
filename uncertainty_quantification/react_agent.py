"""
Simple ReAct Agent using Hugging Face models.
Follows: Thought -> Action -> Observation loop.
"""

import re
from typing import List, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


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

        # Load model
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = "mps" if torch.mps.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        print(f"Loaded on {self.device}")

    def _make_prompt(self) -> str:
        """Create system prompt with tools."""
        tools_text = "\n".join(f"- {name}: {tool.description}"
                               for name, tool in self.tools.items())
        return f"""You solve tasks using tools. Format:

Thought: [reasoning]
Action: [tool_name]
Action Input: [input]

When done:
Thought: I have the answer
Final Answer: [answer]

Tools:
{tools_text}
"""

    def _generate(self, prompt: str) -> str:
        """Generate response from model."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()

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
        print(f"\nTask: {task}\n")

        for i in range(self.max_iterations):
            print(f"--- Step {i+1} ---")

            # Generate
            response = self._generate(prompt)
            print(f"Agent: {response}\n")
            prompt += response + "\n"

            # Check if done
            final = self._parse_final(response)
            if final:
                print(f"Final Answer: {final}\n")
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

                print(obs)
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


def search_tool(knowledge: dict) -> Tool:
    """Knowledge base search."""
    def search(query: str) -> str:
        q = query.lower().strip()
        for key, val in knowledge.items():
            if q in key.lower():
                return val
        return f"Not found: {query}"

    return Tool("Search", search, "Search knowledge base")
