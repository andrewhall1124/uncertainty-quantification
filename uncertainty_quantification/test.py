from uncertainty_quantification.react_agent import ReActAgent, calculator_tool, wikipedia_tool

# Example 1: Simple question (should answer directly without tools)
print("=" * 60)
print("Example 1: Factual question (no tools needed)")
print("=" * 60)
agent = ReActAgent(
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    tools=[],  # No tools - agent should answer from knowledge
    max_iterations=3
)
answer = agent.run("What is the capital of France?")

# Example 2: Math question (needs calculator)
print("\n" + "=" * 60)
print("Example 2: Math problem (calculator tool needed)")
print("=" * 60)
agent = ReActAgent(
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    tools=[calculator_tool()],
    max_iterations=5
)
answer = agent.run("What is 234 * 567 + 123?")

# Example 3: Wikipedia search for real information
print("\n" + "=" * 60)
print("Example 3: Wikipedia search for real-time information")
print("=" * 60)
agent = ReActAgent(
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    tools=[wikipedia_tool()],
    max_iterations=5
)
answer = agent.run("Who is Albert Einstein?")