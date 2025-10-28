from uncertainty_quantification.react_agent import ReActAgent, calculator_tool

# Create an agent
agent = ReActAgent(
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    tools=[calculator_tool()],
)

# Run a task
answer = agent.run("What is 25 * 4 + 100?")