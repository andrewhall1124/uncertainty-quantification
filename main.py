from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from rich.console import Console
from rich.table import Table


class TextGenerator:
    """
    A class to encapsulate text generation with probability tracking.
    """

    def __init__(self, model_name="gpt2"):
        """
        Initialize the text generator with a specific model.

        Args:
            model_name: HuggingFace model to use
        """
        self.console = Console()
        self.console.print(f"Loading model: {model_name}...", style="bold blue")
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()

    def generate_with_probabilities(self, prompt, max_new_tokens=10, top_k=5):
        """
        Generate text and show the probability distribution for each generated token.

        Args:
            prompt: Input text to continue
            max_new_tokens: Number of tokens to generate
            top_k: Number of top probable tokens to display

        Returns:
            Tuple of (generated_text, probabilities_list)
        """
        # Encode the prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        self.console.print(f"\n[bold]Prompt:[/bold] {prompt}\n")

        generated_text = prompt
        probabilities_list = []

        # Generate tokens one at a time
        for step in range(max_new_tokens):
            with torch.no_grad():
                # Get model outputs
                outputs = self.model(input_ids)

                # Get logits for the last token
                logits = outputs.logits[:, -1, :]

                # Convert to probabilities
                probs = torch.softmax(logits, dim=-1)

                # Get top-k tokens and their probabilities
                top_probs, top_indices = torch.topk(probs, top_k)

                # Create table for this step
                table = Table(title=f"Step {step + 1}: Top {top_k} Next Tokens")
                table.add_column("Rank", justify="center", style="cyan")
                table.add_column("Token", justify="left", style="magenta")
                table.add_column("Probability", justify="right", style="green")

                for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0])):
                    token = self.tokenizer.decode([idx])
                    prob_pct = f"{prob.item()*100:.2f}%"
                    style = "bold green" if i == 0 else None
                    table.add_row(str(i+1), repr(token), prob_pct, style=style)

                self.console.print(table)

                # Use greedy decoding (pick most probable token)
                next_token_id = top_indices[0, 0].unsqueeze(0).unsqueeze(0)
                chosen_token = self.tokenizer.decode([next_token_id.item()])
                chosen_prob = top_probs[0, 0].item()

                # Store probability of chosen token
                probabilities_list.append(chosen_prob)

                # Append to input for next iteration
                input_ids = torch.cat([input_ids, next_token_id], dim=-1)
                generated_text += chosen_token

        # Create final summary table
        self.console.print()

        # Add columns for each generated token
        tokens = self.tokenizer.encode(generated_text, return_tensors="pt")[0]
        generated_tokens = tokens[len(self.tokenizer.encode(prompt)):]

        # Split into chunks of 10 columns
        chunk_size = 10
        for i in range(0, len(generated_tokens), chunk_size):
            chunk_tokens = generated_tokens[i:i+chunk_size]
            chunk_probs = probabilities_list[i:i+chunk_size]

            title = f"Generation Summary (Tokens {i+1}-{i+len(chunk_tokens)})" if i == 0 else None
            summary_table = Table(title=title, show_header=False)

            for _ in range(len(chunk_tokens)):
                summary_table.add_column(justify="center")

            # Add tokens row
            token_row = [repr(self.tokenizer.decode([token_id])) for token_id in chunk_tokens]
            summary_table.add_row(*token_row, style="bold yellow")

            # Add probabilities row
            prob_row = [f"{prob:.4f}" for prob in chunk_probs]
            summary_table.add_row(*prob_row, style="green")

            self.console.print(summary_table)
            self.console.print()

        return generated_text, probabilities_list


if __name__ == "__main__":
    # Example usage
    prompt = "Andrew is"
    generator = TextGenerator(model_name="gpt2")
    generator.generate_with_probabilities(
        prompt=prompt,
        max_new_tokens=30,
        top_k=3
    )