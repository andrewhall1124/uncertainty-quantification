import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tabulate import tabulate

class MaxProbUncertainty:
    """
    Simplest uncertainty: 1 - max(P(y|x))
    
    Pros: Fast, no extra computation
    Cons: Often overconfident, doesn't capture epistemic uncertainty
    """
    
    def __init__(self, model_name="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.model.eval()
    
    def get_uncertainty(self, prompt: str, max_new_tokens: int = 20, temperature: float = 1.0, top_p: float = 1.0, seed: int = None):
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Set seed for reproducibility if provided
        if seed is not None:
            torch.manual_seed(seed)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Get probabilities for each token
        token_probs = []
        token_entropies = []
        for scores in outputs.scores:
            probs = F.softmax(scores[0], dim=-1)
            max_prob = probs.max().item()
            
            # Entropy: -Σ p log p
            entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
            
            token_probs.append(max_prob)
            token_entropies.append(entropy)
        
        generated_text = self.tokenizer.decode(
            outputs.sequences[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        print(probs)
        
        return {
            'text': generated_text,
            'avg_confidence': sum(token_probs) / len(token_probs),
            'min_confidence': min(token_probs),
            'avg_entropy': sum(token_entropies) / len(token_entropies),
            'sequence_prob': sum(token_probs) / len(token_probs)  # Geometric mean approx
        }


if __name__ == '__main__':
    # Usage
    uq = MaxProbUncertainty()

    # Generate 10 different outputs with different seeds
    prompt = "The capital of France is"
    results = []

    print(f"Generating 1 output(s) for prompt: '{prompt}'\n")

    for i in range(1):
        result = uq.get_uncertainty(prompt, max_new_tokens=5, temperature=1.0, seed=i)
        results.append({
            'Run': i + 1,
            'Text': result['text'][:50] + '...' if len(result['text']) > 50 else result['text'],
            'Avg Conf': f"{result['avg_confidence']:.4f}",
            'Min Conf': f"{result['min_confidence']:.4f}",
            'Avg Entropy': f"{result['avg_entropy']:.4f}",
            'Seq Prob': f"{result['sequence_prob']:.4f}"
        })

    # Print table
    print(tabulate(results, headers='keys', tablefmt='grid'))

    # Print summary statistics
    avg_confs = [float(r['Avg Conf']) for r in results]
    avg_entropies = [float(r['Avg Entropy']) for r in results]

    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS ACROSS 10 RUNS:")
    print(f"{'='*80}")
    print(f"Average Confidence - Mean: {sum(avg_confs)/len(avg_confs):.4f}, Std: {torch.tensor(avg_confs).std().item():.4f}")
    print(f"Average Entropy    - Mean: {sum(avg_entropies)/len(avg_entropies):.4f}, Std: {torch.tensor(avg_entropies).std().item():.4f}")