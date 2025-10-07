import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple
from collections import defaultdict
import torch.nn.functional as F

class UncertaintyQuantifier:
    """
    Uncertainty quantification for autoregressive LLMs based on:
    "Uncertainty Estimation in Autoregressive Structured Prediction" (Malinin & Gales, 2021)
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    ):
        """
        Args:
            model_name: Hugging Face model identifier
            device: Device to run on
        """
        self.device = device

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()

        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def enable_dropout(self):
        """Enable dropout for MC Dropout ensemble"""
        for module in self.model.modules():
            if module.__class__.__name__.startswith('Dropout'):
                module.train()
    
    def generate_ensemble_outputs(
        self,
        prompt: str,
        num_ensemble: int,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        use_dropout: bool = True
    ) -> List[Dict]:
        """
        Generate multiple outputs to form an ensemble.

        Args:
            prompt: Input prompt
            num_ensemble: Number of ensemble members (forward passes)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            use_dropout: Whether to use MC Dropout

        Returns:
            List of dicts containing tokens, logits, and probabilities
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]

        ensemble_outputs = []

        if use_dropout:
            self.enable_dropout()

        with torch.no_grad():
            for _ in range(num_ensemble):
                # Generate with temperature sampling
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                generated_ids = outputs.sequences[0, input_length:]
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                # Get logits for each generated token
                logits_list = [score[0].cpu() for score in outputs.scores]
                
                ensemble_outputs.append({
                    'text': generated_text,
                    'tokens': generated_ids.cpu(),
                    'logits': logits_list,
                    'input_length': input_length
                })
        
        return ensemble_outputs
    
    def compute_sequence_probability(
        self,
        prompt: str,
        sequence_tokens: torch.Tensor,
        temperature: float = 1.0
    ) -> Tuple[float, List[float]]:
        """
        Compute the log probability of a sequence under the model.
        
        Returns:
            (total_log_prob, token_log_probs)
        """
        # Combine prompt and sequence
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]
        
        full_sequence = torch.cat([
            inputs.input_ids,
            sequence_tokens.unsqueeze(0).to(self.device)
        ], dim=1)
        
        with torch.no_grad():
            outputs = self.model(full_sequence)
            logits = outputs.logits[0]  # [seq_len, vocab_size]
        
        # Compute log probabilities for generated tokens
        token_log_probs = []
        for i, token_id in enumerate(sequence_tokens):
            position = input_length + i - 1
            logit = logits[position] / temperature
            log_prob = F.log_softmax(logit, dim=-1)
            token_log_probs.append(log_prob[token_id].item())
        
        return sum(token_log_probs), token_log_probs
    
    def compute_predictive_entropy(
        self,
        ensemble_outputs: List[Dict],
        use_chain_rule: bool = True
    ) -> Dict[str, float]:
        """
        Compute entropy of the predictive posterior (Total Uncertainty).
        
        Implements both:
        - Chain-rule approximation: Eq. 9 in paper
        - Joint-sequence approximation: Eq. 8 in paper
        
        Returns:
            Dictionary with sequence-level entropy measures
        """
        if use_chain_rule:
            return self._compute_entropy_chain_rule(ensemble_outputs)
        else:
            return self._compute_entropy_joint_sequence(ensemble_outputs)
    
    def _compute_entropy_chain_rule(
        self,
        ensemble_outputs: List[Dict]
    ) -> Dict[str, float]:
        """Chain-rule entropy: average of token-level entropies"""
        # Get minimum length to ensure we compare the same positions across all sequences
        min_length = min(len(out['logits']) for out in ensemble_outputs)

        if min_length == 0:
            return {
                'sequence_entropy_chain': 0.0,
                'token_entropies': []
            }

        token_entropies = []

        for pos in range(min_length):
            # Collect logits at this position across ensemble
            logits_at_pos = []
            for out in ensemble_outputs:
                logits_at_pos.append(out['logits'][pos])

            # Average logits to get predictive posterior
            avg_logits = torch.stack(logits_at_pos).mean(dim=0)
            probs = F.softmax(avg_logits, dim=-1)

            # Compute entropy: -Σ p(y) log p(y)
            # Clamp probabilities to avoid log(0)
            probs = torch.clamp(probs, min=1e-10)
            entropy = -(probs * torch.log(probs)).sum().item()

            # Check for NaN or inf
            if not np.isnan(entropy) and not np.isinf(entropy):
                token_entropies.append(entropy)

        # Return mean only if we have valid entropies
        if token_entropies:
            return {
                'sequence_entropy_chain': np.mean(token_entropies),
                'token_entropies': token_entropies
            }
        else:
            return {
                'sequence_entropy_chain': 0.0,
                'token_entropies': []
            }
    
    def _compute_entropy_joint_sequence(
        self,
        ensemble_outputs: List[Dict]
    ) -> Dict[str, float]:
        """Joint-sequence entropy: based on complete sequence probabilities"""
        # Group identical sequences
        sequence_counts = defaultdict(int)
        sequence_log_probs = {}
        
        for out in ensemble_outputs:
            seq_str = out['text']
            sequence_counts[seq_str] += 1
            
            if seq_str not in sequence_log_probs:
                # Compute log prob for this sequence
                log_prob = sum([
                    F.log_softmax(logits, dim=-1)[token_id].item()
                    for logits, token_id in zip(out['logits'], out['tokens'])
                ])
                sequence_log_probs[seq_str] = log_prob / len(out['tokens'])  # Length normalized
        
        # Compute entropy over sequence distribution
        total = sum(sequence_counts.values())
        entropy = 0.0
        for seq, count in sequence_counts.items():
            p = count / total
            entropy -= p * np.log(p + 1e-10)
        
        return {
            'sequence_entropy_joint': entropy,
            'unique_sequences': len(sequence_counts),
            'sequence_counts': dict(sequence_counts)
        }
    
    def compute_reverse_mutual_information(
        self,
        prompt: str,
        ensemble_outputs: List[Dict],
        temperature: float = 1.0
    ) -> Dict[str, float]:
        """
        Compute Reverse Mutual Information (RMI) - the paper's key contribution!

        RMI: M[y, θ|x, D] = E_P(y|x,D)[E_θ[ln P(y|x,D) / P(y|x,θ)]]

        This measures knowledge uncertainty (epistemic uncertainty).
        Uses token-level approximation for computational feasibility.
        """
        # Token-level RMI approximation
        min_length = min(len(out['logits']) for out in ensemble_outputs)

        if min_length == 0:
            return {
                'rmi_sequence': 0.0,
                'rmi_raw': 0.0,
                'avg_sequence_length': 0.0
            }

        token_rmi_values = []

        for pos in range(min_length):
            # Collect logits at this position across ensemble
            logits_at_pos = []
            token_ids_at_pos = []
            for out in ensemble_outputs:
                logits_at_pos.append(out['logits'][pos])
                token_ids_at_pos.append(out['tokens'][pos].item())

            # Average logits to get predictive posterior P(y_t|x,D)
            avg_logits = torch.stack(logits_at_pos).mean(dim=0)
            predictive_log_probs = F.log_softmax(avg_logits, dim=-1)

            # Compute RMI for this position
            # E_θ[KL(P(y_t|x,D) || P(y_t|x,θ))]
            position_rmi = 0.0
            for logits in logits_at_pos:
                model_log_probs = F.log_softmax(logits, dim=-1)

                # KL divergence: sum over vocab of P(y|x,D) * [log P(y|x,D) - log P(y|x,θ)]
                # Handle -inf in log probs (masked tokens)
                predictive_probs = torch.exp(predictive_log_probs)

                # Mask out -inf values to avoid NaN
                valid_mask = torch.isfinite(predictive_log_probs) & torch.isfinite(model_log_probs)
                if valid_mask.any():
                    kl_term = torch.where(
                        valid_mask & (predictive_probs > 1e-10),
                        predictive_probs * (predictive_log_probs - model_log_probs),
                        torch.zeros_like(predictive_probs)
                    )
                    kl = kl_term.sum().item()
                else:
                    kl = 0.0

                position_rmi += kl

            position_rmi /= len(logits_at_pos)  # Average over models

            # Only add if valid (not NaN or inf)
            if np.isfinite(position_rmi):
                token_rmi_values.append(position_rmi)

        avg_rmi = np.mean(token_rmi_values)
        avg_length = np.mean([len(out['tokens']) for out in ensemble_outputs])

        return {
            'rmi_sequence': avg_rmi,
            'rmi_raw': avg_rmi * avg_length,
            'avg_sequence_length': avg_length,
            'token_rmi_values': token_rmi_values
        }
    
    def compute_mutual_information(
        self,
        ensemble_outputs: List[Dict]
    ) -> Dict[str, float]:
        """
        Compute standard Mutual Information (MI) - traditional epistemic uncertainty.

        MI: I[y, θ|x, D] = H[P(y|x,D)] - E_θ[H[P(y|x,θ)]]
        """
        # Total uncertainty: H[P(y|x,D)]
        total_uncertainty = self._compute_entropy_chain_rule(ensemble_outputs)['sequence_entropy_chain']

        # Expected data uncertainty: E_θ[H[P(y|x,θ)]]
        # We need to compute this only on the common positions (min_length)
        min_length = min(len(out['logits']) for out in ensemble_outputs)

        model_entropies = []
        for out in ensemble_outputs:
            token_entropies = []
            # Only compute entropy for positions up to min_length
            for logits in out['logits'][:min_length]:
                # Softmax already handles -inf by mapping to 0 probability
                probs = F.softmax(logits, dim=-1)

                # Compute entropy, handling log(0) = -inf gracefully
                # We use the fact that lim x->0 of x*log(x) = 0
                log_probs = torch.log(probs + 1e-10)  # Add epsilon to avoid log(0)
                entropy = -(probs * log_probs).sum().item()

                if np.isfinite(entropy) and entropy >= 0:
                    token_entropies.append(entropy)

            if token_entropies:
                model_entropies.append(np.mean(token_entropies))

        expected_data_uncertainty = np.mean(model_entropies) if model_entropies else 0.0

        # MI = Total - Expected Data
        # Note: Chain-rule approximation can sometimes produce negative MI due to
        # sequence-level effects (models agreeing on different tokens). Clamp to 0.
        mi = max(0.0, total_uncertainty - expected_data_uncertainty)

        return {
            'mutual_information': mi,
            'total_uncertainty': total_uncertainty,
            'expected_data_uncertainty': expected_data_uncertainty,
            'mi_raw': total_uncertainty - expected_data_uncertainty  # Include unclamped for debugging
        }
    
    def analyze_uncertainty(
        self,
        prompt: str,
        num_ensemble: int = 10,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        use_dropout: bool = True
    ) -> Dict:
        """
        Complete uncertainty analysis combining all measures.

        Args:
            prompt: Input prompt
            num_ensemble: Number of ensemble members (forward passes)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            use_dropout: Whether to use MC Dropout

        Returns comprehensive uncertainty metrics for a given prompt.
        """
        print(f"Generating {num_ensemble} ensemble predictions...")
        ensemble_outputs = self.generate_ensemble_outputs(
            prompt,
            num_ensemble=num_ensemble,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            use_dropout=use_dropout
        )
        
        print("Computing uncertainty metrics...")
        
        # Total Uncertainty (different approximations)
        entropy_chain = self.compute_predictive_entropy(ensemble_outputs, use_chain_rule=True)
        entropy_joint = self.compute_predictive_entropy(ensemble_outputs, use_chain_rule=False)
        
        # Knowledge Uncertainty (epistemic)
        rmi = self.compute_reverse_mutual_information(prompt, ensemble_outputs, temperature)
        mi = self.compute_mutual_information(ensemble_outputs)
        
        # Compile results
        results = {
            'prompt': prompt,
            'ensemble_size': num_ensemble,
            'predictions': [out['text'] for out in ensemble_outputs],
            
            # Total Uncertainty
            'total_uncertainty': {
                'entropy_chain_rule': entropy_chain['sequence_entropy_chain'],
                'entropy_joint_sequence': entropy_joint['sequence_entropy_joint'],
                'unique_sequences': entropy_joint['unique_sequences']
            },
            
            # Knowledge Uncertainty (Epistemic)
            'knowledge_uncertainty': {
                'reverse_mutual_information': rmi['rmi_sequence'],
                'mutual_information': mi['mutual_information']
            },
            
            # Data Uncertainty (Aleatoric)
            'data_uncertainty': {
                'expected_entropy': mi['expected_data_uncertainty']
            },
            
            # Additional info
            'sequence_diversity': entropy_joint['sequence_counts'],
            'avg_sequence_length': rmi['avg_sequence_length']
        }
        
        return results
    
    def pretty_print_results(self, results: Dict):
        """Print results in a readable format"""
        print("\n" + "="*70)
        print("UNCERTAINTY ANALYSIS RESULTS")
        print("="*70)
        
        print(f"\nPrompt: {results['prompt']}")
        print(f"\nEnsemble Size: {results['ensemble_size']}")
        
        print("\n--- SAMPLE PREDICTIONS ---")
        for i, pred in enumerate(results['predictions'][:3], 1):
            print(f"{i}. {pred}")
        if len(results['predictions']) > 3:
            print(f"... and {len(results['predictions']) - 3} more")
        
        print("\n--- TOTAL UNCERTAINTY (How uncertain overall?) ---")
        tu = results['total_uncertainty']
        print(f"  Chain-Rule Entropy:       {tu['entropy_chain_rule']:.4f}")
        print(f"  Joint-Sequence Entropy:   {tu['entropy_joint_sequence']:.4f}")
        print(f"  Unique Sequences:         {tu['unique_sequences']}/{results['ensemble_size']}")
        
        print("\n--- KNOWLEDGE UNCERTAINTY (Model's lack of understanding) ---")
        ku = results['knowledge_uncertainty']
        print(f"  Reverse Mutual Info (RMI): {ku['reverse_mutual_information']:.4f} ⭐ Paper's contribution!")
        print(f"  Mutual Information (MI):   {ku['mutual_information']:.4f}")
        
        print("\n--- DATA UNCERTAINTY (Inherent task ambiguity) ---")
        du = results['data_uncertainty']
        print(f"  Expected Entropy:         {du['expected_entropy']:.4f}")
        
        print("\n--- INTERPRETATION ---")
        rmi = ku['reverse_mutual_information']
        mi = ku['mutual_information']

        # Interpret RMI (knowledge uncertainty)
        if not np.isnan(rmi) and not np.isinf(rmi):
            if rmi > 0.1:
                print("  ⚠️  HIGH knowledge uncertainty (RMI) - model is uncertain/unfamiliar")
            elif rmi > 0.01:
                print("  ⚡ MODERATE knowledge uncertainty (RMI) - some model disagreement")
            else:
                print("  ✓  LOW knowledge uncertainty (RMI) - models produce similar distributions")

        # Interpret MI (traditional epistemic uncertainty)
        if not np.isnan(mi) and not np.isinf(mi):
            if mi > 1.0:
                print("  ⚠️  HIGH epistemic uncertainty (MI) - significant model disagreement")
            elif mi > 0.3:
                print("  ⚡ MODERATE epistemic uncertainty (MI) - some variance in predictions")
            else:
                print("  ✓  LOW epistemic uncertainty (MI) - models agree on outputs")

        # Interpret diversity
        if tu['unique_sequences'] > results['ensemble_size'] * 0.7:
            print("  🌊 HIGH diversity - many different predictions")
        else:
            print("  🎯 LOW diversity - models converging on similar outputs")
        
        print("\n" + "="*70 + "\n")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_basic():
    """Basic uncertainty quantification example"""
    print("\n🔬 EXAMPLE 1: Basic Uncertainty Analysis\n")
    
    uq = UncertaintyQuantifier(
        model_name="gpt2"
    )

    prompt = "The capital of France is"
    results = uq.analyze_uncertainty(
        prompt=prompt,
        num_ensemble=10,
        max_new_tokens=20,
        temperature=0.8
    )
    
    uq.pretty_print_results(results)
    
    return results


def example_compare_prompts():
    """Compare uncertainty across different prompt types"""
    print("\n🔬 EXAMPLE 2: Comparing Different Prompts\n")
    
    uq = UncertaintyQuantifier(
        model_name="gpt2"
    )

    prompts = [
        "The capital of France is",  # Factual, low uncertainty expected
        "In my opinion, the best movie ever made is",  # Subjective, high uncertainty
        "Translate to French: Hello",  # May have multiple valid answers
        "skdjfhskjdhf ksjdhfksjhdf"  # Nonsense, should have high knowledge uncertainty
    ]

    results_comparison = []

    for prompt in prompts:
        print(f"\n{'='*70}")
        print(f"Analyzing: {prompt}")
        print('='*70)

        results = uq.analyze_uncertainty(
            prompt=prompt,
            num_ensemble=15,
            max_new_tokens=15,
            temperature=0.9
        )
        
        results_comparison.append(results)
        
        # Print summary
        ku = results['knowledge_uncertainty']['reverse_mutual_information']
        tu = results['total_uncertainty']['entropy_chain_rule']
        print(f"\n📊 Knowledge Uncertainty (RMI): {ku:.4f}")
        print(f"📊 Total Uncertainty: {tu:.4f}")
    
    return results_comparison


def example_out_of_domain_detection():
    """Demonstrate OOD detection - key application from paper"""
    print("\n🔬 EXAMPLE 3: Out-of-Domain Detection\n")
    
    uq = UncertaintyQuantifier(
        model_name="gpt2"
    )

    # In-domain: Normal English
    in_domain_prompts = [
        "The weather today is",
        "My favorite food is",
        "In the year 2050, technology will"
    ]

    # Out-of-domain: Foreign language, corrupted text
    out_domain_prompts = [
        "Le chat mange le",  # French
        "Der Hund ist",  # German
        "xkcd qwerty asdfgh"  # Gibberish
    ]

    print("IN-DOMAIN PROMPTS:")
    in_domain_rmi = []
    for prompt in in_domain_prompts:
        results = uq.analyze_uncertainty(prompt, num_ensemble=12, max_new_tokens=10, temperature=0.7)
        rmi = results['knowledge_uncertainty']['reverse_mutual_information']
        in_domain_rmi.append(rmi)
        print(f"  '{prompt}' -> RMI: {rmi:.4f}")

    print("\nOUT-OF-DOMAIN PROMPTS:")
    out_domain_rmi = []
    for prompt in out_domain_prompts:
        results = uq.analyze_uncertainty(prompt, num_ensemble=12, max_new_tokens=10, temperature=0.7)
        rmi = results['knowledge_uncertainty']['reverse_mutual_information']
        out_domain_rmi.append(rmi)
        print(f"  '{prompt}' -> RMI: {rmi:.4f}")
    
    print(f"\n📊 Average RMI - In-Domain: {np.mean(in_domain_rmi):.4f}")
    print(f"📊 Average RMI - Out-Domain: {np.mean(out_domain_rmi):.4f}")
    print(f"📊 Separation: {np.mean(out_domain_rmi) - np.mean(in_domain_rmi):.4f}")
    
    if np.mean(out_domain_rmi) > np.mean(in_domain_rmi) * 1.5:
        print("\n✓ Clear separation! RMI successfully detects out-of-domain inputs.")
    else:
        print("\n⚠ Limited separation. May need more ensemble members or different prompts.")


if __name__ == "__main__":
    # Run examples
    print("="*70)
    print("UNCERTAINTY QUANTIFICATION FOR LLMS")
    print("Based on: Malinin & Gales (ICLR 2021)")
    print("="*70)
    
    # Example 1: Basic usage
    basic_results = example_basic()
    
    # Example 2: Compare different prompts
    # comparison_results = example_compare_prompts()
    
    # Example 3: OOD detection
    # example_out_of_domain_detection()