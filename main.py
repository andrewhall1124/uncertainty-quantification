from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def generate_with_probabilities(prompt, model_name="gpt2", max_new_tokens=10, top_k=5):
    """
    Generate text and show the probability distribution for each generated token.
    
    Args:
        prompt: Input text to continue
        model_name: HuggingFace model to use
        max_new_tokens: Number of tokens to generate
        top_k: Number of top probable tokens to display
    """
    # Load model and tokenizer
    print(f"Loading model: {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()  # Set to evaluation mode
    
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    print(f"\nPrompt: '{prompt}'")
    print("=" * 80)
    
    generated_text = prompt
    
    # Generate tokens one at a time
    for step in range(max_new_tokens):
        with torch.no_grad():
            # Get model outputs
            outputs = model(input_ids)
            
            # Get logits for the last token
            logits = outputs.logits[:, -1, :]
            
            # Convert to probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Get top-k tokens and their probabilities
            top_probs, top_indices = torch.topk(probs, top_k)
            
            print(f"\nStep {step + 1}:")
            print(f"Top {top_k} most likely next tokens:")
            
            for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0])):
                token = tokenizer.decode([idx])
                print(f"  {i+1}. '{token}' - {prob.item():.4f} ({prob.item()*100:.2f}%)")
            
            # Use greedy decoding (pick most probable token)
            next_token_id = top_indices[0, 0].unsqueeze(0).unsqueeze(0)
            chosen_token = tokenizer.decode([next_token_id.item()])
            
            print(f"  → Selected: '{chosen_token}'")
            
            # Append to input for next iteration
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            generated_text += chosen_token
    
    print("\n" + "=" * 80)
    print(f"Final generated text: '{generated_text}'")
    
    return generated_text


if __name__ == "__main__":
    # Example usage
    prompt = "The future of artificial intelligence is"
    generate_with_probabilities(
        prompt=prompt,
        model_name="gpt2",
        max_new_tokens=10,
        top_k=5
    )