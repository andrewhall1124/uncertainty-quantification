from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tabulate import tabulate

def generate(prompt: str) -> dict:
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    inputs = tokenizer(text=prompt, return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    input_length = len(input_ids[0])

    outputs = model.generate(
        inputs=input_ids,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=5,
        temperature=.1,
        do_sample=True,
        output_scores=True,
        return_dict_in_generate=True
    )

    output_ids = outputs.sequences[0, input_length:]
    output_tokens = [tokenizer.decode(output_id) for output_id in output_ids]

    logits_list = [score[0] for score in outputs.scores]
    probs_list = [torch.softmax(logits, dim=-1) for logits in logits_list]

    output_probs = []
    for i, token_id in enumerate(output_ids):
        prob = probs_list[i][token_id].item()
        output_probs.append(prob)

    return {
        'output_ids': output_ids,
        'output_tokens': output_tokens,
        'output_probs': output_probs
    }

def min_probability(token_probs: list) -> float:
    return min(token_probs)

def mean_probability(token_probs: list) -> float:
    return sum(token_probs) / len(token_probs)

def print_output(output_ids: list, output_tokens: list, output_probs: list):
    print(type(output_tokens))
    table_data = [
        [
            output_token, output_id, output_prob
        ]
        for output_token, output_id, output_prob in zip(output_tokens, output_ids, output_probs)
    ]
    print(tabulate(table_data, headers=["Token", "ID", "Probability"]))

def print_results(output_ids: list, output_tokens: list, output_probs: list):
    table_data = [["".join(output_tokens), f"{min_probability(output_probs):.4f}", f"{mean_probability(output_probs):.4f}"]]
    print("=" * 100)
    print(f"Prompt: {prompt}")
    print("=" * 100)
    print()
    print(tabulate(table_data, headers=["Output", "Min Probability", "Mean Probability"]))

if __name__ == '__main__':
    prompt = "The capital of France is"
    output = generate(prompt)
    print_output(**output)
    print_results(**output)


