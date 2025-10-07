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
        temperature=1,
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

def print_output(output_ids: list, output_tokens: list, output_probs: list, n: int):
    table_data = [
        [
            output_token, output_id, output_prob
        ]
        for output_token, output_id, output_prob in zip(output_tokens, output_ids, output_probs)
    ]
    print("=" * 100)
    print(f"Run {n}")
    print("=" * 100)
    print()    
    print(tabulate(table_data, headers=["Token", "ID", "Probability"]))
    print()

def print_results(prompt: str, output_list: list):
    table_data = []
    for i, output in enumerate(output_list, 1):
        table_data.append([
            f"{i}",
            output['output_tokens'],
            f"{min_probability(output['output_probs']):.4f}",
            f"{mean_probability(output['output_probs']):.4f}"
        ])
    print("=" * 100)
    print(f"Prompt: {prompt}")
    print("=" * 100)
    print()
    print(tabulate(table_data, headers=["Run", "Output", "Min Probability", "Mean Probability"]))

if __name__ == '__main__':
    prompt = "The capital of France is"
    n = 5

    output_list = []
    for i in range(1, n + 1):
        output = generate(prompt)
        output_list.append(output)
        print_output(**output, n=i)

    print_results(prompt, output_list)


