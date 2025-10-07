from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tabulate import tabulate

model = AutoModelForCausalLM.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained('gpt2')

prompt = "The capital of France is"

inputs = tokenizer(text=prompt, return_tensors="pt")
input_ids = inputs.input_ids
attention_mask = inputs.attention_mask

max_new_tokens = 5
input_length = len(input_ids[0])

outputs = model.generate(
    inputs=input_ids,
    attention_mask=attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    max_new_tokens=max_new_tokens,
    temperature=1,
    do_sample=True,
    output_scores=True,
    return_dict_in_generate=True
)

output_ids = outputs.sequences[0, input_length:]
output_tokens = tokenizer.decode(output_ids, skip_special_tokens=True)

logits_list = [score[0] for score in outputs.scores]
probs_list = [torch.softmax(logits, dim=-1) for logits in logits_list]

token_probs = []
for i, token_id in enumerate(output_ids):
    prob = probs_list[i][token_id].item()
    token_probs.append(prob)

min_probability = min(token_probs)
mean_probability = sum(token_probs) / len(token_probs)

table_data = [[output_tokens, f"{min_probability:.4f}", f"{mean_probability:.4f}"]]
print("=" * 100)
print(f"Prompt: {prompt}")
print("=" * 100)
print()
print(tabulate(table_data, headers=["Output", "Min Probability", "Mean Probability"]))

