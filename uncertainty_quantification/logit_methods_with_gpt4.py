from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tabulate import tabulate
from langchain_openai import AzureChatOpenAI
from abc import ABC, abstractmethod
from rich import print
from dotenv import load_dotenv
load_dotenv(override=True)

class Model(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> dict:
        pass

class HuggingFaceModel(Model):
    def __init__(self, name: str):
        self.model = AutoModelForCausalLM.from_pretrained(name)
        self.tokenizer = AutoTokenizer.from_pretrained(name)

    def generate(self, prompt: str) -> dict:

        inputs = self.tokenizer(text=prompt, return_tensors="pt")
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        input_length = len(input_ids[0])

        outputs = self.model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=5,
            temperature=1,
            do_sample=True,
            output_scores=True,
            return_dict_in_generate=True
        )

        output_ids = outputs.sequences[0, input_length:]
        output_tokens = [self.tokenizer.decode(output_id) for output_id in output_ids]

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
    
class AzureOpenAIModel(Model):
    def __init__(self, deployment_name: str, api_version: str = "2024-12-01-preview", temperature: float = 1.0, max_tokens: int = 5):
        self.model = AzureChatOpenAI(
            deployment_name=deployment_name,
            openai_api_type="azure",
            openai_api_version=api_version,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=5
        )

    def generate(self, prompt: str) -> dict:
        response = self.model.invoke(prompt)

        output_tokens = []
        output_probs = []

        # Extract tokens and probabilities from logprobs
        logprobs_content = response.response_metadata.get('logprobs', {}).get('content', [])
        for token_data in logprobs_content:
            output_tokens.append(token_data['token'])
            # Convert log probability to probability
            output_probs.append(torch.exp(torch.tensor(token_data['logprob'])).item())

        return {
            'output_ids': list(range(len(output_tokens))),
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
            f"{output_token}", 
            f"{output_id}", 
            f"{output_prob:.4f}"
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
    min_probs = []
    mean_probs = []

    for i, output in enumerate(output_list, 1):
        min_prob = min_probability(output['output_probs'])
        mean_prob = mean_probability(output['output_probs'])
        min_probs.append(min_prob)
        mean_probs.append(mean_prob)

        table_data.append([
            f"{i}",
            output['output_tokens'],
            f"{min_prob:.4f}",
            f"{mean_prob:.4f}"
        ])

    # Add average row
    table_data.append([
        "Avg",
        "",
        f"{sum(min_probs) / len(min_probs):.4f}",
        f"{sum(mean_probs) / len(mean_probs):.4f}"
    ])

    print("=" * 100)
    print(f"Prompt: {prompt}")
    print("=" * 100)
    print()
    print(tabulate(table_data, headers=["Run", "Output", "Min Probability", "Mean Probability"]))
    print()

if __name__ == '__main__':
    prompts = [
        "The capital of France is",  # Factual, low uncertainty expected
        "In my opinion, the best movie ever made is",  # Subjective, high uncertainty
        "skdjfhskjdhf ksjdhfksjhdf"  # Nonsense, should have high knowledge uncertainty
    ]

    model = AzureOpenAIModel(
        deployment_name="gpt-4.1-nano",
        api_version="2024-12-01-preview",
        temperature=1,  # User to set temperature
        max_tokens=10
    )

    n = 1

    for prompt in prompts:

        output_list = []
        for i in range(1, n + 1):
            output = model.generate(prompt)
            output_list.append(output)
            # print_output(**output, n=i)

        print_results(prompt, output_list)


