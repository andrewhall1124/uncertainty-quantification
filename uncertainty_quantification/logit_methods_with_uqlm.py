from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from uqlm import WhiteBoxUQ
import asyncio
import polars as pl

load_dotenv(override=True)

llm = AzureChatOpenAI(
    deployment_name="gpt-4.1-nano",
    openai_api_type="azure",
    openai_api_version="2024-12-01-preview",
    temperature=1,  # User to set temperature
)

prompts = [
    "The capital of France is",  # Factual, low uncertainty expected
    "In my opinion, the best movie ever made is",  # Subjective, high uncertainty
    "skdjfhskjdhf ksjdhfksjhdf",  # Nonsense, should have high knowledge uncertainty
]

wbuq = WhiteBoxUQ(llm=llm)

results = asyncio.run(wbuq.generate_and_score(prompts=prompts))

print(pl.from_pandas(results.to_df()).drop('logprob'))
