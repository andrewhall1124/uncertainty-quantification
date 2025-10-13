from langchain_google_vertexai import ChatVertexAI
llm = ChatVertexAI(model='gemini-pro')

from uqlm import WhiteBoxUQ
wbuq = WhiteBoxUQ(llm=llm, scorers=["min_probability"])

results = wbuq.generate_and_score(prompts=prompts)
results.to_df()