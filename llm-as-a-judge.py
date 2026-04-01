import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
system_prompt = f"""
You are a poet.
"""
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Generate a poem on boyfriend."}
]
response = client.chat.completions.create(
    model = "gpt-3.5-turbo",
    messages = messages
)
poem = response.choices[0].message.content
print(f"Poem: {poem}")

prompt1 = f"""
You are a judge for a poem competition. 
Rate the given poem as winner (1) or loser (0). 
Poem: {poem}
"""
messages1 = [
    {"role": "user", "content": prompt1}
]
judge_llm_response = client.chat.completions.create(
    model="gpt-4",
    messages=messages1
)
print(f"Poem ratings: {judge_llm_response.choices[0].message.content}")