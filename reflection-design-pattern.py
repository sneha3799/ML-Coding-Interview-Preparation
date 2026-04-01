import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Generator
messages = [
    {"role": "system", "content": "You are a romantic poet."},
    {"role": "user", "content": "Generate a short poem about a boyfriend."}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    temperature=0.8
)

poem = response.choices[0].message.content

print("Generated poem:\n")
print(poem)

# Judge
prompt1 = f"""
You are a poetry competition judge.

Return JSON:
{{
 "score": 0 or 1,
 "reason": "short explanation"
}}

Evaluate:
- creativity
- emotional depth
- imagery

Poem:
{poem}
"""

judge = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role":"user","content":prompt1}],
    temperature=0
)

print("\nJudge evaluation:\n")
print(judge.choices[0].message.content)