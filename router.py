# Routing: 80% to small model and 20% to large model

def route(query: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # cheap classifier
        messages=[
            {"role": "system", "content": """You are a query complexity classifier.
Classify the query as 'simple' or 'complex'.

Simple: single fact lookup, basic math, yes/no, short generation
Complex: multi-step reasoning, code generation, analysis, comparison, long documents

Reply with ONLY 'simple' or 'complex'."""},
            {"role": "user", "content": query}
        ],
        max_tokens=5
    )
    
    label = response.choices[0].message.content.strip().lower()
    return "large" if label == "complex" else "small"

def run(query: str):
    model = "gpt-4o" if route(query) == "large" else "gpt-4o-mini"
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": query}]
    )
    print(f"Routed to: {model}")
    print(response.choices[0].message.content)