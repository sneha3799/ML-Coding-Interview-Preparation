# Flow

# User → LLM
# LLM → decides tool needed
# LLM → returns tool_call
# Python → executes tool
# Python → sends result back
# LLM → final answer

# Pipeline

# User query
#     ↓
# LLM reasoning
#     ↓
# Tool call request
#     ↓
# Python executes tool
#     ↓
# Tool result returned
#     ↓
# LLM final answer

import os
import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def current_time():
    now = datetime.now()
    # Format the time as a string (HH:MM:SS)
    return now.strftime("%H:%M:%S")

tools = [
    {
        "type": "function",
        "function": {
            "name": "current_time",
            "description": "used to provide the current time",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

prompt = f"""
You are a helpful assistant. 
Return me the current time. 
"""

messages = [
    {"role": "user", "content": prompt}
]
response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages, 
        tools=tools, 
        tool_choice="auto"
    )
message = response.choices[0].message
messages.append(message)

if message.tool_calls:
    for tool_call in message.tool_calls:
        if tool_call.function.name == "current_time":
            result = current_time()

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

final = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
print(f"Time: {final.choices[0].message.content}")