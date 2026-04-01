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

# Call LLM
# Check tool call
# Execute tool
# Send result back
# Call LLM again

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

# messages = [{"role": "user", "content": "What time is it?"}]

# while True:
#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo", messages=messages, tools=tools
#     )
#     message = response.choices[0].message
#     messages.append(message)

#     if not message.tool_calls:  # Model decided it's done
#         print(message.content)
#         break

#     for tool_call in message.tool_calls:  # Model wants a tool
#         if tool_call.function.name == "current_time":
#             result = current_time()
#             messages.append({
#                 "role": "tool",
#                 "tool_call_id": tool_call.id,
#                 "content": result
#             })
#     # Loop back — model decides next step