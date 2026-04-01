# Multi-step tool-calling agent

import os 
import json

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

tools = [
    {
        "type":"function",
        "function":{
            "name": "multiply",
            "description":"Multiply 2 numbers",
            "parameters":{
                "type":"object",
                "properties":{
                    "a":{
                        "type": "integer", 
                        "description": "first number"
                    }, 
                    "b":{
                        "type": "integer", 
                        "description": "second number"
                    }
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type":"function",
        "function":{
            "name": "add",
            "description":"add 2 numbers",
            "parameters":{
                "type":"object",
                "properties":{
                    "a":{
                        "type": "integer", 
                        "description": "first number"
                    }, 
                    "b":{
                        "type": "integer", 
                        "description": "second number"
                    }
                },
                "required": ["a", "b"]
            }
        }
    }
]

def multiply(a, b):
    return a*b

def add(a, b):
    return a+b

messages = [
    {"role": "system", "content": "You are a tool calling agent."}
]

prompt = f"""
Multiple 5 by 3 and add 2 to it. Return the final answer as an integer value.
"""
messages.append(
    {"role": "user", "content": prompt}
)

while True:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        tools=tools,
        tool_choice="auto", 
        messages=messages
    )
    message = response.choices[0].message
    messages.append(message)

    if not message.tool_calls:
        print(f"Final result: {message.content}")
        break

    if message.tool_calls:
        for tool_call in message.tool_calls:
            args = json.loads(tool_call.function.arguments)

            if tool_call.function.name=="add":
                res = add(**args)

                messages.append({
                    "role":"tool",
                    "tool_call_id": tool_call.id,
                    "content": str(res)
                })

            elif tool_call.function.name=="multiply":
                res = multiply(**args)

                messages.append({
                    "role":"tool",
                    "tool_call_id": tool_call.id,
                    "content": str(res)
                })
            