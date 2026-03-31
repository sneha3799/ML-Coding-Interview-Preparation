# JSON parsing
# LLM with tool-calling 

# import os
# from pydantic import BaseModel, Field
# from typing import Literal
# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv
# load_dotenv()

# client = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.getenv('OPENAI_API_KEY'))

# examples = [
# "Someone used my card without permission",
# "My transfer failed yesterday",
# "I want to increase my credit limit",
# "I forgot my PIN",
# "My card was declined abroad",
# "I see a suspicious $900 transaction",
# "I want to dispute a charge",
# "I need a replacement card",
# "My payment didn't go through",
# "I want to close my account"
# ]

# class OutputParser(BaseModel):
#     intent: str = Field(description="The issue being reported")
#     priority: Literal["low", "medium", "high"] = Field(description="Urgency level")
#     department: Literal["fraud", "payments", "cards", "support", "accounts"] = Field(
#         description="Responsible department"
#     )

# llm = client.with_structured_output(OutputParser)
# for example in examples:
#     response = llm.invoke(example)
#     print(f'Response: {response}')


import os
import json
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

examples = [ 
"Someone used my card without permission",
"My transfer failed yesterday",
"I want to increase my credit limit",
"I forgot my PIN",
"My card was declined abroad",
"I see a suspicious $900 transaction",
"I want to dispute a charge",
"I need a replacement card",
"My payment didn't go through",
"I want to close my account"
]

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

for ex in examples:
    prompt = f"""
        You are a helpful banking assistant. 
        Go through the customer issue {ex} and extract the following information.

        Return ONLY a valid JSON
        {{
        "intent": "",
        "priority": "",
        "department": ""
        }}
        """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role":"user","content":prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        output = response.choices[0].message.content.strip()
        result = json.loads(output)
        print(f'Result: {result}')
    except json.JSONDecodeError:
        print("Invalid JSON returned:")
        print(output)

    except Exception as e:
        print(f"Error: {e}")