# For a structured DB practice setup, add these:

# only allow SELECT
# reject INSERT, UPDATE, DELETE, DROP, ALTER
# scope queries to allowed tables
# optionally add LIMIT
# pass schema explicitly in the prompt

import os
import psycopg2
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def current_time():
    now = datetime.now()
    # Format the time as a string (HH:MM:SS)
    return now.strftime("%H:%M:%S")

SCHEMA = """
Tables:
customers(customer_id, name, email)
accounts(account_id, customer_id, account_type, balance)
transactions(transaction_id, account_id, merchant, amount, date)

Relationships:
accounts.customer_id -> customers.customer_id
transactions.account_id -> accounts.account_id
"""

def generate_sql(question: str) -> str:
    prompt = f"""
You are a SQL assistant.

Convert the user's question into a single PostgreSQL SELECT query.

Rules:
- Only output SQL
- Only use SELECT
- Do not use markdown
- Use only the schema below
- Add LIMIT when appropriate

Schema:
{SCHEMA}

Question:
{question}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return validate_sql(response.choices[0].message.content.strip())

def validate_sql(sql: str):
    sql_upper = sql.strip().upper()

    if not sql_upper.startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed.")

    blocked = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE"]
    if any(word in sql_upper for word in blocked):
        raise ValueError("Unsafe SQL detected.")

# def execute_sql(sql: str):
#     with psycopg2.connect(os.getenv('Database_url')) as conn:
#         with conn.cursor() as cur:
#             cur.execute(sql)
#             rows = cur.fetchall()
#             columns = [desc[0] for desc in cur.description]
#     return columns, rows

# def explain_result(question: str, columns, rows) -> str:
#     prompt = f"""
# You are a helpful banking assistant.

# Given the user's question and the SQL result, answer clearly and concisely.

# Question:
# {question}

# Columns:
# {columns}

# Rows:
# {rows}
# """
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0
#     )
#     return response.choices[0].message.content.strip()

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
    },
    {
        "type": "function",
        "function": {
            "name": "generate_sql",
            "description": "convert text to sql query",
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
Return me the current time and 
get me the last transaction detail from database.
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
        
        elif tool_call.function.name == "text_sql":
            result = generate_sql(client)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

final = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
print(f"Time: {final.choices[0].message.content}")