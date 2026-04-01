import os
import json
from typing import Any, Dict
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# -----------------------------
# Mock banking tools
# -----------------------------
def get_account_balance(account_id: str) -> str:
    fake_db = {
        "ACC123": "$5,240.18",
        "ACC456": "$182.77",
    }
    if account_id not in fake_db:
        raise ValueError(f"Unknown account_id: {account_id}")
    return fake_db[account_id]


def block_card(card_last4: str) -> str:
    if len(card_last4) != 4 or not card_last4.isdigit():
        raise ValueError("card_last4 must be exactly 4 digits.")
    return f"Card ending in {card_last4} has been blocked successfully."


def lookup_recent_transactions(account_id: str, limit: int = 3) -> str:
    fake_txns = {
        "ACC123": [
            {"merchant": "Uber", "amount": "$18.20"},
            {"merchant": "Amazon", "amount": "$42.99"},
            {"merchant": "Starbucks", "amount": "$6.80"},
        ],
        "ACC456": [
            {"merchant": "Walmart", "amount": "$77.10"},
            {"merchant": "Shell", "amount": "$54.20"},
        ],
    }
    if account_id not in fake_txns:
        raise ValueError(f"Unknown account_id: {account_id}")
    return json.dumps(fake_txns[account_id][:limit])


# -----------------------------
# Tool registry
# -----------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_account_balance",
            "description": "Get the current balance for a bank account.",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {
                        "type": "string",
                        "description": "Bank account identifier, e.g. ACC123",
                    }
                },
                "required": ["account_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "block_card",
            "description": "Block a card if the customer reports it lost or stolen.",
            "parameters": {
                "type": "object",
                "properties": {
                    "card_last4": {
                        "type": "string",
                        "description": "Last 4 digits of the card",
                    }
                },
                "required": ["card_last4"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_recent_transactions",
            "description": "Retrieve recent transactions for an account.",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {
                        "type": "string",
                        "description": "Bank account identifier, e.g. ACC123",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of transactions to return",
                        "default": 3,
                    },
                },
                "required": ["account_id"],
                "additionalProperties": False,
            },
        },
    },
]


def execute_tool(tool_name: str, args: Dict[str, Any]) -> str:
    try:
        if tool_name == "get_account_balance":
            return get_account_balance(**args)
        if tool_name == "block_card":
            return block_card(**args)
        if tool_name == "lookup_recent_transactions":
            return lookup_recent_transactions(**args)
        return f"Unknown tool: {tool_name}"
    except Exception as e:
        return f"Tool error: {str(e)}"


def run_agent(user_query: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful banking assistant. "
                "Use tools when needed. "
                "If a tool is required, call it with the right arguments. "
                "After receiving tool results, give a short, clear answer."
            ),
        },
        {"role": "user", "content": user_query},
    ]

    first_response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )

    assistant_message = first_response.choices[0].message
    messages.append(assistant_message)

    if assistant_message.tool_calls:
        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name
            raw_args = tool_call.function.arguments or "{}"

            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}

            result = execute_tool(tool_name, args)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

        final_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
        )
        return final_response.choices[0].message.content

    return assistant_message.content or "No response returned."


if __name__ == "__main__":
    queries = [
        "What is the balance of account ACC123?",
        "Please block my card ending in 4421.",
        "Show me the last 2 transactions for account ACC456.",
        "My card ending in 123 is stolen, block it.",
    ]

    for q in queries:
        print("=" * 80)
        print("USER:", q)
        print("ASSISTANT:", run_agent(q))