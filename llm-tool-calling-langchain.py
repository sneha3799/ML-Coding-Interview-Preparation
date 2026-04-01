from langchain_openai import ChatOpenAI
from langchain.tools import tool
from dotenv import load_dotenv
from datetime import datetime
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv()

@tool
def current_time():
    """Return the current system time."""
    return datetime.now().strftime("%H:%M:%S")

# Create LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# Bind tools
llm_with_tools = llm.bind_tools([current_time])

# Invoke LLM
query = "Tell me the current time"
response = llm_with_tools.invoke(query)

# Store conversation
messages = [HumanMessage(content=query), response]

# Handle tool calls
if response.tool_calls:
    for tool_call in response.tool_calls:

        if tool_call["name"] == "current_time":

            # Execute tool
            result = current_time.invoke(tool_call["args"])

            # Add tool result message
            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"]
                )
            )

# Final LLM call (LangChain completes flow)
final_response = llm_with_tools.invoke(messages)
print(final_response.content)