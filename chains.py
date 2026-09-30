from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
import re

from langchain_openai import ChatOpenAI

load_dotenv()

@tool
def multiply(a: int, b: int) -> int:
    """
    Multiplies two numbers

    Args:
        a (int): first number
        b (int): second number
    """
    print(f"--- Multiplying {a} and {b} ---")
    return a * b

@tool
def add(a: int, b: int) -> int:
    """
    Adds two numbers

    Args:
        a (int): first number
        b (int): second number
    """

    print(f"--- Adding {a} and {b} ---")
    return a + b

@tool
def divide(a: int, b: int) -> float:
    """
    Divides two numbers

    Args:
        a (int): first number
        b (int): second number
    """

    print(f"--- Dividing {a} and {b} ---")
    return a / b

tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            """
                You are a helpful calculator assistant tasked with performing arithemetic operations on a set of inputs.
                You have access to some tools to perform those calculations. You must NEVER do arithmetic yourself. For every calculation, call the appropriate tool.
                If you cannot do the calculation with the provided tools, tell the user.
            """
        ),
        MessagesPlaceholder(variable_name="message_history")
    ])

llm = ChatOpenAI(model="gpt-6-luna")
llm_with_tools = llm.bind_tools(tools)
llm_chain = prompt | llm_with_tools