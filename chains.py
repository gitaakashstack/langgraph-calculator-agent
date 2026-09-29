from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
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
    return a * b

@tool
def add(a: int, b: int) -> int:
    """
    Adds two numbers

    Args:
        a (int): first number
        b (int): second number
    """
    return a + b

@tool
def divide(a: int, b: int) -> float:
    """
    Divides two numbers

    Args:
        a (int): first number
        b (int): second number
    """
    return a / b

tools = [add, multiply, divide]
tools_by_name = {tool.__name__: tool for tool in tools}

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            """
                You are a helpful calculator assistant tasked with performing arithemetic operations on a set of inputs.
                You have access to some tools to perform those calculations.
                If you cannot do the calculation with the provided tools, tell the user.
            """
        )
    ])

llm = ChatOpenAI(model="gpt-6-luna")
llm_with_tools = llm.bind_tools(tools)
llm_chain = prompt | llm