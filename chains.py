from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
import re

from langchain_openai import ChatOpenAI

load_dotenv()

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
llm_chain = prompt | llm