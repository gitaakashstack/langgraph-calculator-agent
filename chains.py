from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
import re

generation_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage("You are a grant application writer. Write a clear, persuasive answer to a given question."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            "You are a critical reviewer. Evaluate the following grant application response on:\n"
               "- Relevance to the question\n"
               "- Clarity and structure\n"
               "- Persuasiveness and tone\n"
               "Give a score out of 10 and suggestions for improvement."
        ),
        MessagesPlaceholder(variable_name="messages"),
        HumanMessage("Please critique the assistant's answer above.")
    ]
)

# The last message with user role "Please critique the assistant's answer above" is needed specfically by Gemini.
# Gemini's generateContent endpoint requires the contents list to end on a user (or function/tool) turn, never on a model turn. If the last message you send in is itself
# an AIMessage (model turn), Gemini rejects the whole request with exactly this error — it won't "continue" a model turn, it can only respond to a user/tool turn.
# What happens: an earlier node produced an AIMessage (e.g. the actor's answer), that got appended to state["messages"], and then the reflect node calls the LLM
# again on the same messages list without adding anything new — so the list going into llm.invoke() ends with ..., AIMessage(...). Gemini refuses that outright.

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)
generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm

def parse_score(text):
    match = re.search(r"score\\s*[:\\-]?\\s*(\\d+(?:\\.\\d+)?)", text, re.I)
    return float(match.group(1)) if match else 0.0