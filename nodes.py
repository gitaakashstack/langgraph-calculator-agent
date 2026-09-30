from typing import Dict, Any

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState

from chains import llm_chain, tools_by_name
from schemas import GraphState


def agent_node(state:GraphState) -> Dict[str, Any]:
    """LLM decides whether to call a tool or not"""

    print("--- Agent Node ---")
    response = llm_chain.invoke({"message_history":state["messages"]})
    return { "messages": [response], "llm_calls": state.get("llm_calls",0) + 1}

def tool_node(state:GraphState) -> Dict[str, Any]:
    """Performs the tool call"""

    print("--- Tool Node ---")
    result = []
    tool_calls = state["messages"][-1].tool_calls
    for tool_call in tool_calls:
        print(f"--- Performing {tool_call["name"]} ---")
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
        
    return { "messages": result }



