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

    response = llm_chain.invoke(state["messages"])
    return { "message": response, "llm_calls": state.get("llm_calls",0) + 1}

def tool_node(state:GraphState) -> Dict[str, Any]:
    """Performs the tool call"""

    result = []
    tool_calls = state["messages"][-1]["tool_calls"]
    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
        
    return { "messages": result }



