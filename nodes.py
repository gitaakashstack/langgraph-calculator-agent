from typing import Dict, Any

from langchain_core.messages import ToolMessage, AIMessage, HumanMessage
from langgraph.prebuilt import ToolNode

from chains.llm_chain import llm_chain, tools_by_name, tools
from chains.verifier import VerificationResponse, verifier_chain
from schemas import GraphState


def agent_node(state:GraphState) -> Dict[str, Any]:
    """LLM decides whether to call a tool or not"""

    print("--- Agent Node ---")
    response = llm_chain.invoke({"message_history":state["messages"]})
    return { "messages": [response]}

def tool_error_handler(error: Exception) -> str:
    if isinstance(error, ValueError):
        return f"Error: {error} Explain this to user instead of calling the tool again"

    return f"Error: {error}"

tool_node = ToolNode(
    tools, handle_tool_errors=tool_error_handler
)

def verify_node(state:GraphState) -> Dict[str, Any]:
    """ Verifies the previous response """

    print(f"--- Verify Node: Attempt {state.get("retries",0)} ---")

    steps = format_steps(state["messages"])

    response: VerificationResponse = verifier_chain.invoke({
        "question": state["question"],
        "steps": steps,
        "answer": state["messages"][-1].content,
    })

    if response.binary_score:
        return {
            "retries": state.get("retries", 0) + 1,
            "verdict": response.binary_score,
        }

    feedback = HumanMessage(
        content=f"""
            Your answer was checked and is wrong: {response.reason} 
            Please fix it using the tools.
        """
    )
    return {
        "messages": [feedback],
        "retries": state.get("retries",0) + 1,
        "verdict": response.binary_score,
    }


def format_steps(messages) -> str:
    lines = []
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls:
                lines.append(f"- Called {tc['name']} with {tc['args']}")
        elif isinstance(m, ToolMessage):
            lines.append(f"  → returned {m.content}")
    return "\n".join(lines) or "(no tools were called)"


