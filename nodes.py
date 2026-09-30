from typing import Dict, Any

from langchain_core.messages import ToolMessage, AIMessage, HumanMessage

from chains.llm_chain import llm_chain, tools_by_name
from chains.verifier import VerificationResponse, verifier_chain
from schemas import GraphState


def agent_node(state:GraphState) -> Dict[str, Any]:
    """LLM decides whether to call a tool or not"""

    print("--- Agent Node ---")
    response = llm_chain.invoke({"message_history":state["messages"]})
    return { "messages": [response]}

def tool_node(state:GraphState) -> Dict[str, Any]:
    """Performs the tool call"""

    print("--- Tool Node ---")
    result = []
    tool_calls = state["messages"][-1].tool_calls
    for tool_call in tool_calls:
        print(f"--- Performing {tool_call["name"]} ---")

        tool = tools_by_name[tool_call["name"]]
        try:
            observation = tool.invoke(tool_call["args"])
            result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
        except ValueError as e:
            result.append(ToolMessage(
                content=f"Error: {e} Explain this to user instead of calling the tool again",
                tool_call_id=tool_call["id"],
                status="error",
            ))
        
    return { "messages": result }

def verify_node(state:GraphState) -> Dict[str, Any]:
    """ Verifies the previous response """

    print(f"--- Verify Node: Attempt {state.get("retries",0)} ---")

    steps = format_steps(state["messages"])

    # Intentionally modifying steps so that verification llm returns incorrect. Set step such that it forgets a step. Comment out the below statement if using another prompt
    if state.get("retries",0) == 0:
        steps = "Called multiply with {'a': 3, 'b': 12.5}\n  → returned 37.5"

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


