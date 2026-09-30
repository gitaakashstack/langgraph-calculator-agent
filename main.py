from typing import Literal

from dotenv import load_dotenv
from langchain.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, END
from nodes import agent_node, tool_node, verify_node
from schemas import GraphState

load_dotenv()

AGENT = "agent"
TOOL = "tool"
VERIFIER = "verifier"

def should_continue(state) -> Literal[TOOL, VERIFIER]:
    """
        Decide if we should continue the loop or stop based upon whether the LLM made a tool call
    """

    print("--- Should Continue to TOOL or VERIFIER ? ---")
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        print("--- Continue to TOOL ---")
        return TOOL
    else:
        print("--- Continue to VERIFIER ---")
        return VERIFIER

def should_verify(state: GraphState)-> Literal[AGENT, END]:
    """
        Decide if the agent should be called again to correct its answer. If already more than 2 retires, the loop is terminated
    """
    print("--- Should Verify ---")

    if state["retries"] > 2:
        print("---Two verification attempted, terminating ---")
        return END

    if state["verdict"]:
        print("--- verification returned correct ---")
        return END
    else:
        print("--- verification failed ---")
        return AGENT


graph = StateGraph(GraphState)
graph.add_node(AGENT, agent_node)
graph.add_node(TOOL, tool_node)
graph.add_node(VERIFIER, verify_node)

graph.set_entry_point(AGENT)

graph.add_edge(TOOL, AGENT)

graph.add_conditional_edges(AGENT, should_continue, path_map={
    VERIFIER: VERIFIER,
    TOOL: TOOL,
})
graph.add_conditional_edges(VERIFIER, should_verify, path_map={
    END: END,
    AGENT: AGENT,
})

app = graph.compile()
app.get_graph().draw_mermaid_png(output_file_path='flow.png')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('Hello Calculator Agent')

    question = "I buy 3 pizzas at 12.50 € each and split the bill between 4 people. How much does each person pay?"
    user_message = HumanMessage(content=question)
    res = app.invoke({
        "messages": [user_message],
        "question": question,
    })

    for m in res["messages"]:
        m.pretty_print()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
