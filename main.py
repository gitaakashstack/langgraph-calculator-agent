from typing import Literal

from dotenv import load_dotenv
from langchain.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, END
from nodes import agent_node, tool_node
from schemas import GraphState

load_dotenv()

AGENT = "agent"
TOOL = "tool"

def should_continue(state) -> Literal[TOOL, END]:
    """
        Decide if we should continue the loop or stop based upon whether the LLM made a tool call
    """

    print("--- Should Continue to TOOL or END ? ---")
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        print("--- Continue to TOOL ---")
        return TOOL
    else:
        print("--- Continue to END ---")
        return END

graph = StateGraph(GraphState)
graph.add_node(AGENT, agent_node)
graph.add_node(TOOL, tool_node)

graph.set_entry_point(AGENT)

graph.add_edge(TOOL, AGENT)

graph.add_conditional_edges(AGENT, should_continue, path_map={
    END: END,
    TOOL: TOOL,
})


app = graph.compile()
app.get_graph().draw_mermaid_png(output_file_path='flow.png')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('Hello Calculator Agent')
    user_message = HumanMessage(content="Add 3 and 4, then multiply the result with 10, then add 5 to result.")
    res = app.invoke({
        "messages": [user_message]
    })

    for m in res["messages"]:
        m.pretty_print()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
