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

    # division by error test
    question1 = "Add 2 and 5, then divide the result by 0"

    # testing verification node failure (intentional failure in verification node)
    question2 = "I buy 3 pizzas at 12.50 € each and split the bill between 4 people. How much does each person pay?"

    # basic test, all success test
    question3 = "Add 3 and 5, multiply the result by 6, then divide the result by 2"

    # gives first result false, movies to verifier and then corrects itself
    question4 = "Add 1 and 2, then perform the factorial of result"

    # breaks down the percentage into available tools as percentage tool is not available. Also happens with power or factorial
    question5 = "A shirt costs 40 € and is 25% off. What's the final price"


    user_message = HumanMessage(content=question4)
    res = app.invoke({
        "messages": [user_message],
        "question": question4,
    })

    for m in res["messages"]:
        m.pretty_print()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
