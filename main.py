from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.graph.message import add_messages

load_dotenv()

from chains import generation_chain, reflection_chain, parse_score

REFLECT = "reflect"
GENERATE = "generate"

def generation_node(state: MessagesState):
    return {"messages": [generation_chain.invoke({"messages": state["messages"]})]}

def reflection_node(state: MessagesState):
    res = reflection_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}

def should_continue(state):
    reflections = [msg.content for msg in state if isinstance(msg, HumanMessage)]
    last = reflections[-1] if reflections else ""
    score = parse_score(last)
    if score >= 8:
        return END
    return REFLECT

graph = StateGraph(MessagesState)
graph.add_node(GENERATE, generation_node)
graph.add_node(REFLECT, reflection_node)
graph.set_entry_point(GENERATE)
graph.add_edge(REFLECT, GENERATE)
graph.add_conditional_edges(GENERATE, should_continue, path_map={
    END: END,
    REFLECT: REFLECT,
})


app = graph.compile()
app.get_graph().draw_mermaid_png(output_file_path='flow.png')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('Hello React LangGraph with Reflection')
    res = app.invoke({
        "messages": [
            HumanMessage(
                content="How will your project positively impact underserved communities?"
            )
        ]
    })
    print(res)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
