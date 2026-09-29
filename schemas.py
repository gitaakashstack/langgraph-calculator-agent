import operator
from typing import TypedDict, Annotated
from langchain.messages import AnyMessage
from pydantic import BaseModel


class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int