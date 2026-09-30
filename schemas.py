import operator
from typing import TypedDict, Annotated
from langchain.messages import AnyMessage
from pydantic import BaseModel


class GraphState(TypedDict):
    """
        Represents the state of our graph.

        Attributes:
            messages: messages
            question: question
            verdict: final verdict after self verification, true means correct, false means incorrect
            retries: number of retries for self verification
        """

    messages: Annotated[list[AnyMessage], operator.add]
    question: str
    verdict: bool
    retries: int