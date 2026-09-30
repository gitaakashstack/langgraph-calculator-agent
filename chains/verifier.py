from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import Field, BaseModel


class VerificationResponse(BaseModel):
    """
        Response for the self verification done
    """

    binary_score: bool = Field(description="All the checks for verification passed, then '1' else '0'")
    reason: str = Field(description="Steps to correct the answer")

verifier_prompt = ChatPromptTemplate.from_messages(
    [
       (
           "system",
            """
            You are a strict reviewer for a calculator agent.

            You receive a math question and the agent's solution steps. Each step shows which
            tool was called, with which arguments, and what it returned.

            IMPORTANT:
            - The tool results are always arithmetically correct. Do NOT recalculate them.
            - Only judge whether the agent chose the right steps.

            Check the following:
            1. Numbers: Did the agent use all the relevant numbers from the question, exactly as given?
            2. Operations: Are the chosen operations correct for what the question describes?
            3. Order: Were the steps done in the right order (e.g. total first, then split)?
            4. Completeness: Is any step missing?
            5. Answer: Does the final result answer exactly what was asked (e.g. price per person, not the total)?
            
            Give a binary score 1 or 0, where 1 is given only if all five checks pass.
            
            If something is wrong, explain in "reason" exactly which step is wrong and what the agent
            should do instead, in one or two sentences. This reason is sent back to the agent,
            so make it a concrete instruction.

            Question: {question}

            Agent's solution steps: {steps}

            Agent's final answer: {answer}
            """
        )
    ]
)

llm = ChatOpenAI(model="gpt-6-luna")
llm_with_structured_output = llm.with_structured_output(VerificationResponse)

verifier_chain = verifier_prompt | llm_with_structured_output