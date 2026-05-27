from typing import TypedDict, List
from langgraph.graph import StateGraph, END

from Chatbot.components.agents.retrieval.retrieval import retrieval_agent
from Chatbot.components.agents.response.response import response_agent
from Chatbot.components.agents.evaluator.evaluator import evaluator_agent


# Shared State
class AgentState(TypedDict):
    query: str
    retrieved_docs: List
    answer: str
    citations: List
    evaluation: str
    retry_count: int
    chat_history: List


# Router Function
def evaluation_router(state):

    evaluation = state["evaluation"]

    retry_count = state.get("retry_count", 0)

    if evaluation == "PASS":
        return "end"

    if retry_count >= 2:
        return "end"

    return "retry"


# Create graph
graph = StateGraph(AgentState)


# Add nodes
graph.add_node(
    "retrieval",
    retrieval_agent
)

graph.add_node(
    "response",
    response_agent
)

graph.add_node(
    "evaluator",
    evaluator_agent
)


# Entry point
graph.set_entry_point("retrieval")


# Flow
graph.add_edge("retrieval", "response")

graph.add_edge("response", "evaluator")


# Conditional flow
graph.add_conditional_edges(
    "evaluator",
    evaluation_router,
    {
        "retry": "response",
        "end": END
    }
)


# Compile
app = graph.compile()