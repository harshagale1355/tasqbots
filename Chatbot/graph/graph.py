from typing import TypedDict, List
from langgraph.graph import StateGraph, END

from Chatbot.components.agents.retrieval.retrieval import retrieval_agent
from Chatbot.components.agents.response.response import response_agent


# Shared State
class AgentState(TypedDict):
    query: str
    retrieved_docs: List
    answer: str
    citations: List


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


# Entry point
graph.set_entry_point("retrieval")


# Flow
graph.add_edge("retrieval", "response")

graph.add_edge("response", END)


# Compile graph
app = graph.compile()