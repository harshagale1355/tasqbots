from Chatbot.components.agents.llm.llm import llm_agent

# Initialize LLM once
llm = llm_agent()


def response_agent(state):

    query = state["query"]

    docs = state["retrieved_docs"]

    # Combine retrieved chunks
    context = "\n\n".join([
        doc.page_content for doc in docs
    ])

    # Create citations
    citations = [
        {
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", "Unknown")
        }
        for doc in docs
    ]

    # Prompt
    prompt = f"""
    You are a helpful AI assistant.

    Answer ONLY using the provided context.

    If the answer is not found in the context,
    say:
    "I could not find the answer in the document."

    Context:
    {context}

    Question:
    {query}

    Provide a concise answer with citations.
    """

    # Call LLM
    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "citations": citations
    }