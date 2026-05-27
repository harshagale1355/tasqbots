from Chatbot.components.agents.llm.llm import llm_agent
from Chatbot.logging.logger import logging
from Chatbot.exception.exception import RAG_Chatbot_Exception
import sys

llm = llm_agent()


def response_agent(state):

    try:
        query = state["query"]

        docs = state["retrieved_docs"]

        chat_history = state.get("chat_history", [])

        # Retrieved context
        context = "\n\n".join([
            doc.page_content for doc in docs
        ])

        # Conversation history
        history_text = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in chat_history
        ])

        citations = [
            {
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "Unknown")
            }
            for doc in docs
        ]

        prompt = f"""
        You are a helpful AI assistant.

        Use the conversation history and retrieved context
        to answer the question.

        Conversation History:
        {history_text}

        Context:
        {context}

        Current Question:
        {query}

        Answer ONLY from the provided context.

        If answer is not found,
        say:
        "I could not find the answer in the document."
        """

        response = llm.invoke(prompt)

        answer = response.content

        # Update memory
        updated_history = chat_history + [
            {
                "role": "user",
                "content": query
            },
            {
                "role": "assistant",
                "content": answer
            }
        ]

        logging.info(f"Generated response for query: '{query[:50]}...' with {len(docs)} documents as context.")

        return {
            "answer": answer,
            "citations": citations,
            "chat_history": updated_history,
            "retry_count": state.get("retry_count", 0) + 1
        }
    except Exception as e:
        raise RAG_Chatbot_Exception(e, sys)