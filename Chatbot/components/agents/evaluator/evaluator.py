from Chatbot.components.agents.llm.llm import llm_agent
from Chatbot.logging.logger import logging
from Chatbot.exception.exception import RAG_Chatbot_Exception
import sys

llm = llm_agent()


def evaluator_agent(state):
    try:

        query = state["query"]

        answer = state["answer"]

        docs = state["retrieved_docs"]

        logging.info(f"Evaluating response for query: '{query[:50]}...'")
        logging.info(f"Number of retrieved documents used for context: {len(docs)}")

        context = "\n\n".join([
            doc.page_content for doc in docs
        ])

        prompt = f"""
        You are an evaluator agent.

        Check whether the answer is:
        - accurate
        - grounded in the context
        - not hallucinated

        Context:
        {context}

        Question:
        {query}

        Answer:
        {answer}

        Reply ONLY with:
        PASS
        or
        FAIL
        """

        evaluation = llm.invoke(prompt)

        result = evaluation.content.strip()

        logging.info("Evaluation SUCCESSFUL")

        return {
            "evaluation": result
        }
    except Exception as e:
        raise RAG_Chatbot_Exception(e,sys)