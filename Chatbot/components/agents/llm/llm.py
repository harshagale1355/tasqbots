from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from Chatbot.logging.logger import logging
import sys
from Chatbot.exception.exception import RAG_Chatbot_Exception

api_key = os.getenv("GROQ_API_KEY")
load_dotenv()

def llm_agent():
    try:
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.1-8b-instant"
        )

        logging.info("LLM Agent initialized with model: 'llama-3.1-8b-instant'")
        return llm
    except Exception as e:
        raise RAG_Chatbot_Exception(e,sys)
        


