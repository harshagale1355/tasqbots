from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

api_key = os.getenv("GROQ_API_KEY")
load_dotenv()

def llm_agent():
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant"
    )
    return llm
