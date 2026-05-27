# Multi-Document RAG Chatbot

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![LangChain](https://img.shields.io/badge/LangChain-Integration-green)
![LangGraph](https://img.shields.io/badge/LangGraph-Agents-orange)

A powerful Multi-Document Retrieval-Augmented Generation (RAG) Chatbot built with **Streamlit**, **LangChain**, and **LangGraph**. This application allows users to upload multiple PDF documents, process them, and ask questions. It uses advanced agentic workflows to retrieve context, generate answers, and evaluate the quality of its responses before presenting them to the user.

## ✨ Features

- **Multi-Document Support:** Upload and process multiple PDF files simultaneously.
- **Agentic Workflow (LangGraph):** Employs a state graph with dedicated agents for:
  - `Retrieval`: Finding the most relevant context.
  - `Response`: Formulating an accurate answer based on retrieved documents.
  - `Evaluator`: Critiquing the generated answer and initiating a retry if the quality isn't satisfactory.
- **Citations & Sources:** Automatically provides citations for the generated answers, including the source document and page number.
- **Conversation Memory:** Remembers chat history for contextual follow-up questions.
- **Dockerized:** Ready to be built and deployed via Docker.

## 🛠️ Technology Stack

- **Frontend:** Streamlit
- **LLM / Orchestration:** LangChain, LangGraph
- **Vector Search:** ChromaDB / FAISS
- **Embeddings:** Sentence Transformers / HuggingFace
- **Document Parsing:** PyPDF2 / pypdf
- **Deployment:** Docker

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- OpenAI or Groq API keys (configured via `.env`)

### Local Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd Tasqbots
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the root directory and add your necessary API keys:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   GROQ_API_KEY=your_groq_api_key
   ```

5. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

### Running with Docker

1. **Build the Docker image:**
   ```bash
   docker build -t tasqbots-rag .
   ```

2. **Run the Docker container:**
   ```bash
   docker run -p 8501:8501 --env-file .env tasqbots-rag
   ```
   
The app will be available at `http://localhost:8501`.

## 📁 Project Structure

```text
Tasqbots/
├── app.py                     # Main Streamlit application entry point
├── Chatbot/                   # Core application logic
│   ├── components/            # Agent modules
│   │   └── agents/
│   │       ├── ingestion/     # PDF processing and vector store ingestion
│   │       ├── retrieval/     # Context retrieval agent
│   │       ├── response/      # Answer generation agent
│   │       └── evaluator/     # Quality check and retry routing agent
│   ├── graph/                 # LangGraph state machine configuration
│   ├── constant/              # Configuration constants
│   ├── exception/             # Custom exception handling
│   └── logging/               # Logging configuration
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker configuration
└── .env                       # Environment variables (not tracked)
```

## 🤝 Usage

1. Open the application in your browser.
2. Expand the sidebar and upload one or more PDF documents.
3. Wait for the processing to finish (the agent will extract text, chunk it, and store embeddings).
4. Type your question in the chat input.
5. Review the detailed answers along with precise document citations!