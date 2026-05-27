from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Create embeddings once
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def retrieval_agent(state):

    query = state["query"]

    # Load vector database
    db = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Create retriever
    retriever = db.as_retriever(
        search_kwargs={"k": 3}
    )

    # Retrieve relevant chunks
    docs = retriever.invoke(query)

    return {
        "retrieved_docs": docs
    }