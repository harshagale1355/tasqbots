from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Create embeddings model once
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def ingestion_agent(file_path: str):

    # Load PDF
    loader = PyPDFLoader(file_path)

    docs = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    # Add metadata manually
    for chunk in chunks:
        chunk.metadata["source"] = file_path

    # Create vector DB
    db = FAISS.from_documents(
        chunks,
        embeddings
    )

    # Save locally
    db.save_local("vectorstore")

    return {
        "status": "success",
        "message": "Vectorstore created successfully",
        "total_chunks": len(chunks)
    }