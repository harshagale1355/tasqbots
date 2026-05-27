import os
from Chatbot.constant.contant_pipeline import MODEL_NAME
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from Chatbot.constant.contant_pipeline import CHUNK_SIZE, CHUNK_OVERLAP
from Chatbot.logging.logger import logging

embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME
)


def ingestion_agent(file_path):

    loader = PyPDFLoader(file_path)

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(docs)

    for chunk in chunks:
        chunk.metadata["source"] = file_path

    # If vectorstore already exists → load and append
    if os.path.exists("vectorstore"):

        db = FAISS.load_local(
            "vectorstore",
            embeddings,
            allow_dangerous_deserialization=True
        )

        db.add_documents(chunks)

    else:

        db = FAISS.from_documents(
            chunks,
            embeddings
        )

    db.save_local("vectorstore")

    logging.info(f"Ingestion completed for file: '{file_path}' with {len(chunks)} chunks created.")

    return {
        "status": "success",
        "chunks": len(chunks)
    }