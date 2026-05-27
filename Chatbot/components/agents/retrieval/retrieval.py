from rank_bm25 import BM25Okapi

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from Chatbot.constant.contant_pipeline import MODEL_NAME
from Chatbot.logging.logger import logging


# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME
)


def retrieval_agent(state):

    query = state["query"]

    # Load Vector DB


    db = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Vector Search


    vector_docs = db.similarity_search(
        query,
        k=3
    )


    # BM25 Search
  

    all_docs = list(db.docstore._dict.values())

    tokenized_docs = [
        doc.page_content.split()
        for doc in all_docs
    ]

    bm25 = BM25Okapi(tokenized_docs)

    tokenized_query = query.split()

    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:3]

    bm25_docs = [
        all_docs[i]
        for i in top_indices
    ]

  
    # Merge Results
   

    combined_docs = []

    seen = set()

    for doc in vector_docs + bm25_docs:

        content = doc.page_content

        if content not in seen:

            seen.add(content)

            combined_docs.append(doc)
    
    logging.info(f"Retrieval completed for query: '{query[:50]}...' with {len(combined_docs)} unique documents retrieved.")

    return {
        "retrieved_docs": combined_docs
    }