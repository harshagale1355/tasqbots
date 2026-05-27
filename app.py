import os
import streamlit as st

from Chatbot.components.agents.ingestion.ingestion import ingestion_agent
from Chatbot.graph.graph import app


st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Multi-Document RAG Chatbot")


# Session state
if "processed" not in st.session_state:
    st.session_state.processed = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Sidebar
with st.sidebar:

    st.header("Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF Files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files and not st.session_state.processed:

        os.makedirs("uploaded_files", exist_ok=True)

        with st.spinner("Processing documents..."):

            for uploaded_file in uploaded_files:

                file_path = os.path.join(
                    "uploaded_files",
                    uploaded_file.name
                )

                # Save file
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Run ingestion
                ingestion_agent(file_path)

        st.session_state.processed = True

        st.success("Documents processed successfully!")

        st.write("### Uploaded Files")

        for file in uploaded_files:
            st.write(f"📄 {file.name}")


# Chat Input
query = st.chat_input(
    "Ask a question about the documents"
)


if query and st.session_state.processed:

    # Add user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": query
    })

    # Generate response
    with st.spinner("Generating answer..."):

        result = app.invoke({
            "query": query
        })

    answer = result["answer"]

    citations = result["citations"]

    # Add assistant message
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "citations": citations
    })


# Display chat history
for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.write(message["content"])

        if message["role"] == "assistant":

            st.write("### Citations")

            for citation in message["citations"]:

                st.write(
                    f"📄 {citation['source']} | "
                    f"Page: {citation['page']}"
                )