import os
from pathlib import Path
import streamlit as st

from rag_engine import load_or_create_vector_store, answer_with_context

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


st.set_page_config(page_title="Local AI Chatbot", page_icon="🤖")

st.title("Local AI Knowledge Chatbot")
st.write("Runs on Ollama and LangChain. All local. No cloud key.")


with st.sidebar:
    st.header("Document source")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )
    build_clicked = st.button("Build knowledge base")

if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None

if build_clicked and uploaded_files:
    saved_paths = []
    for file in uploaded_files:
        file_path = UPLOAD_DIR / file.name
        with open(file_path, "wb") as f:
            f.write(file.read())
        saved_paths.append(str(file_path))

    st.info("Creating vector store. This can take some time on the first run.")
    vector_store = load_or_create_vector_store(saved_paths)
    st.session_state["vector_store"] = vector_store
    st.success("Knowledge base is ready.")

st.subheader("Chat")

query = st.text_input("Ask something about your documents")

if st.button("Ask"):
    if st.session_state["vector_store"] is None:
        st.error("Please upload and build the knowledge base first.")
    elif not query.strip():
        st.warning("Please enter a question.")
    else:
        answer = answer_with_context(query, st.session_state["vector_store"])
        st.markdown("**Answer**")
        st.write(answer)
