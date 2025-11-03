from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import Ollama


def load_documents(file_paths: List[str]):
    docs = []
    for path in file_paths:
        if path.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        else:
            loader = TextLoader(path, encoding="utf-8")
            docs.extend(loader.load())
    return docs


def build_vector_store(documents, persist_directory: str = "storage"):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(documents)

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    vs.persist()
    return vs


def load_or_create_vector_store(file_paths: List[str], persist_directory: str = "storage"):
    # always rebuild from what the user uploaded
    docs = load_documents(file_paths)

    persist_path = Path(persist_directory)
    if persist_path.exists():
        for item in persist_path.iterdir():
            if item.is_file():
                item.unlink()
            else:
                for sub in item.iterdir():
                    sub.unlink()
                item.rmdir()

    return build_vector_store(docs, persist_directory=persist_directory)


def answer_with_context(query: str, vector_store: Chroma, top_k: int = 6) -> str:
    q = query.lower()

    docs = []

    # for abstract or intro, prefer early pages
    if "abstract" in q or "introduction" in q or "background" in q:
        for p in [0, 1, 2]:
            try:
                docs.extend(vector_store.similarity_search(query, k=2, filter={"page": p}))
            except Exception:
                continue

    # if we did not get anything useful, do normal search
    if not docs:
        docs = vector_store.similarity_search(query, k=top_k)

    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
You are answering questions about a research paper.
Use only the context given below.
If the context does not contain the answer say that it is not present in the document.

Context:
{context}

Question: {query}

Answer in two or three sentences:
"""

    llm = Ollama(model="llama3")
    return llm.invoke(prompt)
