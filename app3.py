import streamlit as st
#import os
import tempfile
#from dotenv import load_dotenv
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# PAGE CONFIG
st.set_page_config(page_title="Smart Document Assistant", layout="wide")
st.title("📄 Smart Document Assistant")
st.markdown("## Created by *Sangati Daveedu*")
st.markdown("##### Built an AI-powered Document Q&A system using RAG architecture with LangChain, FAISS, HuggingFace embeddings, and LLaMA 3.1, enabling context-aware question answering from uploaded documents.")
st.markdown("### 📂 Upload the Documents & Ask Your Doubts")


# LOAD ENV
#load_dotenv()
#GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]


# CACHE MODELS
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=GROQ_API_KEY,
        streaming=True
    )

embeddings = load_embeddings()
llm = load_llm()

# SESSION STATE
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# FILE PROCESSING
def process_files(uploaded_files) -> List:
    all_docs = []

    for uploaded_file in uploaded_files:

        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name

        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        elif uploaded_file.name.endswith(".docx"):
            loader = Docx2txtLoader(tmp_path)
        elif uploaded_file.name.endswith(".txt"):
            loader = TextLoader(tmp_path)
        else:
            continue

        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = uploaded_file.name

        all_docs.extend(docs)

    return all_docs

# FILE UPLOAD
uploaded_files = st.file_uploader(
    "Upload PDF, DOCX, or TXT files",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    documents = process_files(uploaded_files)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    if st.session_state.vectorstore:
        st.session_state.vectorstore.add_documents(chunks)
    else:
        st.session_state.vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )

    st.success("✅ Documents processed successfully!")

# CHAT SYSTEM
if st.session_state.vectorstore:

    vectorstore = st.session_state.vectorstore

    # Extract document names
    all_docs = list(vectorstore.docstore._dict.values())
    unique_sources = sorted(
        list(set(doc.metadata.get("source", "Unknown") for doc in all_docs))
    )

    st.markdown("### 📂 Select Documents")

    search_all = st.checkbox("All Documents", value=True)

    selected_sources = st.multiselect(
        "Select one or more documents:",
        unique_sources,
        disabled=search_all
    )

    prompt = ChatPromptTemplate.from_template("""
You are an intelligent AI assistant.

Your task is to answer the user's question using the provided document context.

Rules:
1. First, carefully analyze the provided context.
2. If the answer is clearly found in the context, answer strictly based on it.
3. If the answer is partially found, combine the context with logical reasoning to complete the answer.
4. If the answer is NOT present in the context, use your general knowledge to provide a clear and correct answer.
5. Clearly mention whether the answer is:
   - "From Document Context"
   - "Partially From Document + Analysis"
   - "From General Knowledge"
Context:
{context}

Chat History:
{history}

Question:
{question}
""")

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Bottom chat input
    user_query = st.chat_input("Ask your question...")

    if user_query:

        st.session_state.messages.append({
            "role": "user",
            "content": user_query
        })

        with st.chat_message("user"):
            st.markdown(user_query)

        # Retriever logic
        if search_all:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        else:
            if not selected_sources:
                st.warning("⚠ Please select at least one document.")
                st.stop()

            retriever = vectorstore.as_retriever(
                search_kwargs={
                    "k": 4,
                    "filter": {"source": {"$in": selected_sources}}
                }
            )

        docs = retriever.invoke(user_query)

        context = "\n\n".join([doc.page_content for doc in docs])

        history_text = "\n".join(
            [
                f"{m['role']}: {m['content']}"
                for m in st.session_state.messages
            ]
        )

        chain = prompt | llm

        with st.chat_message("assistant"):
            response_container = st.empty()
            full_response = ""

            for chunk in chain.stream({
                "context": context,
                "history": history_text,
                "question": user_query
            }):
                full_response += chunk.content
                response_container.markdown(full_response + "▌")

            response_container.markdown(full_response)

            # Show sources
            unique_sources_used = list(
                set(doc.metadata["source"] for doc in docs)
            )

            if unique_sources_used:
                st.markdown("### 📚 Sources")
                for source in unique_sources_used:
                    st.markdown(f"- {source}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response
        })