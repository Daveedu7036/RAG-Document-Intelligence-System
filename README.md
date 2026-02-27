# RAG-Document-Intelligence-System
An AI-powered Multi-Document Question Answering System built using RAG architecture, FAISS vector search, HuggingFace embeddings, and LLaMA 3.1 via Groq, enabling context-aware and source-grounded responses from uploaded documents.

# 📄 Smart Document Assistant

AI-powered **Document Question & Answer System** built using **RAG (Retrieval-Augmented Generation)** architecture.

Created by **Sangati Daveedu**

---

## 🚀 Project Overview

Smart Document Assistant is an intelligent document Q&A system that allows users to:

- Upload multiple **PDF, DOCX, and TXT** files
- Ask questions based on uploaded documents
- Select specific documents or search across all documents
- View document sources used in answers
- Get context-aware AI responses with chat history memory

The system uses **RAG architecture**, combining document retrieval with large language models for accurate and contextual answers.

---

## 🔥 Features

### ✅ Multi-Document Upload
Supports:
- PDF
- DOCX
- TXT

### ✅ Document Filtering
- Search across All Documents
- Or select specific documents

### ✅ Context-Aware Answers
- Uses semantic similarity search
- Maintains chat history
- Mentions answer source type:
  - "From Document Context"
  - "Partially From Document + Analysis"
  - "From General Knowledge"

### ✅ Source Display
Shows which documents were used to generate the answer.

### ✅ Streaming Responses
Real-time token streaming using Groq LLaMA 3.1. 

---
## 🎯 Use Cases

- Academic doubt solving  
- Interview preparation  
- Research document analysis  
- Policy & legal document understanding  ..etc

---

## 🛠️ Tech Stack

- Frontend: Streamlit  
- LLM Provider: Groq (LLaMA 3.1)  
- Embeddings: HuggingFace Transformers  
- Vector Database: FAISS  
- Framework: LangChain  

---

## 📦 Libraries Used

- streamlit  
- langchain  
- langchain-community  
- langchain-core  
- langchain-groq  
- faiss-cpu  
- sentence-transformers  
- python-dotenv  
- docx2txt  
- pypdf

----
## 🏗️ Architecture

This project follows the **RAG (Retrieval-Augmented Generation)** pipeline:

1. Document Upload  
2. Text Chunking  
3. Embedding Generation  
4. Vector Storage (FAISS)  
5. Semantic Retrieval  
6. LLM Response Generation  
7. Source Display 



---

## 📂 Project Structure

```
smart-document-assistant/
│
├── app.py
├── .env
├── requirements.txt
└── README.md
```

 Company knowledge base chatbot 

---

## 🔑 Environment Setup

Create a `.env` file in your project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get your API key from:
https://console.groq.com/

---

## ⚙️ Installation Guide

### 1️⃣ Clone Repository

```
git clone <your-repo-link>
cd smart-document-assistant
```

### 2️⃣ Create Virtual Environment (Recommended)

```
python -m venv venv
```

Activate environment:

Windows:
```
venv\Scripts\activate
```

Mac/Linux:
```
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 4️⃣ Run Application

```
streamlit run app.py
```

---

## 🧠 Model Details

### 🔹 Embedding Model
```
sentence-transformers/all-MiniLM-L6-v2
```

### 🔹 LLM Model
```
llama-3.1-8b-instant
```

Served via Groq API for high-speed inference.

---

## 📊 How It Works Internally

- Documents are split into chunks (1000 characters with 200 overlap)
- Each chunk is converted into vector embeddings
- FAISS stores vectors for semantic search
- User query is embedded and matched with relevant chunks
- Retrieved context + chat history is sent to LLaMA 3.1
- AI generates response with streaming output
- Source documents are displayed 

---

## 💡 Future Improvements

- Persistent vector database storage  
- Citation highlighting with page numbers  
- Role-based access  
- Authentication system  
- Deployment on Streamlit Cloud / AWS / Azure  
- Conversation export (PDF)

---

## 👨‍💻 Author

Sangati Daveedu  
B.Tech CSE  
AI & GenAI Developer  

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
