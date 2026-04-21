# 🚀 Research Paper Q&A Agent

## 📌 Overview
This project is an AI-based Research Paper Q&A Assistant that helps users understand research papers and retrieve relevant information using Retrieval-Augmented Generation (RAG).

---

## 🎯 Problem Statement
Researchers spend a lot of time reading papers. This project builds an assistant that:
- Answers research-related questions
- Retrieves relevant papers
- Avoids hallucinated information

---

## 🛠️ Technologies Used
- Python
- LangChain
- LangGraph
- Sentence Transformers
- Streamlit
- Groq API (LLM)

---

## ⚙️ Project Architecture

The system is built using multiple nodes:

- **Memory Node** → stores conversation history  
- **Router Node** → decides action (retrieve/tool/skip)  
- **Retrieval Node** → fetches relevant documents  
- **Tool Node** → simulates web search  
- **Answer Node** → generates response  
- **Evaluation Node** → checks faithfulness  
- **Save Node** → stores final answer  

---

## 🔄 Working Flow

1. User asks a question  
2. System stores it in memory  
3. Router decides:
   - RETRIEVE → use knowledge base  
   - TOOL → use web search  
   - SKIP → normal response  
4. Relevant data is fetched  
5. Answer is generated  
6. Output is evaluated and stored  

---

## 🧪 Testing

The system was tested using:
- Concept-based questions  
- Comparison questions  
- Out-of-scope queries  
- Red-team tests  

---

## ⚠️ Limitations
- Uses mock retrieval instead of real vector database  
- Web search is simulated  
- Depends on API for full AI functionality  

---

## 🚀 Future Improvements
- Integrate real ChromaDB  
- Add real web search API  
- Improve UI  
- Enhance accuracy  

---

## ▶️ How to Run

```bash
pip install langchain langchain-groq langgraph chromadb sentence-transformers streamlit typing-extensions

streamlit run capstone_streamlit.py
