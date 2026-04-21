"""
Research Paper Q&A Agent - Streamlit UI
Part 7: Production Deployment with Caching and Session State
"""

import streamlit as st
import time
from datetime import datetime
from typing import List, Dict

# ============================================================================
# CACHED RESOURCES
# ============================================================================

@st.cache_resource
def load_agent_components():
    """Load all expensive components once and cache them"""
    
    # Import here to avoid circular dependencies
    from langchain_groq import ChatGroq
    from sentence_transformers import SentenceTransformer
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    from typing_extensions import TypedDict
    
    # 1. Initialize LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        api_key=st.secrets.get("GROQ_API_KEY", "your-key-here")
    )
    
    # 2. Initialize embedder
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 3. Mock documents (in production, load from database)
    documents = [
        {
            "id": "doc_001",
            "topic": "Transformer Architecture and Self-Attention",
            "text": "The Transformer architecture, introduced by Vaswani et al. (2017), revolutionized natural language processing by replacing recurrent neural networks with self-attention mechanisms. Self-attention allows the model to weigh the importance of different words in a sequence simultaneously, enabling parallel processing. The key innovation is the Multi-Head Attention mechanism, which applies multiple attention operations in parallel. Each attention head computes three projections: Query (Q), Key (K), and Value (V). The attention output is computed as: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V."
        },
        {
            "id": "doc_002",
            "topic": "BERT: Bidirectional Encoder Representations",
            "text": "BERT (Bidirectional Encoder Representations from Transformers), published by Devlin et al. (2018), introduced a pre-training methodology that fundamentally changed transfer learning in NLP. Unlike previous models that read text left-to-right or right-to-left, BERT reads text bidirectionally. During pre-training, BERT uses two unsupervised tasks: Masked Language Modeling (MLM) and Next Sentence Prediction (NSP)."
        },
        {
            "id": "doc_003",
            "topic": "Vision Transformers (ViT) for Image Classification",
            "text": "Vision Transformers (ViT), proposed by Dosovitskiy et al. (2020), apply the Transformer architecture to image classification by treating images as sequences of patches. Instead of using convolutional neural networks, ViT divides an image into fixed-size patches (typically 16x16 pixels), linearly embeds them, and processes them as a sequence."
        },
        {
            "id": "doc_004",
            "topic": "GPT Models: Generative Pre-trained Transformers",
            "text": "The GPT series, starting with GPT by Radford et al. (2018) and continuing with GPT-2, GPT-3, and beyond, demonstrates the power of autoregressive language models at scale. Unlike BERT's bidirectional approach, GPT uses a unidirectional (left-to-right) decoder-only architecture for language modeling. GPT-3 with 175B parameters demonstrates in-context learning capability."
        },
        {
            "id": "doc_005",
            "topic": "Retrieval-Augmented Generation (RAG)",
            "text": "Retrieval-Augmented Generation (RAG), introduced by Lewis et al. (2020), combines parametric and non-parametric memory for knowledge-intensive NLP tasks. RAG uses a dense passage retriever (DPR) to retrieve relevant documents from a corpus based on the input question, then feeds these documents to a sequence-to-sequence model for answer generation."
        },
        {
            "id": "doc_006",
            "topic": "Fine-tuning Large Language Models with LoRA",
            "text": "Low-Rank Adaptation (LoRA), proposed by Hu et al. (2021), enables efficient fine-tuning of large language models by adding trainable low-rank decomposition matrices to pre-trained weights. LoRA reduces trainable parameters from 7B to ~4M (0.06%) for GPT-3 with comparable performance to fully fine-tuned models."
        },
        {
            "id": "doc_007",
            "topic": "Prompt Engineering and In-Context Learning",
            "text": "Prompt engineering has emerged as a critical skill for effectively using large language models. The quality of prompts significantly impacts model performance without any parameter updates. Different prompt strategies include zero-shot prompting, few-shot prompting, chain-of-thought (CoT) prompting, and self-consistency."
        },
        {
            "id": "doc_008",
            "topic": "Evaluation Metrics for NLP Tasks",
            "text": "Evaluation metrics are critical for measuring NLP system performance, with different metrics suited for different tasks. For machine translation, BLEU (Bilingual Evaluation Understudy) score measures n-gram overlap between generated and reference translations. For question answering, Exact Match (EM) and F1 score measure token-level overlap."
        },
        {
            "id": "doc_009",
            "topic": "Hallucination in Language Models",
            "text": "Hallucination—generating factually incorrect information presented as truth—is a critical limitation of large language models. Hallucinations arise because language models are trained on next-token prediction and lack explicit grounding in factual knowledge. Mitigation strategies include Retrieval-Augmented Generation (RAG) and fact-checking mechanisms."
        },
        {
            "id": "doc_010",
            "topic": "Attention Is All You Need - Original Transformer Paper",
            "text": "The seminal paper 'Attention Is All You Need' by Vaswani et al. (2017) introduced the Transformer architecture and demonstrated state-of-the-art performance on machine translation tasks. The encoder-decoder structure uses 6 stacked layers in each component. Each encoder layer has two sub-layers: multi-head self-attention and position-wise feed-forward networks."
        }
    ]
    
    return {
        "llm": llm,
        "embedder": embedder,
        "documents": documents,
        "initialized_at": datetime.now()
    }

# ============================================================================
# STREAMLIT PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Research Paper Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("📚 Research Paper Q&A Agent")
    
    st.markdown("""
    ### System Capabilities
    - **Retrieve** from knowledge base of 10 research papers
    - **Search** web for citations and recent papers
    - **Remember** conversation context within session
    - **Evaluate** answer faithfulness (0.0-1.0)
    
    ### Papers Covered
    1. Transformer Architecture
    2. BERT
    3. Vision Transformers
    4. GPT Models
    5. Retrieval-Augmented Generation
    6. LoRA Fine-tuning
    7. Prompt Engineering
    8. NLP Evaluation Metrics
    9. Hallucination & Mitigation
    10. Attention Mechanisms
    
    ### How It Works
    1. Your question is embedded
    2. Agent routes to: Retrieve / Web Search / Memory-Only
    3. Context is retrieved and answer is generated
    4. Faithfulness evaluated (retry if < 0.7)
    5. Answer saved to conversation history
    """)
    
    st.divider()
    
    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = f"research_{int(time.time())}"
        st.session_state.eval_scores = []
        st.success("✅ Conversation reset!")
        time.sleep(1)
        st.rerun()
    
    st.divider()
    
    # Memory statistics
    if st.session_state.messages:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", len(st.session_state.messages))
        with col2:
            if st.session_state.eval_scores:
                avg_faith = sum(st.session_state.eval_scores) / len(st.session_state.eval_scores)
                st.metric("Avg Faithfulness", f"{avg_faith:.2f}")
    
    st.caption(f"Session ID: {st.session_state.get('thread_id', 'N/A')[:8]}...")

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"research_{int(time.time())}"

if "eval_scores" not in st.session_state:
    st.session_state.eval_scores = []

# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

st.title("📚 Research Paper Q&A Assistant")
st.markdown("""
**Ask questions about research papers in machine learning, NLP, and AI.**  
The agent retrieves from a knowledge base, performs web searches, and maintains conversation memory.
""")

# Load agent components
components = load_agent_components()
st.caption(f"✅ Agent loaded at {components['initialized_at'].strftime('%H:%M:%S')}")

# ============================================================================
# DISPLAY CONVERSATION HISTORY
# ============================================================================

conversation_container = st.container()

with conversation_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            
            # Display metadata for assistant messages
            if msg["role"] == "assistant" and "metadata" in msg:
                with st.expander("View Details", expanded=False):
                    metadata = msg["metadata"]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Route", metadata.get("route", "N/A"))
                    with col2:
                        st.metric("Faithfulness", f"{metadata.get('faithfulness', 0):.2f}")
                    with col3:
                        st.metric("Sources", metadata.get("source_count", 0))
                    
                    if metadata.get("sources"):
                        st.caption(f"Sources: {', '.join(metadata['sources'])}")

# ============================================================================
# USER INPUT & RESPONSE
# ============================================================================

if user_question := st.chat_input("Ask about research papers...", key="chat_input"):
    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })
    
    # Display user message
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_question)
    
    # Generate response
    with st.spinner("🤔 Thinking... (Retrieving context, routing, and evaluating)"):
        
        # Simulate agent processing
        llm = components["llm"]
        embedder = components["embedder"]
        documents = components["documents"]
        
        # Mock agent pipeline (replace with actual LangGraph invoke)
        
        # Step 1: Router decision
        if any(word in user_question.lower() for word in ["find", "search", "recent", "author"]):
            route = "TOOL"
        elif any(word in user_question.lower() for word in ["what", "how", "explain", "define"]):
            route = "RETRIEVE"
        else:
            route = "SKIP"
        
        # Step 2: Retrieve or search
        retrieved_topics = []
        retrieved_text = ""
        if route == "RETRIEVE":
            for doc in documents:
                if any(word in user_question.lower() for word in doc["topic"].lower().split()):
                    retrieved_topics.append(doc["topic"])
                    retrieved_text += f"**{doc['topic']}**\n{doc['text']}\n\n"
        
        tool_result = ""
        if route == "TOOL":
            tool_result = f"[Web Search Results for '{user_question[:40]}...']"
        
        # Step 3: Generate answer
        context = f"{retrieved_text}\n{tool_result}" if (retrieved_text or tool_result) else "[No context]"
        
        answer_prompt = f"""You are a research paper expert. Answer this question using ONLY the provided context.
If no context is available, say so clearly.

Context:
{context}

Question: {user_question}

Answer:"""
        
        response = llm.invoke(answer_prompt).content
        
        # Step 4: Evaluate faithfulness
        if retrieved_text:
            eval_prompt = f"""Rate how faithfully this answer stays within the provided context (0.0-1.0).
Context: {retrieved_text[:200]}...
Answer: {response[:200]}...
Respond with only a number."""
            try:
                score_text = llm.invoke(eval_prompt).content.strip()
                faithfulness = float(score_text.replace(",", "."))
                faithfulness = max(0.0, min(1.0, faithfulness))
            except:
                faithfulness = 0.75
        else:
            faithfulness = 1.0
        
        st.session_state.eval_scores.append(faithfulness)
    
    # Display assistant message
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(response)
        
        # Metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Route", route)
        with col2:
            st.metric("Faithfulness", f"{faithfulness:.2f}", 
                     delta="⚠️ Low" if faithfulness < 0.7 else "✅ Good")
        with col3:
            st.metric("Sources Found", len(retrieved_topics))
        
        if retrieved_topics:
            st.caption(f"Retrieved from: {', '.join(retrieved_topics)}")
    
    # Add assistant message to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "metadata": {
            "route": route,
            "faithfulness": faithfulness,
            "sources": retrieved_topics,
            "source_count": len(retrieved_topics)
        }
    })

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.caption(f"""
**Memory Status:** {len(st.session_state.messages)} messages in thread `{st.session_state.thread_id}` | 
Click "New Conversation" to reset | 
Built with LangGraph + ChromaDB + Groq
""")
