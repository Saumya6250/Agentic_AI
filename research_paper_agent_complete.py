"""
PART 1: KNOWLEDGE BASE - 10 RESEARCH PAPER DOCUMENTS
Mock documents for Research Paper Q&A Assistant
"""

RESEARCH_DOCUMENTS = [
    {
        "id": "doc_001",
        "topic": "Transformer Architecture and Self-Attention",
        "text": """The Transformer architecture, introduced by Vaswani et al. (2017), revolutionized natural language processing by replacing recurrent neural networks with self-attention mechanisms. Self-attention allows the model to weigh the importance of different words in a sequence simultaneously, enabling parallel processing. The key innovation is the Multi-Head Attention mechanism, which applies multiple attention operations in parallel. Each attention head computes three projections: Query (Q), Key (K), and Value (V). The attention output is computed as: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V. This parallel computation reduces training time significantly compared to RNNs. Transformers use positional encoding to preserve word order information since there are no recurrent connections. The architecture consists of stacked encoder and decoder blocks, each containing a multi-head attention layer and a feed-forward network. This design has become the foundation for models like BERT, GPT, and T5, which have achieved state-of-the-art results across numerous NLP tasks."""
    },
    {
        "id": "doc_002",
        "topic": "BERT: Bidirectional Encoder Representations",
        "text": """BERT (Bidirectional Encoder Representations from Transformers), published by Devlin et al. (2018), introduced a pre-training methodology that fundamentally changed transfer learning in NLP. Unlike previous models that read text left-to-right or right-to-left, BERT reads text bidirectionally. During pre-training, BERT uses two unsupervised tasks: Masked Language Modeling (MLM) and Next Sentence Prediction (NSP). In MLM, 15% of input tokens are randomly masked, and the model predicts them using bidirectional context. In NSP, the model learns to predict whether two sentences are consecutive in the original text. BERT uses WordPiece tokenization, breaking words into subword units to handle out-of-vocabulary words. The model comes in two sizes: BERT-base (110M parameters) and BERT-large (340M parameters). BERT achieves 88.5% accuracy on GLUE benchmark without task-specific architecture changes, proving the effectiveness of pre-trained bidirectional representations. Fine-tuning BERT for downstream tasks requires minimal task-specific modifications, making it highly practical for industry applications."""
    },
    {
        "id": "doc_003",
        "topic": "Vision Transformers (ViT) for Image Classification",
        "text": """Vision Transformers (ViT), proposed by Dosovitskiy et al. (2020), apply the Transformer architecture to image classification by treating images as sequences of patches. Instead of using convolutional neural networks, ViT divides an image into fixed-size patches (typically 16x16 pixels), linearly embeds them, and processes them as a sequence. The model adds learnable class tokens and position embeddings to each patch. The Transformer encoder then processes this sequence bidirectionally. For a 224x224 image with 16x16 patches, this creates 196 patches (14x14 grid). ViT-Base with 12 layers achieves 77.9% top-1 accuracy on ImageNet without using convolutions. A key finding is that ViT requires less inductive bias (assumptions about image structure) compared to CNNs, making it more data-efficient when trained on large datasets like ImageNet-21k. ViT demonstrates that pure Transformer architectures can outperform CNN-based approaches like ResNet, challenging the dominance of convolutional architectures in computer vision for over a decade."""
    },
    {
        "id": "doc_004",
        "topic": "Attention Is All You Need - Original Transformer Paper",
        "text": """The seminal paper 'Attention Is All You Need' by Vaswani et al. (2017) introduced the Transformer architecture and demonstrated state-of-the-art performance on machine translation tasks. The paper compares Transformers to previous sequence-to-sequence models like RNNs and shows significant improvements in both translation quality (BLEU scores) and training efficiency. The encoder-decoder structure uses 6 stacked layers in each component. Each encoder layer has two sub-layers: multi-head self-attention and position-wise feed-forward networks. Each decoder layer has three sub-layers: masked multi-head self-attention, encoder-decoder attention, and feed-forward networks. The masking in decoder self-attention prevents positions from attending to subsequent positions, preserving the autoregressive property. The paper introduces scaled dot-product attention with the scaling factor 1/sqrt(d_k) to prevent gradients from becoming extremely small. With 8 attention heads and d_model=512, the model achieves BLEU score of 28.4 on WMT 2014 English-German translation. The paper also demonstrates that attention mechanisms are more interpretable than RNN hidden states, allowing visualization of which words the model attends to."""
    },
    {
        "id": "doc_005",
        "topic": "GPT Models: Generative Pre-trained Transformers",
        "text": """The GPT series, starting with GPT by Radford et al. (2018) and continuing with GPT-2 (2019), GPT-3 (2020), and beyond, demonstrates the power of autoregressive language models at scale. Unlike BERT's bidirectional approach, GPT uses a unidirectional (left-to-right) decoder-only architecture for language modeling. The pre-training objective is simple: predict the next token given previous tokens. GPT-2 with 1.5B parameters shows that training a large language model on diverse data learns task-specific behaviors without explicit task-specific training. The model achieves strong zero-shot performance on multiple benchmarks without fine-tuning. GPT-3 with 175B parameters demonstrates in-context learning capability: the model can perform new tasks by conditioning on few examples in the prompt without parameter updates. This emergence of few-shot learning ability was unexpected and has significant implications for AI systems. GPT models use Byte-Pair Encoding (BPE) tokenization. The success of GPT models led to the dominance of decoder-only architectures, influencing the design of subsequent large language models like PaLM and LLaMA."""
    },
    {
        "id": "doc_006",
        "topic": "Retrieval-Augmented Generation (RAG)",
        "text": """Retrieval-Augmented Generation (RAG), introduced by Lewis et al. (2020), combines parametric and non-parametric memory for knowledge-intensive NLP tasks. RAG uses a dense passage retriever (DPR) to retrieve relevant documents from a corpus based on the input question, then feeds these documents to a sequence-to-sequence model for answer generation. The retriever uses BERT-based dense embeddings to find relevant passages, while the generator is a BART model fine-tuned on extractive and abstractive QA datasets. RAG achieves 64.2% exact match on SQuAD compared to 61.5% for parametric-only approaches. The hybrid approach addresses the limitation of pure parametric models which cannot easily update knowledge without retraining. RAG enables models to cite sources, improving verifiability and trustworthiness. The retrieval component is differentiable, allowing end-to-end training of both retriever and generator. RAG has inspired numerous variants like Dense Passage Retrieval (DPR), ColBERT, and others that improve retrieval quality. The technique is now foundational in building grounded language models that cite evidence for their claims, which is critical for high-stakes applications like medical or legal domains."""
    },
    {
        "id": "doc_007",
        "topic": "Fine-tuning Large Language Models with LoRA",
        "text": """Low-Rank Adaptation (LoRA), proposed by Hu et al. (2021), enables efficient fine-tuning of large language models by adding trainable low-rank decomposition matrices to pre-trained weights. Instead of fine-tuning all 7B+ parameters in models like GPT-3, LoRA freezes the original weights and adds two trainable matrices: A (input to hidden dimension r) and B (hidden dimension r to output). The weight update is: ΔW = BA, where r << d (original dimension). With r=8, LoRA reduces trainable parameters from 7B to ~4M (0.06%) for GPT-3. LoRA-finetuned models achieve comparable performance to fully fine-tuned models while requiring 10-100x less memory and compute. Fine-tuning takes hours instead of days. LoRA matrices can be easily switched, enabling task-specific model variants sharing the same base weights. The approach has been widely adopted for practical fine-tuning of LLaMA, Alpaca, and other models. LoRA demonstrates that pre-trained models encode generic knowledge in their dense weight matrices, and task-specific behavior can be efficiently captured through low-rank updates. Variants like QLoRA extend this to quantized 4-bit models, further reducing memory requirements."""
    },
    {
        "id": "doc_008",
        "topic": "Prompt Engineering and In-Context Learning",
        "text": """Prompt engineering has emerged as a critical skill for effectively using large language models, as demonstrated by research from Brown et al. (2020) on GPT-3 and follow-up work by Wei et al. (2022) on chain-of-thought prompting. The quality of prompts significantly impacts model performance without any parameter updates. Different prompt strategies include zero-shot prompting (no examples), few-shot prompting (with examples), chain-of-thought (CoT) prompting (asking for step-by-step reasoning), and self-consistency (sampling multiple reasoning paths). Chain-of-thought prompting improves GPT-3 accuracy on arithmetic reasoning from 17% (zero-shot) to 78% (CoT). The phenomenon of in-context learning allows models to adapt to new tasks just by observing examples in the input context. The underlying mechanism likely involves the model using its pre-trained knowledge to implement task-specific algorithms without gradient updates. Effective prompts often include: clear task description, input-output format specification, and relevant examples. Research shows that prompt engineering can match or exceed the performance gains from model scaling in some tasks. However, prompts are fragile and performance varies significantly with minor wording changes, highlighting the need for systematic prompt evaluation methods."""
    },
    {
        "id": "doc_009",
        "topic": "Evaluation Metrics for NLP Tasks",
        "text": """Evaluation metrics are critical for measuring NLP system performance, with different metrics suited for different tasks. For machine translation, BLEU (Bilingual Evaluation Understudy) score by Papineni et al. (2002) measures n-gram overlap between generated and reference translations (range 0-100). However, BLEU has known limitations: it doesn't capture semantic similarity and correlates imperfectly with human judgments. For question answering, Exact Match (EM) and F1 score measure token-level overlap with reference answers. ROUGE metrics (Recall-Oriented Understudy for Gisting Evaluation) are used for summarization, measuring overlap of unigrams, bigrams, and longest common subsequences. For semantic similarity, Spearman correlation with human judgment scores is standard. METEOR (Metric for Evaluation of Translation with Explicit Ordering) improves on BLEU by incorporating synonymy and paraphrasing. Recent neural-based metrics like BERTScore use contextual embeddings to measure semantic similarity. The field has moved toward task-specific metrics and human evaluation for high-stakes applications. A critical insight is that no single automatic metric perfectly correlates with human quality judgments, so best practice involves multiple metrics and human evaluation. Standardized benchmarks like GLUE, SQuAD, and BLEU corpus enable reproducible comparison across research teams."""
    },
    {
        "id": "doc_010",
        "topic": "Hallucination in Language Models and Mitigation Strategies",
        "text": """Hallucination—generating factually incorrect information presented as truth—is a critical limitation of large language models, as documented by Maynez et al. (2020) and Ji et al. (2022). Even high-performing models like GPT-3 and PaLM hallucinate facts, numbers, and citations that never existed. Hallucinations arise because language models are trained on next-token prediction and lack explicit grounding in factual knowledge. They optimize for linguistic fluency, not factual accuracy. Mitigation strategies include: (1) Retrieval-Augmented Generation (RAG), grounding generation in retrieved documents; (2) Fact-checking mechanisms, using separate models to verify claims; (3) Uncertainty quantification, training models to express confidence levels; (4) Constrained decoding, limiting outputs to verifiable facts. The RAGAS framework evaluates hallucination through faithfulness scores—measuring whether generated content is supported by retrieved context. A faithfulness score below 0.7 typically indicates problematic hallucination. For production systems handling critical information (medical, legal, financial), hallucination mitigation is non-negotiable. Recent work on constitutional AI attempts to align models toward truthfulness through reinforcement learning. Understanding and mitigating hallucination remains an active research area, especially as models scale to trillions of parameters."""
    }
]

# ============================================================================
# PART 2: STATE DESIGN - TypedDict
# ============================================================================

from typing_extensions import TypedDict
from typing import List

class ResearchPaperState(TypedDict):
    """State for Research Paper Q&A Agent"""
    # Mandatory base fields
    question: str                    # Current user question
    messages: List[dict]             # Conversation history [{role, content}, ...]
    route: str                       # Router decision: 'retrieve', 'tool', 'skip'
    retrieved: str                   # Context from ChromaDB retrieval
    sources: List[str]              # List of paper topics from retrieval
    tool_result: str                # Result from web search tool
    answer: str                      # Generated answer
    faithfulness: float             # Faithfulness score (0.0-1.0)
    eval_retries: int               # Number of retry attempts
    
    # Domain-specific fields
    paper_context: str              # Additional paper metadata
    num_papers_cited: int           # Count of papers mentioned

# ============================================================================
# PART 3: NODE FUNCTIONS
# ============================================================================

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime
import json

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

# ---- MEMORY NODE ----
def memory_node(state: ResearchPaperState) -> ResearchPaperState:
    """Append question to history, apply sliding window, extract context"""
    messages = state.get("messages", [])
    new_msg = {"role": "user", "content": state["question"]}
    messages.append(new_msg)
    
    # Sliding window: keep last 6 turns (12 messages)
    messages = messages[-12:]
    
    # Extract paper context if user mentions "this paper" or "my paper"
    paper_context = ""
    if "this paper" in state["question"].lower() or "my paper" in state["question"].lower():
        paper_context = "User is asking about their own paper context."
    
    return {
        **state,
        "messages": messages,
        "paper_context": paper_context,
        "num_papers_cited": 0
    }

# ---- ROUTER NODE ----
def router_node(state: ResearchPaperState) -> ResearchPaperState:
    """Decide: retrieve from KB, use web search, or memory-only"""
    
    messages_text = "\n".join([f"{m['role']}: {m['content']}" for m in state["messages"]])
    
    router_prompt = f"""You are a research paper Q&A router. Decide which strategy to use:

RETRIEVE: Question asks about papers in our knowledge base (methodology, findings, contributions)
TOOL: Question asks to find NEW papers, citations, author names, or recent research not in our KB
SKIP: Question is conversational or doesn't need paper knowledge

Current conversation:
{messages_text}

Respond with ONLY one word: RETRIEVE, TOOL, or SKIP"""

    response = llm.invoke(router_prompt).content.strip().upper()
    
    # Ensure valid route
    route = "RETRIEVE" if "RETRIEVE" in response else ("TOOL" if "TOOL" in response else "SKIP")
    
    return {**state, "route": route}

# ---- RETRIEVAL NODE ----
def retrieval_node(state: ResearchPaperState) -> ResearchPaperState:
    """Retrieve relevant papers from ChromaDB"""
    from sentence_transformers import SentenceTransformer
    
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Mock retrieval (in production, query actual ChromaDB collection)
    query_embedding = embedder.encode(state["question"])
    
    # Simulate retrieval scores
    retrieved_topics = []
    if any(term in state["question"].lower() for term in 
           ["transformer", "attention", "bert", "gpt", "architecture"]):
        retrieved_topics = ["Transformer Architecture and Self-Attention", 
                           "BERT: Bidirectional Encoder Representations"]
    elif any(term in state["question"].lower() for term in ["vision", "image", "vit"]):
        retrieved_topics = ["Vision Transformers (ViT) for Image Classification"]
    elif any(term in state["question"].lower() for term in ["rag", "retrieval", "generation"]):
        retrieved_topics = ["Retrieval-Augmented Generation (RAG)"]
    elif any(term in state["question"].lower() for term in ["lora", "fine-tun", "finetun"]):
        retrieved_topics = ["Fine-tuning Large Language Models with LoRA"]
    elif any(term in state["question"].lower() for term in ["prompt", "in-context"]):
        retrieved_topics = ["Prompt Engineering and In-Context Learning"]
    elif any(term in state["question"].lower() for term in ["metric", "evaluat"]):
        retrieved_topics = ["Evaluation Metrics for NLP Tasks"]
    elif any(term in state["question"].lower() for term in ["hallucin", "factual"]):
        retrieved_topics = ["Hallucination in Language Models and Mitigation Strategies"]
    
    retrieved_text = ""
    for i, doc in enumerate(RESEARCH_DOCUMENTS):
        if doc["topic"] in retrieved_topics:
            retrieved_text += f"[{doc['topic']}]\n{doc['text']}\n\n"
    
    if not retrieved_text:
        retrieved_text = "[No relevant papers found in knowledge base]"
    
    return {
        **state,
        "retrieved": retrieved_text,
        "sources": retrieved_topics,
        "num_papers_cited": len(retrieved_topics)
    }

# ---- SKIP RETRIEVAL NODE ----
def skip_retrieval_node(state: ResearchPaperState) -> ResearchPaperState:
    """Skip retrieval for memory-only or tool-only queries"""
    return {
        **state,
        "retrieved": "",
        "sources": [],
        "num_papers_cited": 0
    }

# ---- WEB SEARCH TOOL NODE ----
def tool_node(state: ResearchPaperState) -> ResearchPaperState:
    """Web search for papers, citations, and recent research"""
    
    question = state["question"]
    
    # Mock web search results (in production, use actual API)
    search_results = ""
    
    if "find" in question.lower() or "search" in question.lower():
        search_results = f"""[Web Search Results for: {question[:50]}...]
1. arXiv preprint found: "Advanced Transformer Architectures" (2023)
2. Google Scholar: 2,450 citations for related topic
3. Recent paper on related conference: ICML 2024 Proceedings
4. GitHub implementation with 5.2k stars
Note: These are simulated results. In production, use real search API."""
    elif "author" in question.lower() or "researcher" in question.lower():
        search_results = f"""[Author/Researcher Search Results]
- Found publication records on semantic-scholar and DBLP
- Recent conference presentations at NeurIPS, ICML, ACL
- Citation metrics and h-index available
Note: Actual search would return specific researcher profiles."""
    elif "recent" in question.lower() or "latest" in question.lower():
        search_results = f"""[Recent Research Results]
- Papers from last 6 months on arXiv
- Latest conference proceedings (2024)
- Trending topics in research community
Note: Actual search would return current publications."""
    else:
        search_results = f"""[Web Search Results]
No specific papers found matching the query. Try searching for:
- Specific paper titles
- Author names
- Research topics with publication year
Note: In production, this would query a real search API."""
    
    return {
        **state,
        "tool_result": search_results,
        "sources": ["Web Search"] if search_results else []
    }

# ---- ANSWER NODE ----
def answer_node(state: ResearchPaperState) -> ResearchPaperState:
    """Generate answer using context and/or tool results"""
    
    # Build context
    context = ""
    if state["retrieved"]:
        context += f"KNOWLEDGE BASE CONTEXT:\n{state['retrieved']}\n"
    if state["tool_result"]:
        context += f"WEB SEARCH RESULTS:\n{state['tool_result']}\n"
    
    messages_text = "\n".join([f"{m['role']}: {m['content']}" for m in state["messages"][-4:]])
    
    system_prompt = """You are an expert research paper Q&A assistant. Your job is to answer questions about research papers.

RULES:
1. Answer ONLY from the knowledge base context provided
2. If context is not provided, say "I don't have this information in my knowledge base"
3. Cite paper topics when referencing them
4. For web search results, note that they are from external sources
5. Never fabricate paper details, author names, or citations
6. If unsure, admit uncertainty clearly
7. Keep answers concise and focused"""

    answer_prompt = f"""{system_prompt}

CONTEXT:
{context if context else "[No context available]"}

CONVERSATION:
{messages_text}

Answer the user's last question based ONLY on the context provided. If no relevant context exists, clearly say so."""

    response = llm.invoke(answer_prompt).content.strip()
    
    return {
        **state,
        "answer": response,
        "eval_retries": 0
    }

# ---- FAITHFULNESS EVAL NODE ----
def eval_node(state: ResearchPaperState) -> ResearchPaperState:
    """Evaluate answer faithfulness to retrieved context"""
    
    if not state["retrieved"]:
        # No retrieval, skip eval
        return {**state, "faithfulness": 1.0}
    
    eval_prompt = f"""Rate the faithfulness of this answer based on the provided context.
Faithfulness = whether answer uses ONLY information from the context, without adding external knowledge.

CONTEXT:
{state['retrieved']}

ANSWER:
{state['answer']}

Rate as a decimal 0.0 to 1.0 where:
- 1.0: Answer uses only context information
- 0.7: Answer is mostly from context with minor additions
- 0.5: Answer mixes context with external knowledge
- 0.0: Answer ignores context entirely

Respond with ONLY a number between 0.0 and 1.0"""

    score_text = llm.invoke(eval_prompt).content.strip()
    try:
        score = float(score_text)
        score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
    except:
        score = 0.5  # Default if parsing fails
    
    eval_retries = state.get("eval_retries", 0) + 1
    
    return {
        **state,
        "faithfulness": score,
        "eval_retries": eval_retries
    }

# ---- SAVE NODE ----
def save_node(state: ResearchPaperState) -> ResearchPaperState:
    """Save answer to message history"""
    messages = state.get("messages", [])
    messages.append({"role": "assistant", "content": state["answer"]})
    return {**state, "messages": messages}

# ============================================================================
# PART 4: GRAPH ASSEMBLY
# ============================================================================

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

def route_decision(state: ResearchPaperState) -> str:
    """Decide next node based on router output"""
    route = state.get("route", "SKIP").upper()
    if "RETRIEVE" in route:
        return "retrieve"
    elif "TOOL" in route:
        return "tool"
    else:
        return "skip"

def eval_decision(state: ResearchPaperState) -> str:
    """Decide: retry answer or save"""
    FAITHFULNESS_THRESHOLD = 0.7
    MAX_EVAL_RETRIES = 2
    
    faithfulness = state.get("faithfulness", 1.0)
    retries = state.get("eval_retries", 0)
    
    # If retrieved is empty, skip faithfulness check
    if not state.get("retrieved", ""):
        return "save"
    
    if faithfulness < FAITHFULNESS_THRESHOLD and retries < MAX_EVAL_RETRIES:
        return "answer"  # Retry answer
    else:
        return "save"  # Accept and save

# Create graph
graph = StateGraph(ResearchPaperState)

# Add nodes
graph.add_node("memory", memory_node)
graph.add_node("router", router_node)
graph.add_node("retrieve", retrieval_node)
graph.add_node("skip", skip_retrieval_node)
graph.add_node("tool", tool_node)
graph.add_node("answer", answer_node)
graph.add_node("eval", eval_node)
graph.add_node("save", save_node)

# Set entry point
graph.set_entry_point("memory")

# Add fixed edges
graph.add_edge("memory", "router")
graph.add_edge("retrieve", "answer")
graph.add_edge("skip", "answer")
graph.add_edge("tool", "answer")
graph.add_edge("answer", "eval")
graph.add_edge("save", END)

# Add conditional edges
graph.add_conditional_edges("router", route_decision, {
    "retrieve": "retrieve",
    "tool": "tool",
    "skip": "skip"
})

graph.add_conditional_edges("eval", eval_decision, {
    "answer": "answer",
    "save": "save"
})

# Compile with memory
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

print("✅ Graph compiled successfully!")

# ============================================================================
# PART 5: TESTING - 10 TEST QUESTIONS
# ============================================================================

def ask_question(question: str, thread_id: str = "default_thread"):
    """Helper function to ask a question and get answer"""
    input_state = {
        "question": question,
        "messages": [],
        "route": "",
        "retrieved": "",
        "sources": [],
        "tool_result": "",
        "answer": "",
        "faithfulness": 0.0,
        "eval_retries": 0,
        "paper_context": "",
        "num_papers_cited": 0
    }
    
    result = app.invoke(input_state, config={"configurable": {"thread_id": thread_id}})
    return result

# Test questions
TEST_QUESTIONS = [
    # Domain-specific questions (should retrieve)
    {
        "question": "What is the self-attention mechanism in Transformers?",
        "expected_route": "retrieve",
        "category": "Core Concept"
    },
    {
        "question": "How does BERT differ from GPT?",
        "expected_route": "retrieve",
        "category": "Comparison"
    },
    {
        "question": "What are the main components of Vision Transformers?",
        "expected_route": "retrieve",
        "category": "Architecture"
    },
    {
        "question": "Explain Retrieval-Augmented Generation (RAG)",
        "expected_route": "retrieve",
        "category": "Technique"
    },
    {
        "question": "What is LoRA and why is it useful?",
        "expected_route": "retrieve",
        "category": "Method"
    },
    
    # Web search questions
    {
        "question": "Find recent papers on prompt engineering published in 2024",
        "expected_route": "tool",
        "category": "Paper Search"
    },
    {
        "question": "Who are the authors of the Vision Transformer paper and what are their other works?",
        "expected_route": "tool",
        "category": "Author Research"
    },
    
    # Memory/Conversational (skip retrieval)
    {
        "question": "Thank you for that explanation!",
        "expected_route": "skip",
        "category": "Acknowledgment"
    },
    
    # ❌ OUT-OF-SCOPE QUESTION
    {
        "question": "What is the best way to cook pasta?",
        "expected_route": "skip",
        "category": "Out-of-Scope",
        "is_red_team": True
    },
    
    # ❌ FALSE-PREMISE QUESTION
    {
        "question": "How does the attention mechanism in Transformers use recurrent connections like RNNs?",
        "expected_route": "retrieve",
        "category": "False Premise",
        "is_red_team": True,
        "note": "Agent should correct: Transformers don't use recurrent connections"
    }
]

# Run all tests
print("\n" + "="*80)
print("TESTING PHASE - 10 Test Questions")
print("="*80)

for i, test in enumerate(TEST_QUESTIONS, 1):
    print(f"\n[Test {i}] {test['category']}")
    print(f"Question: {test['question']}")
    
    result = ask_question(test['question'], thread_id=f"test_{i}")
    
    print(f"Route: {result.get('route', 'UNKNOWN')}")
    print(f"Faithfulness: {result.get('faithfulness', 0.0):.2f}")
    print(f"Answer preview: {result.get('answer', '')[:150]}...")
    
    if test.get('is_red_team'):
        print(f"⚠️  RED TEAM TEST: {test.get('note', 'Special adversarial test')}")
    
    print("-" * 80)

print("\n✅ All 10 tests completed!")

# ============================================================================
# PART 7: STREAMLIT UI
# ============================================================================

STREAMLIT_CODE = '''
import streamlit as st
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
import chromadb

@st.cache_resource
def initialize_agent():
    """Load all expensive resources once"""
    # Initialize embedder
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialize LLM
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
    
    # Initialize ChromaDB collection (mock)
    documents = [
        {"id": "doc_001", "topic": "Transformer Architecture and Self-Attention", 
         "text": "The Transformer architecture..."},
        # ... add all 10 documents
    ]
    
    # Compiled graph (from Part 4)
    # app = compiled LangGraph with MemorySaver
    
    return embedder, llm, documents

# Page config
st.set_page_config(
    page_title="Research Paper Q&A Assistant",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "default"

# Load agent
embedder, llm, documents = initialize_agent()

# Sidebar
with st.sidebar:
    st.title("📚 Research Paper Q&A")
    st.markdown("""
    ### About This Assistant
    Ask questions about research papers in computer science:
    - Transformer architectures
    - BERT, GPT, Vision Transformers
    - Retrieval-Augmented Generation
    - Fine-tuning techniques
    - Evaluation metrics
    
    ### Features
    - ✅ Retrieval from knowledge base
    - 🔍 Web search for new papers
    - 💾 Memory across conversation
    - 📊 Faithfulness evaluation
    """)
    
    if st.button("🔄 New Conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = f"thread_{int(time.time())}"
        st.rerun()

# Main chat interface
st.title("📚 Research Paper Q&A Assistant")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if user_input := st.chat_input("Ask a question about research papers..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Get agent response
    with st.spinner("Thinking..."):
        # Invoke agent with thread_id for memory
        # result = app.invoke(state, config={"configurable": {"thread_id": st.session_state.thread_id}})
        
        # Mock response for demonstration
        assistant_response = f"Based on the knowledge base and web search: {user_input[:50]}..."
        faithfulness = 0.85
    
    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    
    with st.chat_message("assistant"):
        st.markdown(assistant_response)
        st.caption(f"Faithfulness Score: {faithfulness:.2f}")
'''

# Print Streamlit code to console
print("\n" + "="*80)
print("PART 7: STREAMLIT CODE (capstone_streamlit.py)")
print("="*80)
print(STREAMLIT_CODE)

print("\n✅ Complete Research Paper Q&A Agent Generated!")
print("\nNext Steps:")
print("1. Copy PART 1-5 code into your Jupyter notebook")
print("2. Save PART 7 Streamlit code as: capstone_streamlit.py")
print("3. Run: streamlit run capstone_streamlit.py")
print("4. Test with the 10 sample questions provided")
