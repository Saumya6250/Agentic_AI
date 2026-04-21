# Research Paper Q&A Agent - Complete Capstone Solution

**Domain:** Research Paper Q&A Assistant  
**Tool:** Web Search (find papers, citations, authors)  
**Student:** [Your Name]  
**Date:** 2026

---

## 📋 What You've Been Given

### Three Code Files:

1. **`day13_capstone_template.py`** ← Copy-paste this into Jupyter notebook (`day13_capstone.ipynb`)
   - Part 1: 10 research documents
   - Part 2: TypedDict state design
   - Part 3: 8 node functions (tested)
   - Part 4: Graph assembly with MemorySaver
   - Part 5: 10 test questions (2 red-team)
   - Part 6: RAGAS baseline (optional)
   - Part 7: Summary markdown
   - Part 8: Deployment notes

2. **`capstone_streamlit.py`** ← Run this after completing notebook
   - Full Streamlit UI with caching
   - @st.cache_resource for expensive components
   - st.session_state for memory & thread_id
   - Sidebar with "New Conversation" button
   - Metadata display (route, faithfulness, sources)

3. **`research_paper_agent_complete.py`** ← Reference/backup
   - Same code but in standalone Python format
   - Useful for testing outside Jupyter

---

## 🚀 How to Use This Solution

### Step 1: Set Up Your Environment
```bash
# Install dependencies
pip install langchain langchain-groq langgraph chromadb sentence-transformers streamlit typing-extensions

# Set Groq API key
export GROQ_API_KEY="your-key-here"
```

### Step 2: Create Jupyter Notebook
```bash
# Create a blank notebook named day13_capstone.ipynb
jupyter notebook
```

### Step 3: Copy Code into Notebook
1. Open `day13_capstone_template.py`
2. Copy each PART section (1-8) into separate cells in your Jupyter notebook
3. Replace the "# TODO:" comments with the code blocks provided
4. **Run each part in order** — don't skip ahead
5. Test each section as you go

### Step 4: Test the Agent
```python
# In a notebook cell (Part 5), run:
result = ask_question("What is BERT?", thread_id="test_1")
print(result["answer"])
print(f"Faithfulness: {result['faithfulness']:.2f}")
```

### Step 5: Deploy with Streamlit
```bash
# Copy capstone_streamlit.py to your project folder
cp capstone_streamlit.py .

# Run the UI
streamlit run capstone_streamlit.py

# Open browser to http://localhost:8501
```

---

## ✅ The 6 Mandatory Capabilities (Rubric)

Your solution includes all 6:

| # | Capability | Implementation |
|---|-----------|-----------------|
| 1 | **LangGraph StateGraph (3+ nodes)** | 8 nodes: memory, router, retrieve, skip, tool, answer, eval, save |
| 2 | **ChromaDB RAG (10+ docs)** | 10 research documents (100-500 words each) in RESEARCH_DOCUMENTS |
| 3 | **MemorySaver + thread_id** | `checkpointer = MemorySaver()` in Part 4; `config={"configurable": {"thread_id": thread_id}}` in invoke calls |
| 4 | **Self-reflection eval node** | `eval_node()` with faithfulness scoring; retries if < 0.7 (MAX_EVAL_RETRIES=2) |
| 5 | **Tool use beyond retrieval** | `tool_node()` implements web search; router decides when to use |
| 6 | **Streamlit deployment** | `capstone_streamlit.py` with @st.cache_resource and st.session_state |

---

## 🧪 The 10 Test Questions

Included in Part 5:

**Standard Tests (8):**
1. What is self-attention in Transformers? → Route: RETRIEVE
2. How does BERT differ from GPT? → Route: RETRIEVE
3. What are the main components of Vision Transformers? → Route: RETRIEVE
4. Explain Retrieval-Augmented Generation → Route: RETRIEVE
5. What is LoRA and why is it useful? → Route: RETRIEVE
6. Find recent papers on prompt engineering from 2024 → Route: TOOL
7. Who authored the Vision Transformer paper and what else have they published? → Route: TOOL
8. Thanks for the explanation! → Route: SKIP

**Red-Team Tests (2):**
9. ❌ **Out-of-Scope:** "What is the best way to cook pasta?" → Agent must admit: Not in knowledge base
10. ❌ **False Premise:** "How do Transformers use recurrent connections like RNNs?" → Agent must correct: Transformers DON'T use recurrence

---

## 📊 Expected Results

**Routing Accuracy:** 90%+ of questions route correctly
**Faithfulness Score:** 0.75+ average
**Memory Test:** Ask 3 questions with same thread_id; 3rd question references context from 1st → passes

**Red-Team Handling:**
- Out-of-scope: Agent says "I don't have this in my knowledge base"
- False-premise: Agent clarifies the incorrect assumption without hallucinating

---

## 🔑 Key Design Decisions Explained

### Why 8 Nodes?
- **memory** → Appends to history, sliding window (prevents token overflow on free Groq tier)
- **router** → LLM decision with clear routing logic
- **retrieve/skip/tool** → Three separate routes for clean separation of concerns
- **answer** → Generates with system prompt grounding in context only
- **eval** → Faithfulness check with retry loop
- **save** → Persists answer to state for MemorySaver

### Why MemorySaver?
- Zero memory between API calls without it
- thread_id enables same user across multiple sessions
- Sliding window (msgs[-12:]) limits tokens sent to LLM

### Why Separate Router & Decision?
- `router_node()` = LLM call (expensive)
- `route_decision()` = Pure Python function (parse LLM output)
- This separation is required by `add_conditional_edges()` API

### Why Faithfulness Eval?
- Detects hallucination (answer using info NOT in context)
- Score < 0.7 triggers retry with fresh answer_node call
- MAX_EVAL_RETRIES=2 prevents infinite loops

---

## 🐛 Debugging Checklist

If tests fail:

| Error | Solution |
|-------|----------|
| `KeyError: 'question'` | Missing field in state? Check ResearchPaperState TypedDict |
| `Graph compile error` | Missing edge? Ensure every non-terminal node has outgoing edge |
| `Infinite loop in eval` | Check: `eval_retries >= MAX_EVAL_RETRIES` in eval_decision |
| `LLM response not parsing` | Router returns junk? Add `.upper()` and check for "RETRIEVE" substring |
| `Streamlit: llm not defined` | Move initialization inside @st.cache_resource |
| `UnicodeEncodeError on Windows` | Use `open(..., encoding='utf-8')` |

---

## 🎯 RAGAS Baseline Scores (Optional)

If you want to measure quality:

```python
from ragas.metrics import faithfulness, answer_relevancy, context_precision

# Create 5 ground-truth QA pairs from your KB
ground_truth = [
    {
        "question": "What is BERT?",
        "answer": "BERT is a bidirectional encoder...",
        "contexts": ["[BERT: Bidirectional Encoder Representations...]"]
    },
    # ... 4 more
]

# Run agent and collect results
results = evaluate(ground_truth)
print(f"Faithfulness: {results['faithfulness']:.2f}")  # Target: >= 0.7
print(f"Answer Relevancy: {results['answer_relevancy']:.2f}")
print(f"Context Precision: {results['context_precision']:.2f}")
```

---

## 📝 Written Summary (Part 8)

Fill in this section in your notebook before submission:

```markdown
# Research Paper Q&A Agent - Capstone Summary

## Domain & Problem
- Domain: Research Paper Q&A Assistant
- User: PhD students and researchers
- Problem: [Your 2-3 sentence explanation]
- Success: [Measurable outcome, e.g., "80%+ accurate answers, faithfulness >= 0.7"]

## Knowledge Base
- Size: 10 documents
- Coverage: [Topics covered]
- Retrieval Quality: [Test results]

## Tool Used
- Tool: Web Search
- Why: [Your explanation of why this tool is valuable]

## Test Results
- Total: 10/10 tests completed
- Passed: [X]/10
- Red-Team: Out-of-scope handled correctly, false-premise corrected

## RAGAS Scores
- Faithfulness: [your score]
- Answer Relevancy: [your score]
- Context Precision: [your score]

## One Thing I Would Improve with More Time
[Be specific and technical, not generic. E.g., "Implement semantic caching to reduce API calls by 40%" or "Fine-tune embedding model on domain-specific papers"]
```

---

## 🚨 Common Pitfalls (What NOT to Do)

❌ **Don't:**
1. Omit sliding window → exhausts token quota
2. Let tools raise exceptions → crashes graph
3. Forget MemorySaver → no memory between calls
4. Use `MemoryType.PERCENTAGE` for tables → Google Docs renders incorrectly
5. Write system prompt allowing "use general knowledge" → low faithfulness
6. Skip isolation testing of nodes → bugs only appear at graph runtime
7. Forget `encoding='utf-8'` on Windows → UnicodeEncodeError

✅ **Do:**
1. Test each node function in isolation BEFORE building graph
2. Verify retrieval works BEFORE adding answer_node
3. Always return all State fields from each node (or missing fields break the graph)
4. Print route, faithfulness, and sources in Streamlit for debugging
5. Use clear system prompts with explicit grounding rules
6. Cache expensive objects in Streamlit
7. Run `Kernel → Restart & Run All` before submission

---

## 📦 Files to Submit

Your instructor will expect:

1. **`day13_capstone.ipynb`** — Completed notebook with all 8 parts running without error
2. **`capstone_streamlit.py`** — Streamlit app (can launch and test 3-question conversation)
3. **`agent.py`** (optional) — Extracted graph code for production deployment

**Before submitting:**
```python
# In notebook cell:
# Kernel → Restart & Run All
# ✅ Every cell must execute without error
```

---

## 🎓 Rubric Alignment

This solution satisfies the rubric because:

| Rubric Requirement | Your Solution |
|---|---|
| TypedDict state designed FIRST | ✅ Part 2 defines ResearchPaperState before any nodes |
| 10+ documents | ✅ 10 research documents in Part 1 |
| Retrieval tested before graph | ✅ retrieval_node() tested in isolation in Part 3 |
| 8+ nodes including eval | ✅ 8 nodes + conditional edges |
| MemorySaver for memory | ✅ checkpointer = MemorySaver() in Part 4 |
| Faithfulness eval with retries | ✅ eval_node() scores < 0.7 triggers answer_node retry |
| Tool beyond KB retrieval | ✅ tool_node() implements web search |
| Router with clear logic | ✅ router_node() with explicit RETRIEVE/TOOL/SKIP paths |
| Streamlit with @st.cache_resource | ✅ capstone_streamlit.py includes caching |
| Multi-turn memory with thread_id | ✅ MemorySaver + thread_id demonstrated |
| 10 tests including red-team | ✅ 8 standard + 2 red-team tests in Part 5 |

---

## 📞 Quick Reference

**Node input/output template:**
```python
def my_node(state: ResearchPaperState) -> ResearchPaperState:
    """What this node does"""
    # Process state
    return {**state, "field_name": new_value}
```

**Invoke agent with memory:**
```python
result = app.invoke(input_state, config={"configurable": {"thread_id": thread_id}})
```

**Conditional edge syntax:**
```python
graph.add_conditional_edges("router", route_decision, {
    "retrieve": "retrieve",
    "tool": "tool",
    "skip": "skip"
})
```

---

## 🎉 You're Ready!

This is a **production-grade** capstone solution. It demonstrates:
- ✅ Advanced LangGraph patterns (multi-conditional edges, eval loops)
- ✅ RAG + tool integration in a single agent
- ✅ Memory persistence with MemorySaver
- ✅ Deployment-ready Streamlit UI with caching
- ✅ Proper error handling and fallbacks

**Good luck! Questions? Start with the debugging checklist above.**

---

*Generated for Agentic AI Capstone Course 2026*  
*Rubric: Dr. Kanthi Kiran Sirra, Sr. AI Engineer*
