# 🧠 Attention-RAG: Dense Semantic Search vs. Custom Multi-Head Attention in LangGraph

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://adiii89-embeddingmodel-vs-custom-attention-app-ygcewi.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green.svg)](https://github.com/langchain-ai/langgraph)

> 🚀 **Live Interactive Demo:** [https://adiii89-embeddingmodel-vs-custom-attention-app-ygcewi.streamlit.app/](https://adiii89-embeddingmodel-vs-custom-attention-app-ygcewi.streamlit.app/)

A scientific, end-to-end Retrieval-Augmented Generation (RAG) system and empirical research benchmark comparing **standard pre-trained Sentence Transformers (`all-MiniLM-L6-v2`)** against a **custom-trained PyTorch Multi-Head Self-Attention Encoder** orchestrated with **LangGraph**.

---

## 🌟 Overview & Key Highlights

- **Custom PyTorch Multi-Head Attention Encoder (`embeddings/attention_encoder.py`):** Built from scratch with token & positional embeddings, 4-head self-attention ($d_k = 32$), LayerNorm/residual connections, masked mean pooling, and L2 normalization ($128$-dim unit vectors).
- **Contrastive Triplet Training:** Trained on Anchor, Positive, and Hard Negative triplets with Triplet Margin Loss ($\alpha = 0.4$) and AdamW optimizer.
- **Dynamic LangGraph Pipeline (`graph/build_graph.py`):** Orchestrates multi-mode retrieval with conditional routing and parallel fan-out / fan-in comparison topology.
- **Side-by-Side Live Benchmarking:** Quantitative evaluation measuring Recall@K ($K=1,2,3$), Mean Reciprocal Rank (MRR), search latency, vector memory footprint, and Jaccard chunk overlap.
- **Interactive Streamlit Web Dashboard (`app.py`):** Multi-mode RAG query interface with dynamic PDF ingestion, side-by-side chunk candidate inspection, and live evaluation.

---

## 🏗️ Architecture & Pipeline Graph

When running in **Compare Mode**, LangGraph executes a **Parallel Fan-Out / Fan-In** pattern:

```mermaid
graph TD
    Start([🟢 START]) --> QNode[query_node<br><i>Validates query & checks retrieval_mode</i>]
    
    QNode -->|mode == 'semantic'| SNode[semantic_retrieval_node<br><i>Sentence Transformer (all-MiniLM-L6-v2)</i>]
    QNode -->|mode == 'attention'| ANode[attention_retrieval_node<br><i>Custom PyTorch Attention (128-dim)</i>]
    QNode -->|mode == 'compare' (Parallel Fan-Out)| SNode
    QNode -->|mode == 'compare' (Parallel Fan-Out)| ANode
    
    SNode -->|mode == 'semantic'| CNode[context_node<br><i>Formats selected context</i>]
    ANode -->|mode == 'attention'| CNode
    
    SNode -->|mode == 'compare' (Fan-In)| CompNode[comparison_node<br><i>Calculates Jaccard overlap & merges chunks</i>]
    ANode -->|mode == 'compare' (Fan-In)| CompNode
    CompNode --> CNode
    
    CNode --> GNode[generation_node<br><i>Grounded generation with citations</i>]
    GNode --> EndNode([🔴 END])

    style QNode fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
    style SNode fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    style ANode fill:#fff8e1,stroke:#fbc02d,stroke-width:2px
    style CompNode fill:#ede7f6,stroke:#512da8,stroke-width:2px
    style CNode fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style GNode fill:#fce4ec,stroke:#c2185b,stroke-width:2px
```

---

## 📊 Empirical Benchmark Results

Measured across test queries on indexed research documents (`data/benchmark_results.json`):

| Evaluation Metric | Semantic Retriever (`all-MiniLM-L6-v2`) | Custom Attention Retriever (`attention_encoder.pt`) | Difference / Advantage |
| :--- | :---: | :---: | :---: |
| **Recall @ 1** | **`50.00%`** (3 / 6 queries) | **`33.33%`** (2 / 6 queries) | Semantic $+16.67\%$ |
| **Recall @ 2** | **`66.67%`** (4 / 6 queries) | **`50.00%`** (3 / 6 queries) | Semantic $+16.67\%$ |
| **Recall @ 3** | **`66.67%`** (4 / 6 queries) | **`66.67%`** (4 / 6 queries) | **`0.00%` (EXACT TIE)** |
| **Mean Reciprocal Rank (MRR)** | **`0.5833`** | **`0.4722`** | Semantic $+0.1111$ |
| **Mean Search Latency** | **`29.26 ms`** | **`2.41 ms`** | Attention is **`26.85 ms` faster ($12.1\times$)** |
| **P95 Latency** | **`32.44 ms`** | **`3.11 ms`** | Attention is **`29.33 ms` faster ($10.4\times$)** |
| **Vector Dimensions** | **`384` floats** | **`128` floats** | Attention saves **$66.7\%$ RAM ($3\times$ smaller)** |
| **Model Parameters** | **`22,713,216`** | **`4,006,272`** | Attention is **$5.67\times$ smaller** |
| **Candidate Overlap (Jaccard)** | **`40.00%`** | **`40.00%`** | Models agree on $40\%$, differ on $60\%$ |

---

## 🔍 Key Findings

1. **Top-3 Recall Tie (`66.67%` vs `66.67%`):** Both models successfully retrieved the correct answer chunk within their Top-3 candidates for 4 out of 6 test queries.
2. **12x Faster Latency:** Custom Attention runs vector inner-product search in **`2.41 ms`** vs **`29.26 ms`** for MiniLM due to a lightweight single attention layer and 128-dim embeddings.
3. **RAM & Storage Savings:** 128-dimensional vectors reduce FAISS index RAM consumption by **$66.7\%$**.
4. **Complementary Retrieval (Hybrid Power):** A Jaccard overlap of **$40\%$** indicates that running both models in parallel in **Compare Mode** eliminates single-model blind spots by merging general pre-trained semantics with specialized attention weights.

---

## 📂 Project Structure

```text
Attention_RAG/
├── app.py                         # Streamlit interactive web dashboard
├── evaluate.py                    # Quantitative evaluation benchmark suite (Recall@K, MRR, Latency)
├── loader.py                      # PDF text extraction & boundary-aware chunking
├── rag.py                         # Grounded prompt engineering & multi-provider LLM generation
├── requirements.txt               # Project dependencies
├── ROADMAP.md                     # Stage-by-stage development tracker
│
├── data/
│   ├── benchmark_results.json     # Raw empirical benchmark output
│   ├── train_pairs.json           # Contrastive triplet training dataset (A, P, N)
│   └── raw/
│       └── sample_rag_paper.pdf   # Benchmark research document
│
├── embeddings/
│   ├── attention_encoder.py       # Custom PyTorch Multi-Head Self-Attention model
│   ├── attention_encoder.pt       # Trained attention model weights checkpoint
│   ├── attention_retriever.py     # FAISS IndexFlatIP retriever using AttentionEncoder
│   └── semantic_retriever.py      # FAISS IndexFlatIP retriever using all-MiniLM-L6-v2
│
└── graph/
    ├── build_graph.py             # LangGraph state graph assembly & execution
    ├── edges.py                   # Dynamic conditional routing & parallel fan-out
    ├── nodes.py                   # Node functions (query, retrieval, comparison, context, generation)
    └── state.py                   # TypedDict RAGState schema
```

---

## 🚀 Quickstart Guide

### 1. Clone & Setup Virtual Environment
```bash
git clone https://github.com/Adiii89/EmbeddingModel-VS-Custom-Attention.git
cd EmbeddingModel-VS-Custom-Attention

python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. (Optional) Set LLM API Keys in `.env`
Create a `.env` file at the root to enable full LLM answer generation:
```env
GEMINI_API_KEY=your_gemini_api_key
# Or:
GROQ_API_KEY=your_groq_api_key
# Or:
OPENAI_API_KEY=your_openai_api_key
```
*(If no API keys are provided, the system automatically uses an intelligent grounded extractive fallback).*

### 3. Run Quantitative Evaluation Benchmark
```bash
python evaluate.py
```

### 4. Launch the Streamlit Web UI Locally
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser or visit the **[Live Demo](https://adiii89-embeddingmodel-vs-custom-attention-app-ygcewi.streamlit.app/)**.

---

## 🛠️ Tech Stack

- **Deep Learning:** PyTorch, HuggingFace Transformers, Sentence-Transformers
- **Vector Search:** FAISS (`IndexFlatIP` on L2-normalized vectors)
- **Orchestration:** LangGraph, LangChain Core
- **Document Ingestion:** PyMuPDF (`fitz`)
- **LLM APIs:** Google GenAI (`gemini-2.0-flash`), Groq (`llama-3.3-70b`), OpenAI (`gpt-4o-mini`)
- **Frontend & Visualization:** Streamlit, NumPy
