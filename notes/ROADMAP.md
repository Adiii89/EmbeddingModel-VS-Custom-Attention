# Project Roadmap & Progress Tracker
## Attention vs Semantic Retrieval RAG Lab (Student Edition, Staged Build)

> **Research Question:**
> *Can a lightweight, self-trained attention-based encoder produce useful semantic representations for retrieval, and how does it compare against a pre-trained Sentence Transformer in terms of retrieval quality, latency, and downstream RAG answer quality?*

---

### 📊 Overall Progress Dashboard

| Stage | Name | Status | Key Deliverables | Notes / Interview Qs |
| :--- | :--- | :---: | :--- | :---: |
| **Stage 0** | **Setup & Project Skeleton** | ✅ Completed | `requirements.txt`, stub structure, `notes/stage0.md` | `notes/stage0.md` |
| **Stage 1** | **Document Loading + Chunking** | ✅ Completed | `loader.py`, `notes/stage1.md` | `notes/stage1.md` |
| **Stage 2** | **LangGraph Skeleton + Semantic Baseline** | ✅ Completed | `graph/state.py`, `nodes.py`, `build_graph.py`, `embeddings/semantic_retriever.py` | `notes/stage2.md` |
| **Stage 3** | **Custom Attention Encoder (Untrained)** | ✅ Completed | `embeddings/attention_encoder.py`, `notes/stage3.md` | `notes/stage3.md` |
| **Stage 4** | **Training the Attention Encoder** | ✅ Completed | `embeddings/train_attention.py`, `data/train_pairs.json`, `attention_encoder.pt` | `notes/stage4.md` |
| **Stage 5** | **Attention Retriever + Conditional Routing** | ✅ Completed | `embeddings/attention_retriever.py`, `graph/edges.py`, `graph/build_graph.py` | `notes/stage5.md` |
| **Stage 6** | **"Compare Both" Branch (Parallel + Merge)** | ✅ Completed | `graph/nodes.py` (comparison node), `build_graph.py` | `notes/stage6.md` |
| **Stage 7** | **Real RAG Generation Node** | ✅ Completed | `rag.py`, LLM generation in `graph/nodes.py` | `notes/stage7.md` |
| **Stage 8** | **Evaluation & Benchmarking** | ✅ Completed | `evaluate.py` (Recall@K, MRR, latency, overlap) | `notes/stage8.md` |
| **Stage 9** | **Streamlit UI Frontend** | ✅ Completed | `app.py` | `notes/stage9.md` |
| **Stage 10** | **Agent + MCP Server (Bonus)** | ⏳ In Progress | `agent/tool.py`, `agent/mcp_server.py` | `notes/stage10.md` |

---

### 📁 Target Project Structure

```text
attention-rag/
├── ROADMAP.md                    # Project tracker and status (this file)
├── app.py                        # Streamlit UI (single entry point)
├── loader.py                     # PDF loading + chunking
│
├── embeddings/                   # Both embedding methods live together
│   ├── semantic_retriever.py     # Sentence-Transformer + vector search
│   ├── attention_encoder.py      # Custom PyTorch attention encoder
│   ├── train_attention.py        # Contrastive training script
│   └── attention_retriever.py    # Trained attention encoder + vector search
│
├── graph/                        # LangGraph pipeline, modular by concern
│   ├── state.py                  # RAGState TypedDict
│   ├── nodes.py                  # query/retrieval/context/generation nodes
│   ├── edges.py                  # conditional routing logic
│   └── build_graph.py            # wires nodes+edges, compiles the graph
│
├── rag.py                        # Prompt building + LLM call
├── evaluate.py                   # Recall@K, MRR, latency, overlap
├── data/
│   ├── raw/                      # uploaded PDFs
│   └── train_pairs.json          # query/positive/negative triples
├── notes/                        # One markdown file per stage with notes & interview Qs
│   ├── stage0.md
│   ├── stage1.md
│   ├── stage2.md
│   ├── stage3.md
│   ├── stage4.md
│   ├── stage5.md
│   ├── stage6.md
│   ├── stage7.md
│   ├── stage8.md
│   ├── stage9.md
│   └── stage10.md
├── requirements.txt
└── README.md
```

---

### 📋 Stage Checklist & Details

#### [ ] Stage 0: Setup & Project Skeleton
- [ ] Review environment and Python version
- [ ] Populate `requirements.txt` with dependencies (`torch`, `sentence-transformers`, `faiss-cpu`, `langgraph`, `langchain-core`, `pymupdf`, `streamlit`, `numpy`, etc.)
- [ ] Create stub folder hierarchy (`embeddings/`, `graph/`, `data/raw/`, `notes/`)
- [ ] Create `notes/stage0.md` with tech stack justification & interview Q&As

#### [ ] Stage 1: Document Loading & Chunking
- [ ] Implement `loader.py` (PyMuPDF text extraction + sliding window/token-aware chunker)
- [ ] Test on a sample PDF and verify clean vs bad splits
- [ ] Create `notes/stage1.md` with chunking decisions, boundary edge cases, and interview Q&As

#### [ ] Stage 2: LangGraph Skeleton + Baseline Semantic Retrieval
- [ ] Implement `graph/state.py` (`RAGState` TypedDict)
- [ ] Implement `embeddings/semantic_retriever.py` (`all-MiniLM-L6-v2` + FAISS `IndexFlatIP`)
- [ ] Implement `graph/nodes.py` (`query_node`, `semantic_retrieval_node`, `context_node`, stub `generation_node`)
- [ ] Implement `graph/build_graph.py` (linear graph: `START -> query_node -> semantic_retrieval_node -> context_node -> generation_node -> END`)
- [ ] Test end-to-end execution of the linear graph
- [ ] Create `notes/stage2.md` with graph diagram, state design rationale, and interview Q&As

#### [ ] Stage 3: Custom Attention Encoder (Untrained)
- [ ] Implement `embeddings/attention_encoder.py` (Tokenizer -> Embedding layer -> `nn.MultiheadAttention` -> Mean pooling -> Linear projection -> L2 normalization)
- [ ] Verify tensor dimensions at each step
- [ ] Create `notes/stage3.md` with tensor shape breakdown, multi-head attention math, and interview Q&As

#### [ ] Stage 4: Training the Attention Encoder
- [ ] Create `data/train_pairs.json` (triplets: anchor query, positive chunk, hard negative chunk)
- [ ] Implement `embeddings/train_attention.py` (Triplet Margin Loss / InfoNCE training loop)
- [ ] Train the encoder and save checkpoint (`embeddings/attention_encoder.pt`)
- [ ] Verify ranking change before vs after training
- [ ] Create `notes/stage4.md` with training dynamics, loss curve notes, and interview Q&As

#### [ ] Stage 5: Attention Retriever Node & Conditional Routing
- [ ] Implement `embeddings/attention_retriever.py` (FAISS indexing with trained attention encoder)
- [ ] Add `attention_retrieval_node` to `graph/nodes.py`
- [ ] Add conditional routing logic in `graph/edges.py` (`retrieval_mode == "semantic"` vs `"attention"`)
- [ ] Update `graph/build_graph.py` to support dynamic routing
- [ ] Test single-mode runs and compare latencies
- [ ] Create `notes/stage5.md` with routing mechanics and interview Q&As

#### [ ] Stage 6: "Compare Both" Branch (Parallel Execution & Merge)
- [ ] Update `graph/edges.py` and `graph/build_graph.py` for parallel branch execution when `retrieval_mode == "compare"`
- [ ] Implement `comparison_node` in `graph/nodes.py` to merge/align results from both retrievers
- [ ] Test compare mode end-to-end
- [ ] Create `notes/stage6.md` with fan-out/fan-in state merge patterns and interview Q&As

#### [ ] Stage 7: Real RAG Generation Node
- [ ] Implement `rag.py` (unified prompt template, LLM integration)
- [ ] Replace stub in `generation_node` with real LLM invocation
- [ ] Test generation with both semantic context and attention context
- [ ] Create `notes/stage7.md` with prompt templates, hallucination mitigation, and interview Q&As

#### [ ] Stage 8: Evaluation & Benchmarking
- [ ] Implement `evaluate.py` (Recall@1, Recall@5, MRR, latency benchmarks, chunk overlap Jaccard)
- [ ] Run benchmark on evaluation test set
- [ ] Generate comparative tables and statistical insights
- [ ] Create `notes/stage8.md` with metrics analysis, findings, and interview Q&As

#### [ ] Stage 9: Streamlit UI Frontend
- [ ] Implement `app.py` (PDF upload, retrieval mode selection, LangGraph execution, side-by-side comparison visualization)
- [ ] Test complete user flow in browser
- [ ] Create `notes/stage9.md` with UI architecture tradeoffs and interview Q&As

#### [ ] Stage 10: Agent + MCP (Bonus)
- [ ] Build agent wrapper with retrieval tools
- [ ] Create optional MCP server definition
- [ ] Create `notes/stage10.md` with tool safety, agent constraints, and interview Q&As
