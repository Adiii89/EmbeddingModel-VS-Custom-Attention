# Stage 5 Notes: Attention Retriever Node & Conditional Routing

## 🌟 1. Updated Pipeline Graph Architecture

In Stage 5, our LangGraph pipeline branches for the first time!
Using **conditional edges**, the graph inspects `state["retrieval_mode"]` and routes execution dynamically:

```mermaid
graph TD
    Start([🟢 START]) --> QNode[query_node<br><i>Validates query & checks retrieval_mode</i>]
    
    QNode -->|mode == 'semantic'| SNode[semantic_retrieval_node<br><i>Sentence Transformer (all-MiniLM-L6-v2)</i>]
    QNode -->|mode == 'attention'| ANode[attention_retrieval_node<br><i>Custom PyTorch Attention (128-dim)</i>]
    
    SNode --> CNode[context_node<br><i>Formats selected context</i>]
    ANode --> CNode
    
    CNode --> GNode[generation_node<br><i>Grounded generation with citations</i>]
    GNode --> EndNode([🔴 END])

    style QNode fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
    style SNode fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    style ANode fill:#fff8e1,stroke:#fbc02d,stroke-width:2px
    style CNode fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style GNode fill:#fce4ec,stroke:#c2185b,stroke-width:2px
```

---

## ⚖️ 2. The Golden Rule of Scientific Benchmarking

To ensure an apples-to-apples comparison between the two retrieval methods:
- **Identical Ingestion:** Both use the exact same PDF text chunks from `loader.py`.
- **Identical Indexing:** Both use FAISS `IndexFlatIP` on L2-normalized vectors (Cosine Similarity).
- **Identical Search Parameters:** Both use `top_k = 2`.
- **Identical Downstream Nodes:** Both feed into the exact same `context_node` and `generation_node`.
- **The Only Variable:** The embedding representation model (`all-MiniLM-L6-v2` vs. `AttentionEncoder`).

---

## ⚡ 3. Retriever Comparison Table

| Metric / Aspect | Semantic Retriever (`all-MiniLM-L6-v2`) | Custom Attention Retriever (`attention_encoder.pt`) |
| :--- | :--- | :--- |
| **Model Origin** | Pre-trained by HuggingFace / Microsoft | Built from scratch & trained on domain triplets |
| **Vector Dimension** | **384** | **128** (3x smaller storage!) |
| **Model Parameters** | **22.7 Million** | **4.0 Million** (5.5x smaller!) |
| **Depth** | 6 Transformer Blocks | 1 Multi-Head Attention Layer |
| **Retrieval Accuracy** | Broad general semantic understanding | Sharp, domain-focused mathematical matching |

---

## 🎤 4. Interview Questions & Simple Answers

### Q: How does LangGraph decide which node runs next in a conditional edge?
**Simple Answer:**
- We pass a routing function (`route_retrieval` in `graph/edges.py`) to `workflow.add_conditional_edges()`.
- When `query_node` finishes, LangGraph passes the updated state to `route_retrieval(state)`.
- The function inspects `state["retrieval_mode"]` and returns the name of the destination node (`"semantic_retrieval_node"` or `"attention_retrieval_node"`).

### Cross-Q: What happens if `retrieval_mode` contains an invalid or unexpected string?
**Simple Answer:**
- In `graph/edges.py`, we implement strict validation:
  ```python
  if mode not in ["semantic", "attention", "compare"]:
      raise ValueError(f"Invalid retrieval_mode '{mode}'...")
  ```
- This ensures the graph **fails fast and loudly** with an informative error message instead of silently routing to an arbitrary node or freezing the pipeline.

### Cross-Q: Why is the custom attention encoder faster and more lightweight during inference?
**Simple Answer:**
- **Fewer Layers:** MiniLM runs 6 stacked transformer layers with heavy feed-forward blocks ($1536$ dim). Our custom encoder runs only 1 attention layer.
- **Smaller Dimension:** Vectors are 128-dim instead of 384-dim, requiring fewer floating-point operations (FLOPs) during dot-product search in FAISS.
