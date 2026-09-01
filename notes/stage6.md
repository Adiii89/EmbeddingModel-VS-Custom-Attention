# Stage 6 Notes: "Compare Both" Branch (Parallel Execution + Merge)

## 🌟 1. Complete & Final Pipeline Graph Topology

In Stage 6, our LangGraph pipeline achieves its **complete, final architecture**.
When `retrieval_mode == "compare"`, the graph executes a **Parallel Fan-Out / Fan-In pattern**:

```mermaid
graph TD
    Start([🟢 START]) --> QNode[query_node<br><i>Validates query & checks retrieval_mode</i>]
    
    QNode -->|mode == 'semantic'| SNode[semantic_retrieval_node<br><i>Sentence Transformer</i>]
    QNode -->|mode == 'attention'| ANode[attention_retrieval_node<br><i>Custom PyTorch Attention</i>]
    QNode -->|mode == 'compare' (Parallel Fan-Out)| SNode
    QNode -->|mode == 'compare' (Parallel Fan-Out)| ANode
    
    SNode -->|mode == 'semantic'| CNode[context_node<br><i>Formats selected context</i>]
    ANode -->|mode == 'attention'| CNode
    
    SNode -->|mode == 'compare' (Fan-In)| CompNode[comparison_node<br><i>Aligns results & computes overlap</i>]
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

## 🔍 2. How `comparison_node` Works

When both retrievers finish in parallel, `comparison_node`:

### 1. Computes Jaccard Chunk Overlap:
$$\text{Overlap Score} = \frac{|\text{Semantic Chunks} \cap \text{Attention Chunks}|}{|\text{Semantic Chunks} \cup \text{Attention Chunks}|}$$

- **High Overlap (e.g. 70%–100%):** Both models agree on what is relevant.
- **Low Overlap (e.g. 0%–30%):** The models retrieved different sections, providing complementary perspectives or highlighting a blind spot in one of the encoders.

### 2. Adds Origin Tags:
- **`[BOTH]`**: Chunk was retrieved by both models (highest confidence, prioritized at the top of the prompt).
- **`[SEMANTIC_ONLY]`**: Chunk was retrieved only by Sentence Transformer.
- **`[ATTENTION_ONLY]`**: Chunk was retrieved only by custom Attention Encoder.

### 3. Deduplication:
Ensures the LLM prompt receives unique, clean paragraphs without redundant copies.

---

## 🎤 3. Interview Questions & Simple Answers

### Q: How do you run two nodes in parallel in LangGraph and merge their outputs into shared state?
**Simple Answer:**
- **Parallel Fan-Out:** In `workflow.add_conditional_edges()`, our routing function returns a list of node names: `["semantic_retrieval_node", "attention_retrieval_node"]`. LangGraph invokes both node functions concurrently.
- **Independent State Keys:** `semantic_retrieval_node` only writes to `semantic_results`, and `attention_retrieval_node` only writes to `attention_results`.
- **Fan-In Merge:** Both nodes route into `comparison_node`, which reads both keys from the state clipboard and merges them safely.

### Cross-Q: What happens if two parallel nodes try to write to the exact same state key at the same time?
**Simple Answer:**
- If two parallel nodes write to the exact same key without a reducer, the last one to finish overwrites the first one (Race Condition).
- To prevent this:
  1. We assign **distinct, independent state keys** (`semantic_results` vs `attention_results`).
  2. Or use a LangGraph **reducer function** (e.g. `Annotated[list, operator.add]`) to append results safely.

### Cross-Q: Why hard-code both branches to run in Compare mode instead of letting an AI agent choose whether to run one?
**Simple Answer:**
- For scientific and benchmarking integrity, "Compare Both" must be **deterministic**.
- If an agent or LLM were allowed to decide, it might skip one retriever based on temperature randomness, ruining the side-by-side benchmark comparison.
