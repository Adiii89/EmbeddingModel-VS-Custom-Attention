# Stage 8 Notes: Quantitative Evaluation & Benchmarking (`evaluate.py`)

## 📊 1. Key Evaluation Metrics Explained

In Stage 8, we turn our pipeline into an empirical AI research benchmark using 4 standard Information Retrieval (IR) metrics:

### 1. Recall@K (e.g. $K=1, 2, 3$)
- **What it is:** The percentage of queries where the true answering paragraph was found in the top-$K$ returned chunks.
- **Formula:**
  $$\text{Recall@K} = \frac{\text{Number of queries with correct chunk in top-}K}{\text{Total Queries}}$$
- **Intuition:** If you ask 10 questions and the correct answer appears in the top-3 results for 9 of them, **$\text{Recall@3} = 90\%$**.

---

### 2. Mean Reciprocal Rank (MRR)
- **What it is:** Measures how close to the #1 top spot the correct chunk was ranked.
- **Formula:**
  $$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{Rank}_i}$$
- **Scoring Rules:**
  - Ranked **1st**: Score = $\frac{1}{1} = \mathbf{1.00}$ (Perfect!)
  - Ranked **2nd**: Score = $\frac{1}{2} = \mathbf{0.50}$
  - Ranked **3rd**: Score = $\frac{1}{3} = \mathbf{0.33}$
  - Not in top candidates: Score = $\mathbf{0.00}$

---

### 3. Latency (Mean & P95 in ms)
- **What it is:** Execution time in milliseconds for tokenizing, embedding, and searching the FAISS index.
- **P95 Latency:** The 95th percentile latency (95% of queries execute faster than this cutoff), measuring real-world stability under load.

---

### 4. Jaccard Chunk Overlap Ratio
- **What it is:** The agreement percentage between the two different retrieval models.
- **Formula:**
  $$\text{Overlap} = \frac{|\text{Semantic Chunks} \cap \text{Attention Chunks}|}{|\text{Semantic Chunks} \cup \text{Attention Chunks}|}$$

---

## ⚖️ 2. Retriever Trade-off Analysis

| Dimension | Semantic Retriever (MiniLM-L6) | Custom Attention Retriever (Our Model) |
| :--- | :--- | :--- |
| **Model Size** | 22.7M parameters | **4.0M parameters (5.5x smaller!)** |
| **Vector Size** | 384 dimensions | **128 dimensions (3x smaller storage!)** |
| **CPU Latency** | ~40–50 ms | **~7–10 ms (5x–6x faster!)** |
| **Generalization** | Broad open-domain vocabulary | Sharp, domain-specific contrastive focus |

---

## 🎤 3. Interview Questions & Simple Answers

### Q: Why is MRR better than simple Accuracy for evaluating a retriever?
**Simple Answer:**
- Accuracy only gives a binary Yes/No: did the retriever find the document or not?
- MRR cares about **ranking quality**. Finding the document at Rank 1 (score 1.0) is much better for the LLM prompt than finding it buried at Rank 10 (score 0.1).

### Cross-Q: Why is Recall@K more critical for RAG than Precision@K?
**Simple Answer:**
- In RAG, the LLM reads all top-$K$ chunks in its prompt context. As long as the true answer is present anywhere in those $K$ chunks (High Recall), the LLM can synthesize a correct answer.
- If the true answer is missing (Low Recall), the LLM is guaranteed to hallucinate, regardless of precision.

### Cross-Q: How does vector dimension affect memory and FAISS search speed in production?
**Simple Answer:**
- Memory usage is directly proportional to dimension: $N \times D \times 4 \text{ bytes}$.
- A 128-dim vector takes 512 bytes per chunk vs. 1,536 bytes for 384-dim (3x memory savings).
- In vector distance computation (Inner Product), each query requires $D$ multiplications and additions per candidate vector. Reducing $D$ from 384 to 128 cuts CPU floating-point operations (FLOPs) by **66%**.
