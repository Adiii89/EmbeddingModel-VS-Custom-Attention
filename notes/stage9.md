# Stage 9 Notes: Interactive Streamlit Web UI (`app.py`)

## 🖥️ 1. What did we build in Stage 9?

In Stage 9, we built a web application and research dashboard in [`app.py`](file:///d:/Projects/RAG/Attention_RAG/app.py):

### 🚀 Tab 1: Interactive RAG Query
1. **Dynamic PDF Ingestion:** Upload any custom PDF or test immediately with the built-in research paper.
2. **Interactive Pipeline Mode Switching:**
   - 🟣 **Compare Both (Parallel Live Benchmark)**
   - 🟢 **Semantic Search (MiniLM-L6)**
   - 🟡 **Custom Attention (PyTorch Model)**
3. **Live Metrics HUD:** Real-time pipeline latency (ms), chunk count, and Jaccard overlap (%).
4. **Side-by-Side Candidate Cards:** View what each model retrieved, with exact similarity scores and origin badges (`[BOTH]`, `[SEMANTIC_ONLY]`, `[ATTENTION_ONLY]`).

---

### 📊 Tab 2: Empirical Benchmark & Scientific Analysis
1. **Exact Measured Benchmark Table:**
   - Recall@1, Recall@2, Recall@3 (with query counts)
   - Mean Reciprocal Rank (MRR)
   - Mean & P95 Latency (ms)
   - Vector Dimensions (384 vs 128) & RAM Savings ($66.7\%$)
   - Parameter Counts (22.7M vs 4.0M)
   - Candidate Jaccard Overlap ($40.00\%$)
2. **The 4 Grounded Scientific Takeaways:**
   - **Top-3 Retrieval Accuracy is Identical:** Both models included the answer chunk in Top-3 for 4/6 queries ($66.67\%$).
   - **MiniLM Pre-training Advantage on Top-1:** Due to $1\text{B}+$ sentence pre-training vs. initial training from scratch.
   - **Custom Model is 12x Faster on CPU:** 1 attention layer & 128 dimensions vs 6 layers & 384 dimensions.
   - **Complementary Agreement ($40\%$ Overlap):** Parallel Compare mode eliminates single-model blind spots.
3. **Dynamic Live Evaluator for ANY Uploaded PDF:**
   - When a user uploads a new PDF, the app dynamically extracts probe queries and computes real-time Recall@K, MRR, Latency, and Jaccard Overlap tailored specifically to that document!

---

## ⚡ 2. Performance & Caching Strategy

```python
@st.cache_resource(show_spinner="Initializing Retriever Models...")
def load_retriever_models():
    sem_retriever = SemanticRetriever()
    attn_retriever = AttentionRetriever()
    return sem_retriever, attn_retriever
```
- `@st.cache_resource` ensures models are loaded into memory once and reused across queries without reloading delay.

---

## 🚀 How to Run the Web App:
```bash
streamlit run app.py
```
👉 Accessible at: `http://localhost:8501`

---

## 🎤 3. Interview Questions & Simple Answers

### Q: How does the dynamic evaluator benchmark a newly uploaded PDF with no existing ground-truth labels?
**Simple Answer:**
- The dynamic engine extracts high-entropy probe queries from each chunk (e.g. leading topic sentences) and tracks whether each retriever successfully ranks the originating chunk within Top-K candidates.
- This provides an immediate relative ranking quality and latency score across both models for any arbitrary document in seconds.
