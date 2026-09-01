# Stage 1 Notes: Document Loading & Chunking

## 1. What did we do in Stage 1?
We built [`loader.py`](file:///d:/Projects/RAG/Attention_RAG/loader.py), which handles the very first step of any RAG pipeline:
1. **Reading the PDF:** It uses `pymupdf` to read the PDF page-by-page and grab the clean text.
2. **Splitting into Chunks:** It cuts long pages of text into smaller, overlapping chunks (around 300–500 characters each).
3. **Saving Metadata:** Every chunk is tagged with its `chunk_id`, `page_number`, and source filename so we always know where each piece of information came from.

---

## 2. Why do we need Overlap?

Imagine reading a sentence that gets cut right in the middle:
- **Without Overlap (Bad Split):**
  - Chunk 1: *"The patient was prescribed aspirin because of severe"*
  - Chunk 2: *"chest pain, but has an allergy to ibuprofen."*
  - *Problem:* Neither chunk has the complete medical picture!
- **With Overlap (Clean Split):**
  - Chunk 1: *"The patient was prescribed aspirin because of severe chest pain."*
  - Chunk 2: *"because of severe chest pain, but has an allergy to ibuprofen."*
  - *Result:* Both chunks retain enough context for the search model to understand the full meaning.

---

## 3. PDF Ingestion: PyMuPDF vs. IBM Docling vs. Others

| Tool | Speed | Table & Layout Handling | Scanned & OCR Support | Best Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **PyMuPDF (`fitz`)** *(Chosen)* | ⚡ **Ultra-Fast** (~5ms/page) | Basic raw text lines. | Needs external OCR tool. | Fast prototyping, clean digital PDFs, lightweight labs. |
| **IBM Docling** | 🐢 **Slower** (~1-4s/page, runs AI vision models) | 🌟 **Top-tier** (converts complex tables into Markdown tables). | Built-in AI vision & OCR for scanned pages. | Enterprise production RAG with complex financial tables and scanned forms. |
| **PyPDF / pypdf** | 🐢 Moderate (pure Python) | Often garbles multi-column text. | No built-in OCR. | Legacy projects or basic text dumps. |

---

## 4. Chunking Strategies & Trade-offs

| Strategy | Pros | Cons | Why we picked it |
| :--- | :--- | :--- | :--- |
| **Fixed Character Split** | Super simple to code. | Cuts words in half (e.g. `retri-` / `eval`). | ❌ Avoided |
| **Sentence-Based Split** | Clean sentences. | Sentence lengths vary wildly; some sentences are 5 words, others are 100 words. | ❌ Hard to batch |
| **Boundary-Aware Window + Overlap** | Cuts at natural breaks (paragraphs $\to$ newlines $\to$ periods $\to$ spaces) and keeps words intact. | Slightly more logic. | ✅ **Chosen** |

---

## 5. Interview Questions & Simple Answers

### Q1: Why chunk documents at all instead of embedding the whole PDF?
**Simple Answer:**
1. **Model Limits:** Embedding models have a maximum token limit (e.g., 256 or 512 tokens). A 20-page PDF simply won't fit into one vector.
2. **Diluted Meaning:** If you compress an entire 20-page document into a single vector, specific details get lost in the average.
3. **Precision:** Chunking lets us find the exact paragraph that answers the user's question, rather than dumping the whole book into the LLM prompt.

### Cross-Q: What happens if chunks are too small vs. too large?
**Simple Answer:**
- **Too small (e.g., 20 words):** The chunk loses its surrounding context. A search might find the chunk, but the LLM won't understand what it refers to.
- **Too large (e.g., 2,000 words):** The embedding gets diluted with unrelated topics, and retrieval accuracy drops. It also wastes expensive LLM context window space.

### Cross-Q: How does our PyMuPDF approach compare to modern tools like IBM Docling?
**Simple Answer:**
- **PyMuPDF:** Super lightweight and 100x faster for standard digital text documents where speed and low compute matter.
- **IBM Docling:** Uses vision-language models (like TableFormer) to parse complex multi-column layouts, convert tables into Markdown tables, and automatically run OCR on scanned PDFs.
- **In our architecture:** `loader.py` is an isolated module. In a production environment with scanned financial statements, we could swap PyMuPDF with Docling inside `loader.py` without modifying the LangGraph pipeline or retrieval models.

### Cross-Q: How do you handle tables or code blocks split across chunk boundaries?
**Simple Answer:**
- **In our loader:** We search for paragraph breaks (`\n\n`) first before breaking at spaces, which keeps small tables and code snippets together.
- **In production:** For complex tables, tools like Docling extract tables as Markdown structures, or we can summarize the table into natural text before indexing.
