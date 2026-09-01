# Stage 7 Notes: Real RAG Generation Node (`rag.py`)

## 🌟 1. What did we build in Stage 7?

In Stage 7, we connected our LangGraph pipeline to **real LLM generation** via [`rag.py`](file:///d:/Projects/RAG/Attention_RAG/rag.py):
1. **Strict Grounding System Prompt (`RAG_SYSTEM_PROMPT`):** Enforces that the model only answers using the retrieved context passages and explicitly cites source files and page numbers (`[Source: document.pdf, Page: X]`).
2. **Multi-Provider LLM Support:** Seamlessly connects to:
   - **Google Gemini** (`gemini-2.0-flash`) via `google-genai`
   - **Groq** (`llama-3.3-70b-versatile`) via `groq`
   - **OpenAI** (`gpt-4o-mini`) via `openai`
   - **Grounded Extractive Fallback:** If no API keys are provided in `.env`, extracts the relevant text and citations cleanly without crashing.
3. **Deterministic Temperature (`temperature = 0.0`):** Minimizes randomness and hallucination risk, ensuring verifiable and grounded answers.

---

## 🛡️ 2. Hallucination Mitigation Techniques

| Technique | Implementation in this project | Why it works |
| :--- | :--- | :--- |
| **Strict Negative Constraint** | *"If context does not contain enough info, explicitly state..."* | Prevents the LLM from making up plausible-sounding false answers. |
| **Zero Temperature ($T=0.0$)** | `temperature=0.0` across all LLM client calls | Forces greedy token decoding for maximum factual determinism. |
| **Origin & Citation Anchoring** | Passages formatted with `[Passage X] (Source, Page, Origin)` | Anchors every claim in the response directly to verifiable coordinates. |
| **Deduplicated Context** | Priority given to chunks found by `[BOTH]` | Ensures the LLM is not confused by redundant or conflicting duplicates. |

---

## 🎤 3. Interview Questions & Simple Answers

### Q: Why do we set `temperature=0.0` in a RAG generation pipeline?
**Simple Answer:**
- In creative writing (like writing stories), a high temperature ($0.7$–$1.0$) adds randomness and variety.
- In RAG systems for research papers, finance, or legal documents, we want **100% factual accuracy and repeatability**. Setting `temperature=0.0` forces the LLM to pick the highest probability grounded words every time.

### Cross-Q: How do you prevent an LLM from hallucinating when the retriever returns irrelevant chunks?
**Simple Answer:**
1. **Explicit Negative Prompting:** Include a strict instruction telling the LLM to admit when context is insufficient.
2. **Threshold Filtering:** In the retriever, drop any chunks whose cosine similarity score falls below a minimum confidence cutoff (e.g. $< 0.40$).
3. **Guardrails / Verification:** Check if the generated response contains cited quotes that actually exist in the retrieved text (Citation Fidelity).

### Cross-Q: How does multi-provider fallback improve system reliability?
**Simple Answer:**
- If an API rate limit or outage occurs on one provider (e.g. Gemini 429 error), the system gracefully cascades to secondary providers (e.g. Groq or OpenAI) or falls back to an offline extractive mode, preventing total pipeline failure.
