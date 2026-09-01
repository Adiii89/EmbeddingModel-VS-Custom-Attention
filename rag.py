"""RAG Generation module: Prompt formatting and LLM answer generation.

Supports multiple LLM providers (Google Gemini, OpenAI, Groq) via API keys
defined in .env or environment variables, with a graceful grounded extractive
fallback when running offline without keys.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# System instruction strictly enforcing grounding and citations
RAG_SYSTEM_PROMPT = """You are an expert, fact-grounded AI research assistant.
Your task is to answer the user's question accurately using ONLY the retrieved context passages provided below.

Strict Guidelines:
1. Answer the question directly and concisely based solely on the provided context.
2. If the context does not contain enough information to answer the question, explicitly state: "The retrieved context does not contain sufficient information to answer this question."
3. Cite your sources in the text using format: [Source: <filename>, Page: <page_number>].
4. Do NOT hallucinate, assume, or extrapolate beyond the provided text facts.
"""


def format_context(context_chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved candidate chunks into structured text with citation metadata.

    Args:
        context_chunks: List of chunk dictionaries from the retrieval/comparison nodes.

    Returns:
        Formatted context string.
    """
    if not context_chunks:
        return "No context passages retrieved."

    formatted_passages = []
    for i, c in enumerate(context_chunks, 1):
        source = c.get("source", "document")
        page = c.get("page_number", 1)
        origin = c.get("origin")
        origin_tag = f" | Retrieved By: {origin}" if origin else ""

        scores = []
        if "similarity_score" in c and c["similarity_score"] is not None:
            scores.append(f"Score={c['similarity_score']:.3f}")
        if "semantic_score" in c and c["semantic_score"] is not None:
            scores.append(f"Sem={c['semantic_score']:.3f}")
        if "attention_score" in c and c["attention_score"] is not None:
            scores.append(f"Attn={c['attention_score']:.3f}")
        score_str = f" | {', '.join(scores)}" if scores else ""

        passage_header = f"--- [Passage {i}] (Source: {source}, Page: {page}{origin_tag}{score_str}) ---"
        formatted_passages.append(f"{passage_header}\n{c.get('text', '').strip()}")

    return "\n\n".join(formatted_passages)


def build_user_prompt(query: str, context_text: str) -> str:
    """Build user message combining query and formatted context.

    Args:
        query: User search question.
        context_text: Formatted context passages.

    Returns:
        Combined prompt string.
    """
    return f"""Context Passages:
{context_text}

User Question:
{query}

Grounded Answer (with citations):"""


def generate_answer(
    query: str,
    context: List[Dict[str, Any]],
    mode: str = "semantic",
    llm_provider: Optional[str] = None,
) -> str:
    """Generate a factually grounded answer using an LLM or extractive fallback.

    Args:
        query: User query string.
        context: List of retrieved context chunk dicts.
        mode: Retrieval mode ('semantic', 'attention', 'compare').
        llm_provider: Optional explicit provider ('gemini', 'openai', 'groq').

    Returns:
        Generated answer text with source citations.
    """
    if not context:
        return f"[{mode.upper()} Mode] No relevant context found to answer the query."

    context_str = format_context(context)
    user_prompt = build_user_prompt(query, context_str)

    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    # 1. Try Google Gemini
    if (llm_provider == "gemini" or not llm_provider) and gemini_key:
        try:
            from google import genai

            client = genai.Client(api_key=gemini_key)
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=f"{RAG_SYSTEM_PROMPT}\n\n{user_prompt}",
                config={"temperature": 0.0},
            )
            return response.text.strip()
        except Exception as e:
            print(f"Gemini API invocation failed: {e}. Falling back...")

    # 2. Try Groq (Ultra-fast Llama-3)
    if (llm_provider == "groq" or not llm_provider) and groq_key:
        try:
            from groq import Groq

            client = Groq(api_key=groq_key)
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": RAG_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.0,
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"Groq API invocation failed: {e}. Falling back...")

    # 3. Try OpenAI
    if (llm_provider == "openai" or not llm_provider) and openai_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": RAG_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API invocation failed: {e}. Falling back...")

    # 4. Extractive Grounded Fallback (Offline / Zero API Key mode)
    # Extracts the most relevant sentences directly from top chunks with full citations
    summary_lines = []
    for i, c in enumerate(context, 1):
        source = c.get("source", "document")
        page = c.get("page_number", 1)
        origin = c.get("origin")
        origin_str = f" [{origin}]" if origin else ""
        summary_lines.append(
            f"Passage {i}{origin_str} (Source: {source}, Page: {page}):\n\"{c.get('text', '')}\""
        )

    offline_notice = (
        "(Note: Set GEMINI_API_KEY, GROQ_API_KEY, or OPENAI_API_KEY in .env for full neural LLM synthesis)\n"
    )
    return (
        f"[Grounded Response - Mode: {mode.upper()}]\n"
        f"Query: \"{query}\"\n\n"
        f"Grounded Source Context ({len(context)} chunks):\n"
        + "\n\n".join(summary_lines)
        + f"\n\n{offline_notice}"
    )


if __name__ == "__main__":
    test_context = [
        {
            "chunk_id": 2,
            "text": "By computing Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V, the model captures fine-grained contextual interactions between words.",
            "source": "sample_rag_paper.pdf",
            "page_number": 1,
            "origin": "BOTH",
            "similarity_score": 0.845,
        },
        {
            "chunk_id": 0,
            "text": "Retrieval-Augmented Generation (RAG) combines external knowledge retrieval with neural text generation.",
            "source": "sample_rag_paper.pdf",
            "page_number": 1,
            "origin": "ATTENTION_ONLY",
            "similarity_score": 0.669,
        },
    ]

    print("Testing format_context()...")
    print(format_context(test_context))
    print("\nTesting generate_answer()...")
    ans = generate_answer("What is the self-attention formula?", test_context, mode="compare")
    print(ans)
