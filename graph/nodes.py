"""Node functions for the LangGraph RAG pipeline.

Each node is a standalone callable that receives the current RAGState
and returns a dictionary containing only the state keys it updates.
"""

from typing import Any, Callable, Dict, List, Optional
from embeddings.attention_retriever import AttentionRetriever
from embeddings.semantic_retriever import SemanticRetriever
from graph.state import RAGState
from rag import generate_answer


def query_node(state: RAGState) -> Dict[str, Any]:
    """Validates the input query and initializes default state fields.

    Args:
        state: Current graph state.

    Returns:
        Updated query and retrieval_mode.
    """
    raw_query = state.get("query", "")
    query = raw_query.strip() if isinstance(raw_query, str) else ""

    if not query:
        raise ValueError("Query string cannot be empty.")

    return {
        "query": query,
        "retrieval_mode": state.get("retrieval_mode", "semantic"),
    }


def make_semantic_retrieval_node(
    retriever: SemanticRetriever, top_k: int = 3
) -> Callable[[RAGState], Dict[str, Any]]:
    """Factory creating semantic_retrieval_node bound to a SemanticRetriever.

    Args:
        retriever: Initialized SemanticRetriever instance.
        top_k: Number of candidate chunks to retrieve.

    Returns:
        LangGraph node function.
    """

    def semantic_retrieval_node(state: RAGState) -> Dict[str, Any]:
        query = state["query"]
        results = retriever.retrieve(query, top_k=top_k)
        return {"semantic_results": results}

    return semantic_retrieval_node


def make_attention_retrieval_node(
    retriever: AttentionRetriever, top_k: int = 3
) -> Callable[[RAGState], Dict[str, Any]]:
    """Factory creating attention_retrieval_node bound to an AttentionRetriever.

    Args:
        retriever: Initialized AttentionRetriever instance.
        top_k: Number of candidate chunks to retrieve.

    Returns:
        LangGraph node function.
    """

    def attention_retrieval_node(state: RAGState) -> Dict[str, Any]:
        query = state["query"]
        results = retriever.retrieve(query, top_k=top_k)
        return {"attention_results": results}

    return attention_retrieval_node


def comparison_node(state: RAGState) -> Dict[str, Any]:
    """Compares and merges results from both Semantic and Attention retrievers.

    Calculates Jaccard chunk overlap, tags chunk origin ([BOTH], [SEMANTIC], [ATTENTION]),
    and merges chunks with shared agreements prioritized at the top.

    Args:
        state: Current graph state containing both semantic_results and attention_results.

    Returns:
        Dictionary updating 'selected_context'.
    """
    sem_results = state.get("semantic_results", [])
    attn_results = state.get("attention_results", [])

    sem_dict = {c["chunk_id"]: c for c in sem_results}
    attn_dict = {c["chunk_id"]: c for c in attn_results}

    sem_ids = set(sem_dict.keys())
    attn_ids = set(attn_dict.keys())

    shared_ids = sem_ids & attn_ids
    union_ids = sem_ids | attn_ids

    merged_chunks = []

    # 1. Priority: Chunks retrieved by BOTH models (Highest agreement)
    for cid in shared_ids:
        chunk = dict(sem_dict[cid])
        chunk["origin"] = "BOTH"
        chunk["semantic_score"] = sem_dict[cid].get("similarity_score")
        chunk["attention_score"] = attn_dict[cid].get("similarity_score")
        merged_chunks.append(chunk)

    # 2. Chunks retrieved only by Semantic model
    for cid in sem_ids - shared_ids:
        chunk = dict(sem_dict[cid])
        chunk["origin"] = "SEMANTIC_ONLY"
        chunk["semantic_score"] = sem_dict[cid].get("similarity_score")
        merged_chunks.append(chunk)

    # 3. Chunks retrieved only by Attention model
    for cid in attn_ids - shared_ids:
        chunk = dict(attn_dict[cid])
        chunk["origin"] = "ATTENTION_ONLY"
        chunk["attention_score"] = attn_dict[cid].get("similarity_score")
        merged_chunks.append(chunk)

    return {"selected_context": merged_chunks}


def context_node(state: RAGState) -> Dict[str, Any]:
    """Prepares and validates selected_context for downstream generation.

    Args:
        state: Current graph state.

    Returns:
        Dictionary updating 'selected_context'.
    """
    mode = state.get("retrieval_mode", "semantic")
    existing_context = state.get("selected_context", [])

    if existing_context:
        return {"selected_context": existing_context}

    if mode == "semantic":
        return {"selected_context": state.get("semantic_results", [])}
    elif mode == "attention":
        return {"selected_context": state.get("attention_results", [])}
    else:
        return {"selected_context": state.get("semantic_results", [])}


def generation_node(state: RAGState) -> Dict[str, Any]:
    """Generates the grounded natural language answer with citations using rag.py.

    Args:
        state: Current graph state.

    Returns:
        Dictionary updating 'answer'.
    """
    query = state["query"]
    context = state.get("selected_context", [])
    mode = state.get("retrieval_mode", "semantic")

    answer = generate_answer(query=query, context=context, mode=mode)
    return {"answer": answer}
