"""LangGraph pipeline construction and compilation with parallel execution and comparison.

Complete Stage 6 final graph topology supporting 3 execution paths:
1. 'semantic':  START -> query_node -> semantic_retrieval_node -> context_node -> generation_node -> END
2. 'attention': START -> query_node -> attention_retrieval_node -> context_node -> generation_node -> END
3. 'compare':   START -> query_node -> [semantic_node, attention_node] -> comparison_node -> context_node -> generation_node -> END
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional
from langgraph.graph import END, START, StateGraph

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embeddings.attention_retriever import AttentionRetriever
from embeddings.semantic_retriever import SemanticRetriever
from graph.edges import route_after_retrieval, route_retrieval
from graph.nodes import (
    comparison_node,
    context_node,
    generation_node,
    make_attention_retrieval_node,
    make_semantic_retrieval_node,
    query_node,
)
from graph.state import RAGState


def build_rag_graph(
    semantic_retriever: SemanticRetriever,
    attention_retriever: AttentionRetriever,
    top_k: int = 3,
) -> Any:
    """Build and compile the complete multi-mode LangGraph RAG pipeline.

    Args:
        semantic_retriever: Initialized SemanticRetriever instance.
        attention_retriever: Initialized AttentionRetriever instance.
        top_k: Number of candidate chunks to retrieve per retriever.

    Returns:
        Compiled LangGraph application supporting semantic, attention, and compare modes.
    """
    workflow = StateGraph(RAGState)

    # 1. Register all nodes
    workflow.add_node("query_node", query_node)
    workflow.add_node(
        "semantic_retrieval_node",
        make_semantic_retrieval_node(semantic_retriever, top_k=top_k),
    )
    workflow.add_node(
        "attention_retrieval_node",
        make_attention_retrieval_node(attention_retriever, top_k=top_k),
    )
    workflow.add_node("comparison_node", comparison_node)
    workflow.add_node("context_node", context_node)
    workflow.add_node("generation_node", generation_node)

    # 2. Add Start Edge
    workflow.add_edge(START, "query_node")

    # 3. Add Conditional Edge from query_node (Supports Single Mode & Parallel Fan-Out)
    workflow.add_conditional_edges(
        "query_node",
        route_retrieval,
        {
            "semantic_retrieval_node": "semantic_retrieval_node",
            "attention_retrieval_node": "attention_retrieval_node",
        },
    )

    # 4. Add Conditional Routing after retrieval (Fan-In to comparison_node in compare mode)
    retrieval_destinations = {
        "comparison_node": "comparison_node",
        "context_node": "context_node",
    }
    workflow.add_conditional_edges(
        "semantic_retrieval_node",
        route_after_retrieval,
        retrieval_destinations,
    )
    workflow.add_conditional_edges(
        "attention_retrieval_node",
        route_after_retrieval,
        retrieval_destinations,
    )

    # 5. Route comparison to context, then to generation and end
    workflow.add_edge("comparison_node", "context_node")
    workflow.add_edge("context_node", "generation_node")
    workflow.add_edge("generation_node", END)

    return workflow.compile()


if __name__ == "__main__":
    from loader import load_and_chunk_pdf

    sample_pdf = PROJECT_ROOT / "data" / "raw" / "sample_rag_paper.pdf"
    if not sample_pdf.exists():
        print(f"Sample PDF not found at {sample_pdf}. Run loader.py first.")
        sys.exit(1)

    print(f"1. Loading chunks from: {sample_pdf.name}")
    chunks = load_and_chunk_pdf(sample_pdf, chunk_size=350, chunk_overlap=70)
    print(f"   Total chunks loaded: {len(chunks)}")

    print("\n2. Indexing Semantic Retriever (Sentence Transformer)...")
    sem_retriever = SemanticRetriever()
    sem_retriever.build_index(chunks)

    print("\n3. Indexing Attention Retriever (Trained Custom Attention)...")
    attn_retriever = AttentionRetriever()
    attn_retriever.build_index(chunks)

    print("\n4. Compiling LangGraph with Parallel Comparison Topology...")
    app = build_rag_graph(sem_retriever, attn_retriever, top_k=2)
    print("   Graph compiled successfully.")

    test_query = "What is the self-attention formula and how are Q, K, V matrices used?"

    # --- TEST 1: Semantic Mode ---
    print("\n" + "=" * 60)
    print(">>> TEST 1: RETRIEVAL_MODE = 'semantic'")
    print("=" * 60)
    res_sem = app.invoke({
        "query": test_query,
        "retrieval_mode": "semantic",
        "semantic_results": [],
        "attention_results": [],
        "selected_context": [],
        "answer": "",
    })
    print(res_sem["answer"])

    # --- TEST 2: Attention Mode ---
    print("\n" + "=" * 60)
    print(">>> TEST 2: RETRIEVAL_MODE = 'attention'")
    print("=" * 60)
    res_attn = app.invoke({
        "query": test_query,
        "retrieval_mode": "attention",
        "semantic_results": [],
        "attention_results": [],
        "selected_context": [],
        "answer": "",
    })
    print(res_attn["answer"])

    # --- TEST 3: Compare Mode (Parallel Fan-Out & Fan-In) ---
    print("\n" + "=" * 60)
    print(">>> TEST 3: RETRIEVAL_MODE = 'compare' (PARALLEL EXECUTION + MERGE)")
    print("=" * 60)
    t0 = time.perf_counter()
    res_compare = app.invoke({
        "query": test_query,
        "retrieval_mode": "compare",
        "semantic_results": [],
        "attention_results": [],
        "selected_context": [],
        "answer": "",
    })
    t_compare = (time.perf_counter() - t0) * 1000
    print(f"Total Execution Time (Both branches in parallel): {t_compare:.2f} ms")
    print(f"\n{res_compare['answer']}")
    print("=" * 60)
