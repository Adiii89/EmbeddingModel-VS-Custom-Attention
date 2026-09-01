"""Conditional routing edges for the LangGraph RAG pipeline.

Handles dynamic branching:
1. route_retrieval: Dispatches from query_node to semantic, attention, or both (compare).
2. route_after_retrieval: Routes to comparison_node in 'compare' mode, or directly to context_node.
"""

from typing import Any, List, Union
from graph.state import RAGState


def route_retrieval(state: RAGState) -> Union[str, List[str]]:
    """Determine the next retrieval node(s) based on retrieval_mode.

    Args:
        state: Current graph state containing 'retrieval_mode'.

    Returns:
        Destination node name or list of node names for parallel fan-out.
    """
    raw_mode = state.get("retrieval_mode", "semantic")
    mode = str(raw_mode).lower().strip() if raw_mode else "semantic"

    if mode == "semantic":
        return "semantic_retrieval_node"
    elif mode == "attention":
        return "attention_retrieval_node"
    elif mode == "compare":
        # Parallel Fan-Out: Execute both retrieval nodes
        return ["semantic_retrieval_node", "attention_retrieval_node"]
    else:
        allowed = ["'semantic'", "'attention'", "'compare'"]
        raise ValueError(
            f"Invalid retrieval_mode '{raw_mode}'. Must be one of: {', '.join(allowed)}."
        )


def route_after_retrieval(state: RAGState) -> str:
    """Determine destination after a retrieval node finishes.

    Args:
        state: Current graph state containing 'retrieval_mode'.

    Returns:
        'comparison_node' if in compare mode (fan-in), else 'context_node'.
    """
    mode = state.get("retrieval_mode", "semantic").lower().strip()
    if mode == "compare":
        return "comparison_node"
    return "context_node"
