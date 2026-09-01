"""State definition for the Attention vs Semantic Retrieval RAG pipeline.

In LangGraph, the State acts as the shared memory passed between nodes.
Each node receives this state, performs its specific task, and returns updated keys.
"""

from typing import Any, Dict, List, TypedDict


class RAGState(TypedDict):
    """The central state dictionary flowing through the LangGraph pipeline."""

    query: str               #The original user question or search prompt.
    retrieval_mode: str      #Active retrieval mode: 'semantic', 'attention', or 'compare'.
    semantic_results: List[Dict[str, Any]]  #Candidate text chunks retrieved by the Sentence Transformer model.
    attention_results: List[Dict[str, Any]]  #Candidate text chunks retrieved by the custom PyTorch Attention model.
    selected_context: List[Dict[str, Any]]   #Final merged/selected chunks passed to the LLM for answer generation.
    answer: str  #The generated answer produced by the generation node.
