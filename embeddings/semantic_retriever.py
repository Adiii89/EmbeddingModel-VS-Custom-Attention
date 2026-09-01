"""Baseline semantic retrieval using Sentence Transformers and FAISS.

Embeds text passages into dense numerical vectors using 'all-MiniLM-L6-v2'
and searches for nearest neighbors using FAISS with cosine similarity (Inner Product on L2-normalized vectors).
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Ensure project root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class SemanticRetriever:
    """Dense semantic retriever using Sentence Transformers and FAISS."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize the embedding model and vector index.

        Args:
            model_name: HuggingFace model identifier for SentenceTransformer.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        if hasattr(self.model, "get_embedding_dimension"):
            self.embedding_dim = self.model.get_embedding_dimension()
        else:
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[Dict[str, Any]] = []

    def build_index(self, chunks: List[Dict[str, Any]]) -> None:
        """Encode text chunks and index them into FAISS.

        Args:
            chunks: List of chunk dictionaries containing 'text' and metadata.
        """
        if not chunks:
            raise ValueError("Cannot build index on empty chunk list.")

        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]

        # Generate normalized embeddings (so dot product equals cosine similarity)
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype(np.float32)

        # Create FAISS Inner Product index
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search the indexed chunks for the most semantically relevant passages.

        Args:
            query: The search question or statement.
            top_k: Maximum number of top chunks to return.

        Returns:
            List of retrieved chunk dictionaries with added 'similarity_score' field.
        """
        if self.index is None or not self.chunks:
            raise ValueError("Retriever index is not built. Call build_index() first.")

        query = query.strip()
        if not query:
            return []

        # Encode and normalize query vector
        query_vec = self.model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype(np.float32)

        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                chunk_copy = dict(self.chunks[idx])
                chunk_copy["similarity_score"] = float(score)
                results.append(chunk_copy)

        return results


if __name__ == "__main__":
    from loader import load_and_chunk_pdf

    sample_pdf = PROJECT_ROOT / "data" / "raw" / "sample_rag_paper.pdf"
    if sample_pdf.exists():
        print(f"Loading chunks from sample PDF: {sample_pdf.name}")
        chunks = load_and_chunk_pdf(sample_pdf, chunk_size=350, chunk_overlap=70)
        
        print(f"Building semantic index with {len(chunks)} chunks using all-MiniLM-L6-v2...")
        retriever = SemanticRetriever()
        retriever.build_index(chunks)

        query = "What is the self-attention mechanism and how does it compute similarity?"
        print(f"\nQuery: \"{query}\"")
        top_chunks = retriever.retrieve(query, top_k=2)

        print("\n--- Retrieved Results ---")
        for i, res in enumerate(top_chunks, 1):
            print(f"\nResult {i} (Similarity Score: {res['similarity_score']:.4f} | Page {res['page_number']} | Chunk #{res['chunk_id']}):")
            print(f"\"{res['text']}\"")
