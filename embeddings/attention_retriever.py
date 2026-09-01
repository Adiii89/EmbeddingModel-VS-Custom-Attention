"""Custom Attention Retriever using trained AttentionEncoder and FAISS.

Loads trained weights from 'embeddings/attention_encoder.pt', embeds text passages
into 128-dimensional L2-normalized dense vectors, and executes sub-millisecond
cosine similarity vector search with FAISS (IndexFlatIP).
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embeddings.attention_encoder import AttentionEncoder


class AttentionRetriever:
    """Dense retriever powered by custom-trained PyTorch AttentionEncoder and FAISS."""

    def __init__(
        self,
        weights_path: Optional[Path] = None,
        embed_dim: int = 128,
        num_heads: int = 4,
        max_seq_len: int = 128,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize tokenizer, load trained AttentionEncoder, and prepare FAISS.

        Args:
            weights_path: Path to attention_encoder.pt weights file.
            embed_dim: Embedding dimension (default: 128).
            num_heads: Number of attention heads (default: 4).
            max_seq_len: Max sequence length (default: 128).
            device: Torch device (CPU or CUDA).
        """
        if weights_path is None:
            weights_path = PROJECT_ROOT / "embeddings" / "attention_encoder.pt"
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.weights_path = Path(weights_path)
        self.device = device
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # 1. Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # 2. Instantiate Model Architecture
        self.model = AttentionEncoder(
            vocab_size=len(self.tokenizer),
            embed_dim=embed_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=0.0,  # 0.0 dropout during inference
        ).to(self.device)

        # 3. Load Trained Weights if available
        if self.weights_path.exists():
            checkpoint = torch.load(str(self.weights_path), map_location=self.device)
            self.model.load_state_dict(checkpoint)
            print(f"Loaded trained attention weights from: {self.weights_path.name}")
        else:
            print(f"Warning: Trained weights not found at {self.weights_path}. Using untrained weights.")

        self.model.eval()
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[Dict[str, Any]] = []

    def build_index(self, chunks: List[Dict[str, Any]]) -> None:
        """Encode text chunks and build FAISS IndexFlatIP index.

        Args:
            chunks: List of chunk dictionaries containing 'text' and metadata.
        """
        if not chunks:
            raise ValueError("Cannot build index on empty chunk list.")

        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]

        # Generate normalized 128-dim embeddings
        embeddings = self.model.encode(
            texts=texts,
            tokenizer=self.tokenizer,
            device=self.device,
            max_length=self.max_seq_len,
        ).astype(np.float32)

        # Build FAISS Inner Product index (Cosine similarity on normalized vectors)
        self.index = faiss.IndexFlatIP(self.embed_dim)
        self.index.add(embeddings)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-K most semantically relevant chunks for a query.

        Args:
            query: User search query string.
            top_k: Number of top candidate chunks to return.

        Returns:
            List of retrieved chunks with similarity_score and metadata.
        """
        if self.index is None or not self.chunks:
            raise ValueError("Retriever index is not built. Call build_index() first.")

        query = query.strip()
        if not query:
            return []

        # Encode query to 128-dim normalized vector
        query_vec = self.model.encode(
            texts=[query],
            tokenizer=self.tokenizer,
            device=self.device,
            max_length=self.max_seq_len,
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

        print(f"\nBuilding attention vector index with {len(chunks)} chunks...")
        retriever = AttentionRetriever()
        retriever.build_index(chunks)

        query = "What is the self-attention formula and how are Q, K, V matrices used?"
        print(f"\nQuery: \"{query}\"")
        top_chunks = retriever.retrieve(query, top_k=2)

        print("\n--- Attention Retriever Results ---")
        for i, res in enumerate(top_chunks, 1):
            print(f"\nResult {i} (Similarity Score: {res['similarity_score']:.4f} | Page {res['page_number']} | Chunk #{res['chunk_id']}):")
            print(f"\"{res['text']}\"")
