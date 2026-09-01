"""Custom PyTorch Multi-Head Attention Encoder (Untrained Baseline).

Implements a lightweight self-attention embedding architecture from scratch:
Tokenizer -> Token & Position Embeddings -> MultiheadAttention -> Residual & LayerNorm
-> Masked Mean Pooling -> Linear Projection -> L2 Normalization.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class AttentionEncoder(nn.Module):
    """Custom PyTorch module implementing Multi-Head Self-Attention for text embeddings."""

    def __init__(
        self,
        vocab_size: int = 30522,
        embed_dim: int = 128,
        num_heads: int = 4,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ) -> None:
        """Initialize AttentionEncoder layers.

        Args:
            vocab_size: Size of tokenizer vocabulary (default: BERT 30,522).
            embed_dim: Hidden representation dimension (must be divisible by num_heads).
            num_heads: Number of parallel attention heads.
            max_seq_len: Maximum sequence length supported by positional encodings.
            dropout: Dropout probability for regularization.
        """
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # 1. Token & Positional Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

        # 2. Multi-Head Self-Attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # 3. Residual & Normalization
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # 4. Feed-Forward Projection
        self.linear_proj = nn.Linear(embed_dim, embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass generating L2-normalized passage embeddings.

        Args:
            input_ids: Tensor of shape (Batch_Size, Seq_Len) containing token IDs.
            attention_mask: Tensor of shape (Batch_Size, Seq_Len) with 1 for tokens, 0 for padding.

        Returns:
            Tuple of:
              - embeddings: Normalized tensor of shape (Batch_Size, embed_dim).
              - attn_weights: Attention distribution weights from multi-head attention.
        """
        batch_size, seq_len = input_ids.shape

        if seq_len > self.max_seq_len:
            input_ids = input_ids[:, : self.max_seq_len]
            attention_mask = attention_mask[:, : self.max_seq_len]
            seq_len = self.max_seq_len

        # 1. Compute Token + Position Embeddings
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)  # Shape: (Batch_Size, Seq_Len, embed_dim)

        # 2. Multi-Head Self-Attention with Key Padding Mask
        # In PyTorch MultiheadAttention, key_padding_mask expects True where values are IGNORED (padding)
        key_padding_mask = (attention_mask == 0)

        attn_out, attn_weights = self.multihead_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )

        # 3. Residual Connection & LayerNorm
        x = self.layer_norm1(x + attn_out)  # Shape: (Batch_Size, Seq_Len, embed_dim)

        # 4. Masked Mean Pooling (Collapse Seq_Len -> 1 vector per sentence)
        mask_expanded = attention_mask.unsqueeze(-1).expand(x.size()).float()
        sum_embeddings = torch.sum(x * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask  # Shape: (Batch_Size, embed_dim)

        # 5. Linear Projection & Normalization
        projected = self.layer_norm2(self.linear_proj(pooled))

        # 6. L2 Normalization (Unit vector where Dot Product == Cosine Similarity)
        normalized_embeddings = F.normalize(projected, p=2, dim=-1)

        return normalized_embeddings, attn_weights

    @torch.no_grad()
    def encode(
        self,
        texts: Union[str, List[str]],
        tokenizer: AutoTokenizer,
        device: Optional[torch.device] = None,
        max_length: int = 128,
    ) -> np.ndarray:
        """Helper to encode raw strings directly into NumPy embedding vectors.

        Args:
            texts: Single text string or list of text strings.
            tokenizer: HuggingFace tokenizer instance.
            device: Target torch device (CPU or CUDA).
            max_length: Max token sequence length.

        Returns:
            NumPy array of shape (N, embed_dim) with L2-normalized embeddings.
        """
        self.eval()
        if isinstance(texts, str):
            texts = [texts]

        if device is None:
            device = next(self.parameters()).device

        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        embeddings, _ = self.forward(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )

        return embeddings.cpu().numpy().astype(np.float32)


if __name__ == "__main__":
    print("--- Stage 3: Testing Custom Attention Encoder ---")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    model = AttentionEncoder(
        vocab_size=len(tokenizer),
        embed_dim=128,
        num_heads=4,
        max_seq_len=128,
    )
    print(f"Total Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    sample_sentences = [
        "What is the self-attention mechanism?",
        "Self-attention computes similarity weights across all token pairs.",
        "Deep neural networks require training on large datasets.",
    ]

    print(f"\nEncoding {len(sample_sentences)} test sentences...")
    embeddings = model.encode(sample_sentences, tokenizer=tokenizer)

    print(f"Output Embeddings Shape: {embeddings.shape}")
    print(f"Vector L2 Norm (Should be ~1.0): {np.linalg.norm(embeddings, axis=1)}")

    # Check cosine similarity on untrained random weights
    similarity_matrix = np.dot(embeddings, embeddings.T)
    print("\nUntrained Cosine Similarity Matrix (Random Weights):")
    print(np.round(similarity_matrix, 4))
    print("\nNote: With random untrained weights, the similarities are arbitrary and noisy.")
    print("In Stage 4, we will train this model with Triplet Loss to rank true positives higher!")
