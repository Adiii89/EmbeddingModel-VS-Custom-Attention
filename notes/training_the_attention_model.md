# Training the Custom Attention Model (Google Colab Notebook & Guide)

> **Document Purpose:** Complete, step-by-step interactive Colab training guide. Every code section in this file matches the exact code cells and explanations used in Google Colab.

---

## 🧭 The 6 Colab Notebook Sections Overview

| Section | Colab Cell Purpose | Key Code / Concepts |
| :--- | :--- | :--- |
| **Section 1** | **Setup & GPU Check** | `torch`, `transformers`, `device = cuda` check |
| **Section 2** | **Training Dataset Creation** | Anchor, Positive, and Hard Negative Triplets |
| **Section 3** | **AttentionEncoder Architecture** | `MultiheadAttention`, `forward()` line-by-line, Mean Pooling, L2 Normalization |
| **Section 4** | **Dataset & DataLoader** | Batching and Tokenizing $(A, P, N)$ into PyTorch Tensors |
| **Section 5** | **The Training Loop** | `TripletMarginLoss`, `AdamW`, Backprop, Loss Curve |
| **Section 6** | **Validation & Weight Download** | Before vs. After Ranking Test & Saving `attention_encoder.pt` |

---

## 📦 Section 1: Setup, GPU Check & Dependencies (Colab Cell 1)

### 📝 Code:
```python
# ==============================================================================
# SECTION 1: Setup & Environment
# ==============================================================================

# 1. Install necessary libraries
!pip install -q torch transformers

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# 2. Check if GPU (CUDA) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")
if torch.cuda.is_available():
    print(f"🎮 GPU Model: {torch.cuda.get_device_name(0)}")
```

---

## 📊 Section 2: Training Dataset Creation (Colab Cell 2)

### 📝 Code:
```python
# ==============================================================================
# SECTION 2: Training Data (Anchor, Positive, Hard Negative Triplets)
# ==============================================================================

TRAIN_DATA = [
    {
        "query": "What is the self-attention formula and how are Q, K, V matrices used?",
        "positive": "Self-attention computes similarity weights across all token pairs in a sequence using query (Q), key (K), and value (V) matrices. By computing Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V, the model captures fine-grained contextual interactions between words.",
        "negative": "We evaluate retrieval performance using Recall@K and Mean Reciprocal Rank (MRR). Latency benchmarks compare execution time across CPU and GPU inference environments."
    },
    {
        "query": "What is Retrieval-Augmented Generation (RAG) and why is it used?",
        "positive": "Retrieval-Augmented Generation (RAG) combines external knowledge retrieval with neural text generation. By grounding generation with retrieved vector embeddings, the model provides accurate, verifiable citations and prevents hallucinations.",
        "negative": "A multi-head attention layer runs self-attention in parallel across multiple representation subspaces, allowing the model to attend to information at different positions."
    },
    {
        "query": "How does multi-head attention work in parallel across subspaces?",
        "positive": "A multi-head attention layer runs self-attention computation in parallel across multiple representation subspaces, allowing the model to jointly attend to information at different positions.",
        "negative": "Traditional semantic search uses pre-trained sentence transformer models such as all-MiniLM-L6-v2 to map passages into dense embedding vectors."
    },
    {
        "query": "What evaluation metrics measure retrieval quality?",
        "positive": "We evaluate retrieval performance using Recall@K and Mean Reciprocal Rank (MRR). Recall measures the proportion of relevant chunks in top-K candidates. MRR measures the reciprocal of the rank position of the first relevant chunk.",
        "negative": "Self-attention computes similarity weights across all token pairs in a sequence using query (Q), key (K), and value (V) matrices."
    },
    {
        "query": "Why do traditional LLMs hallucinate without RAG?",
        "positive": "Traditional LLMs rely solely on their internal training data, which can become outdated and leads to hallucinations when asked about domain-specific facts without external knowledge grounding.",
        "negative": "Contrastive learning pulls positive document chunks closer to the query while pushing hard negative distractor chunks farther apart in Euclidean space."
    },
    {
        "query": "How does contrastive triplet learning train custom encoders?",
        "positive": "Contrastive learning pulls positive document chunks closer to the query while pushing hard negative distractor chunks farther apart in Euclidean space using a margin boundary.",
        "negative": "PyMuPDF is a Python binding to the high-performance C library MuPDF, extracting text structured by blocks, lines, and spans with bounding box coordinates."
    }
]

print(f"✅ Loaded {len(TRAIN_DATA)} training triplets.")
```

---

## 🧠 Section 3: Custom Attention Encoder Architecture (Colab Cell 3)

### 📝 Code:
```python
# ==============================================================================
# SECTION 3: Custom PyTorch Multi-Head Attention Encoder Architecture
# ==============================================================================

class AttentionEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 30522,
        embed_dim: int = 128,
        num_heads: int = 4,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # 1. Token & Position Embedding layers
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.position_embedding = nn.Embedding(num_embeddings=max_seq_len, embedding_dim=embed_dim)

        # 2. Multi-Head Self-Attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # 3. Residual & Normalization
        self.layer_norm1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.dropout = nn.Dropout(p=dropout)

        # 4. Linear Projection & Normalization
        self.linear_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=embed_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        batch_size, seq_len = input_ids.shape

        if seq_len > self.max_seq_len:
            input_ids = input_ids[:, : self.max_seq_len]
            attention_mask = attention_mask[:, : self.max_seq_len]
            seq_len = self.max_seq_len

        # Step 1: Token IDs -> Dense Embeddings + Position Vectors
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        # Step 2: Multi-Head Self-Attention (Masking padded tokens)
        key_padding_mask = (attention_mask == 0)
        attn_out, attn_weights = self.multihead_attn(
            query=x, key=x, value=x,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )

        # Step 3: Residual Connection & LayerNorm
        x = self.layer_norm1(x + attn_out)

        # Step 4: Masked Mean Pooling (Collapse Seq_Len -> 1 Vector)
        mask_expanded = attention_mask.unsqueeze(-1).expand(x.size()).float()
        sum_embeddings = torch.sum(x * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask

        # Step 5: Linear Projection & LayerNorm
        projected = self.layer_norm2(self.linear_proj(pooled))

        # Step 6: L2 Normalization (Length = 1.0, Dot Product == Cosine Similarity)
        normalized = F.normalize(projected, p=2, dim=-1)

        return normalized, attn_weights

    @torch.no_grad()
    def encode(self, texts, tokenizer, max_length=128):
        self.eval()
        if isinstance(texts, str):
            texts = [texts]
        
        dev = next(self.parameters()).device
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(dev)
        embeddings, _ = self.forward(encoded["input_ids"], encoded["attention_mask"])
        return embeddings.cpu().numpy()

# Initialize Tokenizer and Model
print("Loading bert-base-uncased tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

model = AttentionEncoder(
    vocab_size=len(tokenizer),
    embed_dim=128,
    num_heads=4,
    max_seq_len=128,
    dropout=0.1,
).to(device)

print(f"✅ AttentionEncoder instantiated successfully on {device}!")
print(f"   Total Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
```

---

## 🗂️ Section 4: Triplet Dataset & DataLoader (Colab Cell 4)

### 📝 Code:
```python
# ==============================================================================
# SECTION 4: PyTorch Dataset & DataLoader
# ==============================================================================

class TripletDataset(Dataset):
    """Custom PyTorch Dataset that tokenizes (Anchor, Positive, Negative) triplets."""

    def __init__(self, triplets, tokenizer, max_length=128):
        self.triplets = triplets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        item = self.triplets[idx]
        
        # Tokenize Anchor Query
        anchor_enc = self.tokenizer(
            item["query"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Tokenize Positive Document Chunk
        pos_enc = self.tokenizer(
            item["positive"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Tokenize Negative Distractor Chunk
        neg_enc = self.tokenizer(
            item["negative"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "anchor_ids": anchor_enc["input_ids"].squeeze(0),
            "anchor_mask": anchor_enc["attention_mask"].squeeze(0),
            "pos_ids": pos_enc["input_ids"].squeeze(0),
            "pos_mask": pos_enc["attention_mask"].squeeze(0),
            "neg_ids": neg_enc["input_ids"].squeeze(0),
            "neg_mask": neg_enc["attention_mask"].squeeze(0),
        }

# Create Dataset and DataLoader
train_dataset = TripletDataset(TRAIN_DATA, tokenizer=tokenizer, max_length=128)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

print(f"✅ DataLoader created with {len(train_loader)} batches (Batch Size = 2).")
```

---

## ⚡ Section 5: The Training Loop with Triplet Margin Loss (Colab Cell 5)

### 📝 Code:
```python
# ==============================================================================
# SECTION 5: Training Loop (Triplet Loss + AdamW Optimizer)
# ==============================================================================

# 1. Hyperparameters
EPOCHS = 25
LEARNING_RATE = 1e-3
MARGIN = 0.4

# 2. Loss Function & Optimizer
criterion = nn.TripletMarginLoss(margin=MARGIN, p=2.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

print(f"🚀 Starting Training for {EPOCHS} Epochs on {device}...")
print(f"   Loss: TripletMarginLoss(margin={MARGIN}) | Optimizer: AdamW(lr={LEARNING_RATE})")
print("=" * 60)

model.train()
loss_history = []

for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0.0

    for batch in train_loader:
        # Move batch tensors to target device (GPU)
        a_ids = batch["anchor_ids"].to(device)
        a_mask = batch["anchor_mask"].to(device)
        p_ids = batch["pos_ids"].to(device)
        p_mask = batch["pos_mask"].to(device)
        n_ids = batch["neg_ids"].to(device)
        n_mask = batch["neg_mask"].to(device)

        # 1. FORWARD PASS: Pass all 3 through the shared AttentionEncoder (Siamese)
        emb_anchor, _ = model(a_ids, a_mask)
        emb_positive, _ = model(p_ids, p_mask)
        emb_negative, _ = model(n_ids, n_mask)

        # 2. COMPUTE TRIPLET LOSS
        loss = criterion(emb_anchor, emb_positive, emb_negative)

        # 3. BACKWARD PASS (Backpropagation / Autograd)
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 4. OPTIMIZER STEP (Update weights)
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    loss_history.append(avg_loss)

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] ──► Average Triplet Loss: {avg_loss:.4f}")

print("=" * 60)
print(f"🎉 Training Complete! Final Epoch Loss: {loss_history[-1]:.4f}")
```

### 💡 Explanation (Line-by-Line):
1. **`criterion = nn.TripletMarginLoss(margin=0.4, p=2.0)`**:
   - Computes $\max(0, d(A,P) - d(A,N) + 0.4)$. It penalizes the model whenever the Negative chunk is not at least $0.4$ distance farther away than the Positive chunk.
2. **`optimizer = torch.optim.AdamW(...)`**:
   - AdamW with weight decay (`0.01`) prevents weights from growing too large (L2 regularization).
3. **`optimizer.zero_grad()`**:
   - PyTorch accumulates gradients by default. We must reset gradients to zero at every batch before calculating new ones.
4. **`loss.backward()`**:
   - Calculates the calculus gradients $\left(\frac{\partial \mathcal{L}}{\partial W}\right)$ for all 4,006,272 weights.
5. **`torch.nn.utils.clip_grad_norm_(..., max_norm=1.0)`**:
   - Prevents gradients from spiking or exploding.
6. **`optimizer.step()`**:
   - Updates every weight: $W_{\text{new}} = W_{\text{old}} - (\text{lr} \times \text{gradient})$.

---
