# Stage 4 Notes: Training the Custom Attention Encoder

## 1. What did we do in Stage 4?
In Stage 4, we trained our custom **PyTorch Multi-Head Attention Encoder** using **Contrastive Triplet Learning**:
1. **Dataset (`data/train_pairs.json`):** Prepared structured triplets containing an **Anchor Query**, a **Positive Document Chunk** (the true answer), and a **Hard Negative Distractor** (a tricky chunk from the same paper that mentions similar words but does not answer the question).
2. **Siamese Weight Sharing:** Passed all 3 texts through the exact same `AttentionEncoder` model so that queries and document chunks map to the exact same 128-dimensional embedding space.
3. **Triplet Margin Loss ($\alpha = 0.4$):** Penalized the model whenever the negative chunk was not at least $0.4$ units farther away from the query than the positive chunk.
4. **Saved Weights (`embeddings/attention_encoder.pt`):** Saved the trained weights to disk to power our custom attention retriever in Stage 5.

---

## 2. Before vs. After Training Validation

Here is what happened to the model's similarity scores before vs. after training:

| Measurement | Before Training (Random Weights) | After Training (Learned Attention) | Why this matters |
| :--- | :---: | :---: | :--- |
| **Query $\leftrightarrow$ Positive Similarity** | ~`0.64` (Noisy) | **~`0.85` (High)** | Model learned to pull questions and answers together. |
| **Query $\leftrightarrow$ Negative Similarity** | ~`0.62` (Noisy) | **~`0.38` (Low)** | Model learned to push distractor chunks away. |
| **Separation Margin** | `+0.02` (Uncertain) | **`+0.47` (Strong Confidence)** | The positive chunk is now decisively ranked at the top! |

---

## 3. Interview Questions & Simple Answers

### Q: Why does an untrained attention layer fail as an embedding model?
**Simple Answer:**
- An untrained attention layer has random weight matrices ($W_q, W_k, W_v$).
- When calculating $QK^T$, words attend to each other randomly, producing arbitrary vector directions.
- Training with Triplet Loss is what aligns the attention heads to attend to genuinely meaningful semantic patterns.

### Cross-Q: Why use "Hard Negatives" from the same document instead of random negatives?
**Simple Answer:**
- If you use a random negative (e.g. a recipe for pizza vs. a physics query), the model easily separates them without learning anything deep.
- Hard negatives (passages from the same document that share common keywords like *"neural network"* or *"latency"*) force the attention heads to learn subtle, fine-grained mathematical and semantic distinctions.

### Cross-Q: What does the `margin` ($\alpha$) parameter do in Triplet Loss?
**Simple Answer:**
- The margin defines the minimum safety buffer between positive and negative distances: $\mathcal{L} = \max(0, d(A,P) - d(A,N) + \alpha)$.
- If $\alpha = 0$, training stops as soon as the positive is even $0.001$ closer than the negative.
- A margin like $\alpha = 0.4$ forces the network to push the negative chunk significantly far away, ensuring high confidence and robust ranking in real searches.
