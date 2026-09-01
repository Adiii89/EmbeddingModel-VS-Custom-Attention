"""Quantitative Evaluation and Benchmarking Suite for Attention-RAG.

Evaluates and compares:
1. Semantic Retriever (Sentence Transformer: all-MiniLM-L6-v2, 384-dim)
2. Attention Retriever (Custom Trained PyTorch Multi-Head Attention, 128-dim)
3. Parallel Compare Mode (Merged Context)

Metrics computed:
- Recall@1, Recall@2, Recall@3
- Mean Reciprocal Rank (MRR)
- Average & p95 Search Latency (ms)
- Jaccard Chunk Overlap (%)
- Storage & Parameter Footprint
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embeddings.attention_retriever import AttentionRetriever
from embeddings.semantic_retriever import SemanticRetriever
from loader import load_and_chunk_pdf


def compute_mrr_and_recall(
    retrieved_chunks: List[Dict[str, Any]],
    expected_text_keyword: str,
    top_k_levels: List[int] = [1, 2, 3],
) -> Tuple[Dict[int, float], float]:
    """Compute Recall@K at various K levels and Reciprocal Rank for a single query.

    Args:
        retrieved_chunks: List of candidate chunk dicts returned by retriever.
        expected_text_keyword: Core keyword/phrase that must appear in true chunk.
        top_k_levels: List of K thresholds to compute recall.

    Returns:
        Tuple of (Dict[K, recall_score], reciprocal_rank_score).
    """
    keyword_lower = expected_text_keyword.lower()

    # Find the 1-based rank of the first relevant chunk
    found_rank = None
    for rank, chunk in enumerate(retrieved_chunks, 1):
        if keyword_lower in chunk.get("text", "").lower():
            found_rank = rank
            break

    # Reciprocal Rank: 1 / rank if found, else 0.0
    rr = 1.0 / found_rank if found_rank is not None else 0.0

    # Recall@K: 1.0 if found within top-K, else 0.0
    recall_at_k = {}
    for k in top_k_levels:
        recall_at_k[k] = 1.0 if (found_rank is not None and found_rank <= k) else 0.0

    return recall_at_k, rr


def compute_jaccard_overlap(
    results_a: List[Dict[str, Any]], results_b: List[Dict[str, Any]]
) -> float:
    """Compute Jaccard similarity between two sets of retrieved chunk IDs."""
    set_a = {c["chunk_id"] for c in results_a}
    set_b = {c["chunk_id"] for c in results_b}
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def run_benchmark(
    pdf_path: Optional[Path] = None,
    eval_data_path: Optional[Path] = None,
    num_runs: int = 3,
) -> Dict[str, Any]:
    """Run full benchmarking suite and print comparative results table.

    Args:
        pdf_path: Path to sample PDF.
        eval_data_path: Path to train/eval pairs JSON.
        num_runs: Number of latency warmup/averaging passes.

    Returns:
        Benchmark report dictionary.
    """
    if pdf_path is None:
        pdf_path = PROJECT_ROOT / "data" / "raw" / "sample_rag_paper.pdf"
    if eval_data_path is None:
        eval_data_path = PROJECT_ROOT / "data" / "train_pairs.json"

    print("=" * 70)
    print("      ATTENTION-RAG: QUANTITATIVE BENCHMARKING & EVALUATION")
    print("=" * 70)

    # 1. Load document chunks
    print(f"Loading document chunks from: {pdf_path.name}")
    chunks = load_and_chunk_pdf(pdf_path, chunk_size=350, chunk_overlap=70)
    print(f"Total chunks indexed: {len(chunks)}")

    # 2. Load evaluation test cases
    with open(eval_data_path, "r", encoding="utf-8") as f:
        eval_cases = json.load(f)
    print(f"Total test queries: {len(eval_cases)}")

    # 3. Initialize & index retrievers
    print("\nIndexing Semantic Retriever (Sentence Transformer: all-MiniLM-L6-v2)...")
    sem_retriever = SemanticRetriever()
    sem_retriever.build_index(chunks)

    print("Indexing Attention Retriever (Trained Custom AttentionEncoder)...")
    attn_retriever = AttentionRetriever()
    attn_retriever.build_index(chunks)

    # 4. Evaluation storage
    metrics = {
        "semantic": {"r1": [], "r2": [], "r3": [], "mrr": [], "latency": []},
        "attention": {"r1": [], "r2": [], "r3": [], "mrr": [], "latency": []},
        "overlap": [],
    }

    # 5. Run benchmark across all test cases
    for case in eval_cases:
        query = case["query"]
        expected = case["positive"][:50]  # First 50 chars as distinctive signature

        # Warmup and timed runs for Semantic
        sem_latencies = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            sem_chunks = sem_retriever.retrieve(query, top_k=3)
            sem_latencies.append((time.perf_counter() - t0) * 1000)

        r_sem, mrr_sem = compute_mrr_and_recall(sem_chunks, expected)
        metrics["semantic"]["r1"].append(r_sem[1])
        metrics["semantic"]["r2"].append(r_sem[2])
        metrics["semantic"]["r3"].append(r_sem[3])
        metrics["semantic"]["mrr"].append(mrr_sem)
        metrics["semantic"]["latency"].append(float(np.mean(sem_latencies)))

        # Warmup and timed runs for Attention
        attn_latencies = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            attn_chunks = attn_retriever.retrieve(query, top_k=3)
            attn_latencies.append((time.perf_counter() - t0) * 1000)

        r_attn, mrr_attn = compute_mrr_and_recall(attn_chunks, expected)
        metrics["attention"]["r1"].append(r_attn[1])
        metrics["attention"]["r2"].append(r_attn[2])
        metrics["attention"]["r3"].append(r_attn[3])
        metrics["attention"]["mrr"].append(mrr_attn)
        metrics["attention"]["latency"].append(float(np.mean(attn_latencies)))

        # Compute Jaccard Overlap
        overlap = compute_jaccard_overlap(sem_chunks, attn_chunks)
        metrics["overlap"].append(overlap)

    # 6. Aggregate results
    report = {
        "semantic": {
            "recall@1": float(np.mean(metrics["semantic"]["r1"])),
            "recall@2": float(np.mean(metrics["semantic"]["r2"])),
            "recall@3": float(np.mean(metrics["semantic"]["r3"])),
            "mrr": float(np.mean(metrics["semantic"]["mrr"])),
            "latency_mean_ms": float(np.mean(metrics["semantic"]["latency"])),
            "latency_p95_ms": float(np.percentile(metrics["semantic"]["latency"], 95)),
            "embedding_dim": 384,
            "parameters": "22.7M",
        },
        "attention": {
            "recall@1": float(np.mean(metrics["attention"]["r1"])),
            "recall@2": float(np.mean(metrics["attention"]["r2"])),
            "recall@3": float(np.mean(metrics["attention"]["r3"])),
            "mrr": float(np.mean(metrics["attention"]["mrr"])),
            "latency_mean_ms": float(np.mean(metrics["attention"]["latency"])),
            "latency_p95_ms": float(np.percentile(metrics["attention"]["latency"], 95)),
            "embedding_dim": 128,
            "parameters": "4.0M",
        },
        "mean_jaccard_overlap": float(np.mean(metrics["overlap"])),
    }

    # 7. Print Comparative Table
    print("\n" + "=" * 70)
    print("                    BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Evaluation Metric':<28} | {'Semantic (MiniLM)':<18} | {'Custom Attention':<18}")
    print("-" * 70)
    print(f"{'Recall @ 1':<28} | {report['semantic']['recall@1']:>17.2%} | {report['attention']['recall@1']:>17.2%}")
    print(f"{'Recall @ 2':<28} | {report['semantic']['recall@2']:>17.2%} | {report['attention']['recall@2']:>17.2%}")
    print(f"{'Recall @ 3':<28} | {report['semantic']['recall@3']:>17.2%} | {report['attention']['recall@3']:>17.2%}")
    print(f"{'Mean Reciprocal Rank (MRR)':<28} | {report['semantic']['mrr']:>17.4f} | {report['attention']['mrr']:>17.4f}")
    print(f"{'Mean Latency (ms)':<28} | {report['semantic']['latency_mean_ms']:>15.2f} ms | {report['attention']['latency_mean_ms']:>15.2f} ms")
    print(f"{'P95 Latency (ms)':<28} | {report['semantic']['latency_p95_ms']:>15.2f} ms | {report['attention']['latency_p95_ms']:>15.2f} ms")
    print(f"{'Vector Dimensions':<28} | {report['semantic']['embedding_dim']:>18} | {report['attention']['embedding_dim']:>18}")
    print(f"{'Model Parameter Count':<28} | {report['semantic']['parameters']:>18} | {report['attention']['parameters']:>18}")
    print("-" * 70)
    print(f"{'Mean Jaccard Overlap':<28} | {report['mean_jaccard_overlap']:>17.2%} (Chunk agreement)")
    print("=" * 70)

    # Save to JSON
    out_file = PROJECT_ROOT / "data" / "benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved raw benchmark data to: {out_file}")

    return report


if __name__ == "__main__":
    run_benchmark()
