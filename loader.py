"""Document loader and chunking module for the RAG pipeline.

Uses PyMuPDF for fast PDF text extraction and a recursive
boundary-aware chunker to create overlapping text segments with clean word boundaries.
"""

from pathlib import Path
from typing import Any, Dict, List
import pymupdf


def load_pdf(pdf_path: str | Path) -> List[Dict[str, Any]]:
    """Extract text from a PDF file page-by-page.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of dicts containing 'page_number', 'text', and 'source'.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    pages_data = []
    doc = pymupdf.open(str(pdf_path))

    for page_index in range(len(doc)):
        page = doc[page_index]
        text = page.get_text("text").strip()
        if text:  # Ignore empty pages
            pages_data.append({
                "page_number": page_index + 1,
                "text": text,
                "source": pdf_path.name,
            })

    doc.close()
    return pages_data


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    min_chunk_chars: int = 50,
    page_number: int = 1,
    source: str = "",
    start_chunk_id: int = 0,
) -> List[Dict[str, Any]]:
    """Split text into overlapping chunks respecting natural word, sentence, and paragraph boundaries.

    Args:
        text: Raw text to split.
        chunk_size: Target maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.
        min_chunk_chars: Minimum character length to avoid tiny useless fragments.
        page_number: Source page number for metadata tracking.
        source: Source document name for citation tracking.
        start_chunk_id: Initial ID counter for chunk sequencing.

    Returns:
        List of chunk dictionaries with clean text boundaries and metadata.
    """
    cleaned_text = text.strip()
    if not cleaned_text:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be strictly smaller than chunk_size")

    chunks = []
    start = 0
    text_length = len(cleaned_text)
    current_id = start_chunk_id

    while start < text_length:
        end = start + chunk_size

        if end >= text_length:
            chunk_content = cleaned_text[start:].strip()
            if len(chunk_content) >= min_chunk_chars or not chunks:
                chunks.append({
                    "chunk_id": current_id,
                    "text": chunk_content,
                    "page_number": page_number,
                    "source": source,
                    "char_count": len(chunk_content),
                })
            elif chunks and chunk_content:
                # Merge tiny trailing fragment into the previous chunk
                chunks[-1]["text"] += " " + chunk_content
                chunks[-1]["char_count"] = len(chunks[-1]["text"])
            break

        # Search for natural break points in order of semantic preference:
        # Paragraph break (\n\n) -> Line break (\n) -> Sentence end (. / ? / !) -> Word space (' ')
        split_point = -1
        for delimiter in ["\n\n", "\n", ". ", "? ", "! ", " "]:
            pos = cleaned_text.rfind(delimiter, start + chunk_overlap, end)
            if pos != -1:
                split_point = pos + len(delimiter)
                break

        # Fallback if no natural delimiter is found in window
        if split_point == -1:
            split_point = end

        chunk_content = cleaned_text[start:split_point].strip()
        if len(chunk_content) >= min_chunk_chars:
            chunks.append({
                "chunk_id": current_id,
                "text": chunk_content,
                "page_number": page_number,
                "source": source,
                "char_count": len(chunk_content),
            })
            current_id += 1

        # Calculate next start position using overlap
        next_start = split_point - chunk_overlap

        # Snap start forward to word boundary so chunks never start mid-word
        if next_start > start and next_start < text_length:
            if not cleaned_text[next_start - 1].isspace():
                space_pos = cleaned_text.find(" ", next_start)
                if space_pos != -1 and space_pos < split_point:
                    next_start = space_pos + 1
        
        start = max(start + 1, next_start)

    return chunks


def load_and_chunk_pdf(
    pdf_path: str | Path,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    min_chunk_chars: int = 50,
) -> List[Dict[str, Any]]:
    """Convenience function: Loads all pages from a PDF and returns indexed chunks.

    Args:
        pdf_path: Path to the target PDF file.
        chunk_size: Maximum character count per chunk.
        chunk_overlap: Character overlap across adjacent chunks.
        min_chunk_chars: Minimum character length per chunk.

    Returns:
        List of chunk dictionaries with globally incremented chunk_ids.
    """
    pages = load_pdf(pdf_path)
    all_chunks = []
    chunk_counter = 0

    for page in pages:
        page_chunks = chunk_text(
            text=page["text"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_chars=min_chunk_chars,
            page_number=page["page_number"],
            source=page["source"],
            start_chunk_id=chunk_counter,
        )
        all_chunks.extend(page_chunks)
        chunk_counter += len(page_chunks)

    return all_chunks


if __name__ == "__main__":
    pdf_path = Path("data/raw/sample_rag_paper.pdf")
    if pdf_path.exists():
        print(f"Loading and chunking PDF: {pdf_path}")
        chunks = load_and_chunk_pdf(pdf_path, chunk_size=350, chunk_overlap=70)
        print(f"Successfully created {len(chunks)} chunks from PDF:\n")
        for ch in chunks:
            print(f"[Chunk #{ch['chunk_id']} | Page {ch['page_number']} | Length: {ch['char_count']} chars]")
            print(f"\"{ch['text']}\"\n")
