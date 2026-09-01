# Stage 0 Notes: Project Setup and Structure

## 1. What did we do in Stage 0?
In this stage, we set up our workspace and project skeleton.
Instead of creating dozens of complicated folders, we chose a simple, flat structure:
- `embeddings/`: Holds our two search methods (Sentence Transformer and our custom Attention encoder) right next to each other.
- `graph/`: Contains the LangGraph code that controls how queries flow through our system step-by-step.
- `data/`: Where we store raw PDF files and training data.
- `notes/`: Short, simple notes for each stage to help you understand the concepts and prepare for job interviews.

---

## 2. Why did we pick these tools?

| Choice | What we picked | Other options | Simple Reason |
| :--- | :--- | :--- | :--- |
| **Python Environment** | **`venv`** | `conda`, `poetry` | It comes built-in with Python. It is fast, clean, and does not require installing any extra software. |
| **PDF Reader** | **`pymupdf` (PyMuPDF)** | `pdfplumber`, `PyPDF2` | It is written in C under the hood, making it 10 to 20 times faster than pure Python tools. It also preserves font and layout details cleanly. |
| **Vector Search** | **`faiss-cpu`** | `chromadb`, raw `numpy` | It is an ultra-fast vector search library by Meta. It finds the closest matching text chunks in less than a millisecond without running a heavy database server in the background. |
| **Workflow Manager** | **`langgraph`** | Pure Python functions | It lets us visualize and control our pipeline like a flowchart (nodes and branches) and pass data cleanly from step to step. |

---

## 3. Interview Questions & Simple Answers

### Q1: Why use PyMuPDF instead of PyPDF2 or pdfplumber?
**Simple Answer:**
- **Speed:** PyMuPDF is written in C (MuPDF), so it reads large PDFs 10x to 20x faster.
- **Accurate Text:** It extracts text along with font sizes and line positions, which helps us identify headings, paragraphs, and sections properly.

### Cross-Q: What happens if a PDF is just scanned images with no real text?
**Simple Answer:**
- If a page has scanned images instead of text, normal PDF readers will return empty text.
- **How to fix:** We check if the extracted text is empty or very short. If it is, we run OCR (Optical Character Recognition, like Tesseract) or a vision model on the page image to read the text from the picture.

### Cross-Q: Why keep the folder structure simple and flat?
**Simple Answer:**
- For experiments and learning, a flat structure lets us see everything at a glance.
- Both retrievers sit side-by-side in `embeddings/`, and the graph logic sits in `graph/`. This makes it super easy to compare how both work without jumping across 10 nested folders.
