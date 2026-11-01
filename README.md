# Chat with your PDFs: local RAG from scratch
 
Ask questions about your PDFs and get answers grounded in them, with page citations.
Runs fully offline on a Mac with Ollama. The RAG pipeline is hand-written (no LangChain),
and every answer shows the retrieved chunks, their similarity scores and the exact prompt.
 

https://github.com/user-attachments/assets/ce5824c9-452d-4517-ab52-e6074a054a49


 
## Architecture
 
```mermaid
flowchart LR
  subgraph Indexing
    A[PDF] --> B[Load: pypdf] --> C[Chunk: 800 chars, 20% overlap] --> D[Embed: nomic-embed-text] --> E[(Vector store: numpy)]
  end
  subgraph Asking
    Q[Question] --> F[Embed query] --> G[Top-k cosine search] --> H{score >= min?}
    H -- yes --> I[Grounded prompt to llama3.2:3b] --> J[Answer with citations]
    H -- no --> K[I couldn't find that]
  end
  E --> G
```
 
## Tech stack
Python · Ollama · Llama 3.2 (3B) · nomic-embed-text · NumPy · Streamlit · pypdf · pytest
 
## Key design decisions
- **No framework:** every RAG step is written by hand, so each one can be explained and tuned.
- **Own vector store:** normalized embeddings in a NumPy matrix, so search is a single matrix multiply (exact cosine similarity).
- **Two layers against hallucination:** a similarity threshold refuses before the LLM is called, and a grounded prompt forbids outside knowledge and requires page citations.
- **Explainable by design:** the UI shows the retrieved chunks, scores and prompt, so a wrong answer can be traced to retrieval or to generation.
- **Tested offline:** chunking, search ranking, persistence and the refusal path are unit-tested with fake vectors.
 
## What I learned
- [Your finding from the chunk-size experiment]
- [Your finding from the threshold or top-k experiment]
- Math symbols in research PDFs came out of extraction as invalid Unicode and crashed embedding.
  I fixed it with a normalization step in ingestion and a regression test.
 
## Source code
The source is in a private repository and is available on request. Message me on [LinkedIn](YOUR-LINKEDIN-URL).
