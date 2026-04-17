import sys
sys.path.insert(0, ".")
try:
    from services.pdf_parser import extract_text_from_pdf
    print("pdf_parser OK")
    from services.cleaner import clean_text
    print("cleaner OK")
    from services.chunker import chunk_text
    print("chunker OK")
    from services.embeddings import EmbeddingService
    print("embeddings OK")
    from services.vector_store import VectorStore
    print("vector_store OK")
    from services.retrieval import RetrievalService
    print("retrieval OK")
    print("All imports OK")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
