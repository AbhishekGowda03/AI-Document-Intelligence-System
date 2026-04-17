from sentence_transformers import SentenceTransformer
import torch

class EmbeddingService:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

    def get_embeddings(self, texts: list[str]):
        """Generates embeddings for a list of text chunks."""
        # Returns a numpy array by default
        return self.model.encode(texts, show_progress_bar=False)

    def get_embedding(self, text: str):
        """Generates embedding for a single text string."""
        return self.model.encode(text, show_progress_bar=False)
