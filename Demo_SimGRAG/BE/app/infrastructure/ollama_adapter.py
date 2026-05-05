from sentence_transformers import SentenceTransformer

class OllamaEmbeddingAdapter:
    def __init__(self, model_name="nomic-ai/nomic-embed-text-v1", device="cpu"):
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=device)

    def embed_query(self, text: str) -> list[float]:
        try:
            # Prefix with 'search_query: ' as recommended for Nomic embeddings
            prompt = f"search_query: {text}"
            emb = self.model.encode(prompt, normalize_embeddings=True)
            return emb.tolist()
        except Exception as e:
            print(f"Embedding error: {e}")
            return [0.0] * 768
