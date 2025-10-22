from sentence_transformers import SentenceTransformer
import numpy as np

# Charge le modèle (téléchargera automatiquement la première fois)
MODEL_NAME = "all-MiniLM-L6-v2"
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def embed_text(text: str):
    model = get_model()
    vec = model.encode(text, normalize_embeddings=True)
    return vec
