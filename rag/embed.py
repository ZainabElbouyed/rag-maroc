"""Étape 5a : modèles d'embeddings (TF-IDF hors ligne ou sentence-transformers)."""
from sklearn.feature_extraction.text import TfidfVectorizer

from .text_utils import tokenize


def _analyzer(text: str) -> list[str]:
    toks = tokenize(text)
    return toks + [a + "_" + b for a, b in zip(toks, toks[1:])]


class TfidfEmbedder:
    name = "tfidf"

    def __init__(self):
        self.vec = TfidfVectorizer(analyzer=_analyzer, sublinear_tf=True)

    def fit_docs(self, texts):
        return self.vec.fit_transform(texts)

    def encode_query(self, q: str):
        return self.vec.transform([q])


class STEmbedder:
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("pip install sentence-transformers") from exc
        self.name = "st:" + model_name
        self.model = SentenceTransformer(model_name)
        try:
            self.model.max_seq_length = min(int(self.model.max_seq_length or 512), 512)
        except Exception:  # noqa: BLE001
            pass
        self.e5 = "e5" in model_name.lower()

    def fit_docs(self, texts):
        pre = "passage: " if self.e5 else ""
        return self.model.encode([pre + t for t in texts], batch_size=16,
                                 normalize_embeddings=True, show_progress_bar=True)

    def encode_query(self, q: str):
        pre = "query: " if self.e5 else ""
        return self.model.encode([pre + q], normalize_embeddings=True)[0]


def get_embedder(spec: str):
    if spec == "tfidf":
        return TfidfEmbedder()
    if spec.startswith("st:"):
        return STEmbedder(spec[3:])
    raise ValueError(f"Embedding inconnu : {spec}")


def slug(spec: str) -> str:
    return spec.replace(":", "-").replace("/", "_")
