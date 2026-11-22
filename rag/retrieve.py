"""Étape 6a + 7 : recherche dense / BM25 / hybride (RRF), filtres, reranking optionnel."""
import numpy as np

from . import config


class Reranker:
    """Cross-encoder optionnel (nécessite sentence-transformers + téléchargement du modèle)."""
    def __init__(self, name: str = "BAAI/bge-reranker-v2-m3"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(name, max_length=512)

    def rerank(self, query: str, hits: list[dict]) -> list[dict]:
        scores = self.model.predict([(query, h["chunk"]["text"]) for h in hits])
        for h, s in zip(hits, scores):
            h["score"] = float(s)
        return sorted(hits, key=lambda h: -h["score"])


def _ranks(scores: np.ndarray, mask: np.ndarray) -> dict[int, int]:
    idx = [i for i in np.argsort(-scores) if mask[i]]
    return {int(i): r + 1 for r, i in enumerate(idx)}


class Retriever:
    def __init__(self, index, mode: str = config.DEFAULT_MODE, reranker: Reranker | None = None):
        self.index, self.mode, self.reranker = index, mode, reranker

    def _mask(self, filters: dict | None) -> np.ndarray:
        mask = np.ones(len(self.index.chunks), dtype=bool)
        for key, allowed in (filters or {}).items():
            if allowed:
                allowed = set(allowed)
                mask &= np.array([c.get(key) in allowed for c in self.index.chunks])
        return mask

    def search(self, query: str, k: int = config.DEFAULT_K, mode: str | None = None,
               filters: dict | None = None) -> list[dict]:
        mode = mode or self.mode
        mask = self._mask(filters)
        dense = self.index.dense_scores(query) if mode in ("dense", "hybrid") else None
        bm = self.index.bm25.scores(query) if mode in ("bm25", "hybrid") else None
        if mode == "dense":
            final = dense.copy()
        elif mode == "bm25":
            final = bm / (bm.max() or 1.0)
        else:
            rd, rb = _ranks(dense, mask), _ranks(bm, mask)
            final = np.zeros(len(self.index.chunks))
            for i in rd:
                final[i] += 1.0 / (config.RRF_K + rd[i])
            for i in rb:
                if bm[i] > 0:                       # pas de vote BM25 sans terme commun
                    final[i] += 1.0 / (config.RRF_K + rb[i])
        final = np.where(mask, final, -np.inf)
        n_cand = k * 4 if self.reranker else k
        order = [int(i) for i in np.argsort(-final)[:n_cand] if np.isfinite(final[i])]
        hits = [{"chunk": self.index.chunks[i], "score": float(final[i]),
                 "dense": float(dense[i]) if dense is not None else None,
                 "bm25": float(bm[i]) if bm is not None else None} for i in order]
        if self.reranker and hits:
            hits = self.reranker.rerank(query, hits)[:k]
        for r, h in enumerate(hits, 1):
            h["rank"] = r
        return hits
