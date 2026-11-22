"""Étape 5b : index vectoriel (cosinus, numpy/scipy) + index BM25 maison."""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse

from . import config
from .embed import get_embedder, slug
from .text_utils import tokenize


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.inv: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lens = np.zeros(self.n)
        for i, d in enumerate(docs):
            toks = tokenize(d)
            self.lens[i] = len(toks)
            tf: dict[str, int] = defaultdict(int)
            for t in toks:
                tf[t] += 1
            for t, f in tf.items():
                self.inv[t].append((i, f))
        self.avg = float(self.lens.mean()) if self.n else 0.0

    def df_ratio(self, term: str) -> float:
        return len(self.inv.get(term, ())) / max(self.n, 1)

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(self.n)
        for t in set(tokenize(query)):
            post = self.inv.get(t)
            if not post:
                continue
            idf = math.log(1 + (self.n - len(post) + 0.5) / (len(post) + 0.5))
            for i, f in post:
                denom = f + self.k1 * (1 - self.b + self.b * self.lens[i] / (self.avg or 1))
                s[i] += idf * f * (self.k1 + 1) / denom
        return s


class VectorIndex:
    def __init__(self, chunks, embedder, matrix, spec: str, chunking: str):
        self.chunks, self.embedder, self.matrix = chunks, embedder, matrix
        self.spec, self.chunking = spec, chunking
        self.bm25 = BM25([c["embed_text"] for c in chunks])

    @classmethod
    def build(cls, chunks: list[dict], spec: str, chunking: str) -> "VectorIndex":
        emb = get_embedder(spec)
        matrix = emb.fit_docs([c["embed_text"] for c in chunks])
        return cls(chunks, emb, matrix, spec, chunking)

    @staticmethod
    def path_for(spec: str, chunking: str) -> Path:
        return config.INDEX_DIR / f"{slug(spec)}__{chunking}"

    def save(self) -> Path:
        p = self.path_for(self.spec, self.chunking)
        p.mkdir(parents=True, exist_ok=True)
        with open(p / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        (p / "meta.json").write_text(json.dumps(
            {"spec": self.spec, "chunking": self.chunking, "n_chunks": len(self.chunks)}), encoding="utf-8")
        if self.spec != "tfidf":                      # TF-IDF est recalculé au chargement
            np.save(p / "embeddings.npy", self.matrix)
        return p

    @classmethod
    def load(cls, path: Path) -> "VectorIndex":
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        chunks = [json.loads(l) for l in open(path / "chunks.jsonl", encoding="utf-8") if l.strip()]
        emb = get_embedder(meta["spec"])
        if (path / "embeddings.npy").exists():
            matrix = np.load(path / "embeddings.npy")
        else:
            matrix = emb.fit_docs([c["embed_text"] for c in chunks])
        return cls(chunks, emb, matrix, meta["spec"], meta["chunking"])

    def dense_scores(self, query: str) -> np.ndarray:
        q = self.embedder.encode_query(query)
        if sparse.issparse(self.matrix):
            return np.asarray((self.matrix @ q.T).todense()).ravel()
        return np.asarray(self.matrix @ q).ravel()


def list_indexes() -> list[str]:
    if not config.INDEX_DIR.exists():
        return []
    return sorted(p.name for p in config.INDEX_DIR.iterdir() if (p / "meta.json").exists())
