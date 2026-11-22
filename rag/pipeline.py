"""Pipeline RAG complet : question -> recherche -> génération -> réponse + sources."""
import re

from . import config
from .chunk import make_chunks
from .corpus import build_corpus
from .generate import (SYSTEM_PROMPT, build_user_prompt, extractive_answer, get_llm,
                       is_abstention)
from .index import VectorIndex
from .retrieve import Retriever
from .text_utils import tokenize

_CORPUS_CACHE: list[dict] | None = None


def get_corpus(**kw) -> list[dict]:
    global _CORPUS_CACHE
    if _CORPUS_CACHE is None:
        _CORPUS_CACHE = build_corpus(**kw)
    return _CORPUS_CACHE


def ensure_index(spec: str = config.EMBEDDING_BASELINE, chunking: str = config.DEFAULT_CHUNKING,
                 rebuild: bool = False, verbose: bool = True) -> VectorIndex:
    path = VectorIndex.path_for(spec, chunking)
    if (path / "meta.json").exists() and not rebuild:
        return VectorIndex.load(path)
    if verbose:
        print(f"[index] construction {spec} / {chunking} …")
    chunks = make_chunks(get_corpus(verbose=verbose), chunking)
    idx = VectorIndex.build(chunks, spec, chunking)
    idx.save()
    if verbose:
        print(f"[index] {len(chunks)} passages -> {path}")
    return idx


class RAGPipeline:
    def __init__(self, index: VectorIndex, mode: str = config.DEFAULT_MODE, k: int = config.DEFAULT_K,
                 llm: str = "auto", reranker=None, min_coverage: float | None = None):
        self.index, self.mode, self.k = index, mode, k
        self.retriever = Retriever(index, mode, reranker)
        self.llm_pref = llm
        self.llm = get_llm(llm)
        # Sans LLM, on filtre nous-mêmes les questions hors corpus ; avec LLM, c'est le prompt qui décide.
        self.min_coverage = (config.MIN_COVERAGE_EXTRACTIVE if self.llm is None else 0.0) \
            if min_coverage is None else min_coverage

    @property
    def llm_name(self) -> str:
        return self.llm.name if self.llm else "extractif (sans LLM)"

    def _idf(self, term: str) -> float:
        import math
        return math.log(1 + 1 / (self.index.bm25.df_ratio(term) + 1e-3))

    def coverage(self, question: str, hits: list[dict]) -> float:
        """Part des termes informatifs (peu fréquents dans le corpus) retrouvés dans le top-3."""
        terms = {t for t in tokenize(question) if not t[0].isdigit()}
        informative = [t for t in terms if self.index.bm25.df_ratio(t) < 0.2]
        if not informative:
            return 1.0
        found: set[str] = set()
        for h in hits[:3]:
            found |= set(tokenize(h["chunk"]["text"]))
        return sum(t in found for t in informative) / len(informative)

    def reformulate(self, question: str) -> str:
        if not self.llm:
            return question
        try:
            out = self.llm.generate("Réécris la question pour une recherche dans des textes de loi marocains "
                                    "(vocabulaire juridique, sans réponse). Réponds par la question seule.",
                                    question, max_tokens=120)
            return out.strip() or question
        except Exception:  # noqa: BLE001
            return question

    def ask(self, question: str, k: int | None = None, mode: str | None = None,
            filters: dict | None = None, reformulate: bool = False) -> dict:
        k = k or self.k
        query = self.reformulate(question) if reformulate else question
        hits = self.retriever.search(query, k=k, mode=mode, filters=filters)
        res = {"question": question, "query": query, "llm": self.llm_name, "k": k,
               "mode": mode or self.mode, "warning": "", "coverage": None}
        if not hits:
            res.update(answer=config.ABSTAIN_TEXT, abstained=True, sources=[], cited=[])
            return res
        cov = self.coverage(question, hits)
        res["coverage"] = round(cov, 2)
        answer, abstained = "", False
        if cov < self.min_coverage:
            answer, abstained = config.ABSTAIN_TEXT, True
        elif self.llm is None:
            answer = extractive_answer(question, hits, self._idf)
        else:
            try:
                answer = self.llm.generate(SYSTEM_PROMPT, build_user_prompt(question, hits))
                abstained = is_abstention(answer)
            except Exception as exc:  # noqa: BLE001
                res["warning"] = f"Erreur LLM ({exc.__class__.__name__}) : repli sur le mode extractif."
                answer = extractive_answer(question, hits, self._idf)
        cited = sorted({int(n) for n in re.findall(r"\[S(\d+)\]", answer)})
        res.update(answer=answer, abstained=abstained, cited=cited,
                   sources=[] if abstained and self.llm is not None else [self._src(h) for h in hits])
        return res

    @staticmethod
    def _src(h: dict) -> dict:
        c = h["chunk"]
        return {"rank": h["rank"], "score": round(h["score"], 4), "doc_id": c["doc_id"],
                "titre": c["titre"], "numero": c["numero"], "date": c["date"],
                "bo_numero": c["bo_numero"], "url": c["url"], "article": c["article"],
                "page": c["page"], "section": c["section"], "text": c["text"]}
