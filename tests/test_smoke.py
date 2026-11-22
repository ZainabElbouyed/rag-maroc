"""Tests rapides :  python -m unittest discover tests"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.chunk import ART_RE, article_label, split_long  # noqa: E402
from rag.clean import clean_pages, fix_hyphenation, join_lines  # noqa: E402
from rag.text_utils import normalize, tokenize  # noqa: E402


class TestClean(unittest.TestCase):
    def test_hyphenation(self):
        self.assertEqual(fix_hyphenation("la dé-\nclaration"), "la déclaration")

    def test_join_keeps_articles(self):
        t = "Article 1. – Le texte se poursuit\nsur deux lignes.\nArticle 2. – Suite."
        self.assertEqual(join_lines(t).splitlines()[0], "Article 1. – Le texte se poursuit sur deux lignes.")
        self.assertEqual(len(join_lines(t).splitlines()), 2)

    def test_headers_footers_removed(self):
        body = lambda i: "\n".join(f"Ligne {w} du corps de la page {i}" for w in ("alpha", "beta", "gamma", "delta", "epsilon", "zeta"))  # noqa: E731
        pages = [f"EN-TÊTE BO\n{body(i)}\nPage {i}" for i in (1, 2, 3)]
        out = " ".join(clean_pages(pages))
        self.assertNotIn("EN-TÊTE", out)
        self.assertNotIn("Page 2", out)
        self.assertIn("Ligne gamma du corps de la page 2", out)


class TestChunk(unittest.TestCase):
    def test_labels(self):
        for txt, lab in [("Article premier. – x", "1"), ("Article 12 bis : x", "12 bis"), ("ARTICLE 7 x", "7")]:
            self.assertEqual(article_label(ART_RE.match(txt)), lab)

    def test_split_long(self):
        parts = split_long(". ".join(["Phrase " + "mot " * 20] * 30), 400)
        self.assertGreater(len(parts), 1)


class TestText(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize("L’Été  À"), "l'ete a")

    def test_tokenize_plural(self):
        self.assertEqual(tokenize("salariés"), tokenize("salarié"))


class TestPipeline(unittest.TestCase):
    """Nécessite data/raw_pdfs/constitution.pdf (fourni)."""

    def test_end_to_end(self):
        from rag.chunk import make_chunks
        from rag.corpus import build_corpus
        from rag.pipeline import RAGPipeline, ensure_index
        from rag.retrieve import Retriever
        docs = build_corpus(verbose=False, save_processed=False)
        arts = [c for c in make_chunks(docs, "article") if c["article"] not in ("Préambule",)]
        self.assertGreaterEqual(len({c["article"] for c in arts}), 180)      # 180 articles retrouvés
        idx = ensure_index()
        hits = Retriever(idx).search("Dans quel délai le Roi promulgue-t-il la loi ?", k=5)
        self.assertIn("50", [a for h in hits for a in h["chunk"]["articles"]])
        pipe = RAGPipeline(idx, llm="extractive")
        self.assertTrue(pipe.ask("Comment obtenir un permis de conduire ?")["abstained"])
        r = pipe.ask("Pour combien d'années les membres de la Chambre des Représentants sont-ils élus ?", k=5)
        self.assertTrue(r["sources"])


if __name__ == "__main__":
    unittest.main()
