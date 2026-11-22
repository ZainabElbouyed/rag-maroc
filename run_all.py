"""Lance toute la chaîne : corpus -> extraction/nettoyage -> chunking -> index -> évaluation -> rapport.
Usage :  python run_all.py [--embeddings tfidf,st:intfloat/multilingual-e5-small] [--full] [--llm auto|extractive]"""
import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rag import config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default="", help="liste séparée par des virgules")
    ap.add_argument("--full", action="store_true", help="ajoute bge-m3 (lourd, ~2,2 Go)")
    ap.add_argument("--llm", default="auto", choices=["auto", "extractive", "anthropic", "ollama"])
    ap.add_argument("--rebuild", action="store_true", help="reconstruit les index")
    args = ap.parse_args()

    if not list(config.RAW_PDF_DIR.glob("*.pdf")):
        sys.exit(f"Aucun PDF dans {config.RAW_PDF_DIR}. Déposez vos PDF et renseignez data/corpus_tracking.csv "
                 "(voir README). Corpus fictif de test : python scripts/make_demo_corpus.py")

    if args.embeddings:
        embs = [e.strip() for e in args.embeddings.split(",") if e.strip()]
    else:
        embs = [config.EMBEDDING_BASELINE]
        if importlib.util.find_spec("sentence_transformers"):
            embs.append(config.EMBEDDING_E5)
            if args.full:
                embs.append(config.EMBEDDING_BGE)
    if args.rebuild:
        import shutil
        shutil.rmtree(config.INDEX_DIR, ignore_errors=True)

    from rag.evaluate import run_evaluation
    print("== Extraction, nettoyage, chunking, indexation, évaluation ==")
    out = run_evaluation(embs, llm=args.llm)
    g, b = out["gen"], out["best"]
    print("\n== Résumé ==")
    print(f"Meilleure config : {b['embedding']} / {b['chunking']} / {b['mode']}  (hit@5={b['hit@5']:.0%}, MRR={b['mrr']:.2f})")
    print(f"Génération ({g['llm']}) : correct={g['correct_rate']:.0%}  partiel={g['partial_rate']:.0%}  "
          f"refus correct hors corpus={g['abstain_rate']:.0%}")
    print("Fichiers : results/*.csv, results/resultats.json, report/rapport.md, report/script_demo.md")
    print("Interface : streamlit run app.py")


if __name__ == "__main__":
    main()
