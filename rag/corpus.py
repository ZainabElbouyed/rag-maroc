"""Étapes 2-3 : suivi du corpus (CSV) + extraction + nettoyage de tous les PDF."""
import csv
from pathlib import Path

from . import config
from .clean import clean_pages
from .extract import extract_pdf

TRACKING_FIELDS = ["doc_id", "titre", "numero", "date", "bo_numero", "url", "langue",
                   "domaine", "fichier", "source", "note"]


def read_tracking() -> list[dict]:
    if not config.TRACKING_CSV.exists():
        return []
    with open(config.TRACKING_CSV, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_corpus(ocr: bool = True, save_processed: bool = True, verbose: bool = True) -> list[dict]:
    """Retourne une liste de documents : {meta, text, page_starts, n_pages, n_ocr}."""
    rows = {r["fichier"]: r for r in read_tracking() if r.get("fichier")}
    pdfs = sorted(Path(config.RAW_PDF_DIR).glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"Aucun PDF dans {config.RAW_PDF_DIR}. Lancez `python scripts/make_demo_corpus.py` "
            "ou déposez vos PDF et renseignez data/corpus_tracking.csv.")
    for fichier in rows:
        if not (config.RAW_PDF_DIR / fichier).exists() and verbose:
            print(f"  [!] {fichier} est dans le tableau de suivi mais absent de {config.RAW_PDF_DIR}")
    docs = []
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for pdf in pdfs:
        meta = dict(rows.get(pdf.name) or {})
        if not meta:
            if verbose:
                print(f"  [!] {pdf.name} absent du tableau de suivi : métadonnées par défaut")
            meta = {"doc_id": pdf.stem, "titre": pdf.stem.replace("_", " "), "numero": "",
                    "date": "", "bo_numero": "", "url": "", "langue": "fr", "fichier": pdf.name}
        pages, n_ocr = extract_pdf(pdf, ocr=ocr)
        cleaned = clean_pages(pages)
        starts, pos, parts = [], 0, []
        for c in cleaned:
            starts.append(pos)
            parts.append(c)
            pos += len(c) + 1
        text = "\n".join(parts)
        if verbose:
            print(f"  {meta['doc_id']:>4} | {len(pages)} p. | {len(text):>6} car. | OCR: {n_ocr}")
        if len(text.strip()) < 50:
            print(f"  [!] {pdf.name} : texte quasi vide (PDF scanné ? installez Tesseract + pytesseract)")
        if save_processed:
            (config.PROCESSED_DIR / f"{meta['doc_id']}.txt").write_text(text, encoding="utf-8")
        docs.append({"meta": meta, "text": text, "page_starts": starts,
                     "n_pages": len(pages), "n_ocr": n_ocr})
    return docs
