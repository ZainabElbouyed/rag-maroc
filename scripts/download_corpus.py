"""Télécharge les PDF listés dans data/corpus_tracking.csv (colonne `url` renseignée, `fichier` vide ou absent).
Usage : python scripts/download_corpus.py
Ne fait AUCUN scraping : vous collez vous-mêmes les liens de PDF (SGG, Adala…) dans le CSV.
Vérifiez les conditions de réutilisation des sites sources."""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import requests  # noqa: E402

from rag import config  # noqa: E402
from rag.corpus import TRACKING_FIELDS  # noqa: E402


def main():
    rows = list(csv.DictReader(open(config.TRACKING_CSV, encoding="utf-8", newline="")))
    config.RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)
    for r in rows:
        url, fichier = (r.get("url") or "").strip(), (r.get("fichier") or "").strip() or f"{r['doc_id']}.pdf"
        r["fichier"] = fichier
        dest = config.RAW_PDF_DIR / fichier
        if not url or dest.exists():
            continue
        try:
            resp = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0 (projet-etudiant-rag)"})
            resp.raise_for_status()
            if not resp.content.startswith(b"%PDF"):
                raise ValueError("le contenu n'est pas un PDF (page HTML ? lien indirect ?)")
            dest.write_bytes(resp.content)
            print(f"OK   {r['doc_id']} -> {fichier}")
        except Exception as exc:  # noqa: BLE001
            print(f"ÉCHEC {r['doc_id']} ({url}) : {exc}  -> téléchargez-le à la main dans {config.RAW_PDF_DIR}")
    with open(config.TRACKING_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRACKING_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
