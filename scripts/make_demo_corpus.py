"""Génère le corpus de démonstration FICTIF : PDF (avec en-têtes, pieds de page, césures),
tableau de suivi et questions d'évaluation dans data/demo_corpus/ (ne touche pas à votre corpus).
Usage : python scripts/make_demo_corpus.py"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from demo_corpus_data import DOCS, QUESTIONS  # noqa: E402
from rag import config  # noqa: E402
from rag.corpus import TRACKING_FIELDS  # noqa: E402

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase.pdfmetrics import stringWidth
    from reportlab.pdfgen import canvas
except ImportError:
    sys.exit("Installez reportlab : pip install reportlab")

FONT, BOLD, SIZE, LEAD, LINES_PER_PAGE = "Helvetica", "Helvetica-Bold", 10, 14, 18
MARGIN, WIDTH = 60, A4[0] - 120


def wrap(text, hyphen):
    lines, cur = [], ""
    for w in text.split():
        trial = (cur + " " + w).strip()
        if stringWidth(trial, FONT, SIZE) <= WIDTH:
            cur = trial
            continue
        if hyphen and w.isalpha() and len(w) >= 9 and cur:
            for cut in range(len(w) - 3, 3, -1):
                part = w[:cut] + "-"
                if stringWidth(cur + " " + part, FONT, SIZE) <= WIDTH:
                    lines.append(cur + " " + part)
                    cur = w[cut:]
                    break
            else:
                lines.append(cur)
                cur = w
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def render(doc, path):
    body = [(BOLD, f"{doc['kind'].upper()} N° {doc['numero']}"), (FONT, doc["objet"]),
            (FONT, f"Promulguée le {doc['date']} - publiée au Bulletin officiel n° {doc['bo']}"), (FONT, "")]
    n = 0
    for item in doc["items"]:
        if item[0] == "T":
            body += [(FONT, ""), (BOLD, item[1])]
            continue
        n += 1
        head = "Article premier" if n == 1 else f"Article {n}"
        if doc["alone"]:
            body += [(BOLD, head + " :")] + [(FONT, l) for l in wrap(item[1], doc["hyphen"])]
        else:
            body += [(FONT, l) for l in wrap(f"{head}. – {item[1]}", doc["hyphen"])]
    pages = [body[i:i + LINES_PER_PAGE] for i in range(0, len(body), LINES_PER_PAGE)]
    c = canvas.Canvas(str(path), pagesize=A4)
    c.setTitle(f"{doc['kind']} {doc['numero']} (FICTIF)")
    for pno, lines in enumerate(pages, 1):
        c.setFont(FONT, 8)
        c.drawString(MARGIN, A4[1] - 40, "BULLETIN OFFICIEL – DOCUMENT FICTIF DE DÉMONSTRATION")
        y = A4[1] - 80
        for font, line in lines:
            c.setFont(font, SIZE)
            c.drawString(MARGIN, y, line)
            y -= LEAD
        c.setFont(FONT, 8)
        c.drawString(MARGIN, 40, f"BO n° {doc['bo']} – page {pno}")
        c.showPage()
    c.save()


DEMO_DIR = config.DATA_DIR / "demo_corpus"


def main():
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in DOCS:
        render(d, DEMO_DIR / d["fichier"])
        rows.append({"doc_id": d["doc_id"],
                     "titre": f"{d['kind']} n° {d['numero']} {d['objet']} (FICTIF)",
                     "numero": d["numero"], "date": d["date"], "bo_numero": d["bo"], "url": "",
                     "langue": "fr", "domaine": "Droit du travail", "fichier": d["fichier"],
                     "source": "Corpus de démonstration fictif",
                     "note": "Texte inventé : ne pas citer comme droit marocain"})
    with open(DEMO_DIR / "corpus_tracking.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRACKING_FIELDS)
        w.writeheader()
        w.writerows(rows)
    with open(DEMO_DIR / "eval_questions.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "mots_cles", "doc_id", "article", "repondable"])
        for i, (q, kw, doc, art, ok) in enumerate(QUESTIONS, 1):
            w.writerow([f"Q{i:02d}", q, kw, doc, art, ok])
    print(f"{len(DOCS)} PDF fictifs + suivi + {len(QUESTIONS)} questions -> {DEMO_DIR}\n"
          "Pour les utiliser : copiez les PDF dans data/raw_pdfs/ et les 2 CSV dans data/ (remplace votre corpus !).")


if __name__ == "__main__":
    main()
