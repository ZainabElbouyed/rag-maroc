"""Configuration centrale du projet."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
PROCESSED_DIR = DATA_DIR / "processed"
TRACKING_CSV = DATA_DIR / "corpus_tracking.csv"
EVAL_CSV = DATA_DIR / "eval_questions.csv"
INDEX_DIR = ROOT / "index"
RESULTS_DIR = ROOT / "results"
REPORT_DIR = ROOT / "report"

# --- Embeddings -------------------------------------------------------------
# "tfidf"  : baseline lexicale, 100 % hors ligne, aucune dépendance lourde
# "st:..." : modèle sentence-transformers (téléchargé depuis Hugging Face)
EMBEDDING_BASELINE = "tfidf"
EMBEDDING_E5 = "st:intfloat/multilingual-e5-small"   # léger (~470 Mo)
EMBEDDING_BGE = "st:BAAI/bge-m3"                     # meilleur, lourd (~2,2 Go)

# --- Découpage ----------------------------------------------------------------
MAX_ARTICLE_CHARS = 1500      # au-delà, un article est redécoupé
FIXED_CHUNK_CHARS = 600       # variante de comparaison (découpage à taille fixe)
FIXED_OVERLAP_CHARS = 100

# --- Recherche / génération ------------------------------------------------
DEFAULT_MODE = "hybrid"       # dense | bm25 | hybrid
DEFAULT_K = 5
DEFAULT_CHUNKING = "article"  # article | fixed
RRF_K = 60
# Mode sans LLM : si moins de cette part des termes informatifs de la question
# se retrouve dans les passages, le système répond « je ne sais pas ».
MIN_COVERAGE_EXTRACTIVE = 0.5

# --- Critères de succès (cadrage) --------------------------------------------
SUCCESS_CRITERIA = {
    "hit_at_5": 0.80,          # le bon article est dans le top-5
    "correct_rate": 0.70,      # réponses jugées correctes (questions répondables)
    "abstain_rate": 0.80,      # refus correct sur les questions hors corpus
}

ABSTAIN_TEXT = "Je ne sais pas : l'information demandée ne figure pas dans les textes du corpus."
