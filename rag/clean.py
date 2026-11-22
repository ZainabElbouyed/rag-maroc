"""Étape 3b : nettoyage (en-têtes/pieds de page, numéros, césures, retours à la ligne)."""
import math
import re
import unicodedata
from collections import Counter

PAGE_NUM_RE = re.compile(r"^\s*[-–—]*\s*(?:page\s+)?\d{1,4}(?:\s*(?:/|sur)\s*\d{1,4})?\s*[-–—]*\s*$", re.I)
ARTICLE_START = re.compile(r"^\s*(?:Article|ARTICLE)\s+(?:premier|1er|\d+)")
ARTICLE_ALONE = re.compile(r"^\s*(?:Article|ARTICLE)\s+(?:premier|1er|\d+)(?:\s*(?:bis|ter|quater))?\s*[.:]?\s*$")
HEADER_ADALA = re.compile(r"^\s*Royaume du Maroc\b.{0,12}Minist", re.I)
FOOTNOTE_BO = re.compile(r"^\s*\d{1,2}\s*-\s*Bulletin officiel\b", re.I)
PRIVATE_USE = re.compile(r"[\uf000-\uf8ff]")
INTERTITRE = "INTERTITRE : "
STRUCT_START = re.compile(
    r"^\s*(?:(?:TITRE|CHAPITRE|SECTION|SOUS-SECTION|ANNEXE|INTERTITRE)\b"
    r"|(?:Titre|Chapitre|Section)\s+(?:premier|préliminaire|[IVXLC]+\b|\d+))")
LIST_START = re.compile(r"^\s*(?:\d{1,2}\s*[°\)]|[a-z]\)|[-–—•·]\s)")
TERMINAL = re.compile(r"[.;:!?»”\"]\s*$")


def _norm_repeat(line: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", line.lower())).strip()


def detect_repeated_lines(pages: list[str], edge: int = 3, min_ratio: float = 0.5) -> set[str]:
    """Lignes (chiffres neutralisés) qui reviennent en haut/bas de ≥ 50 % des pages."""
    if len(pages) < 2:
        return set()
    counter: Counter = Counter()
    for p in pages:
        lines = [l.strip() for l in p.splitlines() if l.strip()]
        counter.update({_norm_repeat(l) for l in lines[:edge] + lines[-edge:]
                        if not (ARTICLE_START.match(l) or STRUCT_START.match(l))})
    thr = max(2, math.ceil(min_ratio * len(pages)))
    return {k for k, v in counter.items() if v >= thr and len(k) > 3}


def _strip_edges(page: str, repeated: set[str]) -> str:
    lines = page.splitlines()
    idx = [i for i, l in enumerate(lines) if l.strip()]
    edge_idx = set(idx[:3] + idx[-3:])
    kept = []
    for i, l in enumerate(lines):
        s = l.strip()
        if ARTICLE_START.match(s) or STRUCT_START.match(s):
            kept.append(l)                # jamais supprimer un titre d'article ou de structure
            continue
        if i in edge_idx and (_norm_repeat(s) in repeated or PAGE_NUM_RE.match(s)):
            continue
        if FOOTNOTE_BO.match(s) or (i in edge_idx and HEADER_ADALA.match(s)):
            continue
        kept.append(l)
    return "\n".join(kept)


def fix_hyphenation(text: str) -> str:
    text = text.replace("\u00ad\n", "").replace("\u00ad", "")
    return re.sub(r"(?<=[a-zà-ÿ])-\n(?=[a-zà-ÿ])", "", text)


def _starts_new_block(line: str, prev: str) -> bool:
    if STRUCT_START.match(line):
        return True
    if ARTICLE_ALONE.match(line):
        return True                      # « Article 12 » seul sur sa ligne : toujours un nouvel article
    if ARTICLE_START.match(line):
        return bool(TERMINAL.search(prev)) or bool(STRUCT_START.match(prev))
    if LIST_START.match(line):
        return bool(re.search(r"[.;:]\s*$", prev))
    return False


def _find_intertitres(lines: list[str]) -> set[int]:
    """Indices des premières lignes d'intertitres : lignes sans ponctuation finale
    situées juste avant un « Article N » seul sur sa ligne (ex. « De l'organisation du Parlement »)."""
    starts: set[int] = set()
    for i, line in enumerate(lines):
        if not ARTICLE_ALONE.match(line):
            continue
        j, collected = i - 1, []
        while j >= 0 and len(collected) < 5:
            l = lines[j]
            if TERMINAL.search(l) or STRUCT_START.match(l) or ARTICLE_START.match(l):
                break
            collected.append(j)
            j -= 1
        if not collected:
            continue
        # lignes en MAJUSCULES juste sous « TITRE … » = suite du titre, pas un intertitre
        if j >= 0 and STRUCT_START.match(lines[j]) and all(lines[k].isupper() for k in collected):
            continue
        starts.add(collected[-1])
    return starts


def join_lines(text: str) -> str:
    lines = [l.strip() for l in text.splitlines()]
    nonempty = [l for l in lines if l]
    inter = _find_intertitres(nonempty)
    out: list[str] = []
    k = -1
    for line in lines:
        if not line:
            out.append("")
            continue
        k += 1
        if k in inter:
            out.append("\x01" + line)
        elif not out or out[-1] == "":
            out.append(line)
        elif _starts_new_block(line, out[-1].lstrip("\x01")) and not out[-1].startswith("\x01") \
                or (out[-1].startswith("\x01") and ARTICLE_ALONE.match(line)):
            out.append(line)
        elif STRUCT_START.match(line):
            out.append(line)
        else:
            out[-1] = out[-1] + " " + line
    return "\n".join(l.replace("\x01", INTERTITRE) for l in out if l)


def clean_pages(pages: list[str]) -> list[str]:
    pages = [PRIVATE_USE.sub("–", unicodedata.normalize("NFKC", p)) for p in pages]
    repeated = detect_repeated_lines(pages)
    cleaned = []
    for p in pages:
        p = _strip_edges(p, repeated)
        p = fix_hyphenation(p)
        p = join_lines(p)
        p = re.sub(r"[ \t]+", " ", p)
        cleaned.append(p.strip())
    return cleaned
