"""Étape 4 : découpage par structure juridique (article) ou à taille fixe (comparaison)."""
import re
from bisect import bisect_right

from . import config

PREAMBLE_MIN_CHARS = 300
ART_RE = re.compile(
    r"^[ \t]*(?:Article|ARTICLE)\s+(premier|1er|\d+)(?:\s*(bis|ter|quater))?[ \t.:\-–—]*", re.M)
STRUCT_RE = re.compile(
    r"^[ \t]*(TITRE|CHAPITRE|SECTION|SOUS-SECTION|INTERTITRE|Titre|Chapitre|Section)\s+[^\n]*$", re.M)


def article_label(m: re.Match) -> str:
    num = "1" if m.group(1) in ("premier", "1er") else m.group(1)
    return f"{num} {m.group(2)}" if m.group(2) else num


def body_without_heading(text: str) -> str:
    return ART_RE.sub("", text, count=1).strip()


def _page_of(pos: int, page_starts: list[int]) -> int:
    return max(1, bisect_right(page_starts, pos))


def _squash(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def split_long(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    sents = re.split(r"(?<=[.;!?])\s+", text)
    pieces, cur = [], []
    for s in sents:
        if cur and len(" ".join(cur)) + len(s) + 1 > max_chars:
            pieces.append(" ".join(cur))
            cur = [cur[-1], s]      # chevauchement d'une phrase
        else:
            cur.append(s)
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def _mk(meta, cid, article, articles, section, page, text, part=0):
    titre = meta.get("titre", "")
    head = f"{titre} — Article {article}" if article and article != "Préambule" else titre
    ctx = f" ({section})" if section else ""
    return {
        "chunk_id": cid, "doc_id": meta["doc_id"], "titre": titre,
        "numero": meta.get("numero", ""), "date": meta.get("date", ""),
        "bo_numero": meta.get("bo_numero", ""), "url": meta.get("url", ""),
        "article": article, "articles": articles, "section": section,
        "page": page, "part": part, "text": text,
        "embed_text": f"{head}{ctx}\n{text}",
    }


def chunk_by_article(doc: dict, max_chars: int = config.MAX_ARTICLE_CHARS) -> list[dict]:
    meta, text, starts = doc["meta"], doc["text"], doc["page_starts"]
    bounds = [(m.start(), "article", m) for m in ART_RE.finditer(text)]
    if not bounds:
        return chunk_fixed(doc)
    bounds += [(m.start(), "struct", m) for m in STRUCT_RE.finditer(text)]
    bounds.sort(key=lambda b: b[0])
    chunks: list[dict] = []
    pre = _squash(text[:bounds[0][0]])
    if len(pre) >= PREAMBLE_MIN_CHARS:   # un simple bloc titre n'est pas un passage utile
        for j, piece in enumerate(split_long(pre, max_chars)):
            chunks.append(_mk(meta, f"{meta['doc_id']}#pre" + (f"-p{j+1}" if j else ""),
                              "Préambule", ["Préambule"], "", 1, piece, j))
    ctx = {"titre": "", "chapitre": "", "section": ""}
    for i, (pos, kind, m) in enumerate(bounds):
        end = bounds[i + 1][0] if i + 1 < len(bounds) else len(text)
        if kind == "struct":
            line = _squash(m.group(0))
            key = m.group(1).lower().replace("sous-", "").replace("intertitre", "section")
            ctx[key] = line
            if key == "titre":
                ctx["chapitre"] = ctx["section"] = ""
            elif key == "chapitre":
                ctx["section"] = ""
            continue
        label = article_label(m)
        section = " > ".join(v for v in (ctx["titre"], ctx["chapitre"], ctx["section"]) if v)
        body = _squash(text[pos:end])
        for j, piece in enumerate(split_long(body, max_chars)):
            cid = f"{meta['doc_id']}#art{label.replace(' ', '')}" + (f"-p{j+1}" if j else "")
            chunks.append(_mk(meta, cid, label, [label], section, _page_of(pos, starts), piece, j))
    return chunks


def chunk_fixed(doc: dict, size: int = config.FIXED_CHUNK_CHARS,
                overlap: int = config.FIXED_OVERLAP_CHARS) -> list[dict]:
    meta, text, starts = doc["meta"], doc["text"], doc["page_starts"]
    flat = text.replace("\n", " ")  # même longueur : positions conservées
    ams = list(ART_RE.finditer(text))
    spans = [(m.start(), ams[i + 1].start() if i + 1 < len(ams) else len(text), article_label(m))
             for i, m in enumerate(ams)]
    chunks, i, n = [], 0, 0
    while i < len(flat):
        j = min(i + size, len(flat))
        if j < len(flat):
            k = flat.rfind(" ", i + int(size * 0.6), j)
            if k > i:
                j = k
        piece = _squash(flat[i:j])
        if piece:
            labels = [lab for (s, e, lab) in spans if s < j and e > i]
            c = _mk(meta, f"{meta['doc_id']}#fx{n}", labels[0] if labels else "", labels, "",
                    _page_of(i, starts), piece, n)
            chunks.append(c)
            n += 1
        if j >= len(flat):
            break
        i = max(j - overlap, i + 1)
    return chunks


def make_chunks(docs: list[dict], mode: str = config.DEFAULT_CHUNKING) -> list[dict]:
    fn = chunk_by_article if mode == "article" else chunk_fixed
    out = []
    for d in docs:
        out.extend(fn(d))
    return out
