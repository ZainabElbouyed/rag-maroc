"""Normalisation et tokenisation légère du français."""
import re
import unicodedata

STOPWORDS = set("""
a au aux avec ce ces cette cet dans de des du elle elles en est et il ils je la le les leur leurs lui
ma mais me mes moi mon ne ni nos notre nous on ou par pas pour qu que quel quelle quelles quels qui sa se
ses si son sur ta te tes toi ton tu un une vos votre vous y ete etre sont sera seront peut peuvent doit
doivent faut fait selon comment combien quand ainsi alors aussi apres avant sous sans entre vers chez dont
donc car comme plus moins tout tous toute toutes autre autres meme parce cela ca ceci celui celle ceux
celles faire avoir t l d j n s c m ont avoir ete etait
""".split())

TOKEN_RE = re.compile(r"\d+(?:[,.]\d+)*|[a-z]+")


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def normalize(s: str) -> str:
    """minuscules, sans accents, apostrophes droites, espaces compactés."""
    s = s.replace("’", "'").replace("‘", "'").replace("\u00a0", " ")
    s = s.lower().replace("œ", "oe").replace("æ", "ae")
    s = strip_accents(s)
    return re.sub(r"\s+", " ", s).strip()


def stem(t: str) -> str:
    """Racinisation très légère (pluriels, féminins) : suffisante pour du lexical."""
    if any(c.isdigit() for c in t):
        return t
    if len(t) > 4 and t[-1] in "sx":
        t = t[:-1]
    for _ in range(2):
        if len(t) > 4 and t[-1] == "e":
            t = t[:-1]
    return t


def tokenize(text: str) -> list[str]:
    out = []
    for t in TOKEN_RE.findall(normalize(text)):
        if t in STOPWORDS:
            continue
        if len(t) == 1 and not t.isdigit():
            continue
        out.append(stem(t))
    return out
