"""Étape 6b : génération. LLM = Anthropic (API), Ollama (local) ou mode extractif sans LLM."""
import json
import os
import re
import urllib.request

from . import config
from .chunk import body_without_heading
from .text_utils import tokenize

SYSTEM_PROMPT = """Tu es un assistant de recherche dans des textes réglementaires marocains.
Règles strictes :
1. Réponds UNIQUEMENT à partir des extraits numérotés [S1], [S2]… fournis. N'utilise aucune connaissance extérieure.
2. Cite la source de chaque affirmation sous la forme [S1], en précisant le texte et l'article (ex. : « selon l'article 5 de la loi X [S1] »).
3. Si les extraits ne contiennent pas l'information demandée, réponds exactement : « Je ne sais pas : l'information demandée ne figure pas dans les textes du corpus. » et rien d'autre.
4. Réponds en français, de façon concise et précise (chiffres, délais et conditions exacts).
5. Ne donne pas de conseil juridique : tu restitues le contenu des textes fournis."""


def build_user_prompt(question: str, hits: list[dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        c = h["chunk"]
        ref = f"{c['titre']} — Article {c['article']}" if c["article"] else c["titre"]
        blocks.append(f"[S{i}] {ref} (BO {c['bo_numero']}, {c['date']})\n{c['text']}")
    return "EXTRAITS :\n\n" + "\n\n".join(blocks) + f"\n\nQUESTION : {question}"


class AnthropicLLM:
    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = os.environ.get("RAG_LLM_MODEL", "claude-sonnet-5-5")
        self.name = f"anthropic:{self.model}"

    def generate(self, system: str, user: str, max_tokens: int = 700) -> str:
        r = self.client.messages.create(model=self.model, max_tokens=max_tokens, system=system,
                                        messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text").strip()


class OllamaLLM:
    def __init__(self):
        self.host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.model = os.environ.get("OLLAMA_MODEL", "mistral")
        self.name = f"ollama:{self.model}"

    def available(self) -> bool:
        try:
            urllib.request.urlopen(self.host + "/api/tags", timeout=1.5)
            return True
        except Exception:  # noqa: BLE001
            return False

    def generate(self, system: str, user: str, max_tokens: int = 700) -> str:
        body = json.dumps({"model": self.model, "stream": False, "options": {"temperature": 0},
                           "messages": [{"role": "system", "content": system},
                                        {"role": "user", "content": user}]}).encode()
        req = urllib.request.Request(self.host + "/api/chat", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read())["message"]["content"].strip()


def get_llm(preference: str = "auto"):
    """Retourne un LLM ou None (= mode extractif)."""
    if preference == "extractive":
        return None
    if preference in ("auto", "anthropic") and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicLLM()
        except Exception as exc:  # noqa: BLE001
            print(f"[LLM] Anthropic indisponible : {exc}")
    if preference in ("auto", "ollama"):
        o = OllamaLLM()
        if o.available():
            return o
    if preference not in ("auto", "extractive"):
        print(f"[LLM] '{preference}' indisponible : repli sur le mode extractif.")
    return None


def is_abstention(answer: str) -> bool:
    return bool(re.match(r"\s*je ne sais pas", answer, re.I))


def extractive_answer(question: str, hits: list[dict], idf=None) -> str:
    """Sans LLM : restitue les 1-2 phrases les plus proches de la question dans le meilleur passage."""
    c = hits[0]["chunk"]
    body = body_without_heading(c["text"])
    sents = [s for s in re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-Ý0-9«])", body) if s.strip()]
    q = set(tokenize(question))
    scored = sorted(((len(q & set(tokenize(s))), -i, s) for i, s in enumerate(sents)), reverse=True)
    top = [s for sc, _, s in scored[:2] if sc > 0] or sents[:2]
    top.sort(key=lambda s: sents.index(s))
    art = f"article {c['article']}" if c["article"] else "extrait"
    return f"D'après {c['titre']} ({art}) : « {' '.join(top)} » [S1]"
