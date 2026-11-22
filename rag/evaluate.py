"""Étape 8 : évaluation (retrieval, réponses, refus) + comparaison de variantes + rapport."""
import csv
import json
from datetime import date

from . import config
from .corpus import read_tracking
from .pipeline import RAGPipeline, ensure_index
from .retrieve import Retriever
from .text_utils import normalize

KS = (3, 5, 8)


def load_questions(path=config.EVAL_CSV) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["repondable"] = str(r["repondable"]).strip() in ("1", "true", "True")
        r["kw"] = [normalize(k) for k in (r.get("mots_cles") or "").split("|") if k.strip()]
    return rows


def is_hit(chunk: dict, q: dict) -> bool:
    expected = {a.strip() for a in q["article"].split("|")}   # plusieurs articles possibles : « 56|115 »
    return chunk["doc_id"] == q["doc_id"] and bool(expected & set(chunk.get("articles", [])))


def retrieval_metrics(index, questions, mode: str) -> dict:
    r = Retriever(index, mode)
    qs = [q for q in questions if q["repondable"]]
    ranks = []
    for q in qs:
        hits = r.search(q["question"], k=max(KS), mode=mode)
        rank = next((h["rank"] for h in hits if is_hit(h["chunk"], q)), None)
        ranks.append(rank)
    out = {f"hit@{k}": sum(1 for x in ranks if x and x <= k) / len(qs) for k in KS}
    out["mrr"] = sum(1 / x for x in ranks if x) / len(qs)
    out["n"] = len(qs)
    return out


def judge(q: dict, res: dict) -> str:
    if q["repondable"]:
        if res["abstained"]:
            return "refus_a_tort"
        found = [k for k in q["kw"] if k in normalize(res["answer"])]
        if len(found) == len(q["kw"]):
            return "correct"
        return "partiel" if found else "faux"
    return "refus_correct" if res["abstained"] else "hallucination"


def generation_eval(index, questions, mode: str, k: int, llm: str = "auto") -> dict:
    pipe = RAGPipeline(index, mode=mode, k=k, llm=llm)
    rows = []
    for q in questions:
        res = pipe.ask(q["question"])
        rows.append({"id": q["id"], "question": q["question"], "repondable": int(q["repondable"]),
                     "reponse": res["answer"], "verdict_auto": judge(q, res),
                     "sources": " ; ".join(f"{s['doc_id']} art.{s['article']}" for s in res["sources"][:3]),
                     "couverture": res["coverage"], "verdict_manuel": ""})
    ans = [r for r in rows if r["repondable"]]
    trap = [r for r in rows if not r["repondable"]]
    cnt = lambda rs, v: sum(1 for r in rs if r["verdict_auto"] == v)  # noqa: E731
    return {"llm": pipe.llm_name, "rows": rows,
            "correct_rate": cnt(ans, "correct") / len(ans),
            "partial_rate": cnt(ans, "partiel") / len(ans),
            "false_rate": (cnt(ans, "faux") + cnt(ans, "refus_a_tort")) / len(ans),
            "false_abstention_rate": cnt(ans, "refus_a_tort") / len(ans),
            "abstain_rate": cnt(trap, "refus_correct") / max(len(trap), 1),
            "n_answerable": len(ans), "n_trap": len(trap)}


def _pct(x): return f"{100 * x:.0f} %"


def md_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def run_evaluation(embeddings: list[str], chunkings=("article", "fixed"), modes=("dense", "bm25", "hybrid"),
                   llm: str = "auto", verbose: bool = True) -> dict:
    questions = load_questions()
    grid = []
    indexes = {}
    for spec in embeddings:
        for ch in chunkings:
            try:
                idx = ensure_index(spec, ch, verbose=verbose)
            except Exception as exc:  # noqa: BLE001
                print(f"[!] {spec}/{ch} ignoré : {exc.__class__.__name__}: {exc}")
                break
            indexes[(spec, ch)] = idx
            for mode in modes:
                m = retrieval_metrics(idx, questions, mode)
                grid.append({"embedding": spec, "chunking": ch, "mode": mode, "n_chunks": len(idx.chunks), **m})
                if verbose:
                    print(f"  {spec:<38} {ch:<8} {mode:<7} hit@5={m['hit@5']:.2f} mrr={m['mrr']:.2f}")
    if not grid:
        raise RuntimeError("Aucune configuration évaluable.")
    # La config de référence (génération + rapport) utilise le découpage par article, choix du projet ;
    # la variante « fixed » sert uniquement à la comparaison.
    pool = [g for g in grid if g["chunking"] == config.DEFAULT_CHUNKING] or grid
    best = max(pool, key=lambda g: (g["hit@5"], g["mrr"], g["mode"] == "hybrid"))
    gen = generation_eval(indexes[(best["embedding"], best["chunking"])], questions, best["mode"], 5, llm)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    with open(config.RESULTS_DIR / "comparaison_variantes.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(grid[0].keys()))
        w.writeheader()
        w.writerows(grid)
    with open(config.RESULTS_DIR / "reponses_a_valider.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(gen["rows"][0].keys()))
        w.writeheader()
        w.writerows(gen["rows"])
    summary = {"best_config": best, "generation": {k: v for k, v in gen.items() if k != "rows"},
               "criteria": config.SUCCESS_CRITERIA}
    (config.RESULTS_DIR / "resultats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                       encoding="utf-8")
    render_report(grid, best, gen)
    return {"grid": grid, "best": best, "gen": gen}


def render_report(grid, best, gen) -> None:
    tpl = (config.REPORT_DIR / "rapport_template.md").read_text(encoding="utf-8")
    corpus = md_table(["ID", "Titre", "Date", "N° BO", "Langue"],
                      [[r["doc_id"], r["titre"], r["date"], r["bo_numero"], r["langue"]] for r in read_tracking()])
    grid_tab = md_table(["Embedding", "Découpage", "Recherche", "Passages", "Hit@3", "Hit@5", "Hit@8", "MRR"],
                        [[g["embedding"], g["chunking"], g["mode"], g["n_chunks"], _pct(g["hit@3"]),
                          _pct(g["hit@5"]), _pct(g["hit@8"]), f"{g['mrr']:.2f}"] for g in grid])
    gen_tab = md_table(["Indicateur", "Valeur"], [
        ["Réponses correctes (questions répondables)", _pct(gen["correct_rate"])],
        ["Réponses partielles", _pct(gen["partial_rate"])],
        ["Réponses fausses ou refus à tort", _pct(gen["false_rate"])],
        ["dont refus à tort", _pct(gen["false_abstention_rate"])],
        ["Refus correct sur questions hors corpus", _pct(gen["abstain_rate"])]])
    c = config.SUCCESS_CRITERIA
    ok = lambda v, t: "✅ atteint" if v >= t else "❌ non atteint"  # noqa: E731
    crit = md_table(["Critère", "Cible", "Mesuré", "Statut"], [
        ["Hit@5 (bon article dans le top-5)", _pct(c["hit_at_5"]), _pct(best["hit@5"]), ok(best["hit@5"], c["hit_at_5"])],
        ["Réponses correctes", _pct(c["correct_rate"]), _pct(gen["correct_rate"]), ok(gen["correct_rate"], c["correct_rate"])],
        ["Refus correct hors corpus", _pct(c["abstain_rate"]), _pct(gen["abstain_rate"]), ok(gen["abstain_rate"], c["abstain_rate"])]])
    bad = [r for r in gen["rows"] if r["verdict_auto"] in ("faux", "partiel", "refus_a_tort", "hallucination")]
    fails = md_table(["ID", "Question", "Verdict", "Réponse (extrait)"],
                     [[r["id"], r["question"], r["verdict_auto"], r["reponse"][:110].replace("|", "/")] for r in bad]) \
        if bad else "Aucun échec détecté par le juge automatique."
    cfg = (f"embedding `{best['embedding']}`, découpage `{best['chunking']}`, recherche `{best['mode']}`, k = 5, "
           f"génération : {gen['llm']}")
    for key, val in {"DATE": date.today().isoformat(), "CORPUS_TABLE": corpus, "GRID_TABLE": grid_tab,
                     "GEN_TABLE": gen_tab, "CRITERIA_TABLE": crit, "FAILURES": fails, "CONFIG": cfg,
                     "N_DOCS": str(len(read_tracking())),
                     "DOMAIN": ", ".join(sorted({r.get("domaine", "") for r in read_tracking() if r.get("domaine")})) or "à préciser", "N_QUESTIONS": str(gen["n_answerable"] + gen["n_trap"]),
                     "N_TRAP": str(gen["n_trap"])}.items():
        tpl = tpl.replace("{{" + key + "}}", val)
    (config.REPORT_DIR / "rapport.md").write_text(tpl, encoding="utf-8")
    # Script de démonstration pour la soutenance
    good = [r for r in gen["rows"] if r["verdict_auto"] == "correct"][:5]
    trap = [r for r in gen["rows"] if r["verdict_auto"] == "refus_correct"][:1]
    fail = bad[:1]
    lines = ["# Script de démonstration (généré automatiquement)\n",
             "## 1. Questions qui fonctionnent"] + [f"- {r['question']}  \n  → {r['reponse'][:160]}" for r in good]
    lines += ["\n## 2. Question hors périmètre (le système doit refuser)"] + \
             [f"- {r['question']}  \n  → {r['reponse']}" for r in trap]
    lines += ["\n## 3. Cas d'échec à expliquer honnêtement"] + \
             ([f"- {r['question']}  \n  → {r['reponse'][:160]}  \n  Verdict : {r['verdict_auto']}" for r in fail]
              or ["- (aucun échec détecté : testez vos propres questions pièges)"])
    lines += ["\n## Conseil\nAnnoncez d'abord les limites (corpus restreint, pas de valeur juridique), "
              "puis montrez la réponse ET la source."]
    (config.REPORT_DIR / "script_demo.md").write_text("\n".join(lines), encoding="utf-8")
