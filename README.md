# Assistant RAG sur la Constitution marocaine (et autres textes réglementaires)

Pose une question en français, obtient une réponse **avec sa source** (texte, article, page, date, n° du BO).
Chaîne complète : PDF → extraction → nettoyage → découpage par article → embeddings → index → recherche hybride → génération → évaluation → interface → rapport.

Corpus actuel : la **Constitution du Royaume du Maroc (2011)**, version française du portail Adala (`data/raw_pdfs/constitution.pdf`, 58 pages, 180 articles). Le projet attend 10 à 20 textes : voir « Ajouter des textes ».

> Outil pédagogique sans valeur juridique : seule la version publiée au Bulletin officiel fait foi.

## Démarrage rapide (Python 3.10+)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
python run_all.py                  # pipeline + évaluation + rapport (≈ 10 s, hors ligne)
streamlit run app.py               # interface
python -m unittest discover tests  # tests
```

Sans clé API ni modèle téléchargé, le système tourne en **mode extractif** : embeddings TF-IDF + BM25, réponse = phrases les plus proches de la question, avec la source. C'est la baseline.



## Activer les vrais composants

```bash
pip install -r requirements-full.txt
python run_all.py                  # ajoute multilingual-e5-small (téléchargé au 1er lancement)
python run_all.py --full           # ajoute bge-m3 (~2,2 Go)
```

Choisir le LLM (au moins l'un des deux, sinon mode extractif) :

```powershell
# Option A : Ollama (local, gratuit). Exemple utilisé pour les résultats ci-dessous :
ollama pull qwen2.5:7b
$env:OLLAMA_MODEL="qwen2.5:7b"          # Mac/Linux : export OLLAMA_MODEL=qwen2.5:7b
python run_all.py --llm ollama

# Option B : Claude (API payante)
$env:ANTHROPIC_API_KEY="sk-ant-..."     # Mac/Linux : export ANTHROPIC_API_KEY=...
$env:RAG_LLM_MODEL="claude-sonnet-5-5"  # optionnel
python run_all.py --llm anthropic
```

`OLLAMA_MODEL` vaut `mistral` par défaut : si ce modèle n'est pas installé, le projet retombe en mode extractif avec un avertissement. La variable n'est valable que dans la fenêtre PowerShell où elle est définie.

**État des tests :** e5-small et Ollama (`qwen2.5:7b`) ont été exécutés avec succès. `bge-m3`, le reranker et Claude (API) sont implémentés mais **non testés**.

## Résultats mesurés (48 questions : 41 répondables, 7 hors corpus)

| Mesure | Baseline TF-IDF, mode extractif | e5-small + `qwen2.5:7b` |
|---|---|---|
| Le bon article est dans le top-5 (hybride) | 100 % | 100 % |
| Rang moyen du bon article (MRR) | 0,95 | 1,00 |
| Réponses correctes (juge automatique par mots-clés) | 88 % | 93 % (38/41) |
| Refus correct sur les questions hors corpus | 57 % (4/7) | 100 % (7/7) |

Les trois critères de succès (80 % / 70 % / 80 %) sont atteints avec le LLM ; le refus hors corpus ne l'est pas en mode extractif.

Lecture honnête :
- La **recherche** fonctionne très bien sur ce texte, mais elle est **saturée** : toutes les variantes atteignent 100 % de Hit@5. L'avantage d'e5 (MRR 1,00 contre 0,93 à 0,97) porte sur quelques rangs sur 41 questions : indicatif, pas établi.
- Le **LLM améliore nettement le refus hors corpus** par rapport au mode extractif (7/7 contre 4/7). Le mode extractif ne lit que le meilleur passage et son garde-fou lexical laisse passer des questions dont les mots existent dans la Constitution (« peine », « taux », « âge »).
- Le **juge automatique est approximatif** : sur les 3 réponses notées fausses avec le LLM, deux sont correctes à la relecture (Q06 torture, Q41 chef des armées : mots-clés trop rigides). Une est une vraie erreur (Q18) : le modèle répond « vingt ans » alors que l'article 44 fixe la fin de la minorité du Roi à dix-huit ans (vingt ans est l'âge jusqu'auquel le Conseil de Régence reste consultatif). La recherche avait trouvé le bon article ; l'erreur vient de la génération. Après relecture, le score serait d'environ 40/41, **à confirmer en validant `results/reponses_a_valider.csv`** (colonne `verdict_manuel`).
- Les questions sont plutôt faciles, le jeu de test est petit et écrit par nous : ces chiffres ne valent que pour la Constitution seule.
- Deux tentatives d'amélioration du mode extractif (pondération des termes rares, refus si un mot est inconnu du corpus) ont **dégradé** les résultats (88 % → 66 % de réponses correctes) et ont été retirées. Régler ces heuristiques sur le jeu de test fausserait les chiffres : si vous les modifiez, gardez un jeu de test séparé.

## Ajouter des textes (objectif 10 à 20)

1. Portail Adala → Ressources → Références légales et réglementaires → Lois et Législations nationales → catégorie voulue → Parcourir. Téléchargez les PDF en version française. Vérifiez les conditions de réutilisation.
2. Déposez les PDF dans `data/raw_pdfs/`.
3. Ajoutez une ligne par PDF dans `data/corpus_tracking.csv` (`doc_id, titre, numero, date, bo_numero, url, langue, domaine, fichier`). Le numéro du BO est dans la note de bas de page des PDF Adala.
4. Ajoutez des questions dans `data/eval_questions.csv` : `id, question, mots_cles` (séparés par `|`, minuscules sans accents), `doc_id`, `article` (plusieurs possibles : `56|115`), `repondable` (1 ou 0). Prévoyez des questions pièges (0).
5. `python run_all.py --rebuild`, puis contrôlez `data/processed/<doc_id>.txt` : c'est là qu'on repère les mauvaises extractions.
6. PDF scannés : installez Tesseract (langue `fra`) + `pytesseract` (OCR automatique si une page n'a pas de texte).

Particularités déjà gérées (Adala) : puces en caractères spéciaux, numéros de page `- 12 -`, notes de bas de page `1 - Bulletin officiel…`, en-têtes, intertitres (« De l'organisation du Parlement », repérés et notés `INTERTITRE :`), « Article N » seul sur sa ligne, césures. Les 180 articles de la Constitution sont retrouvés, sans manque ni doublon (test automatique). D'autres textes (décrets, arrêtés, lois sans titres) peuvent avoir une mise en page différente : adaptez `rag/clean.py` et `rag/chunk.py`.

## Correspondance avec les étapes

| Étape | Où |
|---|---|
| Cadrage | `docs/CADRAGE.md`, `rag/config.py` (critères) |
| Collecte | `data/corpus_tracking.csv`, `scripts/download_corpus.py` |
| Extraction / nettoyage | `rag/extract.py`, `rag/clean.py` → `data/processed/` |
| Découpage + métadonnées | `rag/chunk.py` |
| Embeddings / indexation | `rag/embed.py`, `rag/index.py` (cosinus numpy + BM25 maison) |
| Recherche / génération | `rag/retrieve.py`, `rag/generate.py`, `rag/pipeline.py` |
| Améliorations | hybride RRF, filtre par texte, reformulation (LLM), reranker (`rag/retrieve.py::Reranker`) |
| Évaluation | `rag/evaluate.py` → `results/`, tableau des variantes |
| Interface | `app.py` |
| Rapport / soutenance | `report/rapport.md`, `report/script_demo.md` (régénérés à chaque `run_all.py`) |

`scripts/make_demo_corpus.py` génère dans `data/demo_corpus/` 10 textes **fictifs** de droit du travail (pour tester la chaîne sur plusieurs documents). Ils n'écrasent pas votre corpus.

Index vectoriel : numpy (cosinus), suffisant pour quelques milliers de passages ; FAISS/Chroma ne seraient utiles qu'à plus grande échelle, à justifier en soutenance.
