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
export ANTHROPIC_API_KEY=sk-ant-...        # Windows : set ANTHROPIC_API_KEY=...
export RAG_LLM_MODEL=claude-sonnet-5-5     # optionnel
python run_all.py                  # la génération passe par Claude
```

Alternative gratuite : Ollama (`ollama pull mistral`), détecté automatiquement (`OLLAMA_MODEL` pour changer).
**Ces composants (e5, bge-m3, Claude, Ollama) n'ont pas pu être exécutés lors de la création du projet** (pas d'accès réseau) : le code est écrit mais non testé. Signalez toute erreur.

## Résultats mesurés (baseline TF-IDF + BM25, mode extractif, 48 questions dont 7 hors corpus)

| Mesure | Résultat |
|---|---|
| Recherche : le bon article est dans le top-5 (hybride) | 100 % (MRR 0,95) |
| Réponses correctes (juge automatique par mots-clés) | 88 % |
| Refus correct sur les questions hors corpus | 57 % (cible 80 % : **non atteinte**) |

Lecture honnête :
- La **recherche** fonctionne très bien sur ce texte : un article de loi est court et son vocabulaire est précis.
- Le **refus hors corpus** est le point faible du mode sans LLM : la règle (couverture des mots de la question) laisse passer des questions dont les mots existent dans la Constitution (« peine », « taux », « âge »). C'est là qu'un LLM avec le prompt strict doit améliorer les choses : à mesurer.
- Le mode extractif ne lit que le meilleur passage : quand le bon article est en 2e ou 3e position (ex. Cour constitutionnelle), la réponse est fausse alors que la recherche a bien trouvé l'article. Un LLM lit les 5 passages.
- Le jeu de test est petit et écrit par nous ; le juge automatique est approximatif. Validez à la main `results/reponses_a_valider.csv` (colonne `verdict_manuel`) avant de citer des chiffres.
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
