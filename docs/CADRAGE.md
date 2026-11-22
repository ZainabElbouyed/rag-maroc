# Cadrage du projet

| Point | Décision |
|---|---|
| Domaine | **Droit constitutionnel** : Constitution du Royaume du Maroc (2011), puis textes liés (lois organiques, etc.). Un domaine unique rend la démonstration cohérente. |
| Langue | **Français** pour le prototype. L'arabe est présenté comme perspective (embeddings et OCR arabes à ajouter). |
| Corpus | 10 à 20 textes du Bulletin officiel (portail Adala). Départ : la Constitution (180 articles) ; à compléter (voir README). |
| Mission | Poser une question en langage naturel → réponse **avec sa source** (texte, article, page, date, n° du BO). |

## Critères de succès chiffrés (modifiables dans `rag/config.py`)

| Critère | Cible |
|---|---|
| Hit@5 : le bon article est dans les 5 premiers passages | ≥ 80 % |
| Réponses jugées correctes (questions répondables) | ≥ 70 % |
| Refus correct (« je ne sais pas ») sur les questions hors corpus | ≥ 80 % |

## Répartition à 3

| Rôle | Fichiers principaux |
|---|---|
| **A – Données** | `data/corpus_tracking.csv`, `rag/extract.py`, `rag/clean.py`, `rag/chunk.py`, `data/processed/` |
| **B – Moteur RAG** | `rag/embed.py`, `rag/index.py`, `rag/retrieve.py`, `rag/generate.py`, `rag/pipeline.py` |
| **C – Interface et évaluation** | `app.py`, `data/eval_questions.csv`, `rag/evaluate.py`, `report/` |

## Planning (5 semaines)

1. Cadrage, corpus collecté, questions de test commencées
2. Texte nettoyé, passages avec métadonnées
3. Pipeline de bout en bout
4. Évaluation, comparaison de variantes, améliorations
5. Interface finale, rapport, soutenance
