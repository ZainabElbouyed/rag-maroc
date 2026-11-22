# Assistant RAG sur des textes réglementaires marocains — Rapport

*Généré le 2026-11-21 — à relire, compléter et adapter par le groupe.*

> **Attention.** Outil pédagogique sans valeur juridique : seule la version publiée au Bulletin officiel fait foi. Les chiffres ci-dessous proviennent du juge automatique par mots-clés et d'un jeu de test écrit par le groupe ; validez-les manuellement avant de les citer.

## 1. Contexte et problème

Les textes juridiques marocains (dahirs, lois, décrets, arrêtés) sont publiés au Bulletin officiel. Ils sont nombreux, longs et techniques : retrouver l'article qui répond à une question précise est long, même pour un professionnel. Un grand modèle de langage utilisé seul peut halluciner, ignorer la version en vigueur d'un texte et ne cite pas de source vérifiable, ce qui est rédhibitoire en droit.

## 2. Principe du RAG

Le RAG (*Retrieval-Augmented Generation*) procède en deux temps : (1) **retrouver** dans une base de textes les passages les plus proches de la question, grâce aux embeddings (vecteurs numériques dont la proximité reflète la proximité de sens) ; (2) **générer** une réponse à partir de ces seuls passages, avec citation des sources et refus de répondre si l'information est absente. La réponse est ainsi vérifiable.

## 3. Données et prétraitement

Périmètre : Droit constitutionnel. Langue : français (l'arabe est une perspective). Corpus : 1 textes.

| ID | Titre | Date | N° BO | Langue |
|---|---|---|---|---|
| CONST | Constitution du Royaume du Maroc (2011) | 2011-07-29 | 5964 bis | fr |

Prétraitement : extraction PyMuPDF (OCR Tesseract si page scannée) ; suppression des en-têtes, pieds de page et numéros de page répétés ; correction des coupures de mots ; reconstitution des paragraphes ; découpage **par article** (unité de sens), avec métadonnées (texte, article, page, date, n° du BO, titre/chapitre) servant à citer la source. Les textes nettoyés sont sauvegardés dans `data/processed/` pour contrôle manuel.

## 4. Architecture et choix techniques

Question → embedding → recherche (dense, BM25 ou hybride par fusion RRF) → top-k passages → prompt strict → LLM → réponse + sources.
Configuration retenue : embedding `st:intfloat/multilingual-e5-small`, découpage `article`, recherche `hybrid`, k = 5, génération : ollama:qwen2.5:7b.

- **Embeddings** : TF-IDF (baseline lexicale hors ligne) ; modèles multilingues `multilingual-e5` et `bge-m3` lorsque `sentence-transformers` est installé.
- **Index** : matrice de vecteurs normalisés (similarité cosinus) + index BM25 maison.
- **Génération** : Claude (API), Ollama (local) ou mode extractif sans LLM. Prompt : répondre uniquement à partir des extraits, citer l'article, dire « je ne sais pas » sinon.
- **Garde-fou hors corpus** (mode extractif) : part des termes informatifs de la question retrouvés dans les passages.

## 5. Évaluation

Jeu de test : 48 questions (dont 7 questions piège absentes du corpus), avec mots-clés et article attendus.

### 5.1 Comparaison des variantes (qualité de la recherche)

| Embedding | Découpage | Recherche | Passages | Hit@3 | Hit@5 | Hit@8 | MRR |
|---|---|---|---|---|---|---|---|
| tfidf | article | dense | 193 | 98 % | 100 % | 100 % | 0.93 |
| tfidf | article | bm25 | 193 | 98 % | 100 % | 100 % | 0.93 |
| tfidf | article | hybrid | 193 | 100 % | 100 % | 100 % | 0.95 |
| tfidf | fixed | dense | 194 | 100 % | 100 % | 100 % | 0.97 |
| tfidf | fixed | bm25 | 194 | 100 % | 100 % | 100 % | 0.95 |
| tfidf | fixed | hybrid | 194 | 100 % | 100 % | 100 % | 0.96 |
| st:intfloat/multilingual-e5-small | article | dense | 193 | 100 % | 100 % | 100 % | 1.00 |
| st:intfloat/multilingual-e5-small | article | bm25 | 193 | 98 % | 100 % | 100 % | 0.93 |
| st:intfloat/multilingual-e5-small | article | hybrid | 193 | 100 % | 100 % | 100 % | 1.00 |
| st:intfloat/multilingual-e5-small | fixed | dense | 194 | 100 % | 100 % | 100 % | 0.97 |
| st:intfloat/multilingual-e5-small | fixed | bm25 | 194 | 100 % | 100 % | 100 % | 0.95 |
| st:intfloat/multilingual-e5-small | fixed | hybrid | 194 | 100 % | 100 % | 100 % | 0.99 |

*Hit@k : le bon article figure dans les k premiers passages. MRR : inverse du rang du bon article.*

### 5.2 Qualité des réponses (configuration retenue)

| Indicateur | Valeur |
|---|---|
| Réponses correctes (questions répondables) | 93 % |
| Réponses partielles | 0 % |
| Réponses fausses ou refus à tort | 7 % |
| dont refus à tort | 0 % |
| Refus correct sur questions hors corpus | 100 % |

Le jugement est **automatique** (présence des mots-clés attendus) : il est approximatif. Validez-le manuellement avec `results/reponses_a_valider.csv` (colonne `verdict_manuel`).

### 5.3 Critères de succès fixés au cadrage

| Critère | Cible | Mesuré | Statut |
|---|---|---|---|
| Hit@5 (bon article dans le top-5) | 80 % | 100 % | ✅ atteint |
| Réponses correctes | 70 % | 93 % | ✅ atteint |
| Refus correct hors corpus | 80 % | 100 % | ✅ atteint |

### 5.4 Cas d'échec

| ID | Question | Verdict | Réponse (extrait) |
|---|---|---|---|
| Q06 | La pratique de la torture est-elle punie ? | faux | Oui, la pratique de la torture est punie selon l'article 22 de la Constitution du Royaume du Maroc [S1]. |
| Q18 | À quel âge le Roi cesse-t-il d’être mineur ? | faux | Le Roi cesse d’être mineur à l’âge de vingt ans accomplis [S1]. |
| Q41 | Qui est le chef suprême des Forces Armées Royales ? | faux | Le Chef Suprême des Forces Armées Royales est le Roi, selon l'article 53 de la Constitution du Royaume du Maro |

## 6. Limites

- Pas de valeur juridique : les textes peuvent avoir été modifiés ou abrogés ; seule la version publiée au BO fait foi.
- Qualité d'extraction dépendante des PDF (scans, tableaux, mise en page).
- Français uniquement ; l'arabe demande un modèle d'embeddings et un OCR adaptés.
- Évaluation sur un petit jeu de test (variance élevée) et juge automatique approximatif.
- Le seuil du garde-fou extractif est heuristique et doit être recalibré sur le vrai corpus.

## 7. Perspectives

Ajout de l'arabe, reranking par cross-encoder, suivi des versions et abrogations, réponses multi-articles, évaluation par un juriste, interface avec historique de conversation.
