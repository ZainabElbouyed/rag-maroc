# Assistant RAG sur des textes réglementaires marocains — Rapport

*Généré le {{DATE}} — à relire, compléter et adapter par le groupe.*

> **Attention.** Outil pédagogique sans valeur juridique : seule la version publiée au Bulletin officiel fait foi. Les chiffres ci-dessous proviennent du juge automatique par mots-clés et d'un jeu de test écrit par le groupe ; validez-les manuellement avant de les citer.

## 1. Contexte et problème

Les textes juridiques marocains (dahirs, lois, décrets, arrêtés) sont publiés au Bulletin officiel. Ils sont nombreux, longs et techniques : retrouver l'article qui répond à une question précise est long, même pour un professionnel. Un grand modèle de langage utilisé seul peut halluciner, ignorer la version en vigueur d'un texte et ne cite pas de source vérifiable, ce qui est rédhibitoire en droit.

## 2. Principe du RAG

Le RAG (*Retrieval-Augmented Generation*) procède en deux temps : (1) **retrouver** dans une base de textes les passages les plus proches de la question, grâce aux embeddings (vecteurs numériques dont la proximité reflète la proximité de sens) ; (2) **générer** une réponse à partir de ces seuls passages, avec citation des sources et refus de répondre si l'information est absente. La réponse est ainsi vérifiable.

## 3. Données et prétraitement

Périmètre : {{DOMAIN}}. Langue : français (l'arabe est une perspective). Corpus : {{N_DOCS}} textes.

{{CORPUS_TABLE}}

Prétraitement : extraction PyMuPDF (OCR Tesseract si page scannée) ; suppression des en-têtes, pieds de page et numéros de page répétés ; correction des coupures de mots ; reconstitution des paragraphes ; découpage **par article** (unité de sens), avec métadonnées (texte, article, page, date, n° du BO, titre/chapitre) servant à citer la source. Les textes nettoyés sont sauvegardés dans `data/processed/` pour contrôle manuel.

## 4. Architecture et choix techniques

Question → embedding → recherche (dense, BM25 ou hybride par fusion RRF) → top-k passages → prompt strict → LLM → réponse + sources.
Configuration retenue : {{CONFIG}}.

- **Embeddings** : TF-IDF (baseline lexicale hors ligne) ; modèles multilingues `multilingual-e5` et `bge-m3` lorsque `sentence-transformers` est installé.
- **Index** : matrice de vecteurs normalisés (similarité cosinus) + index BM25 maison.
- **Génération** : Claude (API), Ollama (local) ou mode extractif sans LLM. Prompt : répondre uniquement à partir des extraits, citer l'article, dire « je ne sais pas » sinon.
- **Garde-fou hors corpus** (mode extractif) : part des termes informatifs de la question retrouvés dans les passages.

## 5. Évaluation

Jeu de test : {{N_QUESTIONS}} questions (dont {{N_TRAP}} questions piège absentes du corpus), avec mots-clés et article attendus.

### 5.1 Comparaison des variantes (qualité de la recherche)

{{GRID_TABLE}}

*Hit@k : le bon article figure dans les k premiers passages. MRR : inverse du rang du bon article.*

### 5.2 Qualité des réponses (configuration retenue)

{{GEN_TABLE}}

Le jugement est **automatique** (présence des mots-clés attendus) : il est approximatif. Validez-le manuellement avec `results/reponses_a_valider.csv` (colonne `verdict_manuel`).

### 5.3 Critères de succès fixés au cadrage

{{CRITERIA_TABLE}}

### 5.4 Cas d'échec

{{FAILURES}}

## 6. Limites

- Pas de valeur juridique : les textes peuvent avoir été modifiés ou abrogés ; seule la version publiée au BO fait foi.
- Qualité d'extraction dépendante des PDF (scans, tableaux, mise en page).
- Français uniquement ; l'arabe demande un modèle d'embeddings et un OCR adaptés.
- Évaluation sur un petit jeu de test (variance élevée) et juge automatique approximatif.
- Le seuil du garde-fou extractif est heuristique et doit être recalibré sur le vrai corpus.

## 7. Perspectives

Ajout de l'arabe, reranking par cross-encoder, suivi des versions et abrogations, réponses multi-articles, évaluation par un juriste, interface avec historique de conversation.
