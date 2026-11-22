"""Étape 9 : interface Streamlit.   Lancer :  streamlit run app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

from rag import config  # noqa: E402
from rag.corpus import read_tracking  # noqa: E402
from rag.evaluate import load_questions  # noqa: E402
from rag.index import VectorIndex, list_indexes  # noqa: E402
from rag.pipeline import RAGPipeline, ensure_index  # noqa: E402

st.set_page_config(page_title="Assistant RAG – textes réglementaires", page_icon="⚖️", layout="wide")


@st.cache_resource(show_spinner="Chargement de l'index…")
def load_index(name: str):
    if name == "(construire tfidf / article)":
        return ensure_index()
    return VectorIndex.load(config.INDEX_DIR / name)


@st.cache_resource(show_spinner="Initialisation du LLM…")
def load_pipeline(index_name: str, llm: str):
    return RAGPipeline(load_index(index_name), llm=llm)


st.title("⚖️ Assistant RAG sur des textes réglementaires")
st.caption("Recherche des passages pertinents puis réponse à partir de ces seuls passages, avec sources. "
           "Outil pédagogique : il ne constitue pas un conseil juridique.")

names = list_indexes() or ["(construire tfidf / article)"]
default = next((n for n in names if n.endswith("__article")), names[0])
with st.sidebar:
    st.header("Paramètres")
    index_name = st.selectbox("Index (embedding / découpage)", names, index=names.index(default))
    mode = st.radio("Recherche", ["hybrid", "dense", "bm25"], horizontal=True)
    k = st.slider("Nombre de passages (k)", 1, 10, config.DEFAULT_K)
    llm_pref = st.selectbox("Génération", ["auto", "extractive", "anthropic", "ollama"],
                            help="auto : Claude si ANTHROPIC_API_KEY est défini, sinon Ollama, sinon mode extractif")
    docs = {r["doc_id"]: r["titre"] for r in read_tracking()}
    chosen = st.multiselect("Filtrer par texte", list(docs), format_func=lambda d: f"{d} – {docs[d][:45]}")
    reformulate = st.checkbox("Reformuler la question (LLM requis)")

pipe = load_pipeline(index_name, llm_pref)
st.sidebar.info(f"Génération : **{pipe.llm_name}**")

tab_qa, tab_corpus = st.tabs(["Poser une question", "Corpus"])
with tab_qa:
    examples = [q["question"] for q in load_questions()][:40] if config.EVAL_CSV.exists() else []
    cols = st.columns(3)
    for i, ex in enumerate(examples[:3]):
        if cols[i].button(ex, key=f"ex{i}"):
            st.session_state["q"] = ex
    question = st.text_input("Votre question", key="q", placeholder="Ex. : Quelle est la durée de la période d'essai ?")
    if question:
        with st.spinner("Recherche et génération…"):
            res = pipe.ask(question, k=k, mode=mode, filters={"doc_id": chosen} if chosen else None,
                           reformulate=reformulate)
        if res["warning"]:
            st.warning(res["warning"])
        (st.warning if res["abstained"] else st.success)(res["answer"])
        st.caption(f"Recherche : {res['mode']} · k={res['k']} · couverture des termes : {res['coverage']}")
        if res["sources"]:
            st.subheader("Sources")
            for s in res["sources"]:
                tag = "✅ " if s["rank"] in res["cited"] else ""
                art = f"art. {s['article']}" if s["article"] else "extrait"
                with st.expander(f"{tag}[S{s['rank']}] {s['titre']} — {art} (p. {s['page']})",
                                 expanded=s["rank"] == 1):
                    st.write(s["text"])
                    st.caption(f"Texte {s['numero']} · date {s['date']} · BO n° {s['bo_numero']} · "
                               f"score {s['score']}" + (f" · {s['url']}" if s["url"] else ""))
with tab_corpus:
    st.dataframe(read_tracking())
