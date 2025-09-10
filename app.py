# =========================================
# Auteur : Arthur Prigent
# Date : 09
# Description : Application Streamlit de chat RAG avec LM Studio
# =========================================

import os
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from build_index import main as rebuild_index_main
import torch

# =========================
# Config & constantes
# =========================
load_dotenv()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "lm-studio")
DEFAULT_MODEL   = os.getenv("LMSTUDIO_MODEL", "llama-3.1-8b-instruct")

DB_DIR = "db"
MAX_CONTEXT_CHARS = 12000  # protection longueur du prompt




def choose_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"

# =========================
# Page Streamlit
# =========================
st.set_page_config(page_title="Titre Page • Chat bot", page_icon="💬")
st.title("💬 RAG AI")
st.caption("Tout reste en local au moment de la requête.")

# =========================
# Sidebar (options)
# =========================
with st.sidebar:
    st.subheader("Options")
    model_name = st.text_input("Modèle (LM Studio)", value=DEFAULT_MODEL)
    k = st.slider("Nombre de passages (k)", 1, 20, 8)
    temperature = st.slider("Température", 0.0, 1.0, 0.2, 0.1)
    use_stream = st.toggle("🔄 Streaming", value=True)
    use_mmr = st.toggle("Diversifier (MMR)", value=True)
    max_tokens = st.number_input("Max tokens réponse", min_value=128, max_value=4096, value=2048, step=64)
    if st.button("🔁 Reconstruire l'index"):
        with st.spinner("Reconstruction de l'index…"):
            rebuild_index_main()
        st.success("Index reconstruit. (Recharge la page si besoin)")

# =========================
# Helpers
# =========================
@st.cache_resource
def get_vector_store():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma

    device = choose_device()
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32 if device == "mps" else 64},
    )
    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)


@st.cache_resource
def get_client():
    """
    Client OpenAI-compatible pointant sur LM Studio.
    """
    return OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

def build_prompt(question: str, docs, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """
    Construit le prompt de RAG : contexte + question + instructions.
    """
    header = (
    "Tu es un assistant qui répond STRICTEMENT à partir des documents fournis.\n"
    "Tous les documents proviennent d’une entreprise (?????) spécialisée dans (metttez ce que vous voulez).\n"
    "Si l'information n'est pas présente, dis que tu ne sais pas.\n"
    "Cite les sources (nom de fichier) utilisées.\n\n"
    "### CONTEXTE\n"
)

    context_parts, total = [], 0
    for d in docs:
        src = d.metadata.get("source", "?")
        rel = d.metadata.get("path", src)
        part = f"[Source: {src} | Path: {rel}]\n{d.page_content}\n"
        if total + len(part) > max_chars:
            break
        context_parts.append(part)
        total += len(part)

    context = header + "\n---\n".join(context_parts)
    question_block = (
        f"\n\n### QUESTION\n{question}\n\n"
        "### INSTRUCTIONS\n"
        "- Réponds en français, clair et concis (puces si utile).\n"
        "- Ne fais aucune supposition hors des documents.\n"
        "- Termine par une section **Sources** listant chaque fichier (et chemin relatif s’il existe).\n"
    )
    return context + question_block

def stream_chat_completion(client, model, messages, temperature=0.2, max_tokens=512):
    """
    Générateur de texte en streaming (API OpenAI-compatible → LM Studio).
    """
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        if chunk and chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if delta and getattr(delta, "content", None):
                yield delta.content

# =========================
# Ressources (DB & Client)
# =========================
db = get_vector_store()
client = get_client()

# Alerte si DB vide
try:
    _ = db._collection.count()  # type: ignore[attr-defined]
    if _ == 0:
        st.warning("La base vectorielle semble vide. Clique sur « Reconstruire l'index » dans la sidebar après avoir placé des fichiers dans `docs/`.")
except Exception:
    # Certaines versions ne permettent pas cet accès direct, on ignore juste.
    pass


if "history" not in st.session_state:
    st.session_state.history = []

# Affichage de l'historique
for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.markdown(msg)

user_msg = st.chat_input("Pose ta question sur tes documents…")

if user_msg:
    # Affiche la question utilisateur
    with st.chat_message("user"):
        st.markdown(user_msg)
    st.session_state.history.append(("user", user_msg))

    # Récupération des passages
    search_kwargs = {"k": k}
    if use_mmr:
        # Sur-prélève un peu pour permettre la diversité MMR
        search_kwargs["fetch_k"] = max(10, 3 * k)
        retriever = db.as_retriever(search_type="mmr", search_kwargs=search_kwargs)
    else:
        retriever = db.as_retriever(search_kwargs=search_kwargs)

    docs = retriever.get_relevant_documents(user_msg)

    # Si rien trouvé, on le dit franchement
    if not docs:
        no_data_msg = "Je n’ai pas trouvé d’information pertinente dans les documents indexés. Essaie d’élargir ta question ou de reconstruire l’index."
        with st.chat_message("assistant"):
            st.markdown(no_data_msg)
        st.session_state.history.append(("assistant", no_data_msg))
    else:
        # Construit le prompt avec contexte + question
        prompt = build_prompt(user_msg, docs)

        # Génération (streaming ou non)
        with st.chat_message("assistant"):
            assistant_placeholder = st.empty()
            full_answer = ""

            if use_stream:
                with st.spinner("Génération…"):
                    for token in stream_chat_completion(
                        client=client,
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=int(max_tokens),
                    ):
                        full_answer += token
                        assistant_placeholder.markdown(full_answer)
            else:
                with st.spinner("Génération…"):
                    resp = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=int(max_tokens),
                    )
                    full_answer = resp.choices[0].message.content
                    assistant_placeholder.markdown(full_answer)

        # Historique
        st.session_state.history.append(("assistant", full_answer))

        # Affiche les passages & sources utilisés
        with st.expander("📎 Passages & sources récupérés"):
            for i, d in enumerate(docs, start=1):
                src = d.metadata.get("source", "?")
                rel = d.metadata.get("path", src)
                st.markdown(f"**{i}. {src}** — `{rel}`")
                st.write(d.page_content[:1000] + ("..." if len(d.page_content) > 1000 else ""))
