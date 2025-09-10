# =========================================
# Auteur : Arthur Prigent
# Date : 01/09/2025
# Description : Script de construction de l'index Chroma à partir des documents dans /docs
# =========================================

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from utils_loaders import load_folder
from langchain_community.embeddings import HuggingFaceEmbeddings
import shutil
import torch



DOCS_DIR = "docs"
DB_DIR = "db"
FULL_REBUILD = True


def choose_device() -> str:
    import torch
    # 1) NVIDIA (PC) — prioritaire si présent
    if torch.cuda.is_available():
        return "cuda"
    # 2) Apple Metal (Mac) — MPS
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    # 3) Fallback
    return "cpu"


def main():


    if FULL_REBUILD and os.path.exists(DB_DIR):
        print("🧹 Suppression complète de la base Chroma…")
        shutil.rmtree(DB_DIR)
    os.makedirs(DB_DIR, exist_ok=True)

    print("📥 Lecture des documents…")
    raw_docs = load_folder(DOCS_DIR)
    if not raw_docs:
        print("Aucun document trouvé dans /docs (y compris sous-dossiers).")
        return

    print(f"✅ Fichiers chargés: {len(raw_docs)}")
    print("✂️ Découpage en chunks…")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    texts, metadatas = [], []
    for d in raw_docs:
        chunks = splitter.split_text(d["content"])
        for chunk in chunks:
            texts.append(chunk)
            metadatas.append(d["metadata"])  # contient 'source' et 'path'

    device = choose_device()
    print(f"🧠 Embeddings BAAI/bge-m3 sur device: {device}")

    embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": device},
    # Sur MPS, garde un batch modéré (ex. 32) ; sur CUDA tu peux monter (64/128 selon VRAM)
    encode_kwargs={"normalize_embeddings": True, "batch_size": 32 if device == "mps" else 64},
    )

    print("🧠 Création de la base Chroma…")
    db = Chroma.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    db.persist()
    print("✅ Index sauvegardé dans /db")

if __name__ == "__main__":
    main()

