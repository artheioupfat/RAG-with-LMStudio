# 💬 RAG Local avec LM Studio

Ce projet met en place un **RAG (Retrieval-Augmented Generation)** 100% local sous Windows et Macos en utilisant :

- [LM Studio](https://lmstudio.ai) pour la génération de texte (LLM local, API compatible OpenAI)
- [ChromaDB](https://www.trychroma.com/) pour l’indexation vectorielle
- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) comme modèle d’embeddings multilingue (FR inclus)
- [Streamlit](https://streamlit.io/) pour l’interface web style ChatGPT

👉 Objectif : poser des questions sur vos propres documents (.pdf, .docx, .txt, .xlsx) et obtenir des réponses basées sur leur contenu.

---

## ⚙️ Arborescence

rag_lmstudio/
│── docs/ # Vos documents (peut contenir des sous-dossiers)
│── db/ # Base vectorielle locale (Chroma)
│── .env # Variables d'environnement (API LM Studio, modèle)
│── requirements.txt # Dépendances Python
│── utils_loaders.py # Fonctions de lecture des fichiers
│── build_index.py # Script d’indexation incrémentale
│── app.py # Interface web Streamlit (chat)
│── README.md # Documentation du projet

---

## 🚀 Installation

### 1. Pré-requis
- **Python 3.10+**
- **NVIDIA GPU** (exploité via PyTorch CUDA) ou CPU
- [LM Studio](https://lmstudio.ai) installé

### 2. Installer les dépendances

pip install -r requirements.txt

(choisissez dans le fichier ce qui vous correpond Windows/Macos)


### 3. Configurer LM Studio
- Ouvrir LM Studio
- Aller dans Developer → Local Server et cliquer sur Start
- Charger un modèle (ex. llama-3.1-8b-instruct)
- Vérifier qu’il apparaît dans la liste des modèles servis
- ⚠️ Décochez Allow local network access pour rester en localhost.



🔑 Configuration .env

Créez un fichier .env à la racine :

OPENAI_BASE_URL=http://127.0.0.1:1234/v1
OPENAI_API_KEY=lm-studio
LMSTUDIO_MODEL=llama-3.1-8b-instruct   # nom exact du modèle dans LM Studio


📥 Indexation des documents

Placez vos fichiers dans le dossier docs/ (les sous-dossiers sont pris en charge).

Lancez : python build_index.py


💬 Lancer l’interface

Lancez : streamlit run app.py

Une page web devrait s'ouvrir sinon : http://localhost:8501





Fonctionnalités :

Interface chat style ChatGPT
Bouton 🔁 Reconstruire l’index (dans la sidebar)
Affichage des sources et passages utilisés


## Auteur

Arthur Prigent# RAG-with-LMStudio
