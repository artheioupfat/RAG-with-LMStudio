# =========================================
# Auteur : Arthur Prigent
# Date : 09
# Description : Chargeur de documents pour divers formats (txt, pdf, docx, xlsx)
# =========================================


import os
import pandas as pd
from pypdf import PdfReader
from docx import Document
from typing import List, Dict

SUPPORTED_EXTS = {".pdf", ".docx", ".txt", ".xlsx"}
MAX_FILE_MB = 50  # sécurité: ignorer fichiers trop gros


def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            pass
    return "\n".join(texts)

def read_docx(path: str) -> str:
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def read_xlsx(path: str) -> str:
    xl = pd.ExcelFile(path)
    out = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        out.append(f"=== Sheet: {sheet} ===\n{df.to_string(index=False)}")
    return "\n\n".join(out)

def load_folder(docs_dir: str) -> List[Dict]:
    """Parcourt récursivement docs_dir et charge tous les fichiers supportés."""
    results = []
    if not os.path.isdir(docs_dir):
        return results

    for root, dirs, files in os.walk(docs_dir):
        # option: ignorer dossiers cachés (._, ., .git, etc.)
        dirs[:] = [d for d in dirs if not d.startswith(".") and not d.startswith("_")]

        for name in files:
            if name.startswith("."):  # cache files
                continue
            path = os.path.join(root, name)
            ext = os.path.splitext(name)[1].lower()
            if ext not in SUPPORTED_EXTS:
                continue

            try:
                size_mb = os.path.getsize(path) / (1024 * 1024)
                if size_mb > MAX_FILE_MB:
                    print(f"Ignoré (fichier > {MAX_FILE_MB}MB): {path}")
                    continue

                if ext == ".txt":
                    content = read_txt(path)
                elif ext == ".pdf":
                    content = read_pdf(path)
                elif ext == ".docx":
                    content = read_docx(path)
                elif ext == ".xlsx":
                    content = read_xlsx(path)
                else:
                    continue

                if content and content.strip():
                    rel = os.path.relpath(path, docs_dir)
                    results.append({
                        "path": path,
                        "content": content,
                        "metadata": {
                            "source": os.path.basename(path),  # nom de fichier
                            "path": rel                         # chemin relatif depuis docs/
                        }
                    })
            except Exception as e:
                print(f"Erreur lecture {path}: {e}")

    return results
