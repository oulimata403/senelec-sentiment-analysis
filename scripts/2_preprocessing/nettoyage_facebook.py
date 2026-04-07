"""
Nettoyage robuste des données Facebook
- Parse manuellement les deux CSV même s'ils sont mal formés
- Harmonise les colonnes et crée un corpus unique
"""

import os
import hashlib
from pathlib import Path

import pandas as pd

from config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


POSTS_FILE = RAW_DATA_DIR / "facebook_posts_commentaires.csv"
KEYWORDS_FILE = RAW_DATA_DIR / "facebook_keywords.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "corpus_facebook_nettoye.csv"


def generer_id_source(texte: str, date_visible: str | None, date_collecte: str | None) -> str:
    base = (texte or "") + (date_visible or "") + (date_collecte or "")
    return hashlib.md5(base.encode("utf-8")).hexdigest()


def nettoyer_texte_minimal(texte: str) -> str:
    if not isinstance(texte, str):
        return ""
    t = texte.replace("\n", " ").replace("\r", " ")
    t = " ".join(t.split())
    return t.strip()


def parse_posts_file() -> pd.DataFrame:
    """
    Parse facebook_posts_commentaires.csv comme texte :
    format attendu par ligne :
    source,type,"texte, qui peut contenir, des virgules",date_visible,date_collecte
    """
    if not POSTS_FILE.exists():
        print(f"⚠️ Fichier non trouvé : {POSTS_FILE}")
        return pd.DataFrame()

    lignes = POSTS_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()
    rows = []

    for line in lignes:
        line = line.strip()
        if not line:
            continue

        parts = line.rsplit(",", 2)
        if len(parts) != 3:
            continue

        reste, date_visible, date_collecte = parts
        parts2 = reste.split(",", 2)
        if len(parts2) != 3:
            continue

        source, type_, texte = parts2

        texte = texte.strip()
        if texte.startswith('"') and texte.endswith('"'):
            texte = texte[1:-1]

        rows.append(
            {
                "source": source,
                "type": type_,
                "texte": nettoyer_texte_minimal(texte),
                "date_visible": date_visible.strip() or None,
                "date_collecte": date_collecte.strip() or None,
            }
        )

    if not rows:
        print("⚠️ Aucune ligne valide trouvée dans facebook_posts_commentaires.csv")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["plateforme"] = "facebook"
    df["strategie"] = None
    df["mot_cle"] = None
    df["id_source"] = df.apply(
        lambda row: generer_id_source(
            row.get("texte"), row.get("date_visible"), row.get("date_collecte")
        ),
        axis=1,
    )
    df["source_detail"] = "posts_commentaires"
    return df


def parse_keywords_file() -> pd.DataFrame:
    """
    Parse facebook_keywords.csv comme texte :
    format attendu :
    source,strategie,mot_cle,"texte, qui peut contenir, des virgules",date_collecte
    """
    if not KEYWORDS_FILE.exists():
        print(f"⚠️ Fichier non trouvé : {KEYWORDS_FILE}")
        return pd.DataFrame()

    lignes = KEYWORDS_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()
    rows = []

    for line in lignes:
        line = line.strip()
        if not line:
            continue

        parts = line.rsplit(",", 1)
        if len(parts) != 2:
            continue

        texte_part, date_collecte = parts
        parts2 = texte_part.split(",", 3)
        if len(parts2) != 4:
            continue

        source, strategie, mot_cle, texte = parts2

        texte = texte.strip()
        if texte.startswith('"') and texte.endswith('"'):
            texte = texte[1:-1]

        rows.append(
            {
                "source": source,
                "strategie": strategie,
                "mot_cle": mot_cle,
                "texte": nettoyer_texte_minimal(texte),
                "date_collecte": date_collecte.strip() or None,
            }
        )

    if not rows:
        print("⚠️ Aucune ligne valide trouvée dans facebook_keywords.csv")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["plateforme"] = "facebook"
    df["type"] = "post"
    df["date_visible"] = None
    df["id_source"] = df.apply(
        lambda row: generer_id_source(
            row.get("texte"), row.get("date_visible"), row.get("date_collecte")
        ),
        axis=1,
    )
    df["source_detail"] = "keywords"
    return df


def main():
    print("📥 Parsing manuel des données Facebook...")

    df_posts = parse_posts_file()
    df_keywords = parse_keywords_file()

    if df_posts.empty and df_keywords.empty:
        print("⚠️ Aucun corpus Facebook chargé, arrêt.")
        return

    colonnes_cibles = [
        "id_source",
        "plateforme",
        "source",
        "source_detail",
        "strategie",
        "mot_cle",
        "type",
        "texte",
        "date_visible",
        "date_collecte",
    ]

    for col in colonnes_cibles:
        if col not in df_posts.columns:
            df_posts[col] = None
        if col not in df_keywords.columns:
            df_keywords[col] = None

    df_posts = df_posts[colonnes_cibles]
    df_keywords = df_keywords[colonnes_cibles]

    df_all = pd.concat([df_posts, df_keywords], ignore_index=True)
    print(f"📊 Total brut (posts + keywords) : {len(df_all)} lignes")

    df_all["texte"] = df_all["texte"].astype(str).apply(nettoyer_texte_minimal)
    df_all = df_all[df_all["texte"].str.len() >= 5]
    print(f"📊 Après filtrage textes courts : {len(df_all)} lignes")

    avant = len(df_all)
    df_all = df_all.drop_duplicates(subset=["id_source"])
    apres = len(df_all)
    print(f"🧹 Doublons supprimés sur id_source : {avant - apres} lignes")

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    df_all.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print("✅ Corpus Facebook nettoyé sauvegardé")
    print(f"📁 Fichier : {OUTPUT_FILE}")
    print(f"📊 Nombre final de lignes : {len(df_all)}")


if __name__ == "__main__":
    main()
