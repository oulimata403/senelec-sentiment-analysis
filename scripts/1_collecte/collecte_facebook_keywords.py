from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import pandas as pd
from datetime import datetime
import time
import os
import re
import hashlib

from config.config import FACEBOOK_SESSION_PATH, FACEBOOK_KEYWORDS_OUTPUT

SEARCH_URL = "https://web.facebook.com/search/posts?q={query}&locale=fr_FR"

MOTS_CLES = [
    "SENELEC",
    "Woyofal",
    "coupure",
    "facture",
    "courant",
    "délestage",
    "panne",
]


def generer_id_source(texte: str, date_visible: str | None = None) -> str:
    """Génère un identifiant unique pour un post à partir du texte et de la date."""
    base = (texte or "") + (date_visible or "")
    return hashlib.md5(base.encode("utf-8")).hexdigest()


def nettoyer_texte(texte: str) -> str:
    if not texte:
        return ""

    t = texte

    # 1) Enlever mention "Facebook"
    t = t.replace("Facebook", "")

    # 2) Supprimer les boutons / textes d'interface
    t = re.sub(
        r"(J’aime|J'aime|Répondre|Partager|Suivre|En voir plus|Voir la traduction)",
        "",
        t,
    )
    t = t.replace("Voir les notifications précédentes", "")

    # 3) Supprimer tout le bloc "Toutes les réactions ... Commenter"
    t = re.sub(r"Toutes les réactions.*?Commenter", "", t, flags=re.DOTALL)

    t = re.sub(r"\b\d+\s*(sem\.?|an[s]?)\b", "", t)

    # 5) Supprimer les séquences de lettres isolées (artefacts)
    t = re.sub(r"(?:\b\w\b\s*){6,}", " ", t)

    # 6) Supprimer séparateur '·' et compresser les espaces
    t = t.replace("·", " ")
    t = re.sub(r"\s+", " ", t)

    return t.strip()


def extraire_texte_post(node):
    """
    Essaie d'extraire le texte du contenu du post, pas tout le bloc.
    Heuristique : chercher des span/div à l'intérieur.
    """
    candidates = node.query_selector_all("span, div")
    best = ""
    for c in candidates:
        try:
            txt = c.inner_text().strip()
            if len(txt) < 20:
                continue
            if "Toutes les réactions" in txt:
                continue
            if "Voir les notifications précédentes" in txt:
                continue
            if len(txt) > len(best):
                best = txt
        except Exception:
            continue

    if not best:
        best = node.inner_text().strip()

    return best


def collecter_par_mot_cle_global(mot_cle: str, scrolls: int = 20):
    donnees = []

    with sync_playwright() as p:
        navigateur = p.chromium.launch(headless=False)
        contexte = navigateur.new_context(storage_state=str(FACEBOOK_SESSION_PATH))
        page = contexte.new_page()

        url = SEARCH_URL.format(query=mot_cle)
        print(f"🌐 Recherche Facebook globale (Publications) : {mot_cle}")
        print(f"URL : {url}")

        try:
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
        except PlaywrightTimeoutError:
            print("⚠️ Timeout sur le chargement — on continue quand même.")

        time.sleep(8)

        for _ in range(scrolls):
            page.mouse.wheel(0, 5000)
            time.sleep(4)

        time.sleep(3)

        blocs = page.query_selector_all("div[role='article']")
        print(f"🔎 {len(blocs)} blocs role='article' pour '{mot_cle}'")

        if len(blocs) == 0:
            blocs = page.query_selector_all("div[data-ad-preview='message']")
            print(f"🔎 {len(blocs)} blocs data-ad-preview='message' pour '{mot_cle}'")

        print(f"🔎 Total blocs candidats pour '{mot_cle}' : {len(blocs)}")

        for node in blocs:
            try:
                brut = extraire_texte_post(node)
                texte = nettoyer_texte(brut)

                if not texte or len(texte) < 30:
                    continue

                if texte.startswith("Voir les notifications précédentes"):
                    continue

                id_src = generer_id_source(texte)

                donnees.append(
                    {
                        "id_source": id_src,
                        "plateforme": "facebook",
                        "source": "global_posts",
                        "strategie": "mot_cle",
                        "mot_cle": mot_cle,
                        "texte": texte,
                        "date_visible": None,
                        "date_collecte": datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                    }
                )
            except Exception:
                continue

        navigateur.close()

    return donnees


if __name__ == "__main__":
    toutes_donnees = []

    for mot in MOTS_CLES:
        print(f"\n📥 Recherche globale – mot-clé : {mot}")
        try:
            res = collecter_par_mot_cle_global(mot)
            if res:
                toutes_donnees.extend(res)
        except Exception as e:
            print(f"⚠️ Erreur pendant la collecte pour le mot-clé '{mot}' : {e}")

    if not toutes_donnees:
        print("⚠️ Aucune donnée collectée, CSV non mis à jour.")
        raise SystemExit(0)

    df = pd.DataFrame(toutes_donnees)
    os.makedirs(FACEBOOK_KEYWORDS_OUTPUT.parent, exist_ok=True)

    if FACEBOOK_KEYWORDS_OUTPUT.exists():
        df.to_csv(
            FACEBOOK_KEYWORDS_OUTPUT,
            mode="a",
            header=False,
            index=False,
            encoding="utf-8",
        )
    else:
        df.to_csv(
            FACEBOOK_KEYWORDS_OUTPUT,
            index=False,
            encoding="utf-8",
        )

    print(f"\n✅ {len(df)} éléments collectés par mots-clés (Publications)")
    print(f"📁 Fichier : {FACEBOOK_KEYWORDS_OUTPUT}")
