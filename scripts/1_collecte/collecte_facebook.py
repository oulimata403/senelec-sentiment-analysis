from playwright.sync_api import sync_playwright
import pandas as pd
from datetime import datetime
import time
import os
import re
import hashlib

from config.config import FACEBOOK_SESSION_PATH, FACEBOOK_POSTS_OUTPUT


URLS = {
    "page": "https://web.facebook.com/senelecofficiel",
    "groupe": "https://web.facebook.com/groups/1198315676888578",
}


def generer_id_source(texte: str, date_visible: str | None = None) -> str:
    """Génère un identifiant unique pour un contenu à partir du texte et de la date."""
    base = (texte or "") + (date_visible or "")
    return hashlib.md5(base.encode("utf-8")).hexdigest()


def nettoyer_texte_brut(texte: str) -> str:
    """Nettoie le texte Facebook (supprime boutons, statuts, dates relatives, etc.)."""
    if not texte:
        return ""

    t = texte

    # retirer boutons d'action
    t = t.replace("J’aime", "").replace("J'aime", "")
    t = t.replace("Répondre", "")
    t = t.replace("Partager", "")
    t = t.replace("Suivre", "")
    t = t.replace("En voir plus", "")
    t = t.replace("Voir la traduction", "")
    t = t.replace("Afficher la suite", "")

    # retirer rôles / statuts fréquents
    for mot in [
        "Super fan",
        "Auteur",
        "Admin",
        "Contributeur(ice) star",
        "Contributeur(ice) Star",
        "Contributeur star",
    ]:
        t = t.replace(mot, "")

    # remplacer certains séparateurs
    t = t.replace("·", " ")

    t = re.sub(r"\b\d+\s*(sem\.?|an[s]?)\b", " ", t)

    # enlever des compteurs simples (ex: "3", "11", souvent des nombres de réactions)
    t = re.sub(r"\b\d+\b", " ", t)

    # compresser les espaces
    t = " ".join(t.split())
    return t.strip()


def cliquer_tous_les_en_voir_plus(page):
    """Clique automatiquement sur tous les 'En voir plus' visibles sur la page."""
    locator = page.get_by_text("En voir plus")
    try:
        count = locator.count()
    except Exception:
        count = 0

    print(f"🔎 {count} boutons 'En voir plus' détectés (globaux)")

    for i in range(count):
        try:
            btn = locator.nth(i)
            if btn.is_visible():
                btn.click()
                page.wait_for_timeout(500)
        except Exception:
            continue


def collecter_posts_et_commentaires(url: str, source: str, scrolls: int = 80):
    donnees = []

    with sync_playwright() as p:
        navigateur = p.chromium.launch(headless=False)
        contexte = navigateur.new_context(storage_state=str(FACEBOOK_SESSION_PATH))
        page = contexte.new_page()

        try:
            page.goto(url, wait_until="domcontentloaded", timeout=120000)
        except Exception as e:
            print(f"⚠️ Erreur chargement {source} : {e}")
            navigateur.close()
            return []

        time.sleep(10)

        # scroll profond pour remonter dans l'historique
        for _ in range(scrolls):
            page.mouse.wheel(0, 5000)
            time.sleep(4)

        # 1) Cliquer automatiquement sur tous les "En voir plus" (posts + commentaires)
        cliquer_tous_les_en_voir_plus(page)

        # 2) Récupérer tous les posts visibles
        posts = page.query_selector_all("div[role='article']")
        print(f"🔎 {len(posts)} posts détectés sur {source}")

        for post in posts:
            try:
                # au cas où certains 'En voir plus' seraient dans le post seulement
                try:
                    post.get_by_text("En voir plus").click()
                    page.wait_for_timeout(300)
                except Exception:
                    pass

                texte_post = post.inner_text().strip()
                texte_post = nettoyer_texte_brut(texte_post)

                # filtrer les posts trop courts
                if len(texte_post) < 30:
                    continue

                date_element = post.query_selector("a[aria-label]")
                date_texte = (
                    date_element.get_attribute("aria-label") if date_element else None
                )

                id_post = generer_id_source(texte_post, date_texte)

                donnees.append(
                    {
                        "id_source": id_post,
                        "plateforme": "facebook",
                        "source": source,
                        "type": "post",
                        "texte": texte_post,
                        "date_visible": date_texte,
                        "date_collecte": datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                    }
                )

                # 3) Essayer d'ouvrir les commentaires pour ce post
                try:
                    btn_more_comments = post.locator("span").filter(
                        has_text=re.compile(
                            "commentaire|commentaires|comments", re.IGNORECASE
                        )
                    )
                    count_btn = btn_more_comments.count()
                    for i in range(count_btn):
                        try:
                            btn_more_comments.nth(i).click()
                            page.wait_for_timeout(1500)
                        except Exception:
                            continue
                except Exception:
                    pass

                try:
                    post.get_by_text("En voir plus").click()
                    page.wait_for_timeout(500)
                except Exception:
                    pass

                # 4) Récupérer les commentaires à l'intérieur du post
                comment_nodes = post.query_selector_all("div[aria-label='Commentaire']") or post.query_selector_all(
                    "div[aria-label='Comment']"
                )

                for c in comment_nodes:
                    try:
                        # déplier les "En voir plus" dans le commentaire
                        try:
                            c.get_by_text("En voir plus").click()
                            page.wait_for_timeout(300)
                        except Exception:
                            pass

                        txt_c = c.inner_text().strip()
                        txt_c = nettoyer_texte_brut(txt_c)
                        if len(txt_c) < 5:
                            continue

                        id_com = generer_id_source(txt_c, None)

                        donnees.append(
                            {
                                "id_source": id_com,
                                "plateforme": "facebook",
                                "source": source,
                                "type": "commentaire",
                                "texte": txt_c,
                                "date_visible": None,
                                "date_collecte": datetime.now().strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                ),
                            }
                        )
                    except Exception:
                        continue

            except Exception:
                continue

        navigateur.close()

    return donnees


if __name__ == "__main__":
    toutes_donnees = []

    for src, url in URLS.items():
        print(f"📥 Collecte Facebook : {src}")
        toutes_donnees.extend(collecter_posts_et_commentaires(url, src))

    df = pd.DataFrame(toutes_donnees)
    os.makedirs(FACEBOOK_POSTS_OUTPUT.parent, exist_ok=True)

    if FACEBOOK_POSTS_OUTPUT.exists():
        df.to_csv(
            FACEBOOK_POSTS_OUTPUT,
            mode="a",
            header=False,
            index=False,
            encoding="utf-8",
        )
    else:
        df.to_csv(
            FACEBOOK_POSTS_OUTPUT,
            mode="w",
            header=True,
            index=False,
            encoding="utf-8",
        )

    print(f"✅ {len(df)} éléments Facebook collectés (posts + commentaires)")
    print(f"📁 Fichier mis à jour : {FACEBOOK_POSTS_OUTPUT}")
