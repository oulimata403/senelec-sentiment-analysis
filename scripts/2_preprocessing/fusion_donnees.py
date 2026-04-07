"""
Fusion complète des données Facebook, Twitter et Enquête
Harmonisation des colonnes et création du corpus unifié
"""

import sys
from pathlib import Path
import pandas as pd
import hashlib
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from utils.logger import setup_logger
from utils.file_handler import save_csv

logger = setup_logger("fusion_donnees")


def generer_id_unique(texte: str, source: str, date: str = None) -> str:
    """Génère un ID unique pour chaque entrée"""
    base = f"{texte}_{source}_{date or datetime.now().isoformat()}"
    return hashlib.md5(base.encode("utf-8")).hexdigest()


def charger_facebook() -> pd.DataFrame:
    """Charge et harmonise les données Facebook"""
    logger.info("Chargement des données Facebook...")
    
    filepath = PROCESSED_DATA_DIR / "corpus_facebook_nettoye.csv"
    
    if not filepath.exists():
        logger.warning(f"Fichier Facebook introuvable : {filepath}")
        return pd.DataFrame()
    
    df = pd.read_csv(filepath, encoding="utf-8")
    
    # Harmonisation colonnes
    df_harmonise = pd.DataFrame({
        "id_unique": df["id_source"],
        "plateforme": "facebook",
        "source": df["source"],
        "type_contenu": df["type"],
        "texte": df["texte"],
        "date_publication": pd.to_datetime(df["date_visible"], errors="coerce"),
        "date_collecte": pd.to_datetime(df["date_collecte"], errors="coerce"),
        "strategie_collecte": df["strategie"].fillna("page_officielle"),
        "mot_cle_recherche": df["mot_cle"],
    })
    
    logger.info(f"✅ Facebook : {len(df_harmonise)} entrées")
    return df_harmonise


def charger_twitter() -> pd.DataFrame:
    """Charge et harmonise les données Twitter"""
    logger.info("Chargement des données Twitter...")
    
    filepath = RAW_DATA_DIR / "twitter_keywords.csv"
    
    if not filepath.exists():
        logger.warning(f"Fichier Twitter introuvable : {filepath}")
        return pd.DataFrame()
    
    df = pd.read_csv(filepath, encoding="utf-8")
    
    # Générer IDs uniques si manquants
    if "id_unique" not in df.columns:
        df["id_unique"] = df.apply(
            lambda row: generer_id_unique(
                row["texte"], 
                "twitter", 
                str(row.get("date_visible"))
            ), 
            axis=1
        )
    
    # Harmonisation colonnes
    df_harmonise = pd.DataFrame({
        "id_unique": df["id_unique"],
        "plateforme": "twitter",
        "source": df["source"],
        "type_contenu": "tweet",
        "texte": df["texte"],
        "date_publication": pd.to_datetime(df["date_visible"], errors="coerce"),
        "date_collecte": pd.to_datetime(df["date_collecte"], errors="coerce"),
        "strategie_collecte": df["strategie"],
        "mot_cle_recherche": df["mot_cle"],
    })
    
    logger.info(f"✅ Twitter : {len(df_harmonise)} entrées")
    return df_harmonise


def charger_enquete() -> pd.DataFrame:
    """Charge et harmonise les données de l'enquête terrain"""
    logger.info("Chargement des données enquête...")
    
    filepath = RAW_DATA_DIR / "Enquête_SENELEC.csv"
    
    if not filepath.exists():
        logger.warning(f"Fichier enquête introuvable : {filepath}")
        return pd.DataFrame()
    
    df = pd.read_csv(filepath, encoding="utf-8")
    
    # Extraire les champs textuels pertinents
    colonnes_texte = [
        "Quels sont les principaux problèmes que vous rencontrez ?\n(Plusieurs réponses possibles)",
        "Quels sont les points positifs que vous notez dans les services de la SENELEC ?",
        "Quelles suggestions d'amélioration proposez-vous pour les services de la SENELEC ?",
        "Si oui, quel type d'expériences partagez-vous le plus souvent ?  "
    ]
    
    # Créer une ligne par réponse textuelle
    rows = []
    for idx, row in df.iterrows():
        for col in colonnes_texte:
            if col in df.columns and pd.notna(row[col]) and str(row[col]).strip():
                texte = str(row[col]).strip()
                
                # Générer ID unique
                id_unique = generer_id_unique(texte, "enquete", str(row["Horodateur"]))
                
                rows.append({
                    "id_unique": id_unique,
                    "plateforme": "enquete",
                    "source": "questionnaire_terrain",
                    "type_contenu": col.split("?")[0].strip(),
                    "texte": texte,
                    "date_publication": pd.to_datetime(row["Horodateur"], errors="coerce"),
                    "date_collecte": pd.to_datetime(row["Horodateur"], errors="coerce"),
                    "strategie_collecte": "enquete_terrain",
                    "mot_cle_recherche": None,
                    # Métadonnées enquête
                    "age": row.get("Votre tranche d'âge ?"),
                    "sexe": row.get("Votre Sexe ?"),
                    "region": row.get("Dans quelle région résidez-vous ?  "),
                    "type_client": row.get("Quel est votre type de client SENELEC ?"),
                    "satisfaction_globale": row.get("De manière générale, êtes-vous satisfait(e) des services de la SENELEC ?"),
                })
    
    df_harmonise = pd.DataFrame(rows)
    
    logger.info(f"✅ Enquête : {len(df_harmonise)} entrées textuelles extraites")
    return df_harmonise


def fusionner_corpus() -> pd.DataFrame:
    """Fusionne toutes les sources en un corpus unique"""
    logger.info("=" * 70)
    logger.info("FUSION DES CORPUS")
    logger.info("=" * 70)
    
    # Charger les 3 sources
    df_facebook = charger_facebook()
    df_twitter = charger_twitter()
    df_enquete = charger_enquete()
    
    # Colonnes communes minimales
    colonnes_communes = [
        "id_unique",
        "plateforme",
        "source",
        "type_contenu",
        "texte",
        "date_publication",
        "date_collecte",
        "strategie_collecte",
        "mot_cle_recherche",
    ]
    
    # Assurer que toutes les colonnes communes existent
    for df in [df_facebook, df_twitter, df_enquete]:
        for col in colonnes_communes:
            if col not in df.columns:
                df[col] = None
    
    # Fusion
    corpus_complet = pd.concat(
        [
            df_facebook[colonnes_communes],
            df_twitter[colonnes_communes],
            df_enquete[colonnes_communes],
        ],
        ignore_index=True
    )
    
    logger.info("\n📊 RÉSUMÉ DE LA FUSION")
    logger.info(f"   Facebook  : {len(df_facebook)} entrées")
    logger.info(f"   Twitter   : {len(df_twitter)} entrées")
    logger.info(f"   Enquête   : {len(df_enquete)} entrées")
    logger.info(f"   TOTAL     : {len(corpus_complet)} entrées")
    
    # Statistiques par plateforme
    logger.info("\n📈 RÉPARTITION PAR PLATEFORME")
    for pf, count in corpus_complet["plateforme"].value_counts().items():
        pct = (count / len(corpus_complet)) * 100
        logger.info(f"   {pf:12s} : {count:5d} ({pct:5.2f}%)")
    
    return corpus_complet


def nettoyer_doublons(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les doublons stricts"""
    logger.info("\n🧹 NETTOYAGE DES DOUBLONS")
    
    avant = len(df)
    
    # Doublons sur id_unique
    df = df.drop_duplicates(subset=["id_unique"], keep="first")
    
    apres = len(df)
    supprimes = avant - apres
    
    logger.info(f"   Doublons supprimés : {supprimes}")
    logger.info(f"   Corpus final       : {apres} entrées")
    
    return df


def verifier_qualite(df: pd.DataFrame) -> None:
    """Vérifie la qualité du corpus fusionné"""
    logger.info("\n✅ VÉRIFICATION QUALITÉ")
    
    # Textes vides
    vides = df["texte"].isna().sum()
    logger.info(f"   Textes vides       : {vides}")
    
    # Textes trop courts
    df["longueur_texte"] = df["texte"].astype(str).str.len()
    courts = (df["longueur_texte"] < 10).sum()
    logger.info(f"   Textes < 10 chars  : {courts}")
    
    # Dates manquantes
    dates_manquantes = df["date_publication"].isna().sum()
    logger.info(f"   Dates manquantes   : {dates_manquantes}")
    
    # Distribution longueurs
    logger.info(f"\n📏 STATISTIQUES LONGUEUR TEXTE")
    logger.info(f"   Minimum  : {df['longueur_texte'].min()}")
    logger.info(f"   Maximum  : {df['longueur_texte'].max()}")
    logger.info(f"   Moyenne  : {df['longueur_texte'].mean():.2f}")
    logger.info(f"   Médiane  : {df['longueur_texte'].median():.2f}")


def sauvegarder_corpus(df: pd.DataFrame) -> Path:
    """Sauvegarde le corpus fusionné"""
    logger.info("\n💾 SAUVEGARDE DU CORPUS")
    
    filepath = PROCESSED_DATA_DIR / "corpus_fusionne_brut.csv"
    save_csv(df, filepath)
    
    logger.info(f"   Fichier : {filepath}")
    logger.info(f"   Lignes  : {len(df)}")
    logger.info(f"   Colonnes: {len(df.columns)}")
    
    return filepath


def main():
    """Point d'entrée principal"""
    print("=" * 70)
    print("🔗 FUSION DES CORPUS - SENELEC")
    print("=" * 70)
    
    try:
        # Fusion
        corpus = fusionner_corpus()
        
        # Nettoyage doublons
        corpus = nettoyer_doublons(corpus)
        
        # Vérification qualité
        verifier_qualite(corpus)
        
        # Sauvegarde
        filepath = sauvegarder_corpus(corpus)
        
        print("\n✅ FUSION TERMINÉE AVEC SUCCÈS")
        print(f"📁 Fichier : {filepath}")
        print(f"📊 Total   : {len(corpus)} entrées")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la fusion : {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    print("=" * 70)


if __name__ == "__main__":
    main()