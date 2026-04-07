"""
Déduplication finale et vérification qualité des données labellisées
Supprime les doublons stricts et quasi-doublons
"""

import sys
from pathlib import Path
import pandas as pd
from difflib import SequenceMatcher

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.config import LABELED_DATA_DIR
from utils.logger import setup_logger
from utils.file_handler import save_csv

logger = setup_logger("deduplication")


def similarite_textes(texte1: str, texte2: str) -> float:
    """Calcule la similarité entre deux textes (0-1)"""
    return SequenceMatcher(None, texte1, texte2).ratio()


def charger_labels() -> pd.DataFrame:
    """Charge les données labellisées"""
    logger.info("📥 Chargement des données labellisées...")
    
    filepath = LABELED_DATA_DIR / "enquete_labellisee.csv"
    
    if not filepath.exists():
        logger.error(f"❌ Fichier introuvable : {filepath}")
        raise FileNotFoundError(filepath)
    
    df = pd.read_csv(filepath, encoding="utf-8")
    logger.info(f"   Lignes chargées : {len(df)}")
    
    return df


def supprimer_doublons_stricts(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les doublons exacts"""
    logger.info("\n🧹 SUPPRESSION DES DOUBLONS STRICTS")
    
    avant = len(df)
    
    # Doublons sur texte exact
    df = df.drop_duplicates(subset=["texte"], keep="first")
    
    apres = len(df)
    supprimes = avant - apres
    
    logger.info(f"   Doublons supprimés : {supprimes}")
    logger.info(f"   Textes conservés   : {apres}")
    
    return df


def supprimer_quasi_doublons(df: pd.DataFrame, seuil: float = 0.95) -> pd.DataFrame:
    """
    Supprime les textes très similaires (>95% de similarité)
    Garde le premier rencontré
    """
    logger.info(f"\n🔍 SUPPRESSION DES QUASI-DOUBLONS (similarité > {seuil:.0%})")
    
    textes_a_garder = []
    indices_a_supprimer = set()
    
    for idx, row in df.iterrows():
        if idx in indices_a_supprimer:
            continue
        
        texte_actuel = str(row["texte"]).lower()
        
        # Comparer avec les textes déjà gardés
        est_doublon = False
        for texte_garde in textes_a_garder:
            sim = similarite_textes(texte_actuel, texte_garde)
            if sim > seuil:
                est_doublon = True
                indices_a_supprimer.add(idx)
                break
        
        if not est_doublon:
            textes_a_garder.append(texte_actuel)
    
    avant = len(df)
    df = df.drop(index=list(indices_a_supprimer))
    df = df.reset_index(drop=True)
    apres = len(df)
    
    logger.info(f"   Quasi-doublons supprimés : {avant - apres}")
    logger.info(f"   Textes conservés         : {apres}")
    
    return df


def verifier_qualite(df: pd.DataFrame) -> dict:
    """Vérifie la qualité du dataset"""
    logger.info("\n✅ VÉRIFICATION QUALITÉ")
    
    stats = {
        "total": len(df),
        "textes_vides": df["texte"].isna().sum(),
        "textes_courts": (df["texte"].str.len() < 10).sum(),
        "labels_manquants": df["label"].isna().sum(),
        "sentiments_manquants": df["sentiment"].isna().sum(),
    }
    
    logger.info(f"   Total textes           : {stats['total']}")
    logger.info(f"   Textes vides           : {stats['textes_vides']}")
    logger.info(f"   Textes < 10 chars      : {stats['textes_courts']}")
    logger.info(f"   Labels manquants       : {stats['labels_manquants']}")
    logger.info(f"   Sentiments manquants   : {stats['sentiments_manquants']}")
    
    return stats


def nettoyer_donnees_invalides(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les données invalides"""
    logger.info("\n🧹 NETTOYAGE DES DONNÉES INVALIDES")
    
    avant = len(df)
    
    df = df[df["texte"].notna()].copy()
    df = df[df["texte"].str.len() >= 10].copy()
    
    # Supprimer labels manquants
    df = df[df["label"].notna()].copy()
    df = df[df["sentiment"].notna()].copy()
    
    apres = len(df)
    
    logger.info(f"   Données invalides supprimées : {avant - apres}")
    logger.info(f"   Données valides conservées   : {apres}")
    
    return df


def analyser_distribution(df: pd.DataFrame) -> None:
    """Analyse la distribution des labels"""
    logger.info("\n📊 DISTRIBUTION DES LABELS")
    
    total = len(df)
    
    for sentiment in ["negative", "neutral", "positive"]:
        count = (df["sentiment"] == sentiment).sum()
        pct = (count / total) * 100
        emoji = {"negative": "😡", "neutral": "😐", "positive": "😊"}.get(sentiment, "")
        logger.info(f"   {emoji} {sentiment:10s} : {count:4d} ({pct:5.2f}%)")
    
    # Distribution par source
    if "source_label" in df.columns:
        logger.info("\n📊 DISTRIBUTION PAR SOURCE")
        for source, count in df["source_label"].value_counts().items():
            pct = (count / total) * 100
            logger.info(f"   {source:25s} : {count:4d} ({pct:5.2f}%)")


def sauvegarder_dataset_final(df: pd.DataFrame) -> Path:
    """Sauvegarde le dataset final dédupliqué"""
    logger.info("\n💾 SAUVEGARDE DU DATASET FINAL")
    
    # Backup de l'original
    filepath_original = LABELED_DATA_DIR / "enquete_labellisee.csv"
    if filepath_original.exists():
        from datetime import datetime
        backup_path = LABELED_DATA_DIR / f"enquete_labellisee_avant_dedup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        import shutil
        shutil.copy(filepath_original, backup_path)
        logger.info(f"   📦 Backup créé : {backup_path.name}")
    
    # Sauvegarder version dédupliquée
    filepath_final = LABELED_DATA_DIR / "dataset_final_dedup.csv"
    save_csv(df, filepath_final)
    
    logger.info(f"   📁 Fichier : {filepath_final}")
    logger.info(f"   📊 Lignes  : {len(df)}")
    
    return filepath_final


def main():
    """Point d'entrée principal"""
    print("="*70)
    print("🧹 DÉDUPLICATION FINALE - SENELEC")
    print("="*70)
    
    try:
        # Charger données
        df = charger_labels()
        
        # Supprimer doublons stricts
        df = supprimer_doublons_stricts(df)
        
        # Supprimer quasi-doublons
        df = supprimer_quasi_doublons(df, seuil=0.95)
        
        # Nettoyer données invalides
        df = nettoyer_donnees_invalides(df)
        
        # Vérifier qualité
        stats = verifier_qualite(df)
        
        # Analyser distribution
        analyser_distribution(df)
        
        # Sauvegarder
        filepath = sauvegarder_dataset_final(df)
        
        print("\n" + "="*70)
        print("✅ DÉDUPLICATION TERMINÉE AVEC SUCCÈS")
        print("="*70)
        print(f"📁 Fichier : {filepath}")
        print(f"📊 Total   : {len(df)} textes labellisés")
        print("="*70)
        
    except Exception as e:
        logger.error(f"❌ Erreur : {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()