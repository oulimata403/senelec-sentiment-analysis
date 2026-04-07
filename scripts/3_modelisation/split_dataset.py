"""
Split stratifié du dataset en train/val/test
Gestion du déséquilibre des classes
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.config import LABELED_DATA_DIR, NLP_CONFIG
from utils.logger import setup_logger
from utils.file_handler import save_csv

logger = setup_logger("split_dataset")


def charger_dataset() -> pd.DataFrame:
    """Charge le dataset dédupliqué"""
    logger.info("📥 Chargement du dataset...")
    
    filepath = LABELED_DATA_DIR / "dataset_final_dedup.csv"
    
    if not filepath.exists():
        logger.warning("⚠️  Dataset dédupliqué introuvable, chargement de l'original...")
        filepath = LABELED_DATA_DIR / "enquete_labellisee.csv"
    
    df = pd.read_csv(filepath, encoding="utf-8")
    logger.info(f"   Lignes chargées : {len(df)}")
    
    return df


def analyser_distribution(df: pd.DataFrame, nom: str = "Dataset") -> None:
    """Analyse la distribution des classes"""
    logger.info(f"\n📊 DISTRIBUTION {nom.upper()}")
    
    total = len(df)
    
    for label in sorted(df["label"].unique()):
        count = (df["label"] == label).sum()
        pct = (count / total) * 100
        sentiment = df[df["label"] == label]["sentiment"].iloc[0]
        emoji = {"negative": "😡", "neutral": "😐", "positive": "😊"}.get(sentiment, "")
        logger.info(f"   {emoji} {sentiment:10s} (label={label}) : {count:4d} ({pct:5.2f}%)")


def split_stratifie(df: pd.DataFrame) -> tuple:
    """
    Split stratifié en train/val/test
    
    Returns:
        (df_train, df_val, df_test)
    """
    logger.info("\n✂️  SPLIT STRATIFIÉ DU DATASET")
    
    # Paramètres
    test_size = NLP_CONFIG.get("test_size", 0.15)
    val_size = NLP_CONFIG.get("val_size", 0.15)
    random_state = NLP_CONFIG.get("random_state", 42)
    
    logger.info(f"   Test  : {test_size:.0%}")
    logger.info(f"   Val   : {val_size:.0%}")
    logger.info(f"   Train : {1 - test_size - val_size:.0%}")
    
    # Premier split : train+val / test
    df_trainval, df_test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state
    )
    
    # Deuxième split : train / val
    val_size_adjusted = val_size / (1 - test_size)  
    
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=val_size_adjusted,
        stratify=df_trainval["label"],
        random_state=random_state
    )
    
    logger.info(f"\n   Train : {len(df_train)} textes ({len(df_train)/len(df)*100:.1f}%)")
    logger.info(f"   Val   : {len(df_val)} textes ({len(df_val)/len(df)*100:.1f}%)")
    logger.info(f"   Test  : {len(df_test)} textes ({len(df_test)/len(df)*100:.1f}%)")
    
    return df_train, df_val, df_test


def calculer_poids_classes(df_train: pd.DataFrame) -> dict:
    """
    Calcule les poids pour rééquilibrer les classes
    
    Returns:
        dict: {label: poids}
    """
    logger.info("\n⚖️  CALCUL DES POIDS DE CLASSES")
    
    # Compter occurrences
    counts = df_train["label"].value_counts().sort_index()
    
    # Calculer poids inversement proportionnels
    total = len(df_train)
    n_classes = len(counts)
    
    poids = {}
    for label, count in counts.items():
        poids[label] = total / (n_classes * count)
    
    logger.info("\n   Poids calculés :")
    for label, weight in poids.items():
        sentiment = df_train[df_train["label"] == label]["sentiment"].iloc[0]
        logger.info(f"      {sentiment:10s} (label={label}) : {weight:.4f}")
    
    return poids


def sauvegarder_splits(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame) -> dict:
    """Sauvegarde les 3 splits"""
    logger.info("\n💾 SAUVEGARDE DES SPLITS")
    
    filepaths = {
        "train": LABELED_DATA_DIR / "train_set.csv",
        "val": LABELED_DATA_DIR / "val_set.csv",
        "test": LABELED_DATA_DIR / "test_set.csv",
    }
    
    save_csv(df_train, filepaths["train"])
    save_csv(df_val, filepaths["val"])
    save_csv(df_test, filepaths["test"])
    
    for nom, filepath in filepaths.items():
        logger.info(f"   {nom:5s} : {filepath}")
    
    return filepaths


def sauvegarder_poids_classes(poids: dict) -> Path:
    """Sauvegarde les poids des classes en JSON"""
    import json
    
    filepath = LABELED_DATA_DIR / "class_weights.json"
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(poids, f, indent=2)
    
    logger.info(f"\n   Poids sauvegardés : {filepath}")
    
    return filepath


def generer_rapport_split(df: pd.DataFrame, df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame) -> None:
    """Génère un rapport détaillé du split"""
    logger.info("\n" + "="*70)
    logger.info("📋 RAPPORT DE SPLIT")
    logger.info("="*70)
    
    # Dataset complet
    analyser_distribution(df, "complet")
    
    # Train
    analyser_distribution(df_train, "train")
    
    # Val
    analyser_distribution(df_val, "validation")
    
    # Test
    analyser_distribution(df_test, "test")
    
    logger.info("\n" + "="*70)


def main():
    """Point d'entrée principal"""
    print("="*70)
    print("✂️  SPLIT DATASET - SENELEC")
    print("="*70)
    
    try:
        # Charger dataset
        df = charger_dataset()
        
        # Analyser distribution initiale
        analyser_distribution(df, "initial")
        
        # Split stratifié
        df_train, df_val, df_test = split_stratifie(df)
        
        # Calculer poids classes
        poids = calculer_poids_classes(df_train)
        
        # Sauvegarder splits
        filepaths = sauvegarder_splits(df_train, df_val, df_test)
        
        # Sauvegarder poids
        filepath_poids = sauvegarder_poids_classes(poids)
        
        # Rapport final
        generer_rapport_split(df, df_train, df_val, df_test)
        
        print("\n" + "="*70)
        print("✅ SPLIT TERMINÉ AVEC SUCCÈS")
        print("="*70)
        print(f"📁 Train : {filepaths['train']}")
        print(f"📁 Val   : {filepaths['val']}")
        print(f"📁 Test  : {filepaths['test']}")
        print(f"📁 Poids : {filepath_poids}")
        print("="*70)
        
    except Exception as e:
        logger.error(f"❌ Erreur : {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()