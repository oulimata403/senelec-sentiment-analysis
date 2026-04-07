"""
Préparation des données d'enquête pour labellisation du sentiment
Extraction des labels de sentiment à partir des questions de satisfaction
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.config import RAW_DATA_DIR, LABELED_DATA_DIR, SENTIMENT_LABELS
from utils.logger import setup_logger
from utils.file_handler import save_csv

logger = setup_logger("preparation_enquete")


class ExtracteurLabels:
    """Extrait les labels de sentiment depuis l'enquête"""
    
    def __init__(self):
        # Mapping des réponses vers sentiments
        self.mapping_satisfaction = {
            # Globalement satisfait
            "Très satisfait(e)": "positive",
            "Satisfait(e)": "positive",
            "Neutre": "neutral",
            "Insatisfait(e)": "negative",
            "Très insatisfait(e)": "negative",
            
            # Variations orthographiques
            "Très satisfait": "positive",
            "Satisfait": "positive",
            "Insatisfait": "negative",
            "Très insatisfait": "negative",
        }
        
        self.mapping_evaluation = {
            # Pour questions d'évaluation
            "Très bien": "positive",
            "Bien": "positive",
            "Moyen": "neutral",
            "Mauvais": "negative",
            "Très mauvais": "negative",
            
            "Excellent": "positive",
            "Bon": "positive",
            "Acceptable": "neutral",
            "Médiocre": "negative",
            "Très médiocre": "negative",
        }
        
        self.mapping_woyofal = {
            # Spécifique Woyofal
            "Très satisfait(e)": "positive",
            "Satisfait(e)": "positive",
            "Neutre": "neutral",
            "Insatisfait(e)": "negative",
            "Très insatisfait(e)": "negative",
        }
    
    def extraire_sentiment_global(self, row: pd.Series) -> str:
        """Extrait le sentiment à partir de la satisfaction globale"""
        satisfaction = row.get("De manière générale, êtes-vous satisfait(e) des services de la SENELEC ?")
        
        if pd.isna(satisfaction):
            return None
        
        return self.mapping_satisfaction.get(str(satisfaction).strip(), None)
    
    def extraire_sentiment_woyofal(self, row: pd.Series) -> str:
        """Extrait le sentiment spécifique Woyofal"""
        satisfaction_woyofal = row.get("Globalement, comment évaluez-vous votre satisfaction vis-à-vis du système Woyofal ?")
        
        if pd.isna(satisfaction_woyofal):
            return None
        
        return self.mapping_woyofal.get(str(satisfaction_woyofal).strip(), None)
    
    def extraire_sentiment_interaction(self, row: pd.Series) -> str:
        """Extrait le sentiment de la dernière interaction"""
        evaluation = row.get("Comment évalueriez-vous votre dernière interaction avec la SENELEC (achat de crédit, coupure, service client, facturation, etc.) ?  ")
        
        if pd.isna(evaluation):
            return None
        
        return self.mapping_evaluation.get(str(evaluation).strip(), None)
    
    def extraire_sentiment_service_client(self, row: pd.Series) -> str:
        """Extrait le sentiment sur le service client"""
        acces_service = row.get("Facilité d'accès au service client (agence, centre d'appel, etc.) :  ")
        
        if pd.isna(acces_service):
            return None
        
        return self.mapping_evaluation.get(str(acces_service).strip(), None)


def charger_enquete() -> pd.DataFrame:
    """Charge les données d'enquête"""
    logger.info("📥 Chargement de l'enquête...")
    
    filepath = RAW_DATA_DIR / "Enquête_SENELEC.csv"
    
    if not filepath.exists():
        logger.error(f"❌ Fichier enquête introuvable : {filepath}")
        raise FileNotFoundError(filepath)
    
    df = pd.read_csv(filepath, encoding="utf-8")
    logger.info(f"   Lignes chargées : {len(df)}")
    
    return df


def extraire_textes_labellises(df_enquete: pd.DataFrame) -> pd.DataFrame:
    """Extrait les textes avec leurs labels de sentiment"""
    logger.info("\n🏷️  EXTRACTION DES LABELS DE SENTIMENT")
    
    extracteur = ExtracteurLabels()
    
    textes_labellises = []
    
    for idx, row in df_enquete.iterrows():
        # 1. Problèmes rencontrés (sentiment négatif par défaut)
        problemes = row.get("Quels sont les principaux problèmes que vous rencontrez ?\n(Plusieurs réponses possibles)")
        if pd.notna(problemes) and str(problemes).strip():
            sentiment_global = extracteur.extraire_sentiment_global(row)
            
            textes_labellises.append({
                "texte": str(problemes).strip(),
                "sentiment": sentiment_global or "negative",  
                "source_label": "problemes_rencontres",
                "type_client": row.get("Quel est votre type de client SENELEC ?"),
                "region": row.get("Dans quelle région résidez-vous ?  "),
                "age": row.get("Votre tranche d'âge ?"),
            })
        
        # 2. Points positifs (sentiment positif par défaut)
        points_positifs = row.get("Quels sont les points positifs que vous notez dans les services de la SENELEC ?")
        if pd.notna(points_positifs) and str(points_positifs).strip():
            textes_labellises.append({
                "texte": str(points_positifs).strip(),
                "sentiment": "positive",  
                "source_label": "points_positifs",
                "type_client": row.get("Quel est votre type de client SENELEC ?"),
                "region": row.get("Dans quelle région résidez-vous ?  "),
                "age": row.get("Votre tranche d'âge ?"),
            })
        
        # 3. Suggestions d'amélioration (neutre ou négatif selon contexte)
        suggestions = row.get("Quelles suggestions d'amélioration proposez-vous pour les services de la SENELEC ?")
        if pd.notna(suggestions) and str(suggestions).strip():
            sentiment_global = extracteur.extraire_sentiment_global(row)
            
            textes_labellises.append({
                "texte": str(suggestions).strip(),
                "sentiment": sentiment_global or "neutral",
                "source_label": "suggestions",
                "type_client": row.get("Quel est votre type de client SENELEC ?"),
                "region": row.get("Dans quelle région résidez-vous ?  "),
                "age": row.get("Votre tranche d'âge ?"),
            })
        
        # 4. Expériences partagées sur réseaux sociaux
        experiences = row.get("Si oui, quel type d'expériences partagez-vous le plus souvent ?  ")
        if pd.notna(experiences) and str(experiences).strip():
            sentiment_global = extracteur.extraire_sentiment_global(row)
            
            textes_labellises.append({
                "texte": str(experiences).strip(),
                "sentiment": sentiment_global or "neutral",
                "source_label": "experiences_reseaux",
                "type_client": row.get("Quel est votre type de client SENELEC ?"),
                "region": row.get("Dans quelle région résidez-vous ?  "),
                "age": row.get("Votre tranche d'âge ?"),
            })
    
    df_labels = pd.DataFrame(textes_labellises)
    
    logger.info(f"   Total textes extraits : {len(df_labels)}")
    
    return df_labels


def analyser_distribution_labels(df: pd.DataFrame) -> None:
    """Analyse la distribution des labels"""
    logger.info("\n📊 DISTRIBUTION DES SENTIMENTS")
    
    distribution = df["sentiment"].value_counts()
    total = len(df)
    
    for sentiment, count in distribution.items():
        pct = (count / total) * 100
        logger.info(f"   {sentiment:12s} : {count:4d} ({pct:5.2f}%)")
    
    # Par source
    logger.info("\n📊 DISTRIBUTION PAR SOURCE")
    crosstab = pd.crosstab(df["source_label"], df["sentiment"], margins=True)
    logger.info("\n" + crosstab.to_string())
    
    # Par type de client
    if "type_client" in df.columns:
        logger.info("\n📊 DISTRIBUTION PAR TYPE CLIENT")
        crosstab2 = pd.crosstab(df["type_client"], df["sentiment"], margins=True)
        logger.info("\n" + crosstab2.to_string())


def nettoyer_textes_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les textes labellisés (minuscules, espaces)"""
    logger.info("\n🧹 NETTOYAGE DES TEXTES LABELLISÉS")
    
    avant = len(df)
    
    # Minuscules
    df["texte"] = df["texte"].str.lower().str.strip()
    
    # Supprimer textes vides
    df = df[df["texte"].str.len() >= 10].copy()
    
    # Supprimer doublons exacts
    df = df.drop_duplicates(subset=["texte"], keep="first")
    
    apres = len(df)
    
    logger.info(f"   Textes conservés : {apres}/{avant}")
    
    return df


def convertir_labels_numeriques(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit les labels texte en numériques pour le ML"""
    logger.info("\n🔢 CONVERSION DES LABELS")
    
    # Mapping vers codes numériques
    df["label"] = df["sentiment"].map(SENTIMENT_LABELS)
    
    # Vérifier labels manquants
    manquants = df["label"].isna().sum()
    if manquants > 0:
        logger.warning(f"   ⚠️  {manquants} labels non mappés")
    
    # Distribution labels numériques
    logger.info("\n   Distribution labels numériques :")
    for label_num, count in df["label"].value_counts().sort_index().items():
        label_text = [k for k, v in SENTIMENT_LABELS.items() if v == label_num][0]
        logger.info(f"      {label_num} ({label_text:8s}) : {count}")
    
    return df


def sauvegarder_donnees_labellisees(df: pd.DataFrame) -> Path:
    """Sauvegarde les données labellisées"""
    logger.info("\n💾 SAUVEGARDE DES DONNÉES LABELLISÉES")
    
    # Colonnes finales
    colonnes_finales = [
        "texte",
        "sentiment",
        "label",
        "source_label",
        "type_client",
        "region",
        "age",
    ]
    
    df_final = df[colonnes_finales].copy()
    
    filepath = LABELED_DATA_DIR / "enquete_labellisee.csv"
    save_csv(df_final, filepath)
    
    logger.info(f"   Fichier : {filepath}")
    logger.info(f"   Lignes  : {len(df_final)}")
    
    return filepath


def creer_statistiques_enquete(df_enquete: pd.DataFrame) -> Dict:
    """Crée des statistiques globales sur l'enquête"""
    logger.info("\n📈 STATISTIQUES GLOBALES ENQUÊTE")
    
    stats = {
        "total_repondants": len(df_enquete),
        "repartition_age": df_enquete["Votre tranche d'âge ?"].value_counts().to_dict(),
        "repartition_region": df_enquete["Dans quelle région résidez-vous ?  "].value_counts().to_dict(),
        "repartition_type_client": df_enquete["Quel est votre type de client SENELEC ?"].value_counts().to_dict(),
        "satisfaction_globale": df_enquete["De manière générale, êtes-vous satisfait(e) des services de la SENELEC ?"].value_counts().to_dict(),
    }
    
    logger.info(f"\n   Total répondants : {stats['total_repondants']}")
    
    logger.info("\n   Répartition type client :")
    for type_client, count in stats["repartition_type_client"].items():
        pct = (count / stats["total_repondants"]) * 100
        logger.info(f"      {type_client} : {count} ({pct:.1f}%)")
    
    return stats


def main():
    """Point d'entrée principal"""
    print("=" * 70)
    print("🏷️  PRÉPARATION ENQUÊTE POUR LABELLISATION - SENELEC")
    print("=" * 70)
    
    try:
        # Charger enquête
        df_enquete = charger_enquete()
        
        # Statistiques globales
        stats = creer_statistiques_enquete(df_enquete)
        
        # Extraire textes labellisés
        df_labels = extraire_textes_labellises(df_enquete)
        
        # Analyser distribution
        analyser_distribution_labels(df_labels)
        
        # Nettoyer
        df_labels = nettoyer_textes_labels(df_labels)
        
        # Convertir labels
        df_labels = convertir_labels_numeriques(df_labels)
        
        # Sauvegarder
        filepath = sauvegarder_donnees_labellisees(df_labels)
        
        print("\n✅ PRÉPARATION ENQUÊTE TERMINÉE")
        print(f"📁 Fichier : {filepath}")
        print(f"📊 Total   : {len(df_labels)} textes labellisés")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la préparation : {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    print("=" * 70)


if __name__ == "__main__":
    main()