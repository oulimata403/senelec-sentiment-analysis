"""
Labellisation semi-automatique avec modèle pré-entraîné
Utilise un modèle de sentiment français pour pré-labelliser les textes
Permet ensuite une validation manuelle rapide
"""

import sys
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.config import PROCESSED_DATA_DIR, LABELED_DATA_DIR, SENTIMENT_LABELS
from utils.logger import setup_logger
from utils.file_handler import save_csv

logger = setup_logger("labellisation_semi_auto")


class LabelliseurSemiAuto:
    """Labelliseur semi-automatique avec modèle pré-entraîné"""
    
    def __init__(self, model_name: str = "cmarkea/distilcamembert-base-sentiment"):
        """
        Initialise le modèle de sentiment
        
        Modèles disponibles :
        - "cmarkea/distilcamembert-base-sentiment" (français, rapide)
        - "nlptown/bert-base-multilingual-uncased-sentiment" (multilingue, 5 étoiles)
        """
        logger.info(f"🤖 Chargement du modèle : {model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            self.classifier = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("✅ Modèle chargé avec succès")
            
            # Mapping des labels du modèle vers nos labels
            self.mapping_labels = {
                # CamemBERT sentiment
                "1 star": "negative",
                "2 stars": "negative",
                "3 stars": "neutral",
                "4 stars": "positive",
                "5 stars": "positive",
                
                # Autres modèles possibles
                "NEGATIVE": "negative",
                "NEUTRAL": "neutral",
                "POSITIVE": "positive",
                "NEG": "negative",
                "NEU": "neutral",
                "POS": "positive",
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle : {e}")
            logger.info("💡 Installez transformers : pip install transformers torch")
            raise
    
    def predire_sentiment(self, texte: str) -> tuple:
        """
        Prédit le sentiment d'un texte
        
        Returns:
            (sentiment, confiance) où sentiment = 'negative', 'neutral', 'positive'
        """
        if not texte or len(texte) < 5:
            return ("neutral", 0.0)
        
        try:
            # Prédiction
            result = self.classifier(texte[:512])[0]  
            
            label_brut = result["label"]
            confiance = result["score"]
            
            # Mapper vers nos labels
            sentiment = self.mapping_labels.get(label_brut, "neutral")
            
            return (sentiment, confiance)
        
        except Exception as e:
            logger.warning(f"Erreur prédiction : {e}")
            return ("neutral", 0.0)
    
    def predire_batch(self, textes: list, batch_size: int = 16) -> list:
        """Prédit le sentiment pour un batch de textes"""
        resultats = []
        
        logger.info(f"🔮 Prédiction de {len(textes)} textes...")
        
        for i in tqdm(range(0, len(textes), batch_size)):
            batch = textes[i:i+batch_size]
            
            for texte in batch:
                sentiment, confiance = self.predire_sentiment(texte)
                resultats.append({
                    "sentiment": sentiment,
                    "confiance": confiance
                })
        
        return resultats


def charger_corpus_pour_labellisation(limite: int = 1000) -> pd.DataFrame:
    """Charge le corpus à pré-labelliser"""
    logger.info("📥 Chargement du corpus...")
    
    filepath = PROCESSED_DATA_DIR / "corpus_avec_langues.csv"
    
    if not filepath.exists():
        logger.error(f"❌ Fichier introuvable : {filepath}")
        raise FileNotFoundError(filepath)
    
    df = pd.read_csv(filepath, encoding="utf-8")
    
    df = df[df["plateforme"] != "enquete"].copy()
    
    # Prioriser textes français pour meilleure qualité
    df_fr = df[df["langue"] == "fr"].copy()
    
    # Échantillonner
    if len(df_fr) > limite:
        df_sample = df_fr.sample(n=limite, random_state=42)
    else:
        df_sample = df_fr
    
    logger.info(f"✅ {len(df_sample)} textes sélectionnés")
    
    return df_sample


def pre_labelliser_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """Pré-labellise le corpus avec le modèle"""
    logger.info("\n🤖 PRÉ-LABELLISATION AUTOMATIQUE")
    
    # Initialiser modèle
    labelliseur = LabelliseurSemiAuto()
    
    # Prédire sentiments
    predictions = labelliseur.predire_batch(df["texte_nettoye"].tolist())
    
    # Ajouter prédictions au DataFrame
    df["sentiment_pred"] = [p["sentiment"] for p in predictions]
    df["confiance_pred"] = [p["confiance"] for p in predictions]
    
    # Statistiques
    logger.info("\n📊 DISTRIBUTION DES PRÉDICTIONS")
    for sentiment, count in df["sentiment_pred"].value_counts().items():
        pct = (count / len(df)) * 100
        logger.info(f"   {sentiment:10s} : {count:4d} ({pct:5.2f}%)")
    
    # Confiance moyenne
    conf_moy = df["confiance_pred"].mean()
    logger.info(f"\n💯 Confiance moyenne : {conf_moy:.2%}")
    
    return df


def filtrer_predictions_confiantes(df: pd.DataFrame, seuil_confiance: float = 0.75) -> tuple:
    """
    Sépare les prédictions confiantes des incertaines
    
    Returns:
        (df_confiantes, df_incertaines)
    """
    logger.info(f"\n🎯 FILTRAGE (seuil de confiance : {seuil_confiance:.0%})")
    
    df_confiantes = df[df["confiance_pred"] >= seuil_confiance].copy()
    df_incertaines = df[df["confiance_pred"] < seuil_confiance].copy()
    
    logger.info(f"   Prédictions confiantes : {len(df_confiantes)} ({len(df_confiantes)/len(df)*100:.1f}%)")
    logger.info(f"   Prédictions incertaines : {len(df_incertaines)} ({len(df_incertaines)/len(df)*100:.1f}%)")
    
    return df_confiantes, df_incertaines


def equilibrer_classes(df: pd.DataFrame, max_par_classe: int = 200) -> pd.DataFrame:
    """Équilibre les classes en limitant le nombre par sentiment"""
    logger.info(f"\n⚖️  ÉQUILIBRAGE DES CLASSES (max {max_par_classe} par classe)")
    
    dfs_equilibres = []
    
    for sentiment in ["negative", "neutral", "positive"]:
        df_sentiment = df[df["sentiment_pred"] == sentiment]
        
        if len(df_sentiment) > max_par_classe:
            # Échantillonner en priorisant les plus confiants
            df_sentiment = df_sentiment.nlargest(max_par_classe, "confiance_pred")
        
        dfs_equilibres.append(df_sentiment)
        logger.info(f"   {sentiment:10s} : {len(df_sentiment)} textes")
    
    df_equilibre = pd.concat(dfs_equilibres, ignore_index=True)
    
    logger.info(f"\n   Total équilibré : {len(df_equilibre)} textes")
    
    return df_equilibre


def convertir_en_labels_finaux(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit les prédictions en labels finaux"""
    logger.info("\n🔄 CONVERSION EN FORMAT FINAL")
    
    df_labels = pd.DataFrame({
        "texte": df["texte_nettoye"],
        "sentiment": df["sentiment_pred"],
        "label": df["sentiment_pred"].map(SENTIMENT_LABELS),
        "source_label": "labellisation_semi_auto",
        "plateforme": df.get("plateforme"),
        "langue": df.get("langue"),
        "confiance_modele": df["confiance_pred"],
        "type_client": None,
        "region": None,
        "age": None,
    })
    
    return df_labels


def sauvegarder_labels_semi_auto(df: pd.DataFrame) -> Path:
    """Sauvegarde les labels générés automatiquement"""
    logger.info("\n💾 SAUVEGARDE DES LABELS")
    
    filepath = LABELED_DATA_DIR / "labels_semi_auto.csv"
    save_csv(df, filepath)
    
    logger.info(f"   📁 Fichier : {filepath}")
    logger.info(f"   📊 Total   : {len(df)} textes")
    
    return filepath


def fusionner_avec_labels_manuels(df_semi_auto: pd.DataFrame) -> pd.DataFrame:
    """Fusionne avec les labels manuels existants"""
    logger.info("\n🔗 FUSION AVEC LABELS EXISTANTS")
    
    filepath_existant = LABELED_DATA_DIR / "enquete_labellisee.csv"
    
    if filepath_existant.exists():
        df_existant = pd.read_csv(filepath_existant, encoding="utf-8")
        logger.info(f"   Labels existants : {len(df_existant)}")
        
        # Harmoniser colonnes
        colonnes_communes = ["texte", "sentiment", "label", "source_label"]
        
        for col in colonnes_communes:
            if col not in df_semi_auto.columns:
                df_semi_auto[col] = None
            if col not in df_existant.columns:
                df_existant[col] = None
        
        # Fusionner
        df_fusionne = pd.concat([df_existant, df_semi_auto], ignore_index=True)
        
        # Supprimer doublons 
        df_fusionne = df_fusionne.drop_duplicates(subset=["texte"], keep="first")
        
        logger.info(f"   Total fusionné : {len(df_fusionne)}")
        
        return df_fusionne
    else:
        logger.info("   Aucun label existant")
        return df_semi_auto


def main():
    """Point d'entrée principal"""
    print("="*70)
    print("🤖 LABELLISATION SEMI-AUTOMATIQUE - SENELEC")
    print("="*70)
    
    try:
        # Paramètres
        LIMITE_TEXTES = 2000  
        SEUIL_CONFIANCE = 0.70  
        MAX_PAR_CLASSE = 150 
        
        # Charger corpus
        df_corpus = charger_corpus_pour_labellisation(limite=LIMITE_TEXTES)
        
        # Pré-labelliser
        df_pred = pre_labelliser_corpus(df_corpus)
        
        # Filtrer prédictions confiantes
        df_confiantes, df_incertaines = filtrer_predictions_confiantes(
            df_pred, 
            seuil_confiance=SEUIL_CONFIANCE
        )
        
        # Équilibrer classes
        df_equilibre = equilibrer_classes(df_confiantes, max_par_classe=MAX_PAR_CLASSE)
        
        # Convertir format final
        df_labels = convertir_en_labels_finaux(df_equilibre)
        
        # Sauvegarder labels semi-auto
        filepath = sauvegarder_labels_semi_auto(df_labels)
        
        # FUSION AUTOMATIQUE 
        print("\n🔗 Fusion automatique avec labels existants...")
        df_fusionne = fusionner_avec_labels_manuels(df_labels)
        filepath_final = LABELED_DATA_DIR / "enquete_labellisee.csv"
        
        # Backup de l'ancien fichier
        if filepath_final.exists():
            from datetime import datetime
            backup_path = LABELED_DATA_DIR / f"enquete_labellisee_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            import shutil
            shutil.copy(filepath_final, backup_path)
            print(f"📦 Backup créé : {backup_path}")
        
        save_csv(df_fusionne, filepath_final)
        
        print("\n" + "="*70)
        print("✅ LABELLISATION SEMI-AUTOMATIQUE TERMINÉE")
        print("="*70)
        print(f"📁 Fichier final : {filepath_final}")
        print(f"📊 Total labels  : {len(df_fusionne)}")
        
        # Distribution finale
        print("\n📊 DISTRIBUTION FINALE :")
        for sentiment, count in df_fusionne["sentiment"].value_counts().items():
            pct = (count / len(df_fusionne)) * 100
            print(f"   {sentiment:10s} : {count:4d} ({pct:5.2f}%)")
        
        print("="*70)
        
    except Exception as e:
        logger.error(f"❌ Erreur : {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()