"""
Prédiction du sentiment sur le corpus complet
Utilise le modèle BERT entraîné
"""

import sys
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.config import (
    PROCESSED_DATA_DIR,
    SENTIMENT_MODEL_DIR,
    EXPORTS_DIR,
    SENTIMENT_LABELS_REVERSE
)
from utils.logger import setup_logger
from utils.file_handler import save_csv

logger = setup_logger("predict_sentiment")


def charger_modele():
    """Charge le modèle entraîné"""
    logger.info("🤖 Chargement du modèle...")
    
    if not SENTIMENT_MODEL_DIR.exists():
        logger.error(f"❌ Modèle introuvable : {SENTIMENT_MODEL_DIR}")
        logger.error("   Exécutez d'abord : train_sentiment_model.py")
        raise FileNotFoundError(SENTIMENT_MODEL_DIR)
    
    tokenizer = AutoTokenizer.from_pretrained(str(SENTIMENT_MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(SENTIMENT_MODEL_DIR))
    
    # Mode évaluation
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    logger.info(f"   Device : {device}")
    logger.info("✅ Modèle chargé")
    
    return model, tokenizer, device


def charger_corpus():
    """Charge le corpus complet nettoyé"""
    logger.info("\n📥 Chargement du corpus...")
    
    filepath = PROCESSED_DATA_DIR / "corpus_avec_langues.csv"
    
    if not filepath.exists():
        logger.error(f"❌ Corpus introuvable : {filepath}")
        raise FileNotFoundError(filepath)
    
    df = pd.read_csv(filepath, encoding="utf-8")
    logger.info(f"   Textes chargés : {len(df)}")
    
    return df


def predire_sentiment(texte: str, model, tokenizer, device, max_length=128):
    """Prédit le sentiment d'un texte"""
    # Tokenisation
    encoding = tokenizer(
        texte,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # Prédiction
    with torch.no_grad():
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Probabilités
        probs = torch.softmax(logits, dim=1)
        
        # Classe prédite
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()
    
    sentiment = SENTIMENT_LABELS_REVERSE[predicted_class]
    
    return sentiment, predicted_class, confidence


def predire_batch(df: pd.DataFrame, model, tokenizer, device, batch_size=32):
    """Prédit le sentiment pour un batch de textes"""
    logger.info("\n🔮 PRÉDICTION DU SENTIMENT")
    logger.info(f"   Batch size : {batch_size}")
    
    sentiments = []
    labels = []
    confidences = []
    
    # Prédire par batch
    for i in tqdm(range(0, len(df), batch_size), desc="Prédiction"):
        batch_texts = df['texte_nettoye'].iloc[i:i+batch_size].tolist()
        
        for texte in batch_texts:
            if pd.isna(texte) or len(str(texte)) < 5:
                sentiments.append("neutral")
                labels.append(1)
                confidences.append(0.0)
                continue
            
            try:
                sentiment, label, confidence = predire_sentiment(
                    str(texte), model, tokenizer, device
                )
                sentiments.append(sentiment)
                labels.append(label)
                confidences.append(confidence)
            except Exception as e:
                logger.warning(f"Erreur prédiction : {e}")
                sentiments.append("neutral")
                labels.append(1)
                confidences.append(0.0)
    
    logger.info("✅ Prédictions terminées")
    
    return sentiments, labels, confidences


def analyser_distribution(df: pd.DataFrame) -> None:
    """Analyse la distribution des sentiments prédits"""
    logger.info("\n📊 DISTRIBUTION DES SENTIMENTS PRÉDITS")
    
    total = len(df)
    
    for sentiment in ["negative", "neutral", "positive"]:
        count = (df["sentiment_pred"] == sentiment).sum()
        pct = (count / total) * 100
        emoji = {"negative": "😡", "neutral": "😐", "positive": "😊"}.get(sentiment, "")
        logger.info(f"   {emoji} {sentiment:10s} : {count:5d} ({pct:5.2f}%)")
    
    # Par plateforme
    logger.info("\n📊 DISTRIBUTION PAR PLATEFORME")
    
    for plateforme in df["plateforme"].unique():
        df_pf = df[df["plateforme"] == plateforme]
        logger.info(f"\n   {plateforme.upper()} ({len(df_pf)} textes) :")
        
        for sentiment in ["negative", "neutral", "positive"]:
            count = (df_pf["sentiment_pred"] == sentiment).sum()
            pct = (count / len(df_pf)) * 100
            logger.info(f"      {sentiment:10s} : {count:5d} ({pct:5.2f}%)")


def sauvegarder_predictions(df: pd.DataFrame) -> Path:
    """Sauvegarde le corpus avec prédictions"""
    logger.info("\n💾 SAUVEGARDE DES PRÉDICTIONS")
    
    filepath = EXPORTS_DIR / "corpus_avec_sentiment.csv"
    save_csv(df, filepath)
    
    logger.info(f"   Fichier : {filepath}")
    logger.info(f"   Lignes  : {len(df)}")
    
    return filepath


def main():
    """Point d'entrée principal"""
    print("="*70)
    print("🔮 PRÉDICTION SENTIMENT - SENELEC")
    print("="*70)
    
    try:
        # Charger modèle
        model, tokenizer, device = charger_modele()
        
        # Charger corpus
        df = charger_corpus()
        
        # Prédire
        sentiments, labels, confidences = predire_batch(
            df, model, tokenizer, device, batch_size=32
        )
        
        # Ajouter prédictions au DataFrame
        df["sentiment_pred"] = sentiments
        df["label_pred"] = labels
        df["confiance_pred"] = confidences
        
        # Analyser distribution
        analyser_distribution(df)
        
        # Sauvegarder
        filepath = sauvegarder_predictions(df)
        
        print("\n" + "="*70)
        print("✅ PRÉDICTION TERMINÉE AVEC SUCCÈS")
        print("="*70)
        print(f"📁 Fichier : {filepath}")
        print(f"📊 Total   : {len(df)} textes")
        print("="*70)
        
    except Exception as e:
        logger.error(f"❌ Erreur : {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()