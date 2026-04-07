"""
Entraînement du modèle BERT pour l'analyse de sentiment 
Avec gestion du déséquilibre des classes
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.config import (
    LABELED_DATA_DIR,
    SENTIMENT_MODEL_DIR,
    NLP_CONFIG,
    FIGURES_DIR
)
from utils.logger import setup_logger

logger = setup_logger("train_sentiment_model")


class SentimentDataset(Dataset):
    """Dataset pour l'analyse de sentiment"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def charger_datasets():
    """Charge les datasets train/val/test"""
    logger.info("📥 Chargement des datasets...")
    
    df_train = pd.read_csv(LABELED_DATA_DIR / "train_set.csv", encoding="utf-8")
    df_val = pd.read_csv(LABELED_DATA_DIR / "val_set.csv", encoding="utf-8")
    df_test = pd.read_csv(LABELED_DATA_DIR / "test_set.csv", encoding="utf-8")
    
    logger.info(f"   Train : {len(df_train)} textes")
    logger.info(f"   Val   : {len(df_val)} textes")
    logger.info(f"   Test  : {len(df_test)} textes")
    
    return df_train, df_val, df_test


def charger_poids_classes():
    """Charge les poids des classes"""
    logger.info("⚖️  Chargement des poids de classes...")
    
    filepath = LABELED_DATA_DIR / "class_weights.json"
    
    with open(filepath, 'r', encoding='utf-8') as f:
        poids = json.load(f)
    
    poids_tensor = torch.tensor([poids[str(i)] for i in range(len(poids))])
    logger.info(f"   Poids : {poids_tensor.tolist()}")
    
    return poids_tensor


def creer_dataloaders(df_train, df_val, df_test, tokenizer, batch_size=16):
    """Crée les DataLoaders"""
    logger.info("\n🔧 Création des DataLoaders...")
    
    max_length = NLP_CONFIG.get("max_length", 128)
    
    train_dataset = SentimentDataset(df_train['texte'].values, df_train['label'].values, tokenizer, max_length)
    val_dataset = SentimentDataset(df_val['texte'].values, df_val['label'].values, tokenizer, max_length)
    test_dataset = SentimentDataset(df_test['texte'].values, df_test['label'].values, tokenizer, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    logger.info("✅ DataLoaders créés")
    
    return train_loader, val_loader, test_loader


def evaluer_modele(model, dataloader, device, class_weights=None):
    """Évalue le modèle"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }, all_preds, all_labels


def entrainer_modele(train_loader, val_loader, test_loader, class_weights):
    """Entraîne le modèle BERT"""
    logger.info("\n🤖 INITIALISATION DU MODÈLE")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"   Device : {device}")
    
    model_name = NLP_CONFIG.get("model_name", "bert-base-multilingual-cased")
    logger.info(f"   Modèle : {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Entraînement
    num_epochs = NLP_CONFIG.get("num_epochs", 10)
    best_f1 = 0
    patience_counter = 0
    
    logger.info("\n🚀 DÉMARRAGE DE L'ENTRAÎNEMENT")
    logger.info("="*70)
    
    for epoch in range(num_epochs):
        logger.info(f"\n📍 Epoch {epoch+1}/{num_epochs}")
        
        # Training
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        logger.info(f"   Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        val_metrics, _, _ = evaluer_modele(model, val_loader, device, class_weights)
        logger.info(f"   Val Loss  : {val_metrics['loss']:.4f}")
        logger.info(f"   Val F1    : {val_metrics['f1']:.4f}")
        logger.info(f"   Val Acc   : {val_metrics['accuracy']:.4f}")
        
        # Early stopping
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            patience_counter = 0
            torch.save(model.state_dict(), SENTIMENT_MODEL_DIR / "best_model.pt")
            logger.info("   ✅ Meilleur modèle sauvegardé")
        else:
            patience_counter += 1
            if patience_counter >= 3:
                logger.info("   🛑 Early stopping")
                break
    
    # Charger meilleur modèle
    model.load_state_dict(torch.load(SENTIMENT_MODEL_DIR / "best_model.pt"))
    
    # Test
    logger.info("\n📊 ÉVALUATION SUR TEST SET")
    test_metrics, test_preds, test_labels = evaluer_modele(model, test_loader, device, class_weights)
    
    logger.info("\n   Résultats :")
    for metric, value in test_metrics.items():
        logger.info(f"      {metric:20s} : {value:.4f}")
    
    return model, tokenizer, test_preds, test_labels


def generer_matrice_confusion(y_true, y_pred):
    """Génère et sauvegarde la matrice de confusion"""
    logger.info("\n📈 GÉNÉRATION MATRICE DE CONFUSION")
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Négatif', 'Neutre', 'Positif'],
                yticklabels=['Négatif', 'Neutre', 'Positif'])
    plt.title('Matrice de Confusion - Test Set')
    plt.ylabel('Vraie Classe')
    plt.xlabel('Classe Prédite')
    
    filepath = FIGURES_DIR / "matrice_confusion.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   Matrice sauvegardée : {filepath}")


def sauvegarder_modele(model, tokenizer):
    """Sauvegarde le modèle"""
    logger.info("\n💾 SAUVEGARDE DU MODÈLE")
    
    SENTIMENT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(str(SENTIMENT_MODEL_DIR))
    tokenizer.save_pretrained(str(SENTIMENT_MODEL_DIR))
    
    logger.info(f"   Modèle sauvegardé : {SENTIMENT_MODEL_DIR}")


def main():
    """Point d'entrée principal"""
    print("="*70)
    print("🤖 ENTRAÎNEMENT MODÈLE BERT - SENELEC")
    print("="*70)
    
    try:
        df_train, df_val, df_test = charger_datasets()
        class_weights = charger_poids_classes()
        
        model_name = NLP_CONFIG.get("model_name", "bert-base-multilingual-cased")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        train_loader, val_loader, test_loader = creer_dataloaders(
            df_train, df_val, df_test, tokenizer, batch_size=16
        )
        
        model, tokenizer, test_preds, test_labels = entrainer_modele(
            train_loader, val_loader, test_loader, class_weights
        )
        
        generer_matrice_confusion(test_labels, test_preds)
        sauvegarder_modele(model, tokenizer)
        
        print("\n" + "="*70)
        print("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
        print("="*70)
        print(f"📁 Modèle : {SENTIMENT_MODEL_DIR}")
        print(f"📊 Matrice: {FIGURES_DIR / 'matrice_confusion.png'}")
        print("="*70)
        
    except Exception as e:
        logger.error(f"❌ Erreur : {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()