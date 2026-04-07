"""
Évaluation Détaillée du Modèle de Sentiment
Métriques, courbes ROC, analyse des erreurs
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.config import (
    LABELED_DATA_DIR,
    SENTIMENT_MODEL_DIR,
    FIGURES_DIR,
    STATISTICS_DIR,
    SENTIMENT_LABELS_REVERSE
)
from utils.logger import setup_logger

logger = setup_logger("evaluate_model")

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def charger_modele():
    """Charge le modèle entraîné"""
    logger.info("🤖 Chargement du modèle...")
    
    tokenizer = AutoTokenizer.from_pretrained(str(SENTIMENT_MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(SENTIMENT_MODEL_DIR))
    
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    logger.info(f"   Device : {device}")
    logger.info("✅ Modèle chargé")
    
    return model, tokenizer, device


def charger_test_set():
    """Charge le test set"""
    logger.info("\n📥 Chargement test set...")
    
    filepath = LABELED_DATA_DIR / "test_set.csv"
    
    if not filepath.exists():
        logger.error(f"❌ Fichier introuvable : {filepath}")
        raise FileNotFoundError(filepath)
    
    df = pd.read_csv(filepath, encoding="utf-8")
    logger.info(f"   Textes test : {len(df)}")
    
    return df


def predire_batch(texts, labels, model, tokenizer, device):
    """Prédit le sentiment pour un batch"""
    predictions = []
    true_labels = []
    probabilities = []
    
    for text, label in zip(texts, labels):
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            predicted_class = torch.argmax(logits, dim=1).item()
        
        predictions.append(predicted_class)
        true_labels.append(label)
        probabilities.append(probs)
    
    return predictions, true_labels, np.array(probabilities)


def generer_rapport_classification(y_true, y_pred):
    """Génère rapport de classification détaillé"""
    logger.info("\n📊 RAPPORT DE CLASSIFICATION")
    
    target_names = ['Négatif', 'Neutre', 'Positif']
    
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=target_names,
        digits=4
    )
    
    logger.info("\n" + report)
    
    # Sauvegarder
    filepath = STATISTICS_DIR / "classification_report.txt"
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"\n   Rapport sauvegardé : {filepath}")
    
    return report


def generer_matrice_confusion_detaillee(y_true, y_pred):
    """Génère matrice de confusion avec pourcentages"""
    logger.info("\n📊 Génération matrice de confusion...")
    
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Matrice en nombres
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Négatif', 'Neutre', 'Positif'],
                yticklabels=['Négatif', 'Neutre', 'Positif'],
                ax=ax1)
    ax1.set_title('Matrice de Confusion (Nombres)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Vraie Classe', fontsize=12)
    ax1.set_xlabel('Classe Prédite', fontsize=12)
    
    # Matrice en pourcentages
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Reds',
                xticklabels=['Négatif', 'Neutre', 'Positif'],
                yticklabels=['Négatif', 'Neutre', 'Positif'],
                ax=ax2)
    ax2.set_title('Matrice de Confusion (Pourcentages)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Vraie Classe', fontsize=12)
    ax2.set_xlabel('Classe Prédite', fontsize=12)
    
    plt.tight_layout()
    
    filepath = FIGURES_DIR / "matrice_confusion_detaillee.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   Sauvegardé : {filepath}")


def generer_courbes_roc(y_true, y_proba):
    """Génère courbes ROC pour chaque classe"""
    logger.info("\n📈 Génération courbes ROC...")
    
    # Binariser labels
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    n_classes = 3
    
    # Calculer ROC pour chaque classe
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Graphique
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#e74c3c', '#95a5a6', '#27ae60']
    labels = ['Négatif', 'Neutre', 'Positif']
    
    for i, color, label in zip(range(n_classes), colors, labels):
        ax.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{label} (AUC = {roc_auc[i]:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Aléatoire (AUC = 0.5)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Taux de Faux Positifs', fontsize=12)
    ax.set_ylabel('Taux de Vrais Positifs', fontsize=12)
    ax.set_title('Courbes ROC par Classe', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    filepath = FIGURES_DIR / "courbes_roc.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   Sauvegardé : {filepath}")
    logger.info(f"   AUC moyen : {np.mean(list(roc_auc.values())):.3f}")


def analyser_erreurs_classification(df_test, y_true, y_pred):
    """Analyse les erreurs de classification"""
    logger.info("\n🔍 ANALYSE DES ERREURS")
    
    df_test['pred'] = y_pred
    df_test['true'] = y_true
    df_test['correct'] = df_test['pred'] == df_test['true']
    
    erreurs = df_test[~df_test['correct']]
    
    logger.info(f"   Total erreurs : {len(erreurs)} / {len(df_test)} ({len(erreurs)/len(df_test)*100:.1f}%)")
    
    # Types d'erreurs
    logger.info("\n   Types d'erreurs fréquentes :")
    erreur_types = erreurs.groupby(['true', 'pred']).size().sort_values(ascending=False)
    
    for (true_label, pred_label), count in erreur_types.head(5).items():
        true_name = SENTIMENT_LABELS_REVERSE[true_label]
        pred_name = SENTIMENT_LABELS_REVERSE[pred_label]
        logger.info(f"      {true_name:10s} → {pred_name:10s} : {count} erreurs")
    
    # Exemples d'erreurs
    logger.info("\n   Exemples d'erreurs :")
    for idx, row in erreurs.head(3).iterrows():
        true_name = SENTIMENT_LABELS_REVERSE[row['true']]
        pred_name = SENTIMENT_LABELS_REVERSE[row['pred']]
        logger.info(f"\n      Vrai: {true_name}, Prédit: {pred_name}")
        logger.info(f"      Texte: {row['texte'][:100]}...")


def generer_rapport_performance(y_true, y_pred, y_proba):
    """Génère rapport de performance complet"""
    logger.info("\n📋 GÉNÉRATION RAPPORT PERFORMANCE")
    
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    rapport = {
        'classe': ['Négatif', 'Neutre', 'Positif', 'Moyenne'],
        'precision': list(precision) + [precision.mean()],
        'recall': list(recall) + [recall.mean()],
        'f1_score': list(f1) + [f1.mean()],
        'support': list(support) + [sum(support)],
        'accuracy': [accuracy] * 4
    }
    
    df_rapport = pd.DataFrame(rapport)
    
    filepath = STATISTICS_DIR / "performance_modele.csv"
    df_rapport.to_csv(filepath, index=False, encoding='utf-8')
    
    logger.info(f"   Rapport sauvegardé : {filepath}")
    logger.info("\n" + df_rapport.to_string(index=False))
    
    return df_rapport


def main():
    """Point d'entrée principal"""
    print("="*70)
    print("📊 ÉVALUATION MODÈLE - SENELEC")
    print("="*70)
    
    try:
        # Charger modèle et données
        model, tokenizer, device = charger_modele()
        df_test = charger_test_set()
        
        # Prédictions
        logger.info("\n🔮 Prédiction sur test set...")
        y_pred, y_true, y_proba = predire_batch(
            df_test['texte'].tolist(),
            df_test['label'].tolist(),
            model, tokenizer, device
        )
        
        # Évaluations
        generer_rapport_classification(y_true, y_pred)
        generer_matrice_confusion_detaillee(y_true, y_pred)
        generer_courbes_roc(y_true, y_proba)
        analyser_erreurs_classification(df_test, y_true, y_pred)
        generer_rapport_performance(y_true, y_pred, y_proba)
        
        print("\n" + "="*70)
        print("✅ ÉVALUATION TERMINÉE")
        print("="*70)
        print(f"📊 Graphiques : {FIGURES_DIR}")
        print(f"📋 Rapports : {STATISTICS_DIR}")
        print("="*70)
        
    except Exception as e:
        logger.error(f"❌ Erreur : {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()