"""
Analyse finale du corpus avec sentiments prédits
Génère statistiques et visualisations
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.config import EXPORTS_DIR, FIGURES_DIR, STATISTICS_DIR
from utils.logger import setup_logger
from utils.file_handler import save_csv

logger = setup_logger("analyse_corpus")

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def charger_corpus_avec_sentiment():
    """Charge le corpus avec sentiments prédits"""
    logger.info("📥 Chargement du corpus...")
    
    filepath = EXPORTS_DIR / "corpus_avec_sentiment.csv"
    
    if not filepath.exists():
        logger.error(f"❌ Fichier introuvable : {filepath}")
        logger.error("   Exécutez d'abord : predict_sentiment.py")
        raise FileNotFoundError(filepath)
    
    df = pd.read_csv(filepath, encoding="utf-8")
    logger.info(f"   Textes chargés : {len(df)}")
    
    return df


def analyser_distribution_globale(df: pd.DataFrame) -> dict:
    """Analyse la distribution globale des sentiments"""
    logger.info("\n📊 ANALYSE DISTRIBUTION GLOBALE")
    
    stats = {}
    total = len(df)
    
    for sentiment in ["negative", "neutral", "positive"]:
        count = (df["sentiment_pred"] == sentiment).sum()
        pct = (count / total) * 100
        stats[sentiment] = {"count": count, "percentage": pct}
        
        emoji = {"negative": "😡", "neutral": "😐", "positive": "😊"}.get(sentiment)
        logger.info(f"   {emoji} {sentiment:10s} : {count:5d} ({pct:5.2f}%)")
    
    return stats


def analyser_par_plateforme(df: pd.DataFrame) -> pd.DataFrame:
    """Analyse par plateforme"""
    logger.info("\n📱 ANALYSE PAR PLATEFORME")
    
    stats_plateformes = []
    
    for plateforme in df["plateforme"].unique():
        df_pf = df[df["plateforme"] == plateforme]
        total_pf = len(df_pf)
        
        logger.info(f"\n   {plateforme.upper()} ({total_pf} textes) :")
        
        for sentiment in ["negative", "neutral", "positive"]:
            count = (df_pf["sentiment_pred"] == sentiment).sum()
            pct = (count / total_pf) * 100
            
            stats_plateformes.append({
                "plateforme": plateforme,
                "sentiment": sentiment,
                "count": count,
                "percentage": pct
            })
            
            logger.info(f"      {sentiment:10s} : {count:5d} ({pct:5.2f}%)")
    
    return pd.DataFrame(stats_plateformes)


def analyser_par_langue(df: pd.DataFrame) -> pd.DataFrame:
    """Analyse par langue"""
    logger.info("\n🌍 ANALYSE PAR LANGUE")
    
    stats_langues = []
    
    for langue in df["langue"].unique():
        df_lg = df[df["langue"] == langue]
        total_lg = len(df_lg)
        
        logger.info(f"\n   {langue.upper()} ({total_lg} textes) :")
        
        for sentiment in ["negative", "neutral", "positive"]:
            count = (df_lg["sentiment_pred"] == sentiment).sum()
            pct = (count / total_lg) * 100
            
            stats_langues.append({
                "langue": langue,
                "sentiment": sentiment,
                "count": count,
                "percentage": pct
            })
            
            logger.info(f"      {sentiment:10s} : {count:5d} ({pct:5.2f}%)")
    
    return pd.DataFrame(stats_langues)


def generer_graphique_distribution(df: pd.DataFrame) -> None:
    """Génère le graphique de distribution globale"""
    logger.info("\n📊 Génération graphique distribution...")
    
    # Compter sentiments et trier dans l'ordre logique
    sentiment_counts = df["sentiment_pred"].value_counts()
    order = ["negative", "neutral", "positive"]  
    
    # Réorganiser pour respecter l'ordre
    counts_ordered = [sentiment_counts.get(sent, 0) for sent in order]
    labels_fr = ['Négatif', 'Neutre', 'Positif']
    
    # Créer graphique
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {"negative": "#e74c3c", "neutral": "#95a5a6", "positive": "#27ae60"}
    bars = ax.bar(labels_fr, counts_ordered, 
                  color=[colors[order[i]] for i in range(3)])
    
    # Annotations
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{int(height)}', ha='center', va='bottom',
                fontsize=12, fontweight='bold')
    
    ax.set_title("Distribution des Sentiments - Corpus Complet", fontsize=14, fontweight='bold')
    ax.set_xlabel("Sentiment", fontsize=12)
    ax.set_ylabel("Nombre de textes", fontsize=12)
    
    plt.tight_layout()
    
    filepath = FIGURES_DIR / "distribution_sentiments.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   Sauvegardé : {filepath}")

def generer_graphique_par_plateforme(stats_pf: pd.DataFrame) -> None:
    """Génère le graphique par plateforme"""
    logger.info("\n📱 Génération graphique par plateforme...")
    
    # Pivot
    pivot = stats_pf.pivot(index='plateforme', columns='sentiment', values='percentage')
    
    # Graphique
    fig, ax = plt.subplots(figsize=(12, 6))
    
    pivot.plot(
        kind='bar',
        ax=ax,
        color=['#e74c3c', '#95a5a6', '#27ae60']
    )
    
    ax.set_title("Distribution des Sentiments par Plateforme", fontsize=14, fontweight='bold')
    ax.set_xlabel("Plateforme", fontsize=12)
    ax.set_ylabel("Pourcentage (%)", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(title="Sentiment", labels=['Négatif', 'Neutre', 'Positif'])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    filepath = FIGURES_DIR / "sentiments_par_plateforme.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   Sauvegardé : {filepath}")


def generer_rapport_complet(df: pd.DataFrame, stats_pf: pd.DataFrame, stats_lg: pd.DataFrame) -> None:
    """Génère un rapport complet en CSV"""
    logger.info("\n📋 Génération rapport complet...")
    
    # Statistiques globales
    stats_globales = df.groupby("sentiment_pred").agg({
        "texte_nettoye": "count",
        "confiance_pred": "mean"
    }).reset_index()
    
    stats_globales.columns = ["sentiment", "count", "confiance_moyenne"]
    
    # Sauvegarder
    filepath_global = STATISTICS_DIR / "stats_globales.csv"
    filepath_pf = STATISTICS_DIR / "stats_par_plateforme.csv"
    filepath_lg = STATISTICS_DIR / "stats_par_langue.csv"
    
    save_csv(stats_globales, filepath_global)
    save_csv(stats_pf, filepath_pf)
    save_csv(stats_lg, filepath_lg)
    
    logger.info(f"   Global      : {filepath_global}")
    logger.info(f"   Plateforme  : {filepath_pf}")
    logger.info(f"   Langue      : {filepath_lg}")


def main():
    """Point d'entrée principal"""
    print("="*70)
    print("📊 ANALYSE FINALE CORPUS - SENELEC")
    print("="*70)
    
    try:
        # Charger corpus
        df = charger_corpus_avec_sentiment()
        
        # Analyses
        stats_globales = analyser_distribution_globale(df)
        stats_pf = analyser_par_plateforme(df)
        stats_lg = analyser_par_langue(df)
        
        # Graphiques
        generer_graphique_distribution(df)
        generer_graphique_par_plateforme(stats_pf)
        
        # Rapport
        generer_rapport_complet(df, stats_pf, stats_lg)
        
        print("\n" + "="*70)
        print("✅ ANALYSE TERMINÉE AVEC SUCCÈS")
        print("="*70)
        print(f"📁 Graphiques : {FIGURES_DIR}")
        print(f"📁 Stats      : {STATISTICS_DIR}")
        print("="*70)
        
    except Exception as e:
        logger.error(f"❌ Erreur : {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()