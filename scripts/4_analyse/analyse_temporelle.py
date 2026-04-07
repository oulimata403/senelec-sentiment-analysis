"""
Analyse Temporelle - Évolution du sentiment dans le temps
Objectif : Répondre à "Comment le sentiment évolue-t-il ?"
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.config import EXPORTS_DIR, FIGURES_DIR, STATISTICS_DIR
from utils.logger import setup_logger
from utils.file_handler import save_csv

logger = setup_logger("analyse_temporelle")

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def charger_corpus_avec_themes():
    """Charge le corpus - SOLUTION TZ-AWARE DEFINITIVE"""
    logger.info("📥 Chargement du corpus...")
    
    filepath = EXPORTS_DIR / "corpus_avec_themes.csv"
    
    if not filepath.exists():
        logger.error(f"❌ Fichier manquant : {filepath}")
        logger.error("   → Exécutez d'abord : python scripts/3_modelisation/predict_themes.py")
        raise FileNotFoundError(filepath)
    
    df = pd.read_csv(filepath, encoding="utf-8")
    logger.info(f"   Lignes lues : {len(df)}")
    
    logger.info("🔄 Conversion dates TZ-AWARE (UTC)...")
    
    df['date_publication'] = pd.to_datetime(
        df['date_publication'], 
        utc=True,              
        errors='coerce'
    )
    
    # Suppression NaT
    nat_count = df['date_publication'].isna().sum()
    df = df.dropna(subset=['date_publication']).reset_index(drop=True)
    
    logger.info(f"   Dates NaT supprimées : {nat_count}")
    logger.info(f"   ✅ dtype : {df['date_publication'].dtype}")
    logger.info(f"   ✅ Textes OK : {len(df)}")
    logger.info(f"   ✅ Période : {df['date_publication'].min().strftime('%Y-%m-%d')} → {df['date_publication'].max().strftime('%Y-%m-%d')}")
    
    return df

def analyser_evolution_sentiment_global(df):
    """Évolution sentiment par semaine"""
    logger.info("\n📈 ÉVOLUTION SENTIMENT PAR SEMAINE")
    
    df_semaine = df[['date_publication', 'sentiment_pred']].copy()
    df_semaine['semaine'] = df_semaine['date_publication'].dt.to_period('W')
    
    sentiment_par_semaine = df_semaine.groupby(['semaine', 'sentiment_pred']).size().unstack(fill_value=0)
    total_par_semaine = sentiment_par_semaine.sum(axis=1)
    sentiment_pct = sentiment_par_semaine.div(total_par_semaine, axis=0) * 100
    
    stats = {
        'semaines': len(sentiment_pct),
        'negatif': sentiment_pct.get('negative', 0).mean(),
        'neutral': sentiment_pct.get('neutral', 0).mean(),
        'positif': sentiment_pct.get('positive', 0).mean()
    }
    
    logger.info(f"   Semaines : {stats['semaines']}")
    logger.info(f"   Négatif : {stats['negatif']:.1f}%")
    logger.info(f"   Neutre  : {stats['neutral']:.1f}%")
    logger.info(f"   Positif : {stats['positif']:.1f}%")
    
    return sentiment_pct

def generer_graphique_evolution_sentiment(sentiment_pct):
    """Graphique principal"""
    logger.info("\n📊 Graphique évolution sentiment...")
    
    fig, ax = plt.subplots(figsize=(15, 8))
    x_dates = sentiment_pct.index.to_timestamp()
    
    for sentiment, color, label, marker in [
        ('negative', '#e74c3c', 'Négatif', 'o'),
        ('neutral', '#95a5a6', 'Neutre', 's'),
        ('positive', '#27ae60', 'Positif', '^')
    ]:
        if sentiment in sentiment_pct.columns:
            ax.plot(x_dates, sentiment_pct[sentiment], 
                   label=label, color=color, linewidth=3, 
                   marker=marker, markersize=8, alpha=0.9)
    
    ax.set_title('Évolution Sentiment SENELEC (2024-2026)', 
                fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Pourcentage (%)', fontsize=14)
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    filepath = FIGURES_DIR / "evolution_sentiment_temporelle.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"   ✅ {filepath}")

def analyser_evolution_par_theme(df):
    """Évolution par thème"""
    logger.info("\n📊 ÉVOLUTION PAR THÈME")
    
    df_theme = df.copy()
    df_theme['mois'] = df_theme['date_publication'].dt.to_period('M')
    
    top_themes = df_theme['theme'].value_counts().head(5).index
    logger.info(f"   Top thèmes : {list(top_themes)}")
    
    evolutions = {}
    for theme in top_themes:
        df_t = df_theme[df_theme['theme'] == theme]
        if len(df_t) >= 10:
            sentiment_par_mois = df_t.groupby(['mois', 'sentiment_pred']).size().unstack(fill_value=0)
            total_par_mois = sentiment_par_mois.sum(axis=1)
            sentiment_pct = sentiment_par_mois.div(total_par_mois, axis=0) * 100
            evolutions[theme] = sentiment_pct
    
    return evolutions

def generer_graphiques_evolution_themes(evolutions):
    """Graphiques thèmes"""
    if not evolutions:
        logger.warning("⚠️  Pas assez de données par thème")
        return
    
    logger.info(f"📈 {len(evolutions)} graphiques thèmes...")
    
    THEMES_FR = {
        'facturation': 'Facturation', 'service': 'Service Client', 
        'panne': 'Pannes', 'prix': 'Prix', 'qualite': 'Qualité'
    }
    
    n_themes = len(evolutions)
    fig, axes = plt.subplots(n_themes, 1, figsize=(15, 5*n_themes))
    if n_themes == 1: axes = [axes]
    
    for i, (theme, sentiment_pct) in enumerate(evolutions.items()):
        ax = axes[i]
        x_dates = sentiment_pct.index.to_timestamp()
        
        for sentiment, color, label in [
            ('negative', '#e74c3c', 'Négatif'),
            ('neutral', '#95a5a6', 'Neutre'),
            ('positive', '#27ae60', 'Positif')
        ]:
            if sentiment in sentiment_pct.columns:
                ax.plot(x_dates, sentiment_pct[sentiment], 
                       label=label, color=color, linewidth=2.5, marker='o')
        
        label_fr = THEMES_FR.get(theme, theme.title())
        ax.set_title(f'{label_fr} ({len(evolutions[theme])} posts)', fontweight='bold')
        ax.set_ylabel('Pourcentage (%)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    filepath = FIGURES_DIR / "evolution_par_theme.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"   ✅ {filepath}")

def detecter_pics_mecontentement(df):
    """Pics mécontentement"""
    logger.info("\n🔥 PICS MÉCONTENTEMENT")
    
    df_neg = df[df['sentiment_pred'] == 'negative'].copy()
    if len(df_neg) == 0:
        logger.warning("Aucun post négatif")
        return
    
    df_neg['jour'] = df_neg['date_publication'].dt.date
    negatifs_par_jour = df_neg.groupby('jour').size()
    
    seuil = negatifs_par_jour.mean() + 2 * negatifs_par_jour.std()
    pics = negatifs_par_jour[negatifs_par_jour > seuil].sort_values(ascending=False)
    
    logger.info(f"   Seuil : {seuil:.1f} négatifs/jour")
    logger.info(f"   Pics trouvés : {len(pics)}")
    
    for jour, count in pics.head(3).items():
        logger.info(f"     {jour} : {count} posts")

def analyser_volume_publications(df):
    """Volume publications"""
    logger.info("\n📊 VOLUME PUBLICATIONS")
    
    df_vol = df.copy()
    df_vol['semaine'] = df_vol['date_publication'].dt.to_period('W')
    volume_par_semaine = df_vol.groupby('semaine').size()
    
    logger.info(f"   Moyenne : {volume_par_semaine.mean():.0f}/semaine")
    logger.info(f"   Maximum : {volume_par_semaine.max()} ({volume_par_semaine.idxmax()})")
    
    fig, ax = plt.subplots(figsize=(15, 6))
    x_dates = volume_par_semaine.index.to_timestamp()
    ax.bar(x_dates, volume_par_semaine.values, color='#3498db', alpha=0.7)
    
    ax.set_title('Volume Publications SENELEC par Semaine', fontweight='bold', fontsize=16)
    ax.set_xlabel('Date')
    ax.set_ylabel('Nombre publications')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    filepath = FIGURES_DIR / "volume_publications_temporel.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"   ✅ {filepath}")

def sauvegarder_statistiques_temporelles(df):
    """Sauvegarde stats"""
    logger.info("\n💾 SAUVEGARDE STATS")
    
    df_stats = df.copy()
    df_stats['mois'] = df_stats['date_publication'].dt.to_period('M')
    
    stats = df_stats.groupby('mois').agg({
        'sentiment_pred': lambda x: (x == 'negative').sum(),
        'texte_nettoye': 'count'
    }).rename(columns={
        'sentiment_pred': 'negatifs', 
        'texte_nettoye': 'total'
    })
    
    stats['pct_negatif'] = (stats['negatifs'] / stats['total']) * 100
    
    filepath = STATISTICS_DIR / "evolution_temporelle.csv"
    stats.to_csv(filepath, encoding='utf-8')
    logger.info(f"   ✅ {filepath} ({len(stats)} périodes)")

def main():
    print("="*80)
    print("📈 ANALYSE TEMPORELLE SENELEC - Ouly TOURÉ")
    print("="*80)
    
    try:
        df = charger_corpus_avec_themes()
        
        sentiment_pct = analyser_evolution_sentiment_global(df)
        generer_graphique_evolution_sentiment(sentiment_pct)
        
        evolutions = analyser_evolution_par_theme(df)
        generer_graphiques_evolution_themes(evolutions)
        
        detecter_pics_mecontentement(df)
        analyser_volume_publications(df)
        sauvegarder_statistiques_temporelles(df)
        
        print("\n" + "="*80)
        print("✅ ANALYSE TEMPORELLE TERMINÉE ✅")
        print(f"📊 Graphiques (4) : {FIGURES_DIR}")
        print(f"📈 Stats : {STATISTICS_DIR / 'evolution_temporelle.csv'}")
        print("="*80)
        
    except Exception as e:
        logger.error(f"❌ ERREUR : {e}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"\n❌ ERREUR : {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
