"""
Analyse Thématique Avancée
Corrélations, insights et analyse approfondie par thème
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.config import EXPORTS_DIR, FIGURES_DIR, STATISTICS_DIR, THEMES
from utils.logger import setup_logger
from utils.file_handler import save_csv

logger = setup_logger("analyse_thematique")

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def charger_corpus_themes():
    """Charge le corpus avec thèmes"""
    logger.info("📥 Chargement corpus...")
    
    filepath = EXPORTS_DIR / "corpus_avec_themes.csv"
    df = pd.read_csv(filepath, encoding='utf-8')
    
    logger.info(f"   Textes : {len(df)}")
    return df


def analyser_themes_detailles(df):
    """Analyse détaillée de chaque thème"""
    logger.info("\n🔍 ANALYSE DÉTAILLÉE PAR THÈME")
    
    rapport = []
    
    for theme in df['theme'].unique():
        df_theme = df[df['theme'] == theme]
        
        # Statistiques
        total = len(df_theme)
        neg = (df_theme['sentiment_pred'] == 'negative').sum()
        neu = (df_theme['sentiment_pred'] == 'neutral').sum()
        pos = (df_theme['sentiment_pred'] == 'positive').sum()
        
        # Par plateforme
        facebook = (df_theme['plateforme'] == 'facebook').sum()
        twitter = (df_theme['plateforme'] == 'twitter').sum()
        enquete = (df_theme['plateforme'] == 'enquete').sum()
        
        label_fr = THEMES.get(theme, {}).get('label_fr', theme)
        
        rapport.append({
            'theme': theme,
            'label_fr': label_fr,
            'total': total,
            'pct_corpus': (total / len(df)) * 100,
            'negative': neg,
            'neutral': neu,
            'positive': pos,
            'pct_negative': (neg / total) * 100,
            'pct_neutral': (neu / total) * 100,
            'pct_positive': (pos / total) * 100,
            'facebook': facebook,
            'twitter': twitter,
            'enquete': enquete,
            'confiance_moyenne': df_theme['topic_probability'].mean()
        })
        
        logger.info(f"\n   📌 {label_fr} ({total} publications)")
        logger.info(f"      Négatif: {neg:4d} ({neg/total*100:5.1f}%)")
        logger.info(f"      Neutre : {neu:4d} ({neu/total*100:5.1f}%)")
        logger.info(f"      Positif: {pos:4d} ({pos/total*100:5.1f}%)")
    
    df_rapport = pd.DataFrame(rapport).sort_values('total', ascending=False)
    
    # Sauvegarder
    filepath = STATISTICS_DIR / "analyse_thematique_detaillee.csv"
    save_csv(df_rapport, filepath)
    
    logger.info(f"\n   ✅ {filepath}")
    
    return df_rapport


def analyser_correlations_themes_sentiments(df):
    """Analyse corrélations entre thèmes et sentiments"""
    logger.info("\n📊 CORRÉLATIONS THÈMES × SENTIMENTS")
    
    crosstab = pd.crosstab(
        df['theme'],
        df['sentiment_pred'],
        normalize='index'
    ) * 100
    
    theme_labels = {t: THEMES.get(t, {}).get('label_fr', t) for t in crosstab.index}
    crosstab.index = [theme_labels[idx] for idx in crosstab.index]
    crosstab.columns = ['Négatif', 'Neutre', 'Positif']
    
    # Graphique
    fig, ax = plt.subplots(figsize=(14, 10))
    
    sns.heatmap(
        crosstab,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn_r',
        cbar_kws={'label': 'Pourcentage (%)'},
        linewidths=1,
        linecolor='white',
        ax=ax,
        annot_kws={'fontsize': 11, 'fontweight': 'bold'}
    )
    
    ax.set_title('Distribution des Sentiments par Thème (%)', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Sentiment', fontsize=14)
    ax.set_ylabel('Thème', fontsize=14)
    
    plt.tight_layout()
    filepath = FIGURES_DIR / "heatmap_themes_sentiments_pct.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"   ✅ {filepath}")


def identifier_themes_critiques(df):
    """Identifie les thèmes les plus problématiques"""
    logger.info("\n🔥 THÈMES CRITIQUES")
    
    # Calculer score de criticité
    theme_stats = []
    
    for theme in df['theme'].unique():
        df_theme = df[df['theme'] == theme]
        
        pct_neg = (df_theme['sentiment_pred'] == 'negative').sum() / len(df_theme) * 100
        volume = len(df_theme)
        
        # Score de criticité = % négatif × log(volume)
        criticity_score = pct_neg * np.log10(volume + 1)
        
        theme_stats.append({
            'theme': theme,
            'label_fr': THEMES.get(theme, {}).get('label_fr', theme),
            'pct_negative': pct_neg,
            'volume': volume,
            'criticity_score': criticity_score
        })
    
    df_criticity = pd.DataFrame(theme_stats).sort_values('criticity_score', ascending=False)
    
    logger.info("\n   Ranking de criticité :")
    for idx, row in df_criticity.head(5).iterrows():
        logger.info(f"      {row['label_fr']:30s} : Score {row['criticity_score']:.1f} ({row['pct_negative']:.1f}% négatif, {row['volume']} posts)")
    
    # Graphique
    fig, ax = plt.subplots(figsize=(12, 8))
    
    top_5 = df_criticity.head(5)
    colors = ['#e74c3c' if x > 50 else '#f39c12' if x > 30 else '#27ae60' 
              for x in top_5['pct_negative']]
    
    bars = ax.barh(range(len(top_5)), top_5['criticity_score'], color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_yticks(range(len(top_5)))
    ax.set_yticklabels(top_5['label_fr'], fontsize=12)
    ax.set_xlabel('Score de Criticité', fontsize=13)
    ax.set_title('Thèmes les Plus Critiques (Score = % Négatif × log(Volume))', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    # Annotations
    for i, (bar, row) in enumerate(zip(bars, top_5.itertuples())):
        ax.text(bar.get_width(), i, f' {row.criticity_score:.1f}', 
                va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    filepath = FIGURES_DIR / "themes_critiques_ranking.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"\n   ✅ {filepath}")
    
    return df_criticity


def analyser_themes_par_plateforme_detaille(df):
    """Analyse détaillée par plateforme"""
    logger.info("\n📱 ANALYSE THÈMES PAR PLATEFORME")
    
    # Crosstab
    crosstab = pd.crosstab(df['plateforme'], df['theme'])
    
    # Renommer thèmes
    theme_labels = {t: THEMES.get(t, {}).get('label_fr', t) for t in crosstab.columns}
    crosstab.columns = [theme_labels[col] for col in crosstab.columns]
    
    # Graphique stacked bar
    fig, ax = plt.subplots(figsize=(14, 8))
    
    crosstab.plot(kind='bar', stacked=True, ax=ax, colormap='tab10', edgecolor='black', linewidth=0.5)
    
    ax.set_title('Distribution des Thèmes par Plateforme', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Plateforme', fontsize=14)
    ax.set_ylabel('Nombre de publications', fontsize=14)
    ax.legend(title='Thème', fontsize=11, title_fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=0, fontsize=13)
    
    plt.tight_layout()
    filepath = FIGURES_DIR / "themes_par_plateforme_stacked.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"   ✅ {filepath}")


def generer_insights_strategiques(df_rapport, df_criticity):
    """Génère insights stratégiques pour décideurs"""
    logger.info("\n💡 INSIGHTS STRATÉGIQUES")
    
    insights = []
    
    # Insight 1: Thème dominant
    theme_dominant = df_rapport.iloc[0]
    insights.append(f"🎯 Le thème '{theme_dominant['label_fr']}' représente {theme_dominant['pct_corpus']:.1f}% du corpus ({theme_dominant['total']} publications)")
    
    # Insight 2: Thème le plus négatif
    theme_plus_negatif = df_rapport.loc[df_rapport['pct_negative'].idxmax()]
    insights.append(f"😡 '{theme_plus_negatif['label_fr']}' est le thème le plus négatif avec {theme_plus_negatif['pct_negative']:.1f}% de sentiment négatif")
    
    # Insight 3: Thème le plus positif
    theme_plus_positif = df_rapport.loc[df_rapport['pct_positive'].idxmax()]
    insights.append(f"😊 '{theme_plus_positif['label_fr']}' est le thème le plus positif avec {theme_plus_positif['pct_positive']:.1f}% de sentiment positif")
    
    # Insight 4: Criticité
    theme_critique = df_criticity.iloc[0]
    insights.append(f"🔥 '{theme_critique['label_fr']}' est le thème le plus critique (score: {theme_critique['criticity_score']:.1f})")
    
    # Afficher
    logger.info("\n   === INSIGHTS CLÉS ===")
    for i, insight in enumerate(insights, 1):
        logger.info(f"   {i}. {insight}")
    
    # Sauvegarder
    insights_text = "\n".join([f"{i}. {insight}" for i, insight in enumerate(insights, 1)])
    filepath = STATISTICS_DIR / "insights_strategiques.txt"
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("INSIGHTS STRATÉGIQUES - ANALYSE THÉMATIQUE SENELEC\n")
        f.write("="*70 + "\n\n")
        f.write(insights_text)
        f.write("\n\n" + "="*70)
    
    logger.info(f"\n   ✅ {filepath}")


def main():
    """Point d'entrée principal"""
    print("="*70)
    print("🔍 ANALYSE THÉMATIQUE AVANCÉE - SENELEC")
    print("="*70)
    
    try:
        df = charger_corpus_themes()
        
        # Analyses
        df_rapport = analyser_themes_detailles(df)
        analyser_correlations_themes_sentiments(df)
        df_criticity = identifier_themes_critiques(df)
        analyser_themes_par_plateforme_detaille(df)
        generer_insights_strategiques(df_rapport, df_criticity)
        
        print("\n" + "="*70)
        print("✅ ANALYSE THÉMATIQUE TERMINÉE")
        print("="*70)
        print(f"📊 Graphiques : {FIGURES_DIR}")
        print(f"📋 Rapport : {STATISTICS_DIR / 'analyse_thematique_detaillee.csv'}")
        print(f"💡 Insights : {STATISTICS_DIR / 'insights_strategiques.txt'}")
        print("="*70)
        
    except Exception as e:
        logger.error(f"❌ Erreur : {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()