"""
Génération de TOUS les graphiques 
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

logger = setup_logger("generer_graphiques")


plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11


def charger_corpus():
    """Charge le corpus complet"""
    logger.info("📥 Chargement corpus...")
    
    filepath = EXPORTS_DIR / "corpus_avec_themes.csv"
    df = pd.read_csv(filepath, encoding="utf-8")
    
    logger.info(f"   Textes : {len(df)}")
    return df


def graphique_1_distribution_sentiments_global(df):
    """Distribution globale des sentiments (PIE + BAR)"""
    logger.info("\n📊 Graphique 1 : Distribution sentiments...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # PIE
    sentiment_counts = df['sentiment_pred'].value_counts()
    colors = ['#e74c3c', '#95a5a6', '#27ae60']
    labels_fr = ['Négatif', 'Neutre', 'Positif']
    
    wedges, texts, autotexts = ax1.pie(
        sentiment_counts.values,
        labels=labels_fr,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 13, 'fontweight': 'bold'}
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(14)
    
    ax1.set_title('Répartition Globale du Sentiment', 
                  fontsize=16, fontweight='bold', pad=20)
    
    # BAR
    bars = ax2.bar(range(len(sentiment_counts)), sentiment_counts.values, 
                   color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_xticks(range(len(sentiment_counts)))
    ax2.set_xticklabels(labels_fr, fontsize=13)
    ax2.set_ylabel('Nombre de publications', fontsize=13)
    ax2.set_title('Distribution des Sentiments', 
                  fontsize=16, fontweight='bold', pad=20)
    ax2.grid(axis='y', alpha=0.3)
    
    # Annotations
    for bar, val in zip(bars, sentiment_counts.values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}\n({val/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    filepath = FIGURES_DIR / "memoire_fig1_distribution_sentiments.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"   ✅ {filepath}")


def graphique_2_themes_par_plateforme(df):
    """Distribution thèmes par plateforme (STACKED BAR)"""
    logger.info("\n📊 Graphique 2 : Thèmes par plateforme...")
    
    # Crosstab
    crosstab = pd.crosstab(df['plateforme'], df['theme'])
    
    # Renommer avec labels FR
    theme_labels = {theme: THEMES.get(theme, {}).get('label_fr', theme) 
                   for theme in crosstab.columns}
    crosstab.columns = [theme_labels[col] for col in crosstab.columns]
    
    # Graphique
    fig, ax = plt.subplots(figsize=(14, 8))
    
    crosstab.plot(kind='bar', stacked=True, ax=ax, 
                  colormap='tab10', edgecolor='black', linewidth=0.5)
    
    ax.set_title('Distribution des Thèmes par Plateforme', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Plateforme', fontsize=14)
    ax.set_ylabel('Nombre de publications', fontsize=14)
    ax.legend(title='Thème', fontsize=11, title_fontsize=12, 
              bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=0, fontsize=13)
    
    plt.tight_layout()
    filepath = FIGURES_DIR / "memoire_fig2_themes_par_plateforme.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"   ✅ {filepath}")


def graphique_3_heatmap_themes_sentiments(df):
    """Heatmap Thèmes × Sentiments (HEATMAP ANNOTÉE)"""
    logger.info("\n📊 Graphique 3 : Heatmap thèmes×sentiments...")
    
    # Crosstab en valeurs absolues
    crosstab = pd.crosstab(df['theme'], df['sentiment_pred'])
    
    # Renommer
    theme_labels = {theme: THEMES.get(theme, {}).get('label_fr', theme) 
                   for theme in crosstab.index}
    crosstab.index = [theme_labels[idx] for idx in crosstab.index]
    crosstab.columns = ['Négatif', 'Neutre', 'Positif']
    
    # Graphique
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(
        crosstab,
        annot=True,
        fmt='d',
        cmap='RdYlGn_r',
        cbar_kws={'label': 'Nombre de publications'},
        linewidths=1,
        linecolor='white',
        ax=ax,
        vmin=0,
        annot_kws={'fontsize': 11, 'fontweight': 'bold'}
    )
    
    ax.set_title('Distribution Thèmes × Sentiments', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Sentiment', fontsize=14)
    ax.set_ylabel('Thème', fontsize=14)
    
    plt.tight_layout()
    filepath = FIGURES_DIR / "memoire_fig3_heatmap_themes_sentiments.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"   ✅ {filepath}")


def graphique_4_top_themes_sentiments(df):
    """Top thèmes avec répartition sentiments (GROUPED BAR)"""
    logger.info("\n📊 Graphique 4 : Top thèmes détaillés...")
    
    
    top_themes = df['theme'].value_counts().head(5).index
    df_top = df[df['theme'].isin(top_themes)]
    
  
    crosstab = pd.crosstab(df_top['theme'], df_top['sentiment_pred'], normalize='index') * 100
    
    
    theme_labels = {theme: THEMES.get(theme, {}).get('label_fr', theme) 
                   for theme in crosstab.index}
    crosstab.index = [theme_labels[idx] for idx in crosstab.index]
    crosstab.columns = ['Négatif', 'Neutre', 'Positif']
    
    
    crosstab = crosstab.sort_values('Négatif', ascending=True)
    
    # Graphique
    fig, ax = plt.subplots(figsize=(14, 8))
    
    crosstab.plot(kind='barh', ax=ax, 
                  color=['#e74c3c', '#95a5a6', '#27ae60'],
                  edgecolor='black', linewidth=1)
    
    ax.set_title('Répartition des Sentiments par Thème (Top 5)', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Pourcentage (%)', fontsize=14)
    ax.set_ylabel('Thème', fontsize=14)
    ax.legend(title='Sentiment', fontsize=12, title_fontsize=13)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    filepath = FIGURES_DIR / "memoire_fig4_top_themes_sentiments.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"   ✅ {filepath}")


def graphique_5_comparaison_woyofal(df):
    """Graphique comparatif Woyofal (TWIN BARS)"""
    logger.info("\n📊 Graphique 5 : Comparaison Woyofal...")
    
    # Séparer
    df_woy = df[df['theme'] == 'woyofal']
    df_autres = df[df['theme'] != 'woyofal']
    
    # Compter
    woy_sent = df_woy['sentiment_pred'].value_counts(normalize=True) * 100
    autres_sent = df_autres['sentiment_pred'].value_counts(normalize=True) * 100
    
    # Préparer données
    categories = ['Négatif', 'Neutre', 'Positif']
    woy_values = [woy_sent.get('negative', 0), 
                  woy_sent.get('neutral', 0), 
                  woy_sent.get('positive', 0)]
    autres_values = [autres_sent.get('negative', 0), 
                     autres_sent.get('neutral', 0), 
                     autres_sent.get('positive', 0)]
    
    # Graphique
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars1 = ax.bar(x - width/2, woy_values, width, 
                   label='Woyofal', color='#e74c3c', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, autres_values, width, 
                   label='Autres Thèmes', color='#3498db', edgecolor='black', linewidth=1.5)
    
    ax.set_title('Comparaison Sentiment : Woyofal vs Autres Thèmes', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel('Pourcentage (%)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=13)
    ax.legend(fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    
    # Annotations
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    filepath = FIGURES_DIR / "memoire_fig5_comparaison_woyofal.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"   ✅ {filepath}")


def graphique_6_evolution_temporelle():
    """Graphique évolution temporelle (déjà généré mais on refait en haute qualité)"""
    logger.info("\n📊 Graphique 6 : Évolution temporelle...")
    
    # Charger stats temporelles
    filepath_stats = STATISTICS_DIR / "evolution_temporelle.csv"
    
    if not filepath_stats.exists():
        logger.warning("   ⚠️ Pas de stats temporelles")
        return
    
    df_temp = pd.read_csv(filepath_stats, index_col=0, encoding='utf-8')
    
    # Calculer pourcentages
    df_temp['pct_negatif'] = (df_temp['negatifs'] / df_temp['total']) * 100
    df_temp['pct_positif'] = ((df_temp['total'] - df_temp['negatifs']) / df_temp['total']) * 100
    
    # Graphique
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x = range(len(df_temp))
    
    ax.plot(x, df_temp['pct_negatif'], 
            label='Négatif', color='#e74c3c', linewidth=3, marker='o', markersize=8)
    ax.plot(x, df_temp['pct_positif'], 
            label='Positif', color='#27ae60', linewidth=3, marker='^', markersize=8)
    
    ax.set_title('Évolution du Sentiment au Fil du Temps', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Période', fontsize=14)
    ax.set_ylabel('Pourcentage (%)', fontsize=14)
    ax.legend(fontsize=13, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.xticks(x[::2], df_temp.index[::2], rotation=45, ha='right', fontsize=10)
    
    plt.tight_layout()
    filepath = FIGURES_DIR / "memoire_fig6_evolution_temporelle.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"   ✅ {filepath}")


def main():
    """Point d'entrée principal"""
    print("="*70)
    print("🎨 GÉNÉRATION GRAPHIQUES MÉMOIRE - SENELEC")
    print("="*70)
    
    try:
        df = charger_corpus()
        
        # Générer tous les graphiques
        graphique_1_distribution_sentiments_global(df)
        graphique_2_themes_par_plateforme(df)
        graphique_3_heatmap_themes_sentiments(df)
        graphique_4_top_themes_sentiments(df)
        graphique_5_comparaison_woyofal(df)
        graphique_6_evolution_temporelle()
        
        print("\n" + "="*70)
        print("✅ TOUS LES GRAPHIQUES GÉNÉRÉS")
        print("="*70)
        print(f"📁 Dossier : {FIGURES_DIR}")
        print("📊 6 graphiques haute qualité (300 DPI)")
        print("="*70)
        
    except Exception as e:
        logger.error(f"❌ Erreur : {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()