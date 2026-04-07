"""
Générateur de Wordclouds par Sentiment et par Thème
Visualisation des mots les plus fréquents
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.config import EXPORTS_DIR, FIGURES_DIR, THEMES
from utils.logger import setup_logger

logger = setup_logger("wordcloud_generator")

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['figure.dpi'] = 300


def charger_corpus():
    """Charge le corpus"""
    logger.info("📥 Chargement corpus...")
    
    filepath = EXPORTS_DIR / "corpus_avec_themes.csv"
    df = pd.read_csv(filepath, encoding="utf-8")
    
    logger.info(f"   Textes : {len(df)}")
    return df


def preparer_stopwords():
    """Prépare stopwords personnalisés"""
    from nltk.corpus import stopwords
    import nltk
    
    try:
        stop_words_fr = set(stopwords.words('french'))
    except LookupError:
        nltk.download('stopwords')
        stop_words_fr = set(stopwords.words('french'))
    
    custom_stops = {
        'senelec', 'électricité', 'courant', 'service', 'services',
        'dafa', 'la', 'nga', 'mu', 'di', 'amna',
        'vraiment', 'beaucoup', 'toujours', 'souvent',
        'merci', 'bonjour', 'bonsoir', 'monsieur', 'madame',
        'plus', 'très', 'bien', 'aussi', 'encore',
    }
    
    return list(stop_words_fr.union(custom_stops))


def generer_wordcloud_par_sentiment(df):
    """Génère 3 wordclouds (négatif, neutre, positif)"""
    logger.info("\n☁️  WORDCLOUDS PAR SENTIMENT")
    
    stopwords = preparer_stopwords()
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    sentiments = [
        ('negative', 'Négatif', '#e74c3c'),
        ('neutral', 'Neutre', '#95a5a6'),
        ('positive', 'Positif', '#27ae60')
    ]
    
    for ax, (sentiment, label, color) in zip(axes, sentiments):
        df_sent = df[df['sentiment_pred'] == sentiment]
        
        if len(df_sent) == 0:
            ax.axis('off')
            continue
        
        text = ' '.join(df_sent['texte_nettoye'].fillna('').astype(str))
        
        wordcloud = WordCloud(
            width=800,
            height=600,
            background_color='white',
            colormap='Reds' if sentiment == 'negative' else 'Greens' if sentiment == 'positive' else 'Greys',
            max_words=100,
            stopwords=stopwords,
            relative_scaling=0.5,
            min_font_size=10,
            collocations=False
        ).generate(text)
        
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'{label} ({len(df_sent)} textes)', 
                    fontsize=16, fontweight='bold', color=color, pad=10)
    
    plt.tight_layout()
    filepath = FIGURES_DIR / "wordclouds_par_sentiment.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"   ✅ {filepath}")


def generer_wordcloud_par_theme_ameliore(df):
    """Génère wordclouds pour les 6 thèmes principaux"""
    logger.info("\n☁️  WORDCLOUDS PAR THÈME (AMÉLIORÉ)")
    
    stopwords = preparer_stopwords()
    
    top_themes = df['theme'].value_counts().head(6).index
    
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    axes = axes.flatten()
    
    for i, theme in enumerate(top_themes):
        df_theme = df[df['theme'] == theme]
        
        text = ' '.join(df_theme['texte_nettoye'].fillna('').astype(str))
        
        wordcloud = WordCloud(
            width=900,
            height=600,
            background_color='white',
            colormap='viridis',
            max_words=80,
            stopwords=stopwords,
            relative_scaling=0.5,
            min_font_size=12,
            collocations=False
        ).generate(text)
        
        axes[i].imshow(wordcloud, interpolation='bilinear')
        axes[i].axis('off')
        
        label_fr = THEMES.get(theme, {}).get('label_fr', theme)
        axes[i].set_title(f'{label_fr}\n({len(df_theme)} publications)', 
                         fontsize=15, fontweight='bold', pad=15)
    
    
    for j in range(len(top_themes), len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    filepath = FIGURES_DIR / "wordclouds_par_theme_ameliore.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"   ✅ {filepath}")


def generer_wordcloud_woyofal_specifique(df):
    """Wordcloud spécifique Woyofal (négatif vs positif)"""
    logger.info("\n☁️  WORDCLOUDS WOYOFAL (NÉGATIF vs POSITIF)")
    
    stopwords = preparer_stopwords()
    stopwords.extend(['woyofal', 'compteur', 'système'])  
    
    df_woy = df[df['theme'] == 'woyofal']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Négatif
    df_woy_neg = df_woy[df_woy['sentiment_pred'] == 'negative']
    text_neg = ' '.join(df_woy_neg['texte_nettoye'].fillna('').astype(str))
    
    wc_neg = WordCloud(
        width=800,
        height=600,
        background_color='white',
        colormap='Reds',
        max_words=60,
        stopwords=stopwords,
        relative_scaling=0.5,
        min_font_size=12
    ).generate(text_neg)
    
    ax1.imshow(wc_neg, interpolation='bilinear')
    ax1.axis('off')
    ax1.set_title(f'Woyofal - Critiques Négatives\n({len(df_woy_neg)} publications)', 
                 fontsize=16, fontweight='bold', color='#e74c3c', pad=15)
    
    # Positif
    df_woy_pos = df_woy[df_woy['sentiment_pred'] == 'positive']
    text_pos = ' '.join(df_woy_pos['texte_nettoye'].fillna('').astype(str))
    
    wc_pos = WordCloud(
        width=800,
        height=600,
        background_color='white',
        colormap='Greens',
        max_words=60,
        stopwords=stopwords,
        relative_scaling=0.5,
        min_font_size=12
    ).generate(text_pos)
    
    ax2.imshow(wc_pos, interpolation='bilinear')
    ax2.axis('off')
    ax2.set_title(f'Woyofal - Points Positifs\n({len(df_woy_pos)} publications)', 
                 fontsize=16, fontweight='bold', color='#27ae60', pad=15)
    
    plt.tight_layout()
    filepath = FIGURES_DIR / "wordcloud_woyofal_comparatif.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"   ✅ {filepath}")


def main():
    """Point d'entrée principal"""
    print("="*70)
    print("☁️  GÉNÉRATION WORDCLOUDS - SENELEC")
    print("="*70)
    
    try:
        df = charger_corpus()
        
        # Générer wordclouds
        generer_wordcloud_par_sentiment(df)
        generer_wordcloud_par_theme_ameliore(df)
        generer_wordcloud_woyofal_specifique(df)
        
        print("\n" + "="*70)
        print("✅ WORDCLOUDS GÉNÉRÉS")
        print("="*70)
        print(f"📁 Dossier : {FIGURES_DIR}")
        print("☁️  3 fichiers créés")
        print("="*70)
        
    except Exception as e:
        logger.error(f"❌ Erreur : {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()