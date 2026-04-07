"""
Topic Modeling avec LDA
Identification de TOUS les thèmes : Coupures, Woyofal, Facturation, Service Client, Qualité
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import pickle
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.config import EXPORTS_DIR, FIGURES_DIR, STATISTICS_DIR, TOPIC_MODEL_DIR, THEMES
from utils.logger import setup_logger
from utils.file_handler import save_csv

logger = setup_logger("topic_modeling")

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

THEMES_COMPLETE = {
    'coupure': {'label_fr': 'Coupures Électricité'},
    'woyofal': {'label_fr': 'Système Woyofal'},
    'facturation': {'label_fr': 'Facturation/Tarifs'},
    'service_client': {'label_fr': 'Service Client'},
    'qualite': {'label_fr': 'Qualité Service'},
    'autre': {'label_fr': 'Thèmes Divers'}
}

def charger_corpus_avec_sentiment():
    """Charge le corpus avec sentiments prédits"""
    logger.info("📥 Chargement du corpus...")
    
    filepath = EXPORTS_DIR / "corpus_avec_sentiment.csv"
    
    if not filepath.exists():
        logger.error(f"❌ Fichier introuvable : {filepath}")
        raise FileNotFoundError(filepath)
    
    df = pd.read_csv(filepath, encoding="utf-8")
    logger.info(f"   Textes chargés : {len(df)}")
    
    return df

def preparer_stopwords_personnalises():
    """Prépare la liste des stopwords"""
    try:
        from nltk.corpus import stopwords
        stop_words_fr = set(stopwords.words('french'))
    except:
        import nltk
        nltk.download('stopwords', quiet=True)
        from nltk.corpus import stopwords
        stop_words_fr = set(stopwords.words('french'))
    
    custom_stops = {
        'senelec', 'électricité', 'courant', 'service',
        'dafa', 'la', 'nga', 'mu', 'di', 'amna',
        'vraiment', 'beaucoup', 'toujours', 'souvent',
        'merci', 'bonjour', 'bonsoir'
    }
    
    return list(stop_words_fr.union(custom_stops))

def entrainer_lda(df, n_topics=8):
    logger.info("\n🤖 ENTRAÎNEMENT LDA")
    logger.info(f"   Nombre de documents : {len(df)}")
    logger.info(f"   Nombre de topics : {n_topics}")
    
    texts = df['texte_nettoye'].fillna('').astype(str).tolist()
    stop_words = preparer_stopwords_personnalises()
    
    logger.info("   Vectorisation TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=1500,
        stop_words=stop_words,
        ngram_range=(1, 3),
        min_df=3,
        max_df=0.6
    )
    
    doc_term_matrix = vectorizer.fit_transform(texts)
    
    logger.info("   Entraînement LDA en cours...")
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=100,
        learning_method='online',
        learning_offset=50.,
        verbose=0
    )
    
    lda_output = lda_model.fit_transform(doc_term_matrix)
    
    logger.info(f"✅ Modèle LDA entraîné : {n_topics} topics")
    
    return lda_model, vectorizer, lda_output, doc_term_matrix

def afficher_top_mots_par_topic(lda_model, vectorizer, n_words=15):
    logger.info("\n📋 TOP MOTS PAR TOPIC")
    
    feature_names = vectorizer.get_feature_names_out()
    topics_mots = {}
    
    for topic_idx, topic in enumerate(lda_model.components_):
        top_indices = topic.argsort()[-n_words:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        topics_mots[topic_idx] = top_words
        
        logger.info(f"\n   Topic {topic_idx}: {', '.join(top_words[:8])}")
    
    return topics_mots

def mapper_topics_vers_themes_intelligent(topics_mots, themes_config=None):
    """✅ VERSION CORRIGÉE : SAFE pour TOUS les thèmes"""
    themes_config = themes_config or THEMES_COMPLETE
    
    logger.info("\n🏷️  MAPPING INTELLIGENT TOPICS → THÈMES MÉTIER")
    
    mapping = {}
    
    theme_keywords_enhanced = {
        'coupure': ['coupure', 'coupures', 'délestage', 'panne', 'blackout', 'pas de courant', 'dysfonctionnements'],
        'woyofal': ['woyofal', 'compteur', 'compteurs', 'code', 'crédit', 'recharge', 'numéro compteur'],
        'facturation': ['facture', 'facturation', 'cher', 'coût', 'tarif', 'prix', 'tranche'],
        'service_client': ['service client', 'agence', 'réclamation', 'plainte', 'agents', 'accueil'],
        'qualite': ['qualité', 'amélioration', 'satisfaction', 'stabilité', 'bon service'],
    }
    
    for topic_id, words in topics_mots.items():
        words_str = ' '.join(words).lower()
        
        scores = {}
        for theme, keywords in theme_keywords_enhanced.items():
            score = sum(3 if k in words[:5] else 2 if k in words[:10] else 1 
                       for k in keywords if k in words_str)
            scores[theme] = score
        
        if max(scores.values()) > 0:
            best_theme = max(scores, key=scores.get)
        else:
            best_theme = 'autre'
        
        mapping[topic_id] = best_theme
        
        if best_theme in themes_config:
            label_fr = themes_config[best_theme]['label_fr']
        else:
            label_fr = f"THÈME {best_theme.upper()}"
        
        score_best = scores.get(best_theme, 0)
        logger.info(f"   Topic {topic_id} → {label_fr:>30} (score: {score_best}, mots: {words[:5]})")
    
    return mapping

def assigner_themes_au_corpus(df, lda_output, mapping):
    logger.info("\n📝 ASSIGNATION DES THÈMES AU CORPUS")
    
    dominant_topics = np.argmax(lda_output, axis=1)
    
    df['topic_id'] = dominant_topics
    df['theme'] = df['topic_id'].map(mapping)
    
    df['topic_probability'] = np.max(lda_output, axis=1)
    
    logger.info("\n📊 DISTRIBUTION DES THÈMES")
    theme_counts = df['theme'].value_counts()
    
    for theme, count in theme_counts.items():
        pct = (count / len(df)) * 100
        label_fr = THEMES_COMPLETE.get(theme, {}).get('label_fr', theme)
        logger.info(f"   {label_fr:>30} : {count:4d} ({pct:5.2f}%)")
    
    return df

def analyser_themes_par_sentiment(df):
    logger.info("\n📊 ANALYSE THÈMES × SENTIMENTS")
    
    crosstab = pd.crosstab(
        df['theme'],
        df['sentiment_pred'],
        normalize='index'
    ) * 100
    
    theme_mapping = {t: THEMES_COMPLETE.get(t, {}).get('label_fr', t) for t in crosstab.index}
    crosstab.index = crosstab.index.map(theme_mapping)
    
    logger.info("\n" + crosstab.round(1).to_string())
    
    filepath = STATISTICS_DIR / "themes_par_sentiment.csv"
    crosstab.to_csv(filepath, encoding='utf-8')
    logger.info(f"\n   Sauvegardé : {filepath}")
    
    return crosstab

def analyser_themes_par_plateforme(df):
    logger.info("\n📱 ANALYSE THÈMES × PLATEFORMES")
    
    crosstab = pd.crosstab(
        df['theme'],
        df['plateforme'],
        normalize='index'
    ) * 100
    
    theme_mapping = {t: THEMES_COMPLETE.get(t, {}).get('label_fr', t) for t in crosstab.index}
    crosstab.index = crosstab.index.map(theme_mapping)
    
    logger.info("\n" + crosstab.round(1).to_string())
    
    filepath = STATISTICS_DIR / "themes_par_plateforme.csv"
    crosstab.to_csv(filepath, encoding='utf-8')
    logger.info(f"\n   Sauvegardé : {filepath}")
    
    return crosstab

def generer_wordclouds_par_theme(df):
    logger.info("\n☁️  GÉNÉRATION WORDCLOUDS PAR THÈME")
    
    all_themes = df['theme'].value_counts().index
    n_themes = min(len(all_themes), 6)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, theme in enumerate(all_themes[:n_themes]):
        textes_theme = df[df['theme'] == theme]['texte_nettoye'].tolist()
        text = ' '.join(textes_theme)
        
        wordcloud = WordCloud(
            width=800, height=400, background_color='white',
            colormap='viridis', max_words=50,
            stopwords=preparer_stopwords_personnalises(),
            relative_scaling=0.5, min_font_size=10
        ).generate(text)
        
        axes[i].imshow(wordcloud, interpolation='bilinear')
        axes[i].axis('off')
        
        label_fr = THEMES_COMPLETE.get(theme, {}).get('label_fr', theme)
        count = len(textes_theme)
        axes[i].set_title(f"{label_fr}\n({count} textes)", fontsize=14, fontweight='bold')
    
    for j in range(n_themes, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    
    filepath = FIGURES_DIR / "wordclouds_par_theme.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   Sauvegardé : {filepath}")

def generer_graphique_themes(df):
    logger.info("\n📊 Génération graphique thèmes...")
    
    theme_counts = df['theme'].value_counts()
    theme_labels = [THEMES_COMPLETE.get(t, {}).get('label_fr', t) for t in theme_counts.index]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['#e74c3c', '#3498db', '#f39c12', '#27ae60', '#9b59b6', '#95a5a6']
    bars = ax.barh(theme_labels, theme_counts.values, color=colors[:len(theme_labels)])
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, f' {int(width)}',
                ha='left', va='center', fontsize=12, fontweight='bold')
    
    ax.set_title("Distribution des Thèmes Principaux", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Nombre de publications", fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    filepath = FIGURES_DIR / "distribution_themes.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   Sauvegardé : {filepath}")

def generer_heatmap_themes_sentiments(df):
    logger.info("\n🔥 Génération heatmap thèmes × sentiments...")
    
    crosstab = pd.crosstab(df['theme'], df['sentiment_pred'])
    
    theme_mapping = {t: THEMES_COMPLETE.get(t, {}).get('label_fr', t) for t in crosstab.index}
    crosstab.index = crosstab.index.map(theme_mapping)
    crosstab.columns = ['Négatif', 'Neutre', 'Positif']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(crosstab, annot=True, fmt='d', cmap='YlOrRd',
                cbar_kws={'label': 'Nombre de publications'},
                linewidths=0.5, ax=ax)
    
    ax.set_title('Distribution Thèmes × Sentiments', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Sentiment', fontsize=12)
    ax.set_ylabel('Thème', fontsize=12)
    
    plt.tight_layout()
    
    filepath = FIGURES_DIR / "heatmap_themes_sentiments.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   Sauvegardé : {filepath}")

def sauvegarder_modele(lda_model, vectorizer):
    logger.info("\n💾 SAUVEGARDE DU MODÈLE")
    
    TOPIC_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(TOPIC_MODEL_DIR / "lda_model.pkl", 'wb') as f:
        pickle.dump(lda_model, f)
    
    with open(TOPIC_MODEL_DIR / "vectorizer.pkl", 'wb') as f:
        pickle.dump(vectorizer, f)
    
    logger.info(f"   Modèle sauvegardé : {TOPIC_MODEL_DIR}")

def sauvegarder_corpus_avec_themes(df):
    logger.info("\n💾 SAUVEGARDE CORPUS AVEC THÈMES")
    
    filepath = EXPORTS_DIR / "corpus_avec_themes.csv"
    save_csv(df, filepath)
    
    logger.info(f"   Fichier : {filepath}")
    logger.info(f"   Lignes  : {len(df)}")

def generer_rapport_themes(df):
    logger.info("\n📋 GÉNÉRATION RAPPORT THÈMES DÉTAILLÉ")
    
    rapport = []
    
    for theme in df['theme'].unique():
        df_theme = df[df['theme'] == theme]
        
        rapport.append({
            'theme': theme,
            'label_fr': THEMES_COMPLETE.get(theme, {}).get('label_fr', theme),
            'total': len(df_theme),
            'pct_corpus': (len(df_theme) / len(df)) * 100,
            'negative': (df_theme['sentiment_pred'] == 'negative').sum(),
            'neutral': (df_theme['sentiment_pred'] == 'neutral').sum(),
            'positive': (df_theme['sentiment_pred'] == 'positive').sum(),
            'pct_negative': ((df_theme['sentiment_pred'] == 'negative').sum() / len(df_theme)) * 100,
            'prob_moyenne': df_theme['topic_probability'].mean(),
            'facebook': (df_theme['plateforme'] == 'facebook').sum(),
            'twitter': (df_theme['plateforme'] == 'twitter').sum(),
            'enquete': (df_theme['plateforme'] == 'enquete').sum(),
        })
    
    df_rapport = pd.DataFrame(rapport)
    df_rapport = df_rapport.sort_values('total', ascending=False)
    
    filepath = STATISTICS_DIR / "rapport_themes_detaille.csv"
    df_rapport.to_csv(filepath, index=False, encoding='utf-8')
    
    logger.info(f"   Rapport sauvegardé : {filepath}")
    logger.info("\n" + df_rapport.to_string(index=False))
    
    return df_rapport

def main():
    """Point d'entrée principal"""
    print("="*70)
    print("🔍 TOPIC MODELING AMÉLIORÉ - SENELEC")
    print("="*70)
    
    try:
        df = charger_corpus_avec_sentiment()
        
        lda_model, vectorizer, lda_output, _ = entrainer_lda(df, n_topics=8)
        
        topics_mots = afficher_top_mots_par_topic(lda_model, vectorizer, n_words=15)
        
        mapping = mapper_topics_vers_themes_intelligent(topics_mots)
        
        df = assigner_themes_au_corpus(df, lda_output, mapping)
        
        analyser_themes_par_sentiment(df)
        analyser_themes_par_plateforme(df)
        
        generer_wordclouds_par_theme(df)
        generer_graphique_themes(df)
        generer_heatmap_themes_sentiments(df)
        
        generer_rapport_themes(df)
        
        sauvegarder_modele(lda_model, vectorizer)
        sauvegarder_corpus_avec_themes(df)
        
        print("\n" + "="*70)
        print("✅ TOPIC MODELING TERMINÉ SANS ERREUR !")
        print("="*70)
        print(f"📁 Corpus : {EXPORTS_DIR / 'corpus_avec_themes.csv'}")
        print(f"📁 Modèle : {TOPIC_MODEL_DIR}")
        print(f"📊 Graphiques : {FIGURES_DIR}")
        print(f"📋 Rapport : {STATISTICS_DIR / 'rapport_themes_detaille.csv'}")
        print("="*70)
        
    except Exception as e:
        logger.error(f"❌ Erreur : {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
