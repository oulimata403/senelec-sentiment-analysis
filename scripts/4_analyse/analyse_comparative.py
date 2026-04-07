"""
Analyse Comparative Woyofal vs Postpayé
Objectif clé du mémoire : Comparer les deux systèmes
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.config import RAW_DATA_DIR, EXPORTS_DIR, FIGURES_DIR, STATISTICS_DIR
from utils.logger import setup_logger
from utils.file_handler import save_csv

logger = setup_logger("analyse_comparative")

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def charger_enquete():
    """Charge les données d'enquête"""
    logger.info("📥 Chargement enquête terrain...")
    
    filepath = RAW_DATA_DIR / "Enquête_SENELEC.csv"
    
    if not filepath.exists():
        logger.error(f"❌ Fichier introuvable : {filepath}")
        raise FileNotFoundError(filepath)
    
    df = pd.read_csv(filepath, encoding="utf-8")
    logger.info(f"   Répondants : {len(df)}")
    
    return df


def charger_corpus_themes():
    """Charge corpus avec thèmes (réseaux sociaux)"""
    logger.info("📥 Chargement corpus réseaux sociaux...")
    
    filepath = EXPORTS_DIR / "corpus_avec_themes.csv"
    
    if not filepath.exists():
        logger.error(f"❌ Fichier introuvable : {filepath}")
        raise FileNotFoundError(filepath)
    
    df = pd.read_csv(filepath, encoding="utf-8")
    logger.info(f"   Publications : {len(df)}")
    
    return df


def analyser_satisfaction_par_type_client_enquete(df_enquete):
    """Analyse satisfaction selon type client (enquête)"""
    logger.info("\n📊 SATISFACTION PAR TYPE CLIENT (ENQUÊTE)")
    
    col_type = "Quel est votre type de client SENELEC ?"
    col_satisfaction = "De manière générale, êtes-vous satisfait(e) des services de la SENELEC ?"
    
    if col_type not in df_enquete.columns or col_satisfaction not in df_enquete.columns:
        logger.warning("⚠️ Colonnes manquantes dans l'enquête")
        return None
    
    # Crosstab
    crosstab = pd.crosstab(
        df_enquete[col_type],
        df_enquete[col_satisfaction],
        normalize='index'
    ) * 100
    
    logger.info("\n" + crosstab.round(1).to_string())
    
    # Sauvegarder
    filepath = STATISTICS_DIR / "satisfaction_par_type_client.csv"
    crosstab.to_csv(filepath, encoding='utf-8')
    
    return crosstab


def comparer_woyofal_vs_postpaye_reseaux(df_corpus):
    """Compare sentiment Woyofal vs autres thèmes sur réseaux"""
    logger.info("\n📱 COMPARAISON WOYOFAL vs AUTRES (RÉSEAUX SOCIAUX)")
    
    # Sentiment Woyofal
    df_woyofal = df_corpus[df_corpus['theme'] == 'woyofal']
    df_autres = df_corpus[df_corpus['theme'] != 'woyofal']
    
    sentiment_woyofal = df_woyofal['sentiment_pred'].value_counts(normalize=True) * 100
    sentiment_autres = df_autres['sentiment_pred'].value_counts(normalize=True) * 100
    
    logger.info("\n   Woyofal :")
    for sent, pct in sentiment_woyofal.items():
        logger.info(f"      {sent:10s} : {pct:5.1f}%")
    
    logger.info("\n   Autres thèmes :")
    for sent, pct in sentiment_autres.items():
        logger.info(f"      {sent:10s} : {pct:5.1f}%")
    
    return sentiment_woyofal, sentiment_autres


def generer_graphique_comparaison_woyofal(sentiment_woyofal, sentiment_autres):
    """Génère graphique comparatif"""
    logger.info("\n📊 Génération graphique comparatif...")
    
    # Préparer données
    categories = ['Woyofal', 'Autres Thèmes']
    negative = [
        sentiment_woyofal.get('negative', 0),
        sentiment_autres.get('negative', 0)
    ]
    neutral = [
        sentiment_woyofal.get('neutral', 0),
        sentiment_autres.get('neutral', 0)
    ]
    positive = [
        sentiment_woyofal.get('positive', 0),
        sentiment_autres.get('positive', 0)
    ]
    
    # Graphique
    x = np.arange(len(categories))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.bar(x - width, negative, width, label='Négatif', color='#e74c3c')
    ax.bar(x, neutral, width, label='Neutre', color='#95a5a6')
    ax.bar(x + width, positive, width, label='Positif', color='#27ae60')
    
    ax.set_title('Comparaison Sentiment : Woyofal vs Autres Thèmes',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Pourcentage (%)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    filepath = FIGURES_DIR / "comparaison_woyofal_vs_autres.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   Sauvegardé : {filepath}")


def test_statistique_chi2(df_corpus):
    """Test Chi² : différence significative Woyofal vs Autres"""
    logger.info("\n📊 TEST STATISTIQUE CHI²")
    
    # Tableau de contingence
    df_corpus['est_woyofal'] = df_corpus['theme'] == 'woyofal'
    
    contingency = pd.crosstab(
        df_corpus['est_woyofal'],
        df_corpus['sentiment_pred']
    )
    
    # Test Chi²
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    
    logger.info(f"\n   Chi² = {chi2:.4f}")
    logger.info(f"   p-value = {p_value:.6f}")
    logger.info(f"   Degrés de liberté = {dof}")
    
    if p_value < 0.05:
        logger.info("   ✅ Différence SIGNIFICATIVE (p < 0.05)")
    else:
        logger.info("   ⚠️  Différence NON significative (p >= 0.05)")
    
    return chi2, p_value


def analyser_problemes_woyofal_enquete(df_enquete):
    """Analyse problèmes Woyofal selon enquête"""
    logger.info("\n🔍 PROBLÈMES WOYOFAL (ENQUÊTE)")
    
    col_problemes = "Quels problèmes avez-vous rencontrés avec le système Woyofal ?\n(Plusieurs réponses possibles)  "
    
    if col_problemes in df_enquete.columns:
        problemes = df_enquete[col_problemes].dropna()
        
        logger.info(f"\n   Répondants ayant rapporté des problèmes : {len(problemes)}")
        
        if len(problemes) > 0:
            sample = problemes.head(5)
            logger.info("\n   Exemples de problèmes :")
            for prob in sample:
                logger.info(f"      - {str(prob)[:80]}...")


def generer_rapport_comparatif(df_enquete, df_corpus):
    """Génère rapport comparatif complet"""
    logger.info("\n📋 GÉNÉRATION RAPPORT COMPARATIF")
    
    rapport = {
        'source': [],
        'categorie': [],
        'sentiment_negatif_pct': [],
        'sentiment_neutre_pct': [],
        'sentiment_positif_pct': [],
        'total': []
    }
    
    # Réseaux sociaux - Woyofal
    df_woy = df_corpus[df_corpus['theme'] == 'woyofal']
    rapport['source'].append('Réseaux Sociaux')
    rapport['categorie'].append('Woyofal')
    rapport['sentiment_negatif_pct'].append((df_woy['sentiment_pred'] == 'negative').sum() / len(df_woy) * 100)
    rapport['sentiment_neutre_pct'].append((df_woy['sentiment_pred'] == 'neutral').sum() / len(df_woy) * 100)
    rapport['sentiment_positif_pct'].append((df_woy['sentiment_pred'] == 'positive').sum() / len(df_woy) * 100)
    rapport['total'].append(len(df_woy))
    
    # Réseaux sociaux - Autres
    df_autres = df_corpus[df_corpus['theme'] != 'woyofal']
    rapport['source'].append('Réseaux Sociaux')
    rapport['categorie'].append('Autres Thèmes')
    rapport['sentiment_negatif_pct'].append((df_autres['sentiment_pred'] == 'negative').sum() / len(df_autres) * 100)
    rapport['sentiment_neutre_pct'].append((df_autres['sentiment_pred'] == 'neutral').sum() / len(df_autres) * 100)
    rapport['sentiment_positif_pct'].append((df_autres['sentiment_pred'] == 'positive').sum() / len(df_autres) * 100)
    rapport['total'].append(len(df_autres))
    
    df_rapport = pd.DataFrame(rapport)
    
    filepath = STATISTICS_DIR / "rapport_comparatif_woyofal.csv"
    df_rapport.to_csv(filepath, index=False, encoding='utf-8')
    
    logger.info(f"   Rapport sauvegardé : {filepath}")
    logger.info("\n" + df_rapport.to_string(index=False))
    
    return df_rapport


def main():
    """Point d'entrée principal"""
    print("="*70)
    print("⚖️  ANALYSE COMPARATIVE WOYOFAL - SENELEC")
    print("="*70)
    
    try:
        # Charger données
        df_enquete = charger_enquete()
        df_corpus = charger_corpus_themes()
        
        # Analyses enquête
        analyser_satisfaction_par_type_client_enquete(df_enquete)
        analyser_problemes_woyofal_enquete(df_enquete)
        
        # Analyses réseaux sociaux
        sentiment_woy, sentiment_autres = comparer_woyofal_vs_postpaye_reseaux(df_corpus)
        generer_graphique_comparaison_woyofal(sentiment_woy, sentiment_autres)
        
        # Test statistique
        test_statistique_chi2(df_corpus)
        
        # Rapport
        generer_rapport_comparatif(df_enquete, df_corpus)
        
        print("\n" + "="*70)
        print("✅ ANALYSE COMPARATIVE TERMINÉE")
        print("="*70)
        print(f"📊 Graphiques : {FIGURES_DIR}")
        print(f"📋 Rapport : {STATISTICS_DIR / 'rapport_comparatif_woyofal.csv'}")
        print("="*70)
        
    except Exception as e:
        logger.error(f"❌ Erreur : {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()