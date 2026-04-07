"""
Analyse Géographique - Répartition par région
Objectif : Identifier les régions les plus critiques
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

from config.config import RAW_DATA_DIR, EXPORTS_DIR, FIGURES_DIR, STATISTICS_DIR
from utils.logger import setup_logger
from utils.file_handler import save_csv

logger = setup_logger("analyse_geographique")

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def charger_enquete():
    """Charge les données d'enquête avec info géographique"""
    logger.info("📥 Chargement enquête terrain...")
    
    filepath = RAW_DATA_DIR / "Enquête_SENELEC.csv"
    
    if not filepath.exists():
        logger.error(f"❌ Fichier introuvable : {filepath}")
        raise FileNotFoundError(filepath)
    
    df = pd.read_csv(filepath, encoding="utf-8")
    logger.info(f"   Répondants : {len(df)}")
    
    return df


def analyser_repartition_geographique(df_enquete):
    """Analyse la répartition géographique des répondants"""
    logger.info("\n🗺️  RÉPARTITION GÉOGRAPHIQUE DES RÉPONDANTS")
    
    col_region = "Dans quelle région résidez-vous ?  "
    
    if col_region not in df_enquete.columns:
        logger.warning("⚠️ Colonne région introuvable")
        return None
    
    repartition = df_enquete[col_region].value_counts()
    
    logger.info(f"\n   Total régions : {len(repartition)}")
    
    for region, count in repartition.items():
        pct = (count / len(df_enquete)) * 100
        logger.info(f"   {region:20s} : {count:4d} ({pct:5.1f}%)")
    
    return repartition


def analyser_satisfaction_par_region(df_enquete):
    """Analyse satisfaction par région"""
    logger.info("\n📊 SATISFACTION PAR RÉGION")
    
    col_region = "Dans quelle région résidez-vous ?  "
    col_satisfaction = "De manière générale, êtes-vous satisfait(e) des services de la SENELEC ?"
    
    if col_region not in df_enquete.columns or col_satisfaction not in df_enquete.columns:
        logger.warning("⚠️ Colonnes manquantes")
        return None
    
    # Crosstab
    crosstab = pd.crosstab(
        df_enquete[col_region],
        df_enquete[col_satisfaction],
        normalize='index'
    ) * 100
    
    logger.info("\n" + crosstab.round(1).to_string())
    
    # Sauvegarder
    filepath = STATISTICS_DIR / "satisfaction_par_region.csv"
    crosstab.to_csv(filepath, encoding='utf-8')
    logger.info(f"\n   Sauvegardé : {filepath}")
    
    return crosstab


def identifier_regions_critiques(df_enquete):
    """Identifie les régions avec le plus d'insatisfaction"""
    logger.info("\n🔥 RÉGIONS CRITIQUES (INSATISFACTION)")
    
    col_region = "Dans quelle région résidez-vous ?  "
    col_satisfaction = "De manière générale, êtes-vous satisfait(e) des services de la SENELEC ?"
    
    if col_region not in df_enquete.columns or col_satisfaction not in df_enquete.columns:
        return None
    
    # Compter insatisfaits par région
    df_insatisfaits = df_enquete[
        df_enquete[col_satisfaction].isin(['Insatisfait(e)', 'Très insatisfait(e)', 'Insatisfait', 'Très insatisfait'])
    ]
    
    insatisfaits_par_region = df_insatisfaits[col_region].value_counts()
    
    # Calculer taux d'insatisfaction
    total_par_region = df_enquete[col_region].value_counts()
    taux_insatisfaction = (insatisfaits_par_region / total_par_region * 100).sort_values(ascending=False)
    
    logger.info("\n   Top 5 régions les plus insatisfaites :")
    for region, taux in taux_insatisfaction.head(5).items():
        count = insatisfaits_par_region.get(region, 0)
        logger.info(f"      {region:20s} : {taux:5.1f}% ({count} insatisfaits)")
    
    return taux_insatisfaction


def analyser_problemes_par_region(df_enquete):
    """Analyse les problèmes rencontrés par région"""
    logger.info("\n⚠️  PROBLÈMES PAR RÉGION")
    
    col_region = "Dans quelle région résidez-vous ?  "
    col_problemes = "Quels sont les principaux problèmes que vous rencontrez ?\n(Plusieurs réponses possibles)"
    
    if col_region not in df_enquete.columns or col_problemes not in df_enquete.columns:
        logger.warning("⚠️ Colonnes manquantes")
        return None
    
    # Analyser fréquence des mentions de problèmes clés
    problemes_cles = {
        'Coupures': ['coupure', 'délestage', 'panne'],
        'Facturation': ['facture', 'cher', 'coût', 'prix'],
        'Woyofal': ['woyofal', 'prépayé', 'compteur'],
        'Service Client': ['service', 'accueil', 'agence']
    }
    
    resultats = []
    
    for region in df_enquete[col_region].unique():
        if pd.isna(region):
            continue
        
        df_region = df_enquete[df_enquete[col_region] == region]
        problemes_region = df_region[col_problemes].dropna().str.lower()
        
        row = {'region': region, 'total_repondants': len(df_region)}
        
        for prob_type, keywords in problemes_cles.items():
            count = sum(problemes_region.str.contains('|'.join(keywords), na=False))
            row[prob_type] = count
        
        resultats.append(row)
    
    df_problemes = pd.DataFrame(resultats)
    
    # Sauvegarder
    filepath = STATISTICS_DIR / "problemes_par_region.csv"
    df_problemes.to_csv(filepath, index=False, encoding='utf-8')
    
    logger.info(f"\n   Fichier sauvegardé : {filepath}")
    logger.info("\n" + df_problemes.to_string(index=False))
    
    return df_problemes


def generer_graphique_repartition_regions(repartition):
    """Génère graphique répartition géographique"""
    logger.info("\n📊 Génération graphique répartition...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Top 10 régions
    top_regions = repartition.head(10)
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_regions)))
    bars = ax.barh(range(len(top_regions)), top_regions.values, color=colors)
    
    ax.set_yticks(range(len(top_regions)))
    ax.set_yticklabels(top_regions.index, fontsize=11)
    ax.set_xlabel('Nombre de répondants', fontsize=12)
    ax.set_title('Répartition Géographique des Répondants', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    # Annotations
    for i, (bar, val) in enumerate(zip(bars, top_regions.values)):
        ax.text(val, i, f' {val}', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    filepath = FIGURES_DIR / "repartition_geographique.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   Sauvegardé : {filepath}")


def generer_heatmap_satisfaction_regions(crosstab):
    """Génère heatmap satisfaction par région"""
    logger.info("\n🔥 Génération heatmap satisfaction...")
    
    if crosstab is None or crosstab.empty:
        logger.warning("⚠️ Pas de données pour heatmap")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    col_order = []
    for col in ['Très insatisfait(e)', 'Insatisfait(e)', 'Neutre', 'Satisfait(e)', 'Très satisfait(e)']:
        if col in crosstab.columns:
            col_order.append(col)
    
    if col_order:
        crosstab_ordered = crosstab[col_order]
    else:
        crosstab_ordered = crosstab
    
    sns.heatmap(
        crosstab_ordered,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        cbar_kws={'label': 'Pourcentage (%)'},
        linewidths=0.5,
        ax=ax
    )
    
    ax.set_title('Satisfaction par Région', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Niveau de satisfaction', fontsize=12)
    ax.set_ylabel('Région', fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    filepath = FIGURES_DIR / "heatmap_satisfaction_regions.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   Sauvegardé : {filepath}")


def generer_carte_chaleur_insatisfaction(taux_insatisfaction):
    """Génère graphique des taux d'insatisfaction"""
    logger.info("\n📊 Génération carte insatisfaction...")
    
    if taux_insatisfaction is None or taux_insatisfaction.empty:
        logger.warning("⚠️ Pas de données")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Top 10
    top_10 = taux_insatisfaction.head(10)
    
    colors = ['#e74c3c' if x > 30 else '#f39c12' if x > 20 else '#27ae60' 
              for x in top_10.values]
    
    bars = ax.barh(range(len(top_10)), top_10.values, color=colors)
    
    ax.set_yticks(range(len(top_10)))
    ax.set_yticklabels(top_10.index, fontsize=11)
    ax.set_xlabel('Taux d\'insatisfaction (%)', fontsize=12)
    ax.set_title('Régions par Taux d\'Insatisfaction', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    # Ligne de référence
    ax.axvline(x=30, color='red', linestyle='--', alpha=0.5, label='Seuil critique (30%)')
    ax.legend()
    
    # Annotations
    for i, (bar, val) in enumerate(zip(bars, top_10.values)):
        ax.text(val, i, f' {val:.1f}%', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    filepath = FIGURES_DIR / "taux_insatisfaction_regions.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   Sauvegardé : {filepath}")


def generer_rapport_geographique(df_enquete, repartition, taux_insatisfaction):
    """Génère rapport géographique complet"""
    logger.info("\n📋 GÉNÉRATION RAPPORT GÉOGRAPHIQUE")
    
    col_region = "Dans quelle région résidez-vous ?  "
    
    rapport = []
    
    for region in repartition.index:
        df_region = df_enquete[df_enquete[col_region] == region]
        
        rapport.append({
            'region': region,
            'repondants': len(df_region),
            'pct_total': (len(df_region) / len(df_enquete)) * 100,
            'taux_insatisfaction': taux_insatisfaction.get(region, 0) if taux_insatisfaction is not None else 0,
        })
    
    df_rapport = pd.DataFrame(rapport)
    df_rapport = df_rapport.sort_values('taux_insatisfaction', ascending=False)
    
    filepath = STATISTICS_DIR / "rapport_geographique.csv"
    df_rapport.to_csv(filepath, index=False, encoding='utf-8')
    
    logger.info(f"   Rapport sauvegardé : {filepath}")
    logger.info("\n" + df_rapport.head(10).to_string(index=False))
    
    return df_rapport


def main():
    """Point d'entrée principal"""
    print("="*70)
    print("🗺️  ANALYSE GÉOGRAPHIQUE - SENELEC")
    print("="*70)
    
    try:
        # Charger enquête
        df_enquete = charger_enquete()
        
        # Analyses
        repartition = analyser_repartition_geographique(df_enquete)
        crosstab = analyser_satisfaction_par_region(df_enquete)
        taux_insatisfaction = identifier_regions_critiques(df_enquete)
        analyser_problemes_par_region(df_enquete)
        
        # Visualisations
        if repartition is not None:
            generer_graphique_repartition_regions(repartition)
        
        if crosstab is not None:
            generer_heatmap_satisfaction_regions(crosstab)
        
        if taux_insatisfaction is not None:
            generer_carte_chaleur_insatisfaction(taux_insatisfaction)
        
        # Rapport
        generer_rapport_geographique(df_enquete, repartition, taux_insatisfaction)
        
        print("\n" + "="*70)
        print("✅ ANALYSE GÉOGRAPHIQUE TERMINÉE")
        print("="*70)
        print(f"📊 Graphiques : {FIGURES_DIR}")
        print(f"📋 Rapport : {STATISTICS_DIR / 'rapport_geographique.csv'}")
        print("="*70)
        
    except Exception as e:
        logger.error(f"❌ Erreur : {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()