"""
Analyse Complète de l'Enquête Terrain
Description: Script d'analyse des données d'enquête SENELEC avec visualisations
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

from config.config import RAW_DATA_DIR, FIGURES_DIR, STATISTICS_DIR
from utils.logger import setup_logger
from utils.file_handler import save_csv

logger = setup_logger("analyse_enquete")

# Configuration matplotlib
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


def charger_enquete():
    """Charge les données d'enquête"""
    logger.info("📥 Chargement enquête terrain...")
    
    filepath = RAW_DATA_DIR / "Enquête_SENELEC.csv"
    
    if not filepath.exists():
        logger.error(f"❌ Fichier introuvable : {filepath}")
        raise FileNotFoundError(f"Fichier non trouvé : {filepath}")
    
    df = pd.read_csv(filepath, encoding="utf-8")
    logger.info(f"   ✅ Répondants : {len(df)}")
    
    return df


def analyser_profil_repondants(df):
    """Analyse le profil sociodémographique complet"""
    logger.info("\n" + "="*50)
    logger.info("👥 PROFIL DES RÉPONDANTS")
    logger.info("="*50)
    
    stats = {}
    
    # === ÂGE ===
    col_age = "Votre tranche d'âge ?"
    if col_age in df.columns:
        logger.info("\n📊 RÉPARTITION PAR ÂGE :")
        age_dist = df[col_age].value_counts().sort_index()
        stats['age'] = age_dist
        
        # Affichage formaté
        for age, count in age_dist.items():
            pct = (count / len(df)) * 100
            barre = '█' * int(pct // 2)
            logger.info(f"   {age:15s} : {count:4d} ({pct:5.1f}%) {barre}")
        
        # Statistiques clés
        jeunes = age_dist.get('18 – 25 ans', 0) + age_dist.get('26 – 35 ans', 0)
        logger.info(f"\n   👉 Moins de 35 ans : {jeunes} répondants ({jeunes/len(df)*100:.1f}%)")
    
    # === SEXE ===
    col_sexe = "Votre Sexe ?"
    if col_sexe in df.columns:
        logger.info("\n📊 RÉPARTITION PAR SEXE :")
        sexe_dist = df[col_sexe].value_counts()
        stats['sexe'] = sexe_dist
        
        for sexe, count in sexe_dist.items():
            pct = (count / len(df)) * 100
            barre = '█' * int(pct // 2)
            logger.info(f"   {sexe:10s} : {count:4d} ({pct:5.1f}%) {barre}")
    
    # === RÉGION ===
    col_region = "Dans quelle région résidez-vous ?  "
    if col_region in df.columns:
        logger.info("\n📊 RÉPARTITION PAR RÉGION :")
        region_dist = df[col_region].value_counts()
        stats['region'] = region_dist
        
        # Top 10 régions
        for region, count in region_dist.head(10).items():
            pct = (count / len(df)) * 100
            barre = '█' * int(pct // 2)
            logger.info(f"   {region:20s} : {count:4d} ({pct:5.1f}%) {barre}")
        
        logger.info(f"\n   👉 Total régions : {len(region_dist)}")
        logger.info(f"   👉 Région dominante : {region_dist.index[0]} ({region_dist.values[0]} répondants)")
    
    return stats


def analyser_type_client(df):
    """Analyse type de client"""
    logger.info("\n" + "="*50)
    logger.info("💳 TYPE DE CLIENT")
    logger.info("="*50)
    
    col_type = "Quel est votre type de client SENELEC ?"
    
    if col_type in df.columns:
        type_dist = df[col_type].value_counts()
        
        logger.info("")
        for type_client, count in type_dist.items():
            pct = (count / len(df)) * 100
            barre = '█' * int(pct // 2)
            logger.info(f"   {type_client:30s} : {count:4d} ({pct:5.1f}%) {barre}")
        
        # Statistiques Woyofal
        woyofal_users = df[col_type].str.contains('Woyofal', na=False).sum()
        logger.info(f"\n   👉 Utilisateurs Woyofal : {woyofal_users} ({woyofal_users/len(df)*100:.1f}%)")
        
        return type_dist
    
    return None


def analyser_satisfaction_globale(df):
    """Analyse satisfaction globale"""
    logger.info("\n" + "="*50)
    logger.info("😊 SATISFACTION GLOBALE")
    logger.info("="*50)
    
    col_satisfaction = "De manière générale, êtes-vous satisfait(e) des services de la SENELEC ?"
    
    if col_satisfaction in df.columns:
        satisfaction_dist = df[col_satisfaction].value_counts()
        
        # Ordre logique
        ordre = ['Très satisfait(e)', 'Satisfait(e)', 'Neutre', 'Insatisfait(e)', 'Très insatisfait(e)']
        satisfaction_dist = satisfaction_dist.reindex(ordre, fill_value=0)
        
        logger.info("")
        for niveau, count in satisfaction_dist.items():
            pct = (count / len(df)) * 100
            emoji = {
                'Très satisfait(e)': '😊',
                'Satisfait(e)': '🙂',
                'Neutre': '😐',
                'Insatisfait(e)': '😕',
                'Très insatisfait(e)': '😡'
            }.get(niveau, '')
            barre = '█' * int(pct // 2)
            logger.info(f"   {emoji} {niveau:25s} : {count:4d} ({pct:5.1f}%) {barre}")
        
        # Calculs clés
        satisfaits = satisfaction_dist.get('Très satisfait(e)', 0) + satisfaction_dist.get('Satisfait(e)', 0)
        insatisfaits = satisfaction_dist.get('Très insatisfait(e)', 0) + satisfaction_dist.get('Insatisfait(e)', 0)
        
        logger.info(f"\n   ✅ Satisfaits : {satisfaits} ({satisfaits/len(df)*100:.1f}%)")
        logger.info(f"   ❌ Insatisfaits : {insatisfaits} ({insatisfaits/len(df)*100:.1f}%)")
        logger.info(f"   😐 Neutres : {satisfaction_dist.get('Neutre', 0)} ({satisfaction_dist.get('Neutre', 0)/len(df)*100:.1f}%)")
        
        return satisfaction_dist
    
    return None


def analyser_satisfaction_woyofal(df):
    """Analyse satisfaction spécifique Woyofal"""
    logger.info("\n" + "="*50)
    logger.info("💳 SATISFACTION WOYOFAL")
    logger.info("="*50)
    
    col_satisfaction = "Globalement, comment évaluez-vous votre satisfaction vis-à-vis du système Woyofal ?"
    col_type = "Quel est votre type de client SENELEC ?"
    
    if col_satisfaction in df.columns and col_type in df.columns:
        # Filtrer utilisateurs Woyofal
        df_woyofal = df[df[col_type].str.contains('Woyofal', na=False)]
        
        logger.info(f"\n   👥 Utilisateurs Woyofal : {len(df_woyofal)}")
        
        if len(df_woyofal) > 0:
            satisfaction_dist = df_woyofal[col_satisfaction].value_counts()
            
            # Ordre logique
            ordre = ['Très satisfait(e)', 'Satisfait(e)', 'Neutre', 'Insatisfait(e)', 'Très insatisfait(e)']
            satisfaction_dist = satisfaction_dist.reindex(ordre, fill_value=0)
            
            logger.info("")
            for niveau, count in satisfaction_dist.items():
                pct = (count / len(df_woyofal)) * 100
                barre = '█' * int(pct // 2)
                logger.info(f"   {niveau:25s} : {count:4d} ({pct:5.1f}%) {barre}")
            
            # Calculs clés
            satisfaits = satisfaction_dist.get('Très satisfait(e)', 0) + satisfaction_dist.get('Satisfait(e)', 0)
            insatisfaits = satisfaction_dist.get('Très insatisfait(e)', 0) + satisfaction_dist.get('Insatisfait(e)', 0)
            
            logger.info(f"\n   ✅ Satisfaits Woyofal : {satisfaits} ({satisfaits/len(df_woyofal)*100:.1f}%)")
            logger.info(f"   ❌ Insatisfaits Woyofal : {insatisfaits} ({insatisfaits/len(df_woyofal)*100:.1f}%)")
            
            return satisfaction_dist
    
    return None


def analyser_problemes_rencontres(df):
    """Analyse les problèmes rencontrés par les usagers"""
    logger.info("\n" + "="*50)
    logger.info("⚠️ PROBLÈMES RENCONTRÉS")
    logger.info("="*50)
    
    col_problemes = "Quels sont les principaux problèmes que vous rencontrez ?\n(Plusieurs réponses possibles)"
    
    if col_problemes in df.columns:
        problemes = df[col_problemes].dropna()
        
        logger.info(f"\n   📝 Répondants ayant signalé des problèmes : {len(problemes)} / {len(df)} ({len(problemes)/len(df)*100:.1f}%)")
        
        # Analyse des catégories de problèmes
        problemes_text = ' '.join(problemes.astype(str).str.lower())
        
        categories = {
            'Coupures': ['coupure', 'délestage', 'panne', 'électricité', 'courant'],
            'Woyofal': ['woyofal', 'prépayé', 'code', 'compteur', 'recharge'],
            'Service Client': ['service', 'accueil', 'agence', 'réclamation', 'client'],
            'Facturation': ['facture', 'cher', 'coût', 'prix', 'tarif']
        }
        
        logger.info("\n   📊 CATÉGORIES DE PROBLÈMES :")
        
        problemes_data = []
        for categorie, keywords in categories.items():
            count = sum(problemes_text.count(keyword) for keyword in keywords)
            pct = (count / len(problemes)) * 100 if len(problemes) > 0 else 0
            barre = '█' * int(pct // 2)
            logger.info(f"   {categorie:15s} : {count:4d} mentions ({pct:5.1f}%) {barre}")
            problemes_data.append({'catégorie': categorie, 'mentions': count})
        
        return pd.DataFrame(problemes_data)
    
    return None


def generer_graphique_age(age_dist, total):
    """Génère graphique répartition par âge"""
    logger.info("\n📊 Génération graphique répartition par âge...")
    
    if age_dist is None or age_dist.empty:
        logger.warning("⚠️ Pas de données âge")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Préparation données
    ages = age_dist.index.tolist()
    counts = age_dist.values
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    
    # Barres
    bars = ax.bar(ages, counts, color=colors[:len(ages)], edgecolor='black', linewidth=1, alpha=0.8)
    
    # Personnalisation
    ax.set_xlabel('Tranche d\'âge', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('Nombre de répondants', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_title('Répartition par âge des répondants', fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(counts) * 1.15)
    
    # Annotations
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        pct = (count / total) * 100
        
        # Valeur
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    jeunes = counts[0] + counts[1]
    ax.axhline(y=jeunes, color='#27ae60', linestyle='--', linewidth=2, alpha=0.5)
    ax.text(len(ages)-0.5, jeunes + 5, f'Moins de 35 ans: {jeunes} ({jeunes/total*100:.1f}%)', 
            fontsize=10, color='#27ae60', fontweight='bold')
    
    plt.xticks(rotation=0, fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    
    # Sauvegarde
    filepath = FIGURES_DIR / "repartition_age_enquete.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"   ✅ Sauvegardé : {filepath}")


def generer_graphique_genre(sexe_dist, total):
    """Génère graphique répartition par genre"""
    logger.info("\n📊 Génération graphique répartition par genre...")
    
    if sexe_dist is None or sexe_dist.empty:
        logger.warning("⚠️ Pas de données genre")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # === PIE CHART ===
    colors = ['#3498db', '#e74c3c', '#95a5a6']
    labels = sexe_dist.index.tolist()
    
    pie_data = sexe_dist.copy()
    pie_labels = labels.copy()
    
    if 'Préfère ne pas répondre' in pie_data.index:
        pie_data = pie_data.drop('Préfère ne pas répondre')
        pie_labels = [l for l in labels if l != 'Préfère ne pas répondre']
    
    wedges, texts, autotexts = ax1.pie(
        pie_data.values,
        labels=pie_labels,
        autopct='%1.1f%%',
        colors=colors[:len(pie_data)],
        startangle=90,
        textprops={'fontsize': 11},
        explode=[0.05, 0.02]
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    ax1.set_title('Répartition par Genre', fontsize=14, fontweight='bold', pad=20)
    
    # === BAR CHART ===
    bars = ax2.bar(sexe_dist.index, sexe_dist.values, 
                   color=colors[:len(sexe_dist)], edgecolor='black', linewidth=1.5, alpha=0.8)
    
    ax2.set_ylabel('Nombre de répondants', fontsize=12, fontweight='bold')
    ax2.set_title('Distribution par Genre', fontsize=14, fontweight='bold', pad=20)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Annotations
    for bar, val in zip(bars, sexe_dist.values):
        height = bar.get_height()
        pct = (val / total) * 100
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xticks(rotation=0, ha='center')
    plt.tight_layout()
    
    # Sauvegarde
    filepath = FIGURES_DIR / "repartition_genre_enquete.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"   ✅ Sauvegardé : {filepath}")


def generer_graphique_satisfaction(satisfaction_dist, total):
    """Génère graphique satisfaction globale"""
    logger.info("\n📊 Génération graphique satisfaction globale...")
    
    if satisfaction_dist is None or satisfaction_dist.empty:
        logger.warning("⚠️ Pas de données satisfaction")
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Couleurs par niveau
    colors = []
    for niveau in satisfaction_dist.index:
        if 'Très satisfait' in niveau:
            colors.append('#2ecc71')  
        elif 'Satisfait' in niveau:
            colors.append('#27ae60')   
        elif 'Neutre' in niveau:
            colors.append('#95a5a6')   
        elif 'Très insatisfait' in niveau:
            colors.append('#e74c3c')   
        else:
            colors.append('#e67e22')   
    
    # Barres
    bars = ax.bar(range(len(satisfaction_dist)), satisfaction_dist.values, 
                  color=colors, edgecolor='black', linewidth=1.5, alpha=0.9)
    
    # Personnalisation
    ax.set_xticks(range(len(satisfaction_dist)))
    ax.set_xticklabels(satisfaction_dist.index, rotation=30, ha='right', fontsize=11)
    ax.set_ylabel('Nombre de répondants', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_title('Satisfaction Globale des Services SENELEC', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(satisfaction_dist.values) * 1.15)
    
    # Annotations
    for bar, val in zip(bars, satisfaction_dist.values):
        height = bar.get_height()
        pct = (val / total) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    satisfaits = satisfaction_dist.get('Satisfait(e)', 0) + satisfaction_dist.get('Très satisfait(e)', 0)
    insatisfaits = satisfaction_dist.get('Insatisfait(e)', 0) + satisfaction_dist.get('Très insatisfait(e)', 0)
    
    ax.axhline(y=satisfaits, color='#27ae60', linestyle='--', linewidth=2, alpha=0.5, 
               label=f'Satisfaits: {satisfaits} ({satisfaits/total*100:.1f}%)')
    ax.axhline(y=insatisfaits, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.5,
               label=f'Insatisfaits: {insatisfaits} ({insatisfaits/total*100:.1f}%)')
    
    ax.legend(loc='upper right', fontsize=11)
    
    plt.tight_layout()
    
    # Sauvegarde
    filepath = FIGURES_DIR / "satisfaction_globale_enquete.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"   ✅ Sauvegardé : {filepath}")


def generer_graphique_type_client(type_dist, total):
    """Génère graphique type de client"""
    logger.info("\n📊 Génération graphique type client...")
    
    if type_dist is None or type_dist.empty:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # === PIE CHART ===
    colors = ['#3498db', '#e74c3c', '#f39c12']
    wedges, texts, autotexts = ax1.pie(
        type_dist.values,
        labels=type_dist.index,
        autopct='%1.1f%%',
        colors=colors[:len(type_dist)],
        startangle=90,
        textprops={'fontsize': 11},
        explode=[0.05, 0.02, 0.02]
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    ax1.set_title('Répartition par Type de Client', fontsize=14, fontweight='bold', pad=20)
    
    # === BAR CHART ===
    bars = ax2.bar(type_dist.index, type_dist.values, 
                   color=colors[:len(type_dist)], edgecolor='black', linewidth=1.5, alpha=0.8)
    
    ax2.set_ylabel('Nombre de répondants', fontsize=12, fontweight='bold')
    ax2.set_title('Distribution par Type de Client', fontsize=14, fontweight='bold', pad=20)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Annotations
    for bar, val in zip(bars, type_dist.values):
        height = bar.get_height()
        pct = (val / total) * 100
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    
    # Sauvegarde
    filepath = FIGURES_DIR / "repartition_type_client.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"   ✅ Sauvegardé : {filepath}")


def generer_rapport_enquete(df):
    """Génère rapport statistique complet"""
    logger.info("\n" + "="*50)
    logger.info("📋 GÉNÉRATION RAPPORT ENQUÊTE")
    logger.info("="*50)
    
    rapport = []
    
    # Nombre total
    rapport.append({'indicateur': 'Total répondants', 'valeur': len(df), 'details': ''})
    
    # Âge
    col_age = "Votre tranche d'âge ?"
    if col_age in df.columns:
        age_dist = df[col_age].value_counts()
        rapport.append({'indicateur': 'Tranche d\'âge dominante', 
                       'valeur': age_dist.index[0], 
                       'details': f'{age_dist.values[0]} répondants'})
    
    # Genre
    col_sexe = "Votre Sexe ?"
    if col_sexe in df.columns:
        sexe_dist = df[col_sexe].value_counts()
        rapport.append({'indicateur': 'Genre dominant', 
                       'valeur': sexe_dist.index[0], 
                       'details': f'{sexe_dist.values[0]} répondants ({sexe_dist.values[0]/len(df)*100:.1f}%)'})
    
    # Type client
    col_type = "Quel est votre type de client SENELEC ?"
    if col_type in df.columns:
        type_dist = df[col_type].value_counts()
        type_dominant = type_dist.index[0]
        type_count = type_dist.values[0]
        rapport.append({'indicateur': 'Type client dominant', 
                       'valeur': type_dominant, 
                       'details': f'{type_count} répondants ({type_count/len(df)*100:.1f}%)'})
        
        # Utilisateurs Woyofal
        woyofal_users = df[col_type].str.contains('Woyofal', na=False).sum()
        rapport.append({'indicateur': 'Utilisateurs Woyofal', 
                       'valeur': f"{woyofal_users/len(df)*100:.1f}%", 
                       'details': f"{woyofal_users} répondants"})
    
    # Région
    col_region = "Dans quelle région résidez-vous ?  "
    if col_region in df.columns:
        region_dist = df[col_region].value_counts()
        rapport.append({'indicateur': 'Région dominante', 
                       'valeur': region_dist.index[0], 
                       'details': f'{region_dist.values[0]} répondants ({region_dist.values[0]/len(df)*100:.1f}%)'})
    
    # Satisfaction globale
    col_satisfaction = "De manière générale, êtes-vous satisfait(e) des services de la SENELEC ?"
    if col_satisfaction in df.columns:
        satisfaits = df[col_satisfaction].isin(['Satisfait(e)', 'Très satisfait(e)']).sum()
        insatisfaits = df[col_satisfaction].isin(['Insatisfait(e)', 'Très insatisfait(e)']).sum()
        
        rapport.append({'indicateur': 'Taux de satisfaction globale', 
                       'valeur': f"{satisfaits/len(df)*100:.1f}%", 
                       'details': f"{satisfaits} satisfaits sur {len(df)}"})
        
        rapport.append({'indicateur': "Taux d'insatisfaction globale", 
                       'valeur': f"{insatisfaits/len(df)*100:.1f}%", 
                       'details': f"{insatisfaits} insatisfaits sur {len(df)}"})
    
    # Satisfaction Woyofal
    col_woy_satisfaction = "Globalement, comment évaluez-vous votre satisfaction vis-à-vis du système Woyofal ?"
    if col_woy_satisfaction in df.columns and col_type in df.columns:
        df_woyofal = df[df[col_type].str.contains('Woyofal', na=False)]
        if len(df_woyofal) > 0:
            woy_satisfaits = df_woyofal[col_woy_satisfaction].isin(['Satisfait(e)', 'Très satisfait(e)']).sum()
            woy_insatisfaits = df_woyofal[col_woy_satisfaction].isin(['Insatisfait(e)', 'Très insatisfait(e)']).sum()
            
            rapport.append({'indicateur': 'Satisfaction Woyofal', 
                           'valeur': f"{woy_satisfaits/len(df_woyofal)*100:.1f}%", 
                           'details': f"{woy_satisfaits} satisfaits"})
            
            rapport.append({'indicateur': "Insatisfaction Woyofal", 
                           'valeur': f"{woy_insatisfaits/len(df_woyofal)*100:.1f}%", 
                           'details': f"{woy_insatisfaits} insatisfaits"})
    
    # DataFrame rapport
    df_rapport = pd.DataFrame(rapport)
    
    # Sauvegarde
    filepath = STATISTICS_DIR / "rapport_enquete_terrain.csv"
    df_rapport.to_csv(filepath, index=False, encoding='utf-8')
    
    logger.info(f"\n   ✅ Rapport sauvegardé : {filepath}")
    logger.info("\n" + "="*50)
    logger.info("📊 RÉCAPITULATIF DES INDICATEURS")
    logger.info("="*50)
    logger.info("\n" + df_rapport.to_string(index=False))
    
    return df_rapport


def main():
    """Point d'entrée principal"""
    print("\n" + "="*70)
    print("📋 ANALYSE ENQUÊTE TERRAIN - SENELEC")
    print("="*70 + "\n")
    
    try:
        # Charger enquête
        df = charger_enquete()
        total = len(df)
        
        # Analyses
        stats_profil = analyser_profil_repondants(df)
        type_dist = analyser_type_client(df)
        satisfaction_dist = analyser_satisfaction_globale(df)
        analyser_satisfaction_woyofal(df)
        problemes_df = analyser_problemes_rencontres(df)
        
        # Générer tous les graphiques
        if 'age' in stats_profil:
            generer_graphique_age(stats_profil['age'], total)
        
        if 'sexe' in stats_profil:
            generer_graphique_genre(stats_profil['sexe'], total)
        
        generer_graphique_satisfaction(satisfaction_dist, total)
        generer_graphique_type_client(type_dist, total)
        
        # Rapport final
        generer_rapport_enquete(df)
        
        # Résumé final
        print("\n" + "="*70)
        print("✅ ANALYSE ENQUÊTE TERMINÉE AVEC SUCCÈS")
        print("="*70)
        print(f"\n📊 GRAPHIQUES GÉNÉRÉS :")
        print(f"   - répartition_age_enquete.png")
        print(f"   - répartition_genre_enquete.png")
        print(f"   - satisfaction_globale_enquete.png")
        print(f"   - repartition_type_client.png")
        print(f"\n📋 RAPPORT : {STATISTICS_DIR / 'rapport_enquete_terrain.csv'}")
        print("\n" + "="*70)
        
    except Exception as e:
        logger.error(f"❌ Erreur : {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()