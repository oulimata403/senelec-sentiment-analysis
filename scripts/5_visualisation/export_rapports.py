"""
Export des rapports et statistiques en formats multiples
HTML, Excel, Synthèse TXT
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.config import (
    EXPORTS_DIR, 
    FIGURES_DIR, 
    STATISTICS_DIR, 
    REPORTS_DIR,
    THEMES
)
from utils.logger import setup_logger

logger = setup_logger("export_rapports")


def creer_rapport_html_complet():
    """Crée un rapport HTML interactif complet"""
    logger.info("\n📄 CRÉATION RAPPORT HTML COMPLET")
    
    # Charger données pour KPIs
    df_corpus = pd.read_csv(EXPORTS_DIR / "corpus_avec_themes.csv", encoding='utf-8')
    
    # Calculer KPIs RÉELS
    total_publications = 2739  
    
    # Distribution sentiments 
    pct_negatif = 66.81  
    pct_neutre = 7.92
    pct_positif = 25.26
    
    # Thèmes (selon topic_modeling log)
    theme_dominant = "Système Woyofal"
    pct_woyofal = 57.9
    
    # KPIs HTML
    kpis_html = f"""
    <div class="kpi-card">
        <div class="kpi-label">Total Publications</div>
        <div class="kpi-value">{total_publications:,}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Sentiment Négatif</div>
        <div class="kpi-value sentiment-negative">{pct_negatif:.1f}%</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Sentiment Positif</div>
        <div class="kpi-value sentiment-positive">{pct_positif:.1f}%</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Thème Dominant</div>
        <div class="kpi-value" style="font-size: 1.5em;">{theme_dominant}</div>
    </div>
    """
    
    # Tables
    def df_to_html_table(filepath_str, max_rows=10):
        filepath = Path(filepath_str)
        if not filepath.exists():
            return "<p>Données non disponibles</p>"
        
        try:
            data = pd.read_csv(filepath, encoding='utf-8').head(max_rows)
            return data.to_html(index=False, classes='data-table', border=0)
        except:
            return "<p>Erreur de chargement</p>"
    
    # Stats files
    stats_files = {
        'themes': STATISTICS_DIR / "rapport_themes_detaille.csv",
        'evolution': STATISTICS_DIR / "evolution_temporelle.csv",
        'regions': STATISTICS_DIR / "rapport_geographique.csv",
        'enquete': STATISTICS_DIR / "rapport_enquete_terrain.csv",
        'woyofal': STATISTICS_DIR / "rapport_comparatif_woyofal.csv",
    }
    
    html_content = """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapport Analyse SENELEC - Ouly TOURÉ</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        header p {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        .meta {{
            background: #f8f9fa;
            padding: 20px 40px;
            border-bottom: 3px solid #667eea;
        }}
        
        .meta-item {{
            display: inline-block;
            margin-right: 30px;
            font-size: 0.95em;
        }}
        
        .meta-item strong {{
            color: #667eea;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .section {{
            margin-bottom: 40px;
            padding: 30px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 5px solid #667eea;
        }}
        
        .section h2 {{
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.8em;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        
        .section h3 {{
            color: #764ba2;
            margin: 20px 0 10px 0;
            font-size: 1.3em;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #e0e0e0;
        }}
        
        tr:hover {{
            background: #f5f5f5;
        }}
        
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .kpi-card {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s;
        }}
        
        .kpi-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        }}
        
        .kpi-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            margin: 10px 0;
        }}
        
        .kpi-label {{
            color: #666;
            font-size: 1em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .highlight-box {{
            background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
            border-left: 4px solid #667eea;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        
        .highlight-box strong {{
            color: #667eea;
        }}
        
        footer {{
            background: #2c3e50;
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        footer p {{
            margin: 5px 0;
        }}
        
        .sentiment-positive {{ color: #27ae60; font-weight: bold; }}
        .sentiment-negative {{ color: #e74c3c; font-weight: bold; }}
        .sentiment-neutral {{ color: #95a5a6; font-weight: bold; }}
        
        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            .container {{
                box-shadow: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Rapport d'Analyse de Sentiment</h1>
            <p>Perception des Services SENELEC à l'ère de l'IA</p>
        </header>
        
        <div class="meta">
            <div class="meta-item"><strong>Auteur :</strong> Ouly TOURÉ</div>
            <div class="meta-item"><strong>Université :</strong> UCAD</div>
            <div class="meta-item"><strong>Date :</strong> {date}</div>
            <div class="meta-item"><strong>Année :</strong> 2024-2025</div>
        </div>
        
        <div class="content">
            <!-- SECTION 1: RÉSUMÉ EXÉCUTIF -->
            <div class="section">
                <h2>📌 Résumé Exécutif</h2>
                
                <div class="kpi-grid">
                    {kpis}
                </div>
                
                <div class="highlight-box">
                    <strong>🎯 Conclusion Principale :</strong> {conclusion_principale}
                </div>
            </div>
            
            <!-- SECTION 2: ANALYSE THÉMATIQUE -->
            <div class="section">
                <h2>🔍 Analyse Thématique</h2>
                {table_themes}
            </div>
            
            <!-- SECTION 3: ANALYSE COMPARATIVE WOYOFAL -->
            <div class="section">
                <h2>⚖️ Analyse Comparative Woyofal</h2>
                {table_woyofal}
                
                <div class="highlight-box">
                    <strong>📊 Test Statistique :</strong> Chi² = 132.26, p-value < 0.001. La différence de sentiment entre Woyofal et les autres thèmes est <strong>statistiquement hautement significative</strong>.
                </div>
            </div>
            
            <!-- SECTION 4: ANALYSE GÉOGRAPHIQUE -->
            <div class="section">
                <h2>🗺️ Répartition Géographique</h2>
                {table_regions}
            </div>
            
            <!-- SECTION 5: ÉVOLUTION TEMPORELLE -->
            <div class="section">
                <h2>📈 Évolution Temporelle</h2>
                {evolution_summary}
            </div>
            
            <!-- SECTION 6: ENQUÊTE TERRAIN -->
            <div class="section">
                <h2>📋 Résultats Enquête Terrain</h2>
                {table_enquete}
            </div>
            
            <!-- SECTION 7: PERFORMANCE MODÈLE -->
            <div class="section">
                <h2>🤖 Performance du Modèle BERT</h2>
                {performance_modele}
            </div>
        </div>
        
        <footer>
            <p><strong>Mémoire de Master 2 - Business Intelligence</strong></p>
            <p>Université Cheikh Anta Diop de Dakar (UCAD)</p>
            <p>© 2025 Ouly TOURÉ - Tous droits réservés</p>
        </footer>
    </div>
</body>
</html>
"""
    
    # Performance modèle 
    performance_html = """
    <table>
        <thead>
            <tr>
                <th>Métrique</th>
                <th>Valeur</th>
            </tr>
        </thead>
        <tbody>
            <tr><td>Accuracy (Test Set)</td><td class="sentiment-positive">86.15%</td></tr>
            <tr><td>F1-Score</td><td>86.52%</td></tr>
            <tr><td>Precision</td><td>88.92%</td></tr>
            <tr><td>Recall</td><td>86.15%</td></tr>
            <tr><td>Loss (Test)</td><td>0.3849</td></tr>
            <tr><td>Modèle</td><td>bert-base-multilingual-cased</td></tr>
            <tr><td>Epochs entraînés</td><td>7 (Early Stopping)</td></tr>
        </tbody>
    </table>
    """
    
    # Générer HTML
    html_final = html_content.format(
        date=datetime.now().strftime("%d/%m/%Y"),
        kpis=kpis_html,
        conclusion_principale=f"Le sentiment négatif domine avec {pct_negatif:.1f}%, principalement lié au système Woyofal qui concentre {pct_woyofal:.1f}% du corpus et affiche 75.0% de sentiment négatif.",
        table_themes=df_to_html_table(stats_files['themes']),
        table_woyofal=df_to_html_table(stats_files['woyofal']),
        table_regions=df_to_html_table(stats_files['regions']),
        evolution_summary="<p>Analyse de <strong>106 semaines</strong> (2024-01-01 → 2026-01-17) montrant un sentiment négatif moyen de <strong>81.1%</strong> avec des pics identifiés les 25/11/2025 (41 posts négatifs) et 16/01/2026 (29 posts négatifs).</p>",
        table_enquete=df_to_html_table(stats_files['enquete']),
        performance_modele=performance_html
    )
    
    # Sauvegarder
    filepath = REPORTS_DIR / "rapport_analyse_senelec.html"
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_final)
    
    logger.info(f"   ✅ {filepath}")
    return filepath


def exporter_statistiques_excel():
    """Exporte toutes les stats dans un fichier Excel multi-onglets"""
    logger.info("\n📊 EXPORT EXCEL MULTI-ONGLETS")
    
    filepath = REPORTS_DIR / "statistiques_senelec_complete.xlsx"
    
    # Créer writer Excel
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        
        # Onglet 1: Résumé RÉEL
        df_resume = pd.DataFrame({
            'Indicateur': [
                'Total Publications Analysées',
                'Période Analysée',
                'Sentiment Négatif (%)',
                'Sentiment Positif (%)',
                'Sentiment Neutre (%)',
                'Thème Principal',
                '% Thème Principal',
                'Plateforme Dominante',
                'Nombre de Thèmes Identifiés',
                'Modèle Utilisé',
                'Accuracy Modèle',
                'Nombre Semaines Analysées',
                'Répondants Enquête Terrain'
            ],
            'Valeur': [
                '2,739',
                '2024-01-01 → 2026-01-17',
                '66.81%',
                '25.26%',
                '7.92%',
                'Système Woyofal',
                '57.9%',
                'Facebook (68.9%)',
                '4',
                'BERT Multilingue',
                '86.15%',
                '106',
                '490'
            ]
        })
        df_resume.to_excel(writer, sheet_name='Résumé', index=False)
        
        # Onglet 2: Thèmes
        if (STATISTICS_DIR / "rapport_themes_detaille.csv").exists():
            df_themes = pd.read_csv(STATISTICS_DIR / "rapport_themes_detaille.csv", encoding='utf-8')
            df_themes.to_excel(writer, sheet_name='Thèmes', index=False)
        
        # Onglet 3: Évolution Temporelle
        if (STATISTICS_DIR / "evolution_temporelle.csv").exists():
            df_evolution = pd.read_csv(STATISTICS_DIR / "evolution_temporelle.csv", encoding='utf-8')
            df_evolution.to_excel(writer, sheet_name='Évolution Temporelle', index=False)
        
        # Onglet 4: Géographie
        if (STATISTICS_DIR / "rapport_geographique.csv").exists():
            df_geo = pd.read_csv(STATISTICS_DIR / "rapport_geographique.csv", encoding='utf-8')
            df_geo.to_excel(writer, sheet_name='Géographie', index=False)
        
        # Onglet 5: Woyofal
        if (STATISTICS_DIR / "rapport_comparatif_woyofal.csv").exists():
            df_woy = pd.read_csv(STATISTICS_DIR / "rapport_comparatif_woyofal.csv", encoding='utf-8')
            df_woy.to_excel(writer, sheet_name='Woyofal', index=False)
        
        # Onglet 6: Enquête
        if (STATISTICS_DIR / "rapport_enquete_terrain.csv").exists():
            df_enq = pd.read_csv(STATISTICS_DIR / "rapport_enquete_terrain.csv", encoding='utf-8')
            df_enq.to_excel(writer, sheet_name='Enquête', index=False)
        
        # Onglet 7: Performance Modèle
        df_perf = pd.DataFrame({
            'Métrique': [
                'Accuracy',
                'Precision',
                'Recall',
                'F1-Score',
                'Loss (Test)',
                'AUC Moyen',
                'Total Erreurs Test',
                'Taux Erreur'
            ],
            'Valeur': [
                '86.15%',
                '88.92%',
                '86.15%',
                '86.52%',
                '0.3849',
                '93.6%',
                '9/65',
                '13.8%'
            ]
        })
        df_perf.to_excel(writer, sheet_name='Performance Modèle', index=False)
    
    logger.info(f"   ✅ {filepath}")
    return filepath


def creer_synthese_executive():
    """Crée une synthèse exécutive d'1 page avec VRAIES données"""
    logger.info("\n📄 SYNTHÈSE EXÉCUTIVE")
    
    synthese = f"""
╔══════════════════════════════════════════════════════════════════════╗
║                     SYNTHÈSE EXÉCUTIVE                                ║
║        Analyse Perception SENELEC - Ouly TOURÉ - UCAD 2025           ║
╚══════════════════════════════════════════════════════════════════════╝

📊 DONNÉES ANALYSÉES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Total publications : 2,739
- Période : 2024-01-01 → 2026-01-17 (106 semaines)
- Sources : Facebook (68.9%), Enquête (24.6%), Twitter (6.5%)
- Méthode : Modèle BERT multilingue (Accuracy: 86.15%)

🎯 RÉSULTATS CLÉS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Sentiment Négatif : 66.81% (1,830 publications)
✓ Sentiment Positif : 25.26% (692 publications)
✓ Sentiment Neutre  : 7.92% (217 publications)

🔍 ANALYSE THÉMATIQUE (4 thèmes identifiés)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Système Woyofal         : 57.9% (1,586 pub.) - 75.0% NÉGATIF ⚠️
2. Coupures d'Électricité  : 23.6% (647 pub.)   - 60.7% négatif
3. Service Client          : 13.4% (367 pub.)   - 46.9% négatif
4. Facturation/Tarifs      : 5.1% (139 pub.)    - 54.0% négatif

⚖️ CONSTAT MAJEUR : WOYOFAL (Test Chi² p < 0.001)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Le système Woyofal présente un TAUX DE REJET CRITIQUE :
→ 75.0% de sentiment négatif (vs 55.5% pour autres thèmes)
→ Différence statistiquement significative (Chi² = 132.26, p < 0.001)
→ Score de criticité : 240.1 (plus haut de tous les thèmes)

🗺️ ANALYSE GÉOGRAPHIQUE (490 répondants enquête)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Régions les plus critiques (taux insatisfaction > 50%) :
1. Ziguinchor    : 63.6% insatisfaits
2. Fatick        : 59.1% insatisfaits
3. Saint-Louis   : 57.6% insatisfaits
4. Kaolack       : 52.2% insatisfaits

📈 ÉVOLUTION TEMPORELLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Sentiment négatif moyen : 81.1% sur 106 semaines
- Volume moyen : 26 publications/semaine
- Pics de mécontentement identifiés :
  - 25 novembre 2025 : 41 publications négatives
  - 16 janvier 2026 : 29 publications négatives

🤖 PERFORMANCE MODÈLE BERT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Accuracy : 86.15% (Test Set : 65 exemples)
- F1-Score : 86.52%
- Precision : 88.92%
- Recall : 86.15%
- Modèle : bert-base-multilingual-cased
- Entraînement : 7 epochs (Early Stopping, 302 exemples train)

📋 ENQUÊTE TERRAIN (490 répondants)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Âge dominant : 18-25 ans (40.6%)
- Genre : Hommes (59.8%)
- Type client : Prépayé Woyofal (73.7%)
- Taux satisfaction global : 31.8%
- Taux insatisfaction global : 39.6%
- Satisfaction Woyofal : 31.9%
- Insatisfaction Woyofal : 45.4%

💡 RECOMMANDATIONS STRATÉGIQUES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. PRIORITÉ ABSOLUE : Refonte communication Woyofal
   - Campagne pédagogique sur les avantages du prépayé
   - Simplification processus de recharge
   - Programme de fidélisation spécifique

2. Renforcement Service Client (46.9% insatisfaction)
   - Formation accrue des agents
   - Amélioration temps de réponse

3. Ciblage Régional : Interventions Ziguinchor, Fatick, Saint-Louis
   - Actions ciblées dans les régions à >50% insatisfaction

4. Monitoring Temps Réel
   - Système d'alerte sur pics de mécontentement
   - Tableau de bord sentiment en temps réel

5. Gestion Coupures (60.7% insatisfaction)
   - Communication anticipée des coupures programmées
   - Réduction fréquence coupures

═══════════════════════════════════════════════════════════════════════
📊 MÉTHODOLOGIE COMPLÈTE
═══════════════════════════════════════════════════════════════════════
Phase 1 : Collecte - 3,223 publications brutes (Facebook, Twitter, Enquête)
Phase 2 : Prétraitement - Nettoyage, détection langue → 2,739 publications
Phase 3 : Labellisation - Semi-automatique + manuelle → 432 exemples
Phase 4 : Modélisation - BERT fine-tuning (70% train, 15% val, 15% test)
Phase 5 : Topic Modeling - LDA 8 topics → 4 thèmes métier
Phase 6 : Analyses - Temporelle, géographique, thématique, comparative

═══════════════════════════════════════════════════════════════════════
Date : {datetime.now().strftime("%d/%m/%Y %H:%M")}
Contact : ouly.toure@ucad.edu.sn
Encadrant : Pr Aliou BOLY
═══════════════════════════════════════════════════════════════════════
"""
    
    filepath = REPORTS_DIR / "synthese_executive.txt"
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(synthese)
    
    logger.info(f"   ✅ {filepath}")
    print(synthese)


def main():
    """Point d'entrée principal"""
    print("="*70)
    print("📤 EXPORT RAPPORTS - SENELEC")
    print("="*70)
    
    try:
        # Créer dossier rapports
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Générer rapports
        creer_synthese_executive()
        html_file = creer_rapport_html_complet()
        excel_file = exporter_statistiques_excel()
        
        print("\n" + "="*70)
        print("✅ RAPPORTS EXPORTÉS")
        print("="*70)
        print(f"📄 HTML  : {html_file}")
        print(f"📊 Excel : {excel_file}")
        print(f"📋 Synthèse : {REPORTS_DIR / 'synthese_executive.txt'}")
        print("="*70)
        
    except Exception as e:
        logger.error(f"❌ Erreur : {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()