"""
Page Analyse Géographique - Dashboard SENELEC
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.config import RAW_DATA_DIR, FIGURES_DIR


@st.cache_data
def load_enquete_data():
    """Charge les données d'enquête avec info géographique"""
    try:
        filepath = RAW_DATA_DIR / "Enquête_SENELEC.csv"
        if filepath.exists():
            return pd.read_csv(filepath, encoding='utf-8')
        else:
            return generer_donnees_demo()
    except Exception as e:
        st.error(f"Erreur chargement enquête : {e}")
        return generer_donnees_demo()


def generer_donnees_demo():
    """Génère des données de démonstration pour l'analyse géographique"""
    np.random.seed(42)
    
    regions = [
        'Dakar', 'Thiès', 'Saint-Louis', 'Ziguinchor', 'Kaolack', 
        'Fatick', 'Kaffrine', 'Diourbel', 'Louga', 'Matam',
        'Tambacounda', 'Kédougou', 'Kolda', 'Sédhiou'
    ]
    
    data = []
    
    for region in regions:
        # Nombre variable de répondants par région
        if region == 'Dakar':
            n = 150
        elif region in ['Thiès', 'Saint-Louis']:
            n = 60
        elif region in ['Ziguinchor', 'Kaolack', 'Diourbel']:
            n = 40
        else:
            n = np.random.randint(15, 30)
        
        for _ in range(n):
            # Distribution de satisfaction variable selon région
            if region in ['Ziguinchor', 'Fatick', 'Saint-Louis']:
                # Régions plus insatisfaites
                probs = [0.02, 0.15, 0.15, 0.45, 0.23]
            elif region in ['Dakar', 'Thiès']:
                # Régions plus mitigées
                probs = [0.05, 0.30, 0.25, 0.25, 0.15]
            else:
                # Régions plus satisfaites
                probs = [0.10, 0.35, 0.25, 0.20, 0.10]
            
            satisfaction = np.random.choice(
                ['Très satisfait(e)', 'Satisfait(e)', 'Neutre', 'Insatisfait(e)', 'Très insatisfait(e)'],
                p=probs
            )
            data.append({
                'region': region,
                'satisfaction': satisfaction
            })
    
    return pd.DataFrame(data)


def calculer_indicateurs_avances(df_region, col_region, col_satisfaction):
    """
    Calcule des indicateurs avancés par région
    """
    indicateurs = {}
    
    for region in df_region[col_region].unique():
        df_r = df_region[df_region[col_region] == region]
        total = len(df_r)
        
        # Métriques de satisfaction
        satisfaits = df_r[df_r[col_satisfaction].isin(['Satisfait(e)', 'Très satisfait(e)'])].shape[0]
        insatisfaits = df_r[df_r[col_satisfaction].isin(['Insatisfait(e)', 'Très insatisfait(e)'])].shape[0]
        neutres = df_r[df_r[col_satisfaction] == 'Neutre'].shape[0]
        
        # Taux
        taux_satisfaction = (satisfaits / total * 100) if total > 0 else 0
        taux_insatisfaction = (insatisfaits / total * 100) if total > 0 else 0
        taux_neutre = (neutres / total * 100) if total > 0 else 0
        
        # Indice composite (de -100 à +100)
        indice_satisfaction = taux_satisfaction - taux_insatisfaction
        
        # Score de criticité 
        criticite = taux_insatisfaction * (total / 100)
        
        indicateurs[region] = {
            'region': region,
            'total': total,
            'satisfaits': satisfaits,
            'insatisfaits': insatisfaits,
            'neutres': neutres,
            'taux_satisfaction': round(taux_satisfaction, 1),
            'taux_insatisfaction': round(taux_insatisfaction, 1),
            'taux_neutre': round(taux_neutre, 1),
            'indice_satisfaction': round(indice_satisfaction, 1),
            'criticite': round(criticite, 1)
        }
    
    return indicateurs


def show_geographie(df):
    """
    Affiche l'analyse géographique ultra-complète
    """
    
    # HEADER
    # st.header("🗺️ Analyse Géographique Approfondie")
    st.markdown("""
    Analyse territoriale complète de la perception des services SENELEC.
    Cette analyse permet d'identifier les disparités régionales et de prioriser les interventions.
    """)
    
    st.markdown("---")
    
    # ================================================
    # CHARGEMENT DES DONNÉES
    # ================================================
    
    df_enquete = load_enquete_data()
    
    if df_enquete is None or len(df_enquete) == 0:
        st.error("❌ Impossible de charger les données géographiques")
        return
    
    # Identifier les colonnes
    col_region = None
    for col in df_enquete.columns:
        if 'région' in col.lower() or 'region' in col.lower():
            col_region = col
            break
    
    col_satisfaction = None
    for col in df_enquete.columns:
        if 'satisfait' in col.lower() or 'satisfaction' in col.lower():
            col_satisfaction = col
            break
    
    if col_region is None:
        if 'region' in df_enquete.columns:
            col_region = 'region'
        elif 'Région' in df_enquete.columns:
            col_region = 'Région'
        else:
            st.error("❌ Colonne région non trouvée")
            st.write("Colonnes disponibles:", list(df_enquete.columns))
            return
    
    st.sidebar.success(f"✅ Données géographiques chargées: {len(df_enquete)} répondants")
    st.sidebar.info(f"📊 {df_enquete[col_region].nunique()} régions identifiées")
    
    # ================================================
    # VUE D'ENSEMBLE GÉOGRAPHIQUE
    # ================================================
    
    st.markdown("## 📊 Vue d'Ensemble Territoriale")
    
    # Distribution par région
    region_counts = df_enquete[col_region].value_counts().reset_index()
    region_counts.columns = ['region', 'repondants']
    region_counts['pourcentage'] = (region_counts['repondants'] / len(df_enquete) * 100).round(1)
    region_counts = region_counts.sort_values('repondants', ascending=False)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🗺️ Total Régions", df_enquete[col_region].nunique())
    
    with col2:
        st.metric("👥 Total Répondants", f"{len(df_enquete):,}")
    
    with col3:
        region_max = region_counts.iloc[0]
        st.metric("📈 Région principale", region_max['region'], f"{region_max['repondants']} rép.")
    
    with col4:
        if len(region_counts) > 1:
            cv = region_counts['repondants'].std() / region_counts['repondants'].mean()
            st.metric("📊 Disparité (CV)", f"{cv:.2f}")
        else:
            st.metric("📊 Disparité (CV)", "N/A")
    
    st.markdown("---")
    
    # ================================================
    # ANALYSE 1: RÉPARTITION DES RÉPONDANTS
    # ================================================
    
    st.markdown("## 📍 Répartition des Répondants")
    
    tab_rep1, tab_rep2, tab_rep3 = st.tabs([
        "📊 Graphique", 
        "📋 Tableau détaillé", 
        "📈 Statistiques"
    ])
    
    with tab_rep1:
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            fig_regions = go.Figure(go.Bar(
                x=region_counts.head(15)['repondants'],
                y=region_counts.head(15)['region'],
                orientation='h',
                marker=dict(
                    color=region_counts.head(15)['repondants'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Nombre")
                ),
                text=region_counts.head(15)['repondants'],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Répondants: %{x}<br>%{customdata}%<extra></extra>',
                customdata=region_counts.head(15)['pourcentage']
            ))
            
            fig_regions.update_layout(
                title="Distribution des Répondants par Région",
                xaxis_title="Nombre de Répondants",
                yaxis_title="",
                height=600,
                yaxis={'categoryorder':'total ascending'}
            )
            
            st.plotly_chart(fig_regions, use_container_width=True)
        
        with col_g2:
            # Pie chart
            fig_pie = px.pie(
                region_counts.head(8),
                values='repondants',
                names='region',
                title="Top 8 Régions (% du total)",
                hole=0.3
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(height=600)
            
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with tab_rep2:
        st.dataframe(
            region_counts.style.format({
                'repondants': '{:,.0f}',
                'pourcentage': '{:.1f}%'
            }).background_gradient(subset=['repondants'], cmap='Blues'),
            use_container_width=True,
            hide_index=True
        )
    
    with tab_rep3:
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        
        with col_stat1:
            st.metric("📊 Moyenne", f"{region_counts['repondants'].mean():.0f}")
            st.metric("📊 Médiane", f"{region_counts['repondants'].median():.0f}")
        
        with col_stat2:
            st.metric("📊 Écart-type", f"{region_counts['repondants'].std():.0f}")
            st.metric("📊 Minimum", f"{region_counts['repondants'].min()}")
        
        with col_stat3:
            st.metric("📊 Maximum", f"{region_counts['repondants'].max()}")
            st.metric("📊 Total", f"{region_counts['repondants'].sum()}")
    
    st.markdown("---")
    
    # ================================================
    # ANALYSE 2: SATISFACTION PAR RÉGION
    # ================================================
    
    if col_satisfaction is not None:
        st.markdown("## 😊 Satisfaction par Région")
        
        # Calculer indicateurs avancés
        indicateurs_dict = calculer_indicateurs_avances(df_enquete, col_region, col_satisfaction)
        df_indicateurs = pd.DataFrame(indicateurs_dict.values())
        
        tab_sat1, tab_sat2, tab_sat3 = st.tabs([
            "📊 Heatmap", 
            "🎯 Classement", 
            "📈 Détail par région"
        ])
        
        with tab_sat1:
            # Heatmap des niveaux de satisfaction
            crosstab = pd.crosstab(
                df_enquete[col_region],
                df_enquete[col_satisfaction],
                normalize='index'
            ) * 100
            
            ordre_colonnes = ['Très satisfait(e)', 'Satisfait(e)', 'Neutre', 'Insatisfait(e)', 'Très insatisfait(e)']
            colonnes_presentes = [c for c in ordre_colonnes if c in crosstab.columns]
            crosstab = crosstab[colonnes_presentes]
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=crosstab.values,
                x=crosstab.columns,
                y=crosstab.index,
                colorscale='RdYlGn',
                text=crosstab.values.round(1),
                texttemplate='%{text}%',
                textfont={"size": 10, "color": "black"},
                colorbar=dict(title="Pourcentage (%)")
            ))
            
            fig_heatmap.update_layout(
                title="Distribution de la Satisfaction par Région (%)",
                xaxis_title="Niveau de Satisfaction",
                yaxis_title="Région",
                height=600
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with tab_sat2:
            col_class1, col_class2 = st.columns(2)
            
            with col_class1:
                # Top régions satisfaites
                top_satisfaites = df_indicateurs.nlargest(5, 'taux_satisfaction')[['region', 'taux_satisfaction', 'total']]
                
                fig_top = px.bar(
                    top_satisfaites,
                    x='taux_satisfaction',
                    y='region',
                    orientation='h',
                    title="🏆 Top 5 Régions les Plus Satisfaites",
                    text='taux_satisfaction',
                    color='taux_satisfaction',
                    color_continuous_scale='Greens'
                )
                fig_top.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig_top.update_layout(height=400)
                
                st.plotly_chart(fig_top, use_container_width=True)
            
            with col_class2:
                # Top régions insatisfaites
                top_insatisfaites = df_indicateurs.nlargest(5, 'taux_insatisfaction')[['region', 'taux_insatisfaction', 'total']]
                
                fig_bottom = px.bar(
                    top_insatisfaites,
                    x='taux_insatisfaction',
                    y='region',
                    orientation='h',
                    title="⚠️ Top 5 Régions les Plus Insatisfaites",
                    text='taux_insatisfaction',
                    color='taux_insatisfaction',
                    color_continuous_scale='Reds'
                )
                fig_bottom.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig_bottom.update_layout(height=400)
                
                st.plotly_chart(fig_bottom, use_container_width=True)
        
        with tab_sat3:
            # Sélecteur de région
            selected_region = st.selectbox(
                "Choisir une région pour voir le détail:",
                df_indicateurs['region'].unique()
            )
            
            if selected_region:
                data_region = df_indicateurs[df_indicateurs['region'] == selected_region].iloc[0]
                
                col_d1, col_d2, col_d3 = st.columns(3)
                
                with col_d1:
                    st.metric("📊 Total répondants", f"{int(data_region['total']):,}")
                    st.metric("😊 Satisfaits", f"{int(data_region['satisfaits'])} ({data_region['taux_satisfaction']:.1f}%)")
                
                with col_d2:
                    st.metric("😡 Insatisfaits", f"{int(data_region['insatisfaits'])} ({data_region['taux_insatisfaction']:.1f}%)")
                    st.metric("😐 Neutres", f"{int(data_region['neutres'])} ({data_region['taux_neutre']:.1f}%)")
                
                with col_d3:
                    st.metric("📈 Indice satisfaction", f"{data_region['indice_satisfaction']:.1f}")
                    st.metric("🔥 Criticité", f"{data_region['criticite']:.1f}")
                
                # Graphique en camembert pour la région
                fig_detail = go.Figure(data=[go.Pie(
                    labels=['Satisfaits', 'Neutres', 'Insatisfaits'],
                    values=[data_region['satisfaits'], data_region['neutres'], data_region['insatisfaits']],
                    marker=dict(colors=['#27ae60', '#f39c12', '#e74c3c']),
                    hole=0.4,
                    textinfo='label+percent'
                )])
                
                fig_detail.update_layout(title=f"Distribution Satisfaction - {selected_region}", height=400)
                st.plotly_chart(fig_detail, use_container_width=True)
    
    st.markdown("---")
    
    # ================================================
    # ANALYSE 3: RÉGIONS CRITIQUES ET PRIORITAIRES
    # ================================================
    
    if col_satisfaction is not None and 'df_indicateurs' in locals():
        st.markdown("## 🚨 Analyse des Zones Critiques")
        
        # Calculer matrice de criticité
        df_indicateurs['priorite'] = pd.cut(
            df_indicateurs['taux_insatisfaction'],
            bins=[0, 30, 50, 100],
            labels=['🟢 Faible', '🟠 Modérée', '🔴 Élevée']
        )
        
        # Matrice volume vs insatisfaction
        fig_bubble = px.scatter(
            df_indicateurs,
            x='total',
            y='taux_insatisfaction',
            size='criticite',
            color='taux_insatisfaction',
            text='region',
            title="Matrice de Priorisation: Volume vs Insatisfaction",
            labels={
                'total': "Nombre de répondants",
                'taux_insatisfaction': "Taux d'insatisfaction (%)"
            },
            color_continuous_scale='Reds',
            size_max=60
        )
        
        fig_bubble.update_traces(textposition='top center')
        fig_bubble.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="Seuil critique")
        fig_bubble.add_vline(x=df_indicateurs['total'].median(), line_dash="dash", line_color="blue", annotation_text="Médiane volume")
        
        fig_bubble.update_layout(height=600)
        st.plotly_chart(fig_bubble, use_container_width=True)
        
        # Tableau des priorités
        st.markdown("### 📋 Matrice des Priorités d'Intervention")
        
        priorites = df_indicateurs.sort_values('taux_insatisfaction', ascending=False)
        
        def color_priorite(val):
            if val == '🔴 Élevée':
                return 'background-color: #ffcdd2'
            elif val == '🟠 Modérée':
                return 'background-color: #fff3cd'
            else:
                return 'background-color: #d4edda'
        
        st.dataframe(
            priorites[['region', 'total', 'taux_satisfaction', 'taux_insatisfaction', 'indice_satisfaction', 'priorite']]
            .style.format({
                'total': '{:,.0f}',
                'taux_satisfaction': '{:.1f}%',
                'taux_insatisfaction': '{:.1f}%',
                'indice_satisfaction': '{:.1f}'
                }).map(lambda v: color_priorite(v) if isinstance(v, str) else "", subset=["priorite"]),
            use_container_width=True,
            hide_index=True
        )
        
        # Alertes
        st.markdown("### 🚨 Alertes Automatiques")
        
        col_alert1, col_alert2, col_alert3 = st.columns(3)
        
        regions_critiques = priorites[priorites['taux_insatisfaction'] > 50]
        regions_attention = priorites[(priorites['taux_insatisfaction'] > 30) & (priorites['taux_insatisfaction'] <= 50)]
        
        with col_alert1:
            st.error(f"🔴 **{len(regions_critiques)} régions critiques**\n\nTaux > 50%")
            for _, row in regions_critiques.iterrows():
                st.write(f"- {row['region']}: {row['taux_insatisfaction']:.1f}%")
        
        with col_alert2:
            st.warning(f"🟠 **{len(regions_attention)} régions sous surveillance**\n\nTaux 30-50%")
            for _, row in regions_attention.iterrows():
                st.write(f"- {row['region']}: {row['taux_insatisfaction']:.1f}%")
        
        with col_alert3:
            regions_ok = priorites[priorites['taux_insatisfaction'] <= 30]
            st.success(f"🟢 **{len(regions_ok)} régions stables**\n\nTaux ≤ 30%")
            for _, row in regions_ok.head(3).iterrows():
                st.write(f"- {row['region']}: {row['taux_insatisfaction']:.1f}%")
    
    st.markdown("---")
    
    # ================================================
    # ANALYSE 4: COMPARAISONS STATISTIQUES
    # ================================================
    
    if col_satisfaction is not None and 'df_indicateurs' in locals():
        st.markdown("## 📊 Analyses Statistiques Avancées")
        
        # Préparer données pour ANOVA
        satisfaction_map = {
            'Très satisfait(e)': 5,
            'Satisfait(e)': 4,
            'Neutre': 3,
            'Insatisfait(e)': 2,
            'Très insatisfait(e)': 1
        }
        
        df_temp = df_enquete.copy()
        df_temp['score'] = df_temp[col_satisfaction].map(satisfaction_map)
        df_temp = df_temp.dropna(subset=['score'])
        
        # Top 5 régions pour comparaison
        top_regions = df_indicateurs.nlargest(5, 'total')['region'].tolist()
        df_anova = df_temp[df_temp[col_region].isin(top_regions)]
        
        if len(df_anova) > 0 and len(top_regions) > 1:
            # ANOVA
            groupes = [df_anova[df_anova[col_region] == r]['score'].dropna() for r in top_regions]
            groupes = [g for g in groupes if len(g) > 0]
            
            if len(groupes) >= 2:
                f_stat, p_value = stats.f_oneway(*groupes)
                
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                
                with col_stat1:
                    st.metric("📊 Test ANOVA", f"F = {f_stat:.3f}")
                
                with col_stat2:
                    st.metric("🎯 p-value", f"{p_value:.6f}")
                
                with col_stat3:
                    if p_value < 0.05:
                        st.error("🔴 Différences significatives entre régions")
                    else:
                        st.success("🟢 Pas de différence significative")
                
                fig_box = px.box(
                    df_anova,
                    x=col_region,
                    y='score',
                    title="Distribution des Scores de Satisfaction par Région",
                    labels={'score': 'Score satisfaction (1-5)', col_region: 'Région'},
                    color=col_region
                )
                fig_box.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig_box, use_container_width=True)
    
    st.markdown("---")
    
    # ================================================
    # ANALYSE 5: RECOMMANDATIONS PAR RÉGION
    # ================================================
    
    st.markdown("## 💡 Recommandations Territoriales")
    
    if col_satisfaction is not None and 'df_indicateurs' in locals():
        col_rec1, col_rec2 = st.columns(2)
        
        with col_rec1:
            st.markdown("### 🎯 Actions Prioritaires")
            
            top_3_critiques = priorites.head(3)
            st.error(f"**Intervention immédiate dans:**")
            for _, row in top_3_critiques.iterrows():
                st.markdown(f"""
                **📍 {row['region']}**  
                - Taux insatisfaction: {row['taux_insatisfaction']:.1f}%
                - Volume: {int(row['total'])} répondants
                - Action: Diagnostic terrain + Plan d'urgence
                """)
        
        with col_rec2:
            st.markdown("### 📋 Plans d'Action Spécifiques")
            
            # Recommandations par profil de région
            st.info("""
            **🔵 Régions à forte insatisfaction**
            - Délégation de compétences aux agences locales
            - Renforcement des équipes techniques
            - Comités d'usagers régionaux
            - Suivi hebdomadaire des indicateurs
            """)
            
            st.success("""
            **🟢 Régions performantes**
            - Capitaliser sur les bonnes pratiques
            - Programmes de formation pour les autres régions
            - Études de cas pour transfert de compétences
            """)
    
    st.markdown("---")
    
    # ================================================
    # EXPORT DES DONNÉES
    # ================================================
    
    st.markdown("## 💾 Export des Analyses")
    
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    
    with col_exp1:
        if col_satisfaction is not None:
            crosstab_export = pd.crosstab(df_enquete[col_region], df_enquete[col_satisfaction])
            csv_cross = crosstab_export.to_csv(encoding='utf-8-sig')
            st.download_button(
                label="📥 Matrice satisfaction",
                data=csv_cross,
                file_name=f"satisfaction_par_region_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col_exp2:
        if 'df_indicateurs' in locals():
            csv_indics = df_indicateurs.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 Indicateurs par région",
                data=csv_indics,
                file_name=f"indicateurs_regionaux_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col_exp3:
        csv_brut = df_enquete.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 Données brutes",
            data=csv_brut,
            file_name=f"donnees_geographiques_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )