"""
Page Comparaison Woyofal vs Autres Thèmes - Dashboard SENELEC
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
from scipy.stats import chi2_contingency, norm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def safe_datetime_convert(df, col='date_publication'):
    """Conversion sécurisée des dates"""
    if col in df.columns:
        try:
            # utc=True gère les timezones mixtes
            df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)
            df[col] = df[col].dt.tz_localize(None)
        except:
            df[col] = pd.NaT
    return df


def safe_division(numerateur, denominateur, default=0):
    """Division sécurisée pour éviter les erreurs de division par zéro"""
    if denominateur == 0 or denominateur is None or pd.isna(denominateur):
        return default
    return numerateur / denominateur


def calculer_intervalle_confiance(successes, trials, confidence=0.95):
    """
    Calcule l'intervalle de confiance pour une proportion
    (méthode Wilson score - sans statsmodels)
    """
    if trials == 0:
        return (0, 0)
    
    p = successes / trials
    z = norm.ppf(1 - (1 - confidence) / 2)
    
    denominator = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * trials)) / trials) / denominator
    
    return (center - margin, center + margin)


def calculer_metriques_avancees(df):
    """
    Calcule des métriques avancées pour l'analyse comparative
    """
    metriques = {
        'total': 0, 'neg': 0, 'pos': 0, 'neu': 0,
        'neg_pct': 0, 'pos_pct': 0, 'neu_pct': 0,
        'ratio_neg_pos': 0, 'indice_satisfaction': 0, 'criticite': 0,
        'engagement': 0, 'polarite': 0
    }
    
    if df is None or len(df) == 0:
        return metriques
    
    if 'sentiment_pred' in df.columns:
        df['sentiment_pred'] = df['sentiment_pred'].astype(str).str.lower()
        
        total = len(df)
        neg = (df['sentiment_pred'] == 'negative').sum()
        pos = (df['sentiment_pred'] == 'positive').sum()
        neu = (df['sentiment_pred'] == 'neutral').sum()
        
        metriques['total'] = total
        metriques['neg'] = neg
        metriques['pos'] = pos
        metriques['neu'] = neu
        
        metriques['neg_pct'] = safe_division(neg * 100, total)
        metriques['pos_pct'] = safe_division(pos * 100, total)
        metriques['neu_pct'] = safe_division(neu * 100, total)
        
        metriques['ratio_neg_pos'] = safe_division(neg, pos, default=neg if pos == 0 and neg > 0 else 0)
        metriques['indice_satisfaction'] = safe_division((pos - neg) * 100, total)
        metriques['criticite'] = total * metriques['neg_pct'] / 100
        metriques['engagement'] = total * (1 - metriques['neu_pct'] / 100)
        metriques['polarite'] = metriques['pos_pct'] - metriques['neg_pct']
    
    return metriques


def preparer_donnees_temporelles(df, date_col='date_publication'):
    """Prépare les données temporelles de façon sécurisée"""
    if df is None or len(df) == 0 or date_col not in df.columns:
        return None
    
    df_temp = df.copy()
    
    # Conversion sécurisée des dates avec utc=True
    df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce', utc=True)
    df_temp[date_col] = df_temp[date_col].dt.tz_localize(None)
    
    # Supprimer les dates invalides
    df_temp = df_temp.dropna(subset=[date_col])
    
    if len(df_temp) == 0:
        return None
    
    # Ajouter colonnes temporelles
    df_temp['annee_mois'] = df_temp[date_col].dt.to_period('M').astype(str)
    df_temp['semaine'] = df_temp[date_col].dt.isocalendar().week
    df_temp['mois'] = df_temp[date_col].dt.month
    df_temp['annee'] = df_temp[date_col].dt.year
    df_temp['trimestre'] = df_temp[date_col].dt.quarter
    df_temp['jour_semaine'] = df_temp[date_col].dt.dayofweek
    
    return df_temp


def show_comparaison(df):
    """
    Affiche la comparaison Woyofal vs Autres Thèmes - Version ultra-complète sans statsmodels
    """
    
    if df is None or len(df) == 0:
        st.error("❌ Données non disponibles")
        return
    
    if 'theme' not in df.columns:
        st.error("❌ Colonne 'theme' manquante")
        st.write("Colonnes disponibles:", list(df.columns))
        return
    
    # Normaliser les dates
    df = safe_datetime_convert(df.copy())
    
    # Filtrer les données
    df_woyofal = df[df['theme'] == 'woyofal'].copy()
    df_autres = df[df['theme'] != 'woyofal'].copy()
    
    total_woy = len(df_woyofal)
    total_autres = len(df_autres)
    total_global = len(df)
    
    # ================================================
    # HEADER
    # ================================================
    
    st.markdown("""
    Cette analyse compare en profondeur les perceptions du système **Woyofal (prépayé)** 
    avec l'ensemble des **autres services SENELEC** (coupures, facturation, service client, etc.).
    """)
    
    # Statistiques globales
    col_global1, col_global2, col_global3 = st.columns(3)
    with col_global1:
        st.metric("📊 Total Corpus", f"{total_global:,}")
    with col_global2:
        st.metric("💳 Woyofal", f"{total_woy:,}", f"{total_woy/total_global*100:.1f}%" if total_global > 0 else "0%")
    with col_global3:
        st.metric("🔌 Autres", f"{total_autres:,}", f"{total_autres/total_global*100:.1f}%" if total_global > 0 else "0%")
    
    st.markdown("---")
    
    # ================================================
    # ANALYSE 1: MÉTRIQUES COMPARATIVES 
    # ================================================
    
    # Calculer métriques 
    metriques_woy = calculer_metriques_avancees(df_woyofal)
    metriques_autres = calculer_metriques_avancees(df_autres)
    
    st.markdown("## 📊 Tableau de Bord Comparatif")
    
    # KPIs principaux
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    
    with col_kpi1:
        st.metric(
            "😡 Négativité Woyofal",
            f"{metriques_woy['neg_pct']:.1f}%",
            delta=f"{metriques_woy['neg_pct'] - metriques_autres['neg_pct']:+.1f} pts",
            delta_color="inverse"
        )
    
    with col_kpi2:
        st.metric(
            "😊 Positivité Woyofal",
            f"{metriques_woy['pos_pct']:.1f}%",
            delta=f"{metriques_woy['pos_pct'] - metriques_autres['pos_pct']:+.1f} pts"
        )
    
    with col_kpi3:
        st.metric(
            "📈 Indice Satisfaction",
            f"{metriques_woy['indice_satisfaction']:.1f}",
            delta=f"{metriques_woy['indice_satisfaction'] - metriques_autres['indice_satisfaction']:+.1f}"
        )
    
    with col_kpi4:
        st.metric(
            "🔥 Criticité",
            f"{metriques_woy['criticite']:.0f}",
            delta=f"{metriques_woy['criticite'] - metriques_autres['criticite']:+.0f}",
            delta_color="inverse"
        )
    
    st.markdown("---")
    
    # ================================================
    # ANALYSE 2: DIAGNOSTIC AVANCÉ
    # ================================================
    
    st.markdown("## 🚨 Diagnostic Automatique Approfondi")
    
    diff_neg = metriques_woy['neg_pct'] - metriques_autres['neg_pct']
    ratio_woy = metriques_woy['ratio_neg_pos']
    
    col_diag1, col_diag2, col_diag3 = st.columns(3)
    
    with col_diag1:
        if diff_neg > 15:
            st.error(f"🔴 **Écart CRITIQUE**\n\nWoyofal {diff_neg:.1f} pts plus négatif")
        elif diff_neg > 8:
            st.warning(f"🟠 **Écart SIGNIFICATIF**\n\nWoyofal {diff_neg:.1f} pts plus négatif")
        elif diff_neg > 3:
            st.info(f"🔵 **Écart MODÉRÉ**\n\nWoyofal {diff_neg:.1f} pts plus négatif")
        else:
            st.success(f"🟢 **Écart FAIBLE**\n\nDifférence de {diff_neg:.1f} pts")
    
    with col_diag2:
        if ratio_woy > 4:
            st.error(f"🔴 **Déséquilibre EXTRÊME**\n\n{ratio_woy:.1f} négatifs pour 1 positif")
        elif ratio_woy > 2.5:
            st.warning(f"🟠 **Déséquilibre FORT**\n\n{ratio_woy:.1f} négatifs pour 1 positif")
        elif ratio_woy > 1.5:
            st.info(f"🔵 **Déséquilibre MODÉRÉ**\n\n{ratio_woy:.1f} négatifs pour 1 positif")
        else:
            st.success(f"🟢 **Équilibre**\n\n{ratio_woy:.1f} négatifs pour 1 positif")
    
    with col_diag3:
        if total_woy > total_autres * 1.5:
            st.info(f"📊 **Surcharge Woyofal**\n\n{total_woy/total_autres:.1f}x plus de publications")
        elif total_woy < total_autres * 0.5:
            st.info(f"📊 **Sous-représentation**\n\nWoyofal seulement {total_woy/total_global*100:.1f}%")
        else:
            st.success(f"📊 **Équilibre volumétrique**\n\nRépartition équilibrée")
    
    st.markdown("---")
    
    # ================================================
    # ANALYSE 3: VISUALISATIONS 
    # ================================================
    
    st.markdown("## 📈 Analyses Visuelles Approfondies")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Distribution", 
        "📊 Comparaison", 
        "📈 Évolution", 
        "🔍 Analyse Détaillée"
    ])
    
    with tab1:
        col_v1, col_v2 = st.columns(2)
        
        with col_v1:
            fig_pies = make_subplots(
                rows=1, cols=2,
                specs=[[{"type": "domain"}, {"type": "domain"}]],
                subplot_titles=("💳 Woyofal", "🔌 Autres Services")
            )
            
            fig_pies.add_trace(go.Pie(
                labels=['😡 Négatif', '😐 Neutre', '😊 Positif'],
                values=[metriques_woy['neg'], metriques_woy['neu'], metriques_woy['pos']],
                marker=dict(colors=['#e74c3c', '#95a5a6', '#27ae60']),
                textinfo='label+percent',
                hole=0.4,
                name="Woyofal"
            ), row=1, col=1)
            
            fig_pies.add_trace(go.Pie(
                labels=['😡 Négatif', '😐 Neutre', '😊 Positif'],
                values=[metriques_autres['neg'], metriques_autres['neu'], metriques_autres['pos']],
                marker=dict(colors=['#e74c3c', '#95a5a6', '#27ae60']),
                textinfo='label+percent',
                hole=0.4,
                name="Autres"
            ), row=1, col=2)
            
            fig_pies.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig_pies, use_container_width=True)
        
        with col_v2:
            categories = ['Négatif', 'Neutre', 'Positif']
            woy_values = [metriques_woy['neg_pct'], metriques_woy['neu_pct'], metriques_woy['pos_pct']]
            autres_values = [metriques_autres['neg_pct'], metriques_autres['neu_pct'], metriques_autres['pos_pct']]
            
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                name='Woyofal',
                x=categories,
                y=woy_values,
                marker_color='#e74c3c',
                text=[f'{v:.1f}%' for v in woy_values],
                textposition='outside'
            ))
            fig_bar.add_trace(go.Bar(
                name='Autres',
                x=categories,
                y=autres_values,
                marker_color='#3498db',
                text=[f'{v:.1f}%' for v in autres_values],
                textposition='outside'
            ))
            
            fig_bar.update_layout(
                title="Comparaison des Pourcentages",
                barmode='group',
                height=500,
                yaxis_title="Pourcentage (%)"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab2:
        col_rad1, col_rad2 = st.columns(2)
        
        with col_rad1:
            categories_radar = ['Négatif', 'Positif', 'Neutre', 'Volume', 'Criticité']
            
            woy_radar = [
                min(metriques_woy['neg_pct'] / 100, 1),
                min(metriques_woy['pos_pct'] / 100, 1),
                min(metriques_woy['neu_pct'] / 100, 1),
                min(metriques_woy['total'] / max(1, total_global), 1),
                min(metriques_woy['criticite'] / max(1, metriques_autres['criticite'] * 1.5), 1)
            ]
            
            autres_radar = [
                min(metriques_autres['neg_pct'] / 100, 1),
                min(metriques_autres['pos_pct'] / 100, 1),
                min(metriques_autres['neu_pct'] / 100, 1),
                min(metriques_autres['total'] / max(1, total_global), 1),
                min(metriques_autres['criticite'] / max(1, metriques_woy['criticite'] * 1.5), 1)
            ]
            
            fig_radar = go.Figure()
            
            fig_radar.add_trace(go.Scatterpolar(
                r=woy_radar,
                theta=categories_radar,
                fill='toself',
                name='Woyofal',
                marker_color='#e74c3c'
            ))
            
            fig_radar.add_trace(go.Scatterpolar(
                r=autres_radar,
                theta=categories_radar,
                fill='toself',
                name='Autres',
                marker_color='#3498db'
            ))
            
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title="Comparaison Multidimensionnelle",
                height=500
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
        
        with col_rad2:
            st.markdown("### 📊 Indicateurs Synthétiques")
            
            metrics_df = pd.DataFrame({
                'Indicateur': ['Négativité', 'Positivité', 'Indice Satis.', 'Criticité', 'Ratio N/P'],
                'Woyofal': [
                    f"{metriques_woy['neg_pct']:.1f}%",
                    f"{metriques_woy['pos_pct']:.1f}%",
                    f"{metriques_woy['indice_satisfaction']:.1f}",
                    f"{metriques_woy['criticite']:.0f}",
                    f"{metriques_woy['ratio_neg_pos']:.2f}"
                ],
                'Autres': [
                    f"{metriques_autres['neg_pct']:.1f}%",
                    f"{metriques_autres['pos_pct']:.1f}%",
                    f"{metriques_autres['indice_satisfaction']:.1f}",
                    f"{metriques_autres['criticite']:.0f}",
                    f"{metriques_autres['ratio_neg_pos']:.2f}"
                ],
                'Écart': [
                    f"{metriques_woy['neg_pct'] - metriques_autres['neg_pct']:+.1f} pts",
                    f"{metriques_woy['pos_pct'] - metriques_autres['pos_pct']:+.1f} pts",
                    f"{metriques_woy['indice_satisfaction'] - metriques_autres['indice_satisfaction']:+.1f}",
                    f"{metriques_woy['criticite'] - metriques_autres['criticite']:+.0f}",
                    f"{metriques_woy['ratio_neg_pos'] - metriques_autres['ratio_neg_pos']:+.2f}"
                ]
            })
            
            # ✅ CORRECTION : .map() au lieu de .applymap() (deprecated depuis pandas 2.1)
            st.dataframe(
                metrics_df.style.map(
                    lambda x: 'color: red' if isinstance(x, str) and x.startswith('+') else '',
                    subset=['Écart']
                ),
                use_container_width=True,
                hide_index=True
            )
    
    with tab3:
        st.markdown("### 📈 Évolution Temporelle Comparative")
        
        df_woy_temp = preparer_donnees_temporelles(df_woyofal)
        df_autres_temp = preparer_donnees_temporelles(df_autres)
        
        if df_woy_temp is not None and df_autres_temp is not None:
            woy_monthly = df_woy_temp.groupby('annee_mois').apply(
                lambda x: pd.Series({
                    'neg_pct': (x['sentiment_pred'].astype(str).str.lower() == 'negative').mean() * 100,
                    'volume': len(x)
                })
            ).reset_index()
            
            autres_monthly = df_autres_temp.groupby('annee_mois').apply(
                lambda x: pd.Series({
                    'neg_pct': (x['sentiment_pred'].astype(str).str.lower() == 'negative').mean() * 100,
                    'volume': len(x)
                })
            ).reset_index()
            
            timeline_df = pd.merge(
                woy_monthly, autres_monthly, 
                on='annee_mois', 
                suffixes=('_woy', '_autres'),
                how='outer'
            ).fillna(0)
            
            if len(timeline_df) > 0:
                fig_timeline = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    subplot_titles=("Évolution du Sentiment Négatif (%)", "Volume de Publications"),
                    row_heights=[0.6, 0.4]
                )
                
                fig_timeline.add_trace(go.Scatter(
                    x=timeline_df['annee_mois'],
                    y=timeline_df['neg_pct_woy'],
                    name='Woyofal',
                    mode='lines+markers',
                    line=dict(color='#e74c3c', width=3)
                ), row=1, col=1)
                
                fig_timeline.add_trace(go.Scatter(
                    x=timeline_df['annee_mois'],
                    y=timeline_df['neg_pct_autres'],
                    name='Autres',
                    mode='lines+markers',
                    line=dict(color='#3498db', width=3)
                ), row=1, col=1)
                
                fig_timeline.add_trace(go.Bar(
                    x=timeline_df['annee_mois'],
                    y=timeline_df['volume_woy'],
                    name='Woyofal',
                    marker_color='#e74c3c',
                    opacity=0.7
                ), row=2, col=1)
                
                fig_timeline.add_trace(go.Bar(
                    x=timeline_df['annee_mois'],
                    y=timeline_df['volume_autres'],
                    name='Autres',
                    marker_color='#3498db',
                    opacity=0.7
                ), row=2, col=1)
                
                fig_timeline.update_layout(height=700, hovermode='x unified')
                fig_timeline.update_xaxes(title_text="", row=1, col=1)
                fig_timeline.update_xaxes(title_text="Mois", row=2, col=1)
                fig_timeline.update_yaxes(title_text="% Négatif", row=1, col=1)
                fig_timeline.update_yaxes(title_text="Volume", row=2, col=1)
                
                st.plotly_chart(fig_timeline, use_container_width=True)
            else:
                st.info("Données temporelles insuffisantes")
        else:
            st.info("Données temporelles non disponibles")
    
    with tab4:
        st.markdown("### 🔍 Analyse Statistique Détaillée")
        
        if len(df_woyofal) > 0 and len(df_autres) > 0:
            woy_neg = (df_woyofal['sentiment_pred'].astype(str).str.lower() == 'negative').sum()
            woy_non_neg = len(df_woyofal) - woy_neg
            autres_neg = (df_autres['sentiment_pred'].astype(str).str.lower() == 'negative').sum()
            autres_non_neg = len(df_autres) - autres_neg
            
            contingency = pd.DataFrame([
                [woy_neg, woy_non_neg],
                [autres_neg, autres_non_neg]
            ], index=['Woyofal', 'Autres'], columns=['Négatif', 'Non Négatif'])
            
            chi2, p_value, dof, expected = chi2_contingency(contingency)
            
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                st.metric("📊 Chi²", f"{chi2:.3f}")
                st.metric("📈 Degrés de liberté", dof)
            
            with col_stat2:
                st.metric("🎯 p-value", f"{p_value:.6f}")
                if p_value < 0.001:
                    st.error("🔴 HAUTEMENT SIGNIFICATIF")
                elif p_value < 0.05:
                    st.warning("🟠 SIGNIFICATIF")
                else:
                    st.success("🟢 NON significatif")
            
            with col_stat3:
                ci_woy = calculer_intervalle_confiance(woy_neg, len(df_woyofal))
                ci_autres = calculer_intervalle_confiance(autres_neg, len(df_autres))
                
                st.metric("📊 IC Woyofal (95%)", f"[{ci_woy[0]*100:.1f}%, {ci_woy[1]*100:.1f}%]")
                st.metric("📊 IC Autres (95%)", f"[{ci_autres[0]*100:.1f}%, {ci_autres[1]*100:.1f}%]")
    
    st.markdown("---")
    
    # ================================================
    # ANALYSE 4: COMPOSITION DES AUTRES THÈMES
    # ================================================
    
    if 'theme' in df_autres.columns:
        st.markdown("## 📋 Composition des Autres Thèmes")
        
        theme_counts = df_autres['theme'].value_counts().reset_index()
        theme_counts.columns = ['theme', 'count']
        theme_counts['pourcentage'] = (theme_counts['count'] / total_autres * 100).round(1) if total_autres > 0 else 0
        
        sentiments_theme = []
        for theme in theme_counts['theme']:
            df_t = df_autres[df_autres['theme'] == theme]
            neg_pct = (df_t['sentiment_pred'].astype(str).str.lower() == 'negative').mean() * 100
            sentiments_theme.append(neg_pct)
        
        theme_counts['negativite'] = sentiments_theme
        
        col_comp1, col_comp2 = st.columns(2)
        
        with col_comp1:
            if len(theme_counts) > 0:
                fig_comp = px.pie(
                    theme_counts,
                    values='count',
                    names='theme',
                    title="Répartition des Autres Thèmes",
                    hole=0.3
                )
                st.plotly_chart(fig_comp, use_container_width=True)
        
        with col_comp2:
            st.dataframe(
                theme_counts[['theme', 'count', 'pourcentage', 'negativite']].sort_values('count', ascending=False)
                .style.format({
                    'count': '{:,.0f}',
                    'pourcentage': '{:.1f}%',
                    'negativite': '{:.1f}%'
                }),
                use_container_width=True,
                hide_index=True
            )
    
    st.markdown("---")
    
    # ================================================
    # ANALYSE 5: RECOMMANDATIONS STRATÉGIQUES
    # ================================================
    
    st.markdown("## 💡 Recommandations Stratégiques")
    
    col_rec1, col_rec2 = st.columns(2)
    
    with col_rec1:
        st.markdown("### 🎯 Actions Prioritaires Woyofal")
        
        if metriques_woy['neg_pct'] > 70:
            st.error("""
            **🔴 CRISE MAJEURE - ACTION IMMÉDIATE**
            
            1. **Suspension temporaire** du déploiement
            2. **Audit technique complet** du système
            3. **Cellule de crise** dédiée
            4. **Communication de crise** immédiate
            5. **Hotline d'urgence** 24/7
            """)
        elif metriques_woy['neg_pct'] > 55:
            st.warning("""
            **🟠 PROBLÈMES STRUCTURELS**
            
            1. **Révision du système** de recharge
            2. **Application mobile** simplifiée
            3. **Formation renforcée** des agents
            4. **Campagne d'information** nationale
            5. **SMS d'alerte** avant épuisement
            """)
        elif metriques_woy['neg_pct'] > 40:
            st.info("""
            **🔵 AMÉLIORATIONS REQUISES**
            
            1. **Optimisation UX** de l'interface
            2. **FAQ interactive** en ligne
            3. **Monitoring** en temps réel
            4. **Études satisfaction** mensuelles
            """)
        else:
            st.success("""
            **🟢 SITUATION SATISFAISANTE**
            
            1. **Monitoring** continu
            2. **Partage des bonnes pratiques**
            3. **Innovation** incrémentale
            """)
    
    with col_rec2:
        st.markdown("### 📋 Actions Transversales")
        
        if diff_neg > 15:
            st.error("""
            **🔴 ÉCART CRITIQUE DÉTECTÉ**
            
            1. **Analyse comparative** approfondie
            2. **Identification** des causes profondes
            3. **Plan de convergence** Woyofal-Autres
            4. **Transfert** des bonnes pratiques
            """)
        elif diff_neg > 8:
            st.warning("""
            **🟠 ÉCART SIGNIFICATIF**
            
            1. **Étude** des facteurs de succès des autres
            2. **Adaptation** des processus
            3. **Formation croisée** des équipes
            4. **Objectifs** de réduction d'écart
            """)
        else:
            st.success("""
            **🟢 ÉCART MAÎTRISÉ**
            
            1. **Harmonisation** des services
            2. **Standardisation** des processus
            3. **Veille** des bonnes pratiques
            """)
    
    st.markdown("---")
    
    # ================================================
    # EXPORT DES DONNÉES
    # ================================================
    
    st.markdown("## 💾 Export des Analyses")
    
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    
    with col_exp1:
        if len(df_woyofal) > 0:
            csv_woy = df_woyofal.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 Données Woyofal",
                data=csv_woy,
                file_name=f"woyofal_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col_exp2:
        if len(df_autres) > 0:
            csv_autres = df_autres.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 Données Autres Thèmes",
                data=csv_autres,
                file_name=f"autres_themes_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col_exp3:
        rapport_data = {
            'Indicateur': ['Total', 'Négatif', 'Neutre', 'Positif', '% Négatif', '% Positif', 'Ratio N/P', 'Criticité'],
            'Woyofal': [
                metriques_woy['total'],
                metriques_woy['neg'],
                metriques_woy['neu'],
                metriques_woy['pos'],
                metriques_woy['neg_pct'],
                metriques_woy['pos_pct'],
                metriques_woy['ratio_neg_pos'],
                metriques_woy['criticite']
            ],
            'Autres': [
                metriques_autres['total'],
                metriques_autres['neg'],
                metriques_autres['neu'],
                metriques_autres['pos'],
                metriques_autres['neg_pct'],
                metriques_autres['pos_pct'],
                metriques_autres['ratio_neg_pos'],
                metriques_autres['criticite']
            ]
        }
        
        df_rapport = pd.DataFrame(rapport_data)
        csv_rapport = df_rapport.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 Rapport Comparatif",
            data=csv_rapport,
            file_name=f"rapport_comparatif_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )