"""
Page Vue d'ensemble - Dashboard SENELEC
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from dashboard.components.kpi_cards import show_kpi_cards
from dashboard.components.charts import create_sentiment_pie_chart, create_timeline_chart, create_platform_bar_chart
from dashboard.components.filters import (
    create_date_filter, 
    create_platform_filter, 
    create_sentiment_filter,
    create_theme_filter,
    apply_filters,
    normalize_dates
)


def safe_division(numerateur, denominateur, default=0):
    """Division sécurisée pour éviter les erreurs de division par zéro"""
    if denominateur == 0 or denominateur is None or pd.isna(denominateur):
        return default
    return numerateur / denominateur


def calculer_indicateurs_avances(df):
    """
    Calcule des indicateurs avancés pour la vue d'ensemble
    """
    indicateurs = {}
    
    if df is None or len(df) == 0:
        return indicateurs
    
    total = len(df)
    
    # Indicateurs généraux
    indicateurs['total'] = total
    
    # Sentiment
    if 'sentiment_pred' in df.columns:
        df['sentiment_pred'] = df['sentiment_pred'].astype(str).str.lower()
        
        neg = (df['sentiment_pred'] == 'negative').sum()
        pos = (df['sentiment_pred'] == 'positive').sum()
        neu = (df['sentiment_pred'] == 'neutral').sum()
        
        indicateurs['neg'] = neg
        indicateurs['pos'] = pos
        indicateurs['neu'] = neu
        indicateurs['neg_pct'] = safe_division(neg * 100, total)
        indicateurs['pos_pct'] = safe_division(pos * 100, total)
        indicateurs['neu_pct'] = safe_division(neu * 100, total)
        indicateurs['ratio_neg_pos'] = safe_division(neg, pos, default=neg if pos == 0 and neg > 0 else 0)
        indicateurs['indice_satisfaction'] = safe_division((pos - neg) * 100, total)
        indicateurs['criticite_globale'] = total * indicateurs['neg_pct'] / 100
    
    # Plateformes
    if 'plateforme' in df.columns:
        platform_counts = df['plateforme'].value_counts()
        indicateurs['plateformes'] = platform_counts.to_dict()
        indicateurs['nb_plateformes'] = len(platform_counts)
    
    # Thèmes
    if 'theme' in df.columns:
        theme_counts = df['theme'].value_counts()
        indicateurs['themes'] = theme_counts.to_dict()
        indicateurs['nb_themes'] = len(theme_counts)
        indicateurs['theme_dominant'] = theme_counts.index[0] if len(theme_counts) > 0 else "N/A"
        indicateurs['pct_theme_dominant'] = safe_division(theme_counts.iloc[0] * 100, total)
    
    # Temporalité
    if 'date_publication' in df.columns:
        dates_valides = df['date_publication'].dropna()
        if len(dates_valides) > 0:
            indicateurs['date_min'] = dates_valides.min()
            indicateurs['date_max'] = dates_valides.max()
            indicateurs['duree_jours'] = (dates_valides.max() - dates_valides.min()).days
            indicateurs['moyenne_par_jour'] = safe_division(total, indicateurs['duree_jours'])
    
    return indicateurs


def show_overview(df):
    """
    Affiche la vue d'ensemble - Version ultra-complète
    """
    
    if df is None:
        st.error("❌ Impossible de charger les données")
        return
    
    # 1. Normaliser les dates
    df = normalize_dates(df.copy())
    total_original = len(df)
    
    # 2. Calculer indicateurs avancés sur les données brutes
    indicateurs_bruts = calculer_indicateurs_avances(df)
    
    # 3. Afficher les dates disponibles
    if 'date_publication' in df.columns:
        dates_valides = df['date_publication'].dropna()
        if len(dates_valides) > 0:
            date_min = dates_valides.min().strftime('%d/%m/%Y')
            date_max = dates_valides.max().strftime('%d/%m/%Y')
            st.sidebar.success(f"📅 Période: {date_min} → {date_max}")
            st.sidebar.info(f"📊 Total: {total_original} publications")
    
    # ================================================
    # HEADER
    # ================================================
    
    # st.header("🏠 Vue d'Ensemble")
    st.markdown("""
    **Analyse globale des perceptions des usagers SENELEC**
    
    Cette page présente une synthèse complète de l'ensemble des données collectées,
    avec des indicateurs clés et des visualisations interactives.
    """)
    
    st.markdown("---")
    
    # ================================================
    # FILTRES AVANCÉS
    # ================================================
    
    st.markdown("## 🔍 Filtres Avancés")
    
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    
    with col_f1:
        platform_filter = create_platform_filter(df)
    
    with col_f2:
        date_range = create_date_filter(df)
    
    with col_f3:
        sentiment_filter = create_sentiment_filter(df)
    
    with col_f4:
        theme_filter = create_theme_filter(df) if 'theme' in df.columns else []
    
    # Appliquer les filtres
    filters_desc = []
    if platform_filter != "Toutes":
        filters_desc.append(f"📱 Plateforme: {platform_filter}")
    if date_range is not None:
        filters_desc.append(f"📅 Dates: {date_range[0]} → {date_range[1]}")
    if sentiment_filter and len(sentiment_filter) < 3:
        filters_desc.append(f"😊 Sentiments: {sentiment_filter}")
    if theme_filter and len(theme_filter) < df['theme'].nunique() if 'theme' in df.columns else False:
        filters_desc.append(f"🔍 Thèmes: {len(theme_filter)} sélectionnés")
    
    df_filtered = apply_filters(
        df, 
        platform_filter=platform_filter,
        date_range=date_range,
        sentiment_filter=sentiment_filter,
        themes=theme_filter
    )
    
    # Calculer indicateurs sur données filtrées
    indicateurs = calculer_indicateurs_avances(df_filtered)
    
    # ================================================
    # RÉSUMÉ DU FILTRAGE
    # ================================================
    
    st.markdown("---")
    
    col_res1, col_res2, col_res3 = st.columns([2, 1, 1])
    
    with col_res1:
        st.markdown(f"## 📊 **{len(df_filtered):,} publications**")
    
    with col_res2:
        pct_garde = safe_division(len(df_filtered) * 100, total_original)
        if len(df_filtered) < total_original:
            st.warning(f"Exclues: {total_original - len(df_filtered)} ({100-pct_garde:.1f}%)")
        else:
            st.success("✅ Toutes les données")
    
    with col_res3:
        st.metric("Taux de rétention", f"{pct_garde:.1f}%")
    
    if filters_desc:
        with st.expander("🔍 Filtres actifs"):
            for f in filters_desc:
                st.write(f"• {f}")
    
    st.markdown("---")
    
    # ================================================
    # KPI PRINCIPAUX 
    # ================================================
    
    if len(df_filtered) > 0:
        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        
        with col_kpi1:
            st.metric(
                "😡 Sentiment Négatif",
                f"{indicateurs.get('neg_pct', 0):.1f}%",
                delta=f"{indicateurs.get('neg', 0):,} publications",
                delta_color="inverse"
            )
        
        with col_kpi2:
            st.metric(
                "😊 Sentiment Positif",
                f"{indicateurs.get('pos_pct', 0):.1f}%",
                delta=f"{indicateurs.get('pos', 0):,} publications"
            )
        
        with col_kpi3:
            st.metric(
                "😐 Sentiment Neutre",
                f"{indicateurs.get('neu_pct', 0):.1f}%",
                delta=f"{indicateurs.get('neu', 0):,} publications"
            )
        
        with col_kpi4:
            st.metric(
                "📊 Total Publications",
                f"{indicateurs.get('total', 0):,}",
                delta=f"{indicateurs.get('nb_themes', 0)} thèmes"
            )
    
    st.markdown("---")
    
    # ================================================
    # ANALYSE 1: VUE D'ENSEMBLE GÉNÉRALE
    # ================================================
    
    st.markdown("## 📈 Vue d'Ensemble Générale")
    
    tab_overview1, tab_overview2, tab_overview3 = st.tabs([
        "📊 Distribution Globale",
        "📱 Analyse par Plateforme",
        "📈 Tendances Temporelles"
    ])
    
    with tab_overview1:
        col_ov1, col_ov2 = st.columns(2)
        
        with col_ov1:
            # Pie chart distribution des sentiments
            if len(df_filtered) > 0:
                fig_pie = create_sentiment_pie_chart(df_filtered)
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.warning("Aucune donnée")
        
        with col_ov2:
            # Carte de répartition des sentiments 
            if len(df_filtered) > 0:
                fig_donut = go.Figure(data=[go.Pie(
                    labels=['Négatif', 'Neutre', 'Positif'],
                    values=[indicateurs.get('neg', 0), indicateurs.get('neu', 0), indicateurs.get('pos', 0)],
                    marker=dict(colors=['#e74c3c', '#95a5a6', '#27ae60']),
                    hole=0.6,
                    textinfo='none',
                    hovertemplate='<b>%{label}</b><br>%{value} publications<br>%{percent}<extra></extra>'
                )])
                
                fig_donut.add_annotation(
                    text=f"Total<br>{indicateurs.get('total', 0)}",
                    x=0.5, y=0.5,
                    font_size=20,
                    font_family="Arial",
                    showarrow=False
                )
                
                fig_donut.update_layout(
                    title="Distribution avec Total",
                    height=400,
                    showlegend=True
                )
                
                st.plotly_chart(fig_donut, use_container_width=True)
    
    with tab_overview2:
        # Analyse par plateforme
        if 'plateforme' in df_filtered.columns:
            col_plat1, col_plat2 = st.columns(2)
            
            with col_plat1:
                # Bar chart plateforme
                fig_platform = create_platform_bar_chart(df_filtered)
                st.plotly_chart(fig_platform, use_container_width=True)
            
            with col_plat2:
                # Distribution sentiment par plateforme
                platform_sentiment = pd.crosstab(
                    df_filtered['plateforme'],
                    df_filtered['sentiment_pred'],
                    normalize='index'
                ) * 100
                
                fig_platform_pct = go.Figure()
                
                colors = {'negative': '#e74c3c', 'neutral': '#95a5a6', 'positive': '#27ae60'}
                
                for sentiment in ['negative', 'neutral', 'positive']:
                    if sentiment in platform_sentiment.columns:
                        fig_platform_pct.add_trace(go.Bar(
                            name={'negative': 'Négatif', 'neutral': 'Neutre', 'positive': 'Positif'}[sentiment],
                            x=platform_sentiment.index,
                            y=platform_sentiment[sentiment],
                            marker_color=colors[sentiment],
                            text=platform_sentiment[sentiment].round(1).astype(str) + '%',
                            textposition='inside'
                        ))
                
                fig_platform_pct.update_layout(
                    title="Distribution des Sentiments par Plateforme (%)",
                    barmode='stack',
                    height=400,
                    yaxis_title="Pourcentage (%)"
                )
                
                st.plotly_chart(fig_platform_pct, use_container_width=True)
    
    with tab_overview3:
        # Évolution temporelle
        if 'date_publication' in df_filtered.columns and df_filtered['date_publication'].notna().any():
            # Timeline chart
            fig_timeline = create_timeline_chart(df_filtered)
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Statistiques temporelles
            if 'duree_jours' in indicateurs and indicateurs['duree_jours'] > 0:
                col_temp1, col_temp2, col_temp3 = st.columns(3)
                
                with col_temp1:
                    st.metric("📅 Début", indicateurs['date_min'].strftime('%d/%m/%Y'))
                with col_temp2:
                    st.metric("📅 Fin", indicateurs['date_max'].strftime('%d/%m/%Y'))
                with col_temp3:
                    st.metric("⏱️ Durée", f"{indicateurs['duree_jours']} jours")
                
                # Détection des pics
                df_temp = df_filtered.copy()
                df_temp['date'] = df_filtered['date_publication'].dt.date
                daily_counts = df_temp.groupby('date').size().reset_index(name='count')
                
                if len(daily_counts) > 0:
                    moyenne = daily_counts['count'].mean()
                    ecart_type = daily_counts['count'].std()
                    seuil_pic = moyenne + 2 * ecart_type
                    
                    pics = daily_counts[daily_counts['count'] > seuil_pic]
                    
                    if len(pics) > 0:
                        st.info(f"🔥 **{len(pics)} pics d'activité détectés** (>{seuil_pic:.1f} publications/jour)")
    
    st.markdown("---")
    
    # ================================================
    # ANALYSE 2: ANALYSE THÉMATIQUE
    # ================================================
    
    if 'theme' in df_filtered.columns and len(df_filtered) > 0:
        st.markdown("## 🔍 Analyse Thématique")
        
        tab_theme1, tab_theme2 = st.tabs(["📊 Distribution des Thèmes", "📈 Sentiment par Thème"])
        
        with tab_theme1:
            col_th1, col_th2 = st.columns(2)
            
            with col_th1:
                # Top thèmes
                theme_counts = df_filtered['theme'].value_counts().reset_index()
                theme_counts.columns = ['theme', 'count']
                theme_counts['pourcentage'] = (theme_counts['count'] / len(df_filtered) * 100).round(1)
                
                fig_themes = px.bar(
                    theme_counts.head(5),
                    x='count',
                    y='theme',
                    orientation='h',
                    title="Top 5 Thèmes",
                    text='pourcentage',
                    color='count',
                    color_continuous_scale='Viridis'
                )
                fig_themes.update_traces(texttemplate='%{text}%', textposition='outside')
                fig_themes.update_layout(height=500)
                
                st.plotly_chart(fig_themes, use_container_width=True)
            
            with col_th2:
                # Pie chart des thèmes
                fig_theme_pie = px.pie(
                    theme_counts.head(6),
                    values='count',
                    names='theme',
                    title="Répartition des Thèmes Principaux",
                    hole=0.3
                )
                fig_theme_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_theme_pie.update_layout(height=500)
                
                st.plotly_chart(fig_theme_pie, use_container_width=True)
        
        with tab_theme2:
            # Sentiment par thème
            theme_sentiment = pd.crosstab(
                df_filtered['theme'],
                df_filtered['sentiment_pred'],
                normalize='index'
            ) * 100
            
            # Trier par volume
            top_themes = df_filtered['theme'].value_counts().head(8).index
            theme_sentiment_top = theme_sentiment.loc[top_themes]
            
            fig_theme_sent = go.Figure()
            
            colors = {'negative': '#e74c3c', 'neutral': '#95a5a6', 'positive': '#27ae60'}
            
            for sentiment in ['negative', 'neutral', 'positive']:
                if sentiment in theme_sentiment_top.columns:
                    fig_theme_sent.add_trace(go.Bar(
                        name={'negative': 'Négatif', 'neutral': 'Neutre', 'positive': 'Positif'}[sentiment],
                        x=theme_sentiment_top.index,
                        y=theme_sentiment_top[sentiment],
                        marker_color=colors[sentiment],
                        text=theme_sentiment_top[sentiment].round(1).astype(str) + '%',
                        textposition='inside'
                    ))
            
            fig_theme_sent.update_layout(
                title="Distribution des Sentiments par Thème (%)",
                barmode='stack',
                height=500,
                xaxis_title="Thème",
                yaxis_title="Pourcentage (%)",
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig_theme_sent, use_container_width=True)
    
    st.markdown("---")
    
    # ================================================
    # ANALYSE 3: INDICATEURS AVANCÉS
    # ================================================
    
    st.markdown("## 📊 Indicateurs Avancés")
    
    col_adv1, col_adv2, col_adv3, col_adv4 = st.columns(4)
    
    with col_adv1:
        st.metric(
            "📈 Indice Satisfaction",
            f"{indicateurs.get('indice_satisfaction', 0):.1f}",
            help="(Positif - Négatif) / Total * 100"
        )
    
    with col_adv2:
        st.metric(
            "⚖️ Ratio Négatif/Positif",
            f"{indicateurs.get('ratio_neg_pos', 0):.2f}",
            help="Nombre de négatifs pour 1 positif"
        )
    
    with col_adv3:
        st.metric(
            "🔥 Criticité Globale",
            f"{indicateurs.get('criticite_globale', 0):.0f}",
            help="Volume × % Négatif / 100"
        )
    
    with col_adv4:
        st.metric(
            "📊 Diversité Thématique",
            f"{indicateurs.get('nb_themes', 0)} thèmes",
            delta=f"Dominant: {indicateurs.get('theme_dominant', 'N/A')[:15]}"
        )
    
    # Métriques supplémentaires
    if 'moyenne_par_jour' in indicateurs:
        col_sup1, col_sup2, col_sup3 = st.columns(3)
        
        with col_sup1:
            st.metric("📅 Moyenne par jour", f"{indicateurs['moyenne_par_jour']:.1f}")
        
        with col_sup2:
            if 'nb_plateformes' in indicateurs:
                st.metric("📱 Plateformes actives", indicateurs['nb_plateformes'])
        
        with col_sup3:
            if 'pct_theme_dominant' in indicateurs:
                st.metric("🎯 Thème dominant", f"{indicateurs['pct_theme_dominant']:.1f}%")
    
    st.markdown("---")
    
    # ================================================
    # ANALYSE 4: STATISTIQUES DÉTAILLÉES
    # ================================================
    
    with st.expander("📋 Statistiques Détaillées"):
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            st.markdown("**😊 Distribution des Sentiments**")
            if 'sentiment_pred' in df_filtered.columns:
                sentiment_dist = df_filtered['sentiment_pred'].value_counts()
                for sent, count in sentiment_dist.items():
                    pct = (count / len(df_filtered)) * 100
                    emoji = {'negative': '😡', 'neutral': '😐', 'positive': '😊'}.get(str(sent).lower(), '📊')
                    st.write(f"{emoji} {sent.capitalize()}: {count:,} ({pct:.1f}%)")
            
            st.markdown("---")
            
            st.markdown("**📱 Distribution des Plateformes**")
            if 'plateforme' in df_filtered.columns:
                platform_dist = df_filtered['plateforme'].value_counts()
                for pf, count in platform_dist.items():
                    pct = (count / len(df_filtered)) * 100
                    emoji = {'facebook': '📘', 'twitter': '🐦', 'enquete': '📋'}.get(str(pf).lower(), '📱')
                    st.write(f"{emoji} {pf.capitalize()}: {count:,} ({pct:.1f}%)")
        
        with col_s2:
            st.markdown("**📅 Période d'Analyse**")
            if 'date_publication' in df_filtered.columns and df_filtered['date_publication'].notna().any():
                date_min = df_filtered['date_publication'].min()
                date_max = df_filtered['date_publication'].max()
                st.write(f"📅 Début: {date_min.strftime('%d/%m/%Y')}")
                st.write(f"📅 Fin: {date_max.strftime('%d/%m/%Y')}")
                duree = (date_max - date_min).days
                st.write(f"⏱️ Durée: {duree} jours")
                
                # Statistiques temporelles avancées
                df_temp = df_filtered.copy()
                df_temp['date'] = df_filtered['date_publication'].dt.date
                daily_counts = df_temp.groupby('date').size()
                
                if len(daily_counts) > 0:
                    st.write(f"📊 Moyenne: {daily_counts.mean():.1f} pub/jour")
                    st.write(f"📈 Max: {daily_counts.max()} pub")
                    st.write(f"📉 Min: {daily_counts.min()} pub")
            
            st.markdown("---")
            
            st.markdown("**🔍 Top Mots-Clés**")
            # Trouver colonne de texte
            text_col = None
            for col in ['texte_nettoye', 'texte', 'content']:
                if col in df_filtered.columns:
                    text_col = col
                    break
            
            if text_col:
                all_text = ' '.join(df_filtered[text_col].dropna().astype(str).str.lower())
                mots_importants = ['senelec', 'woyofal', 'coupure', 'facture', 'service', 'client', 'probleme', 'cher', 'merci']
                
                freq_mots = {}
                for mot in mots_importants:
                    freq_mots[mot] = all_text.count(mot)
                
                mots_df = pd.DataFrame(list(freq_mots.items()), columns=['Mot', 'Fréquence']).sort_values('Fréquence', ascending=False)
                st.dataframe(mots_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # ================================================
    # ANALYSE 5: DERNIÈRES PUBLICATIONS
    # ================================================
    
    with st.expander("📰 Aperçu des dernières publications"):
        # Vérifier quelles colonnes de texte existent
        text_column = None
        for col in ['texte_nettoye', 'texte', 'content', 'message', 'text']:
            if col in df_filtered.columns:
                text_column = col
                break
        
        if text_column is None:
            st.info("Aucune colonne de texte trouvée")
            st.write("Colonnes disponibles:", list(df_filtered.columns))
        else:
            # Trier par date si disponible
            if 'date_publication' in df_filtered.columns:
                df_recent = df_filtered.sort_values('date_publication', ascending=False).head(8)
            else:
                df_recent = df_filtered.head(8)
            
            for idx, row in df_recent.iterrows():
                # Sentiment emoji
                sentiment = row.get('sentiment_pred', '')
                if pd.isna(sentiment):
                    sentiment_emoji = '📊'
                else:
                    sentiment_emoji = {'negative': '😡', 'neutral': '😐', 'positive': '😊'}.get(str(sentiment).lower(), '📊')
                
                # Platform emoji
                platform = row.get('plateforme', '')
                if pd.isna(platform):
                    platform_emoji = '📱'
                else:
                    platform_emoji = {'facebook': '📘', 'twitter': '🐦', 'enquete': '📋'}.get(str(platform).lower(), '📱')
                
                # Theme
                theme = row.get('theme', 'Sujet')
                if pd.isna(theme):
                    theme = 'Sujet'
                
                # Date
                date_str = ""
                if 'date_publication' in row and pd.notna(row['date_publication']):
                    date_str = f" - {pd.to_datetime(row['date_publication']).strftime('%d/%m/%Y')}"
                
                # Texte
                texte = str(row.get(text_column, ''))
                if pd.isna(texte) or texte == '' or texte == 'nan':
                    texte = "[Texte non disponible]"
                else:
                    texte = texte[:150] + "..." if len(texte) > 150 else texte
                
                st.markdown(f"""
                {sentiment_emoji} {platform_emoji} **{theme}**{date_str}  
                > {texte}
                """)
                st.markdown("---")
    
    st.markdown("---")
    
    # ================================================
    # ANALYSE 6: RECOMMANDATIONS AUTOMATIQUES
    # ================================================
    
    st.markdown("## 💡 Recommandations Automatiques")
    
    col_rec1, col_rec2 = st.columns(2)
    
    with col_rec1:
        st.markdown("### 🎯 Priorités d'Action")
        
        if indicateurs.get('neg_pct', 0) > 60:
            st.error("""
            **🔴 PRIORITÉ ABSOLUE**
            - Taux de négativité critique (>60%)
            - Audit global recommandé
            - Plan d'action d'urgence
            """)
        elif indicateurs.get('neg_pct', 0) > 45:
            st.warning("""
            **🟠 ATTENTION REQUISE**
            - Négativité élevée
            - Analyse approfondie nécessaire
            - Actions correctives ciblées
            """)
        else:
            st.success("""
            **🟢 SITUATION SOUS CONTRÔLE**
            - Négativité maîtrisée
            - Monitoring continu
            - Optimisations incrémentales
            """)
    
    with col_rec2:
        st.markdown("### 📊 Points d'Attention")
        
        if 'theme_dominant' in indicateurs and indicateurs['pct_theme_dominant'] > 50:
            st.info(f"🎯 **Thème dominant**: {indicateurs['theme_dominant']} ({indicateurs['pct_theme_dominant']:.1f}%)")
        
        if 'ratio_neg_pos' in indicateurs and indicateurs['ratio_neg_pos'] > 2:
            st.warning(f"⚠️ **Déséquilibre**: {indicateurs['ratio_neg_pos']:.1f} négatifs pour 1 positif")
        
        if 'criticite_globale' in indicateurs and indicateurs['criticite_globale'] > 500:
            st.error(f"🔥 **Criticité élevée**: {indicateurs['criticite_globale']:.0f}")
    
    st.markdown("---")
    
    # ================================================
    # EXPORT DES DONNÉES
    # ================================================
    
    st.markdown("## 💾 Export des Analyses")
    
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    
    with col_exp1:
        # Données filtrées
        csv_filtered = df_filtered.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 Données filtrées (CSV)",
            data=csv_filtered,
            file_name=f"senelec_data_filtre_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_exp2:
        # Rapport d'indicateurs
        if indicateurs:
            df_rapport = pd.DataFrame([indicateurs]).T.reset_index()
            df_rapport.columns = ['Indicateur', 'Valeur']
            csv_rapport = df_rapport.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 Indicateurs clés (CSV)",
                data=csv_rapport,
                file_name=f"indicateurs_cles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col_exp3:
        # Distribution des thèmes
        if 'theme' in df_filtered.columns:
            theme_export = df_filtered['theme'].value_counts().reset_index()
            theme_export.columns = ['theme', 'count']
            theme_export['pourcentage'] = (theme_export['count'] / len(df_filtered) * 100).round(1)
            csv_themes = theme_export.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 Distribution thèmes (CSV)",
                data=csv_themes,
                file_name=f"distribution_themes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )