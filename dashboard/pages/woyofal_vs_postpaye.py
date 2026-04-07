"""
Page Woyofal vs Postpayé - Dashboard SENELEC
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import re
from collections import Counter
from scipy.stats import chi2_contingency
from datetime import datetime


def safe_division(numerateur, denominateur, default=0):
    """Division sécurisée"""
    if denominateur == 0 or denominateur is None or pd.isna(denominateur):
        return default
    return numerateur / denominateur


def normalize_dates(df, date_column='date_publication'):
    """
    Normalise les dates en supprimant les fuseaux horaires
    """
    if date_column in df.columns:
        try:
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            if hasattr(df[date_column].dtype, 'tz'):
                df[date_column] = df[date_column].dt.tz_localize(None)
        except:
            df[date_column] = pd.NaT
    return df


def preparer_donnees_temporelles(df, date_col='date_publication'):
    """
    Prépare les données temporelles de façon sécurisée
    """
    if df is None or len(df) == 0 or date_col not in df.columns:
        return None
    
    df_temp = df.copy()
    
    df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
    
    # Supprimer les dates invalides
    df_temp = df_temp.dropna(subset=[date_col])
    
    if len(df_temp) == 0:
        return None
    
    if hasattr(df_temp[date_col].dtype, 'tz'):
        df_temp[date_col] = df_temp[date_col].dt.tz_localize(None)
    
    # Ajouter colonne mois
    df_temp['mois'] = df_temp[date_col].dt.to_period('M').astype(str)
    
    return df_temp


def extraire_mots_cles_postpaye(textes):
    """
    Extrait les mots-clés qui indiquent le système postpayé
    """
    mots_cles_postpaye = {
        'Facture': ['facture', 'facturation', 'facturé', 'facturer'],
        'Paiement': ['paiement', 'payer', 'payé', 'règlement', 'échéance'],
        'Mensuel': ['mensuel', 'mois', 'mensualité', 'abonnement'],
        'Compteur': ['compteur', 'index', 'relevé', 'consommation'],
        'Tarif': ['tarif', 'prix', 'coût', 'cher', 'kW', 'kWh'],
        'Retard': ['retard', 'délai', 'impayé', 'coupure', 'délestage'],
        'Agence': ['agence', 'guichet', 'boutique', 'point de vente'],
        'Service client': ['service', 'client', 'accueil', 'réclamation', 'plainte']
    }
    
    resultats = {categorie: 0 for categorie in mots_cles_postpaye.keys()}
    
    for texte in textes:
        if pd.isna(texte) or texte == '':
            continue
        texte_lower = str(texte).lower()
        for categorie, mots in mots_cles_postpaye.items():
            for mot in mots:
                if mot in texte_lower:
                    resultats[categorie] += 1
                    break
    
    return resultats


def extraire_mots_cles_woyofal(textes):
    """
    Extrait les mots-clés spécifiques à Woyofal
    """
    mots_cles_woyofal = {
        'Code': ['code', 'code non reconnu', 'code incorrect', 'code invalide'],
        'Recharge': ['recharge', 'crédit', 'acheter', 'achat', 'unité'],
        'Compteur prépayé': ['compteur woyofal', 'compteur prépayé', 'compteur bloqué'],
        'Solde': ['solde', 'crédit', 'épuisé', 'insuffisant'],
        'Tranche': ['tranche', 'tarif woyofal', 'prix woyofal'],
        'Application': ['app', 'application', 'mobile', 'sms']
    }
    
    resultats = {categorie: 0 for categorie in mots_cles_woyofal.keys()}
    
    for texte in textes:
        if pd.isna(texte) or texte == '':
            continue
        texte_lower = str(texte).lower()
        for categorie, mots in mots_cles_woyofal.items():
            for mot in mots:
                if mot in texte_lower:
                    resultats[categorie] += 1
                    break
    
    return resultats


def identifier_publications_postpaye(df, text_column):
    """
    Identifie les publications liées au système postpayé
    """
    if text_column is None or text_column not in df.columns:
        return pd.DataFrame()
    
    # Mots-clés postpayé 
    mots_postpaye = [
        'facture', 'facturation', 'mensuel', 'abonnement', 
        'paiement', 'échéance', 'relevé', 'index', 'compteur',
        'régularisation', 'consommation', 'kW', 'kWh',
        'tarif', 'prix', 'coût', 'cher', 'retard', 'impayé',
        'agence', 'guichet', 'réclamation', 'plainte'
    ]
    
    # Mots à exclure 
    mots_exclus = ['woyofal', 'prépayé', 'prepaye', 'code', 'recharge']
    
    mask_postpaye = df[text_column].astype(str).str.lower().apply(
        lambda x: any(mot in x for mot in mots_postpaye)
    )
    
    mask_exclure = df[text_column].astype(str).str.lower().apply(
        lambda x: any(mot in x for mot in mots_exclus)
    )
    
    df_postpaye = df[mask_postpaye & ~mask_exclure].copy()
    
    return df_postpaye


def calculer_metriques_avancees(df):
    """Calcule des métriques avancées"""
    metriques = {
        'total': 0, 'neg': 0, 'pos': 0, 'neu': 0,
        'neg_pct': 0, 'pos_pct': 0, 'neu_pct': 0,
        'ratio_neg_pos': 0, 'indice_satisfaction': 0, 'criticite': 0
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
    
    return metriques


def show_woyofal_vs_postpaye(df):
    """
    Affiche la comparaison Woyofal vs Postpayé
    """
    
    if df is None or len(df) == 0:
        st.error("❌ Données non disponibles")
        return
    
    # Normaliser les dates du dataframe principal
    if 'date_publication' in df.columns:
        df = normalize_dates(df.copy())
    
    # st.header("💳 Analyse Comparative : Woyofal (Prépayé) vs Postpayé")
    
    st.markdown("""
    Cette page compare en profondeur les perceptions des usagers selon leur mode de paiement :
    
    - **💳 Woyofal (Prépayé)** : Système de prépaiement électronique
    - **📋 Postpayé (Facture classique)** : Paiement après consommation (identification par mots-clés)
    
    *L'identification du système postpayé se fait par analyse automatique des mots-clés dans les publications.*
    """)
    
    st.markdown("---")
    
    # ================================================
    # ÉTAPE 1: IDENTIFIER LES PUBLICATIONS
    # ================================================
    
    # 1.1 Publications Woyofal (par thème)
    df_woyofal = pd.DataFrame()
    if 'theme' in df.columns:
        df_woyofal = df[df['theme'] == 'woyofal'].copy()
        if len(df_woyofal) > 0:
            st.sidebar.success(f"✅ Woyofal (thème): {len(df_woyofal)} publications")
    
    # 1.2 Identifier la colonne de texte
    text_column = None
    for col in ['texte_nettoye', 'texte', 'content', 'message']:
        if col in df.columns:
            text_column = col
            break
    
    # 1.3 Publications Postpayé 
    df_postpaye = pd.DataFrame()
    if text_column:
        df_postpaye = identifier_publications_postpaye(df, text_column)
        if len(df_postpaye) > 0:
            st.sidebar.success(f"✅ Postpayé (mots-clés): {len(df_postpaye)} publications")
    
    total_woy = len(df_woyofal)
    total_post = len(df_postpaye)
    
    if total_woy == 0 or total_post == 0:
        st.warning("⚠️ Données insuffisantes pour la comparaison")
        if total_woy == 0:
            st.error("Aucune publication Woyofal trouvée")
        if total_post == 0:
            st.error("Aucune publication Postpayé trouvée (vérifiez les mots-clés)")
        return
    
    # ================================================
    # ÉTAPE 2: ANALYSE DES MOTS-CLÉS
    # ================================================
    
    with st.expander("🔍 Détail de l'identification des publications"):
        col_mc1, col_mc2 = st.columns(2)
        
        with col_mc1:
            st.markdown("### 📋 Mots-clés Postpayé")
            if text_column:
                mots_post = extraire_mots_cles_postpaye(df_postpaye[text_column])
                
                df_mots_post = pd.DataFrame([
                    {'Catégorie': cat, 'Occurrences': count}
                    for cat, count in mots_post.items() if count > 0
                ]).sort_values('Occurrences', ascending=False)
                
                if not df_mots_post.empty:
                    st.dataframe(df_mots_post, use_container_width=True, hide_index=True)
                    
                    fig_mots_post = px.bar(
                        df_mots_post,
                        x='Occurrences',
                        y='Catégorie',
                        orientation='h',
                        title="Fréquence des catégories Postpayé",
                        color='Occurrences',
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig_mots_post, use_container_width=True)
        
        with col_mc2:
            st.markdown("### 💳 Mots-clés Woyofal")
            if text_column:
                mots_woy = extraire_mots_cles_woyofal(df_woyofal[text_column])
                
                df_mots_woy = pd.DataFrame([
                    {'Catégorie': cat, 'Occurrences': count}
                    for cat, count in mots_woy.items() if count > 0
                ]).sort_values('Occurrences', ascending=False)
                
                if not df_mots_woy.empty:
                    st.dataframe(df_mots_woy, use_container_width=True, hide_index=True)
                    
                    fig_mots_woy = px.bar(
                        df_mots_woy,
                        x='Occurrences',
                        y='Catégorie',
                        orientation='h',
                        title="Fréquence des catégories Woyofal",
                        color='Occurrences',
                        color_continuous_scale='Reds'
                    )
                    st.plotly_chart(fig_mots_woy, use_container_width=True)
    
    st.markdown("---")
    
    # ================================================
    # ÉTAPE 3: MÉTRIQUES COMPARATIVES
    # ================================================
    
    metriques_woy = calculer_metriques_avancees(df_woyofal)
    metriques_post = calculer_metriques_avancees(df_postpaye)
    
    st.markdown("## 📊 Tableau de Bord Comparatif")
    
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    
    with col_kpi1:
        st.metric(
            "💳 Publications Woyofal",
            f"{metriques_woy['total']:,}",
            delta=f"{metriques_woy['total']/total_woy*100:.1f}% du total" if total_woy > 0 else "0%"
        )
    
    with col_kpi2:
        st.metric(
            "📋 Publications Postpayé",
            f"{metriques_post['total']:,}",
            delta=f"{metriques_post['total']/total_post*100:.1f}% du total" if total_post > 0 else "0%"
        )
    
    with col_kpi3:
        diff_neg = metriques_woy['neg_pct'] - metriques_post['neg_pct']
        st.metric(
            "😡 Écart Négativité",
            f"{diff_neg:+.1f} pts",
            delta=f"{abs(diff_neg):.1f} pts",
            delta_color="inverse"
        )
    
    with col_kpi4:
        diff_indice = metriques_woy['indice_satisfaction'] - metriques_post['indice_satisfaction']
        st.metric(
            "📈 Écart Indice Satisfaction",
            f"{diff_indice:+.1f}",
            delta=f"{abs(diff_indice):.1f}",
            delta_color="inverse"
        )
    
    st.markdown("---")
    
    # ================================================
    # ÉTAPE 4: VISUALISATIONS
    # ================================================
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Distribution", 
        "📊 Comparaison", 
        "🔍 Analyse Détaillée",
        "📈 Évolution Temporelle"
    ])
    
    with tab1:
        col_dist1, col_dist2 = st.columns(2)
        
        with col_dist1:
            # Pie charts
            fig_pies = make_subplots(
                rows=1, cols=2,
                specs=[[{"type": "domain"}, {"type": "domain"}]],
                subplot_titles=("💳 Woyofal", "📋 Postpayé")
            )
            
            fig_pies.add_trace(go.Pie(
                labels=['😡 Négatif', '😐 Neutre', '😊 Positif'],
                values=[metriques_woy['neg'], metriques_woy['neu'], metriques_woy['pos']],
                marker=dict(colors=['#e74c3c', '#95a5a6', '#27ae60']),
                textinfo='label+percent',
                hole=0.4
            ), row=1, col=1)
            
            fig_pies.add_trace(go.Pie(
                labels=['😡 Négatif', '😐 Neutre', '😊 Positif'],
                values=[metriques_post['neg'], metriques_post['neu'], metriques_post['pos']],
                marker=dict(colors=['#e74c3c', '#95a5a6', '#27ae60']),
                textinfo='label+percent',
                hole=0.4
            ), row=1, col=2)
            
            fig_pies.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig_pies, use_container_width=True)
        
        with col_dist2:
            # Bar chart
            categories = ['Négatif', 'Neutre', 'Positif']
            woy_values = [metriques_woy['neg_pct'], metriques_woy['neu_pct'], metriques_woy['pos_pct']]
            post_values = [metriques_post['neg_pct'], metriques_post['neu_pct'], metriques_post['pos_pct']]
            
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
                name='Postpayé',
                x=categories,
                y=post_values,
                marker_color='#3498db',
                text=[f'{v:.1f}%' for v in post_values],
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
        # Radar chart
        col_rad1, col_rad2 = st.columns(2)
        
        with col_rad1:
            categories_radar = ['Volume', 'Négativité', 'Positivité', 'Criticité']
            
            # Normaliser les valeurs
            max_volume = max(metriques_woy['total'], metriques_post['total'])
            max_criticite = max(metriques_woy['criticite'], metriques_post['criticite'])
            
            woy_radar = [
                metriques_woy['total'] / max_volume if max_volume > 0 else 0,
                metriques_woy['neg_pct'] / 100,
                metriques_woy['pos_pct'] / 100,
                metriques_woy['criticite'] / max_criticite if max_criticite > 0 else 0
            ]
            
            post_radar = [
                metriques_post['total'] / max_volume if max_volume > 0 else 0,
                metriques_post['neg_pct'] / 100,
                metriques_post['pos_pct'] / 100,
                metriques_post['criticite'] / max_criticite if max_criticite > 0 else 0
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
                r=post_radar,
                theta=categories_radar,
                fill='toself',
                name='Postpayé',
                marker_color='#3498db'
            ))
            
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title="Comparaison Multidimensionnelle",
                height=500
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
        
        with col_rad2:
            # Tableau comparatif
            comparaison_data = {
                'Indicateur': ['Total publications', 'Négatif', 'Neutre', 'Positif', 
                              '% Négatif', '% Positif', 'Ratio N/P', 'Indice Satisfaction', 'Criticité'],
                'Woyofal': [
                    f"{metriques_woy['total']:,}",
                    f"{metriques_woy['neg']:,}",
                    f"{metriques_woy['neu']:,}",
                    f"{metriques_woy['pos']:,}",
                    f"{metriques_woy['neg_pct']:.1f}%",
                    f"{metriques_woy['pos_pct']:.1f}%",
                    f"{metriques_woy['ratio_neg_pos']:.2f}",
                    f"{metriques_woy['indice_satisfaction']:.1f}",
                    f"{metriques_woy['criticite']:.0f}"
                ],
                'Postpayé': [
                    f"{metriques_post['total']:,}",
                    f"{metriques_post['neg']:,}",
                    f"{metriques_post['neu']:,}",
                    f"{metriques_post['pos']:,}",
                    f"{metriques_post['neg_pct']:.1f}%",
                    f"{metriques_post['pos_pct']:.1f}%",
                    f"{metriques_post['ratio_neg_pos']:.2f}",
                    f"{metriques_post['indice_satisfaction']:.1f}",
                    f"{metriques_post['criticite']:.0f}"
                ]
            }
            
            df_comp = pd.DataFrame(comparaison_data)
            st.dataframe(df_comp, use_container_width=True, hide_index=True)
    
    with tab3:
        # Analyse détaillée
        col_det1, col_det2 = st.columns(2)
        
        with col_det1:
            st.markdown("### 💳 Top mots Woyofal")
            if text_column:
                all_text_woy = ' '.join(df_woyofal[text_column].dropna().astype(str).str.lower())
                mots_woy_importants = ['code', 'recharge', 'compteur', 'crédit', 'tranche', 
                                       'senelec', 'woyofal', 'problème', 'marche', 'bloqué']
                
                freq_woy = {}
                for mot in mots_woy_importants:
                    freq_woy[mot] = all_text_woy.count(mot)
                
                df_freq_woy = pd.DataFrame(list(freq_woy.items()), columns=['Mot', 'Fréquence']).sort_values('Fréquence', ascending=False)
                st.dataframe(df_freq_woy, use_container_width=True, hide_index=True)
        
        with col_det2:
            st.markdown("### 📋 Top mots Postpayé")
            if text_column:
                all_text_post = ' '.join(df_postpaye[text_column].dropna().astype(str).str.lower())
                mots_post_importants = ['facture', 'paiement', 'mensuel', 'compteur', 'tarif',
                                        'cher', 'retard', 'agence', 'réclamation', 'service']
                
                freq_post = {}
                for mot in mots_post_importants:
                    freq_post[mot] = all_text_post.count(mot)
                
                df_freq_post = pd.DataFrame(list(freq_post.items()), columns=['Mot', 'Fréquence']).sort_values('Fréquence', ascending=False)
                st.dataframe(df_freq_post, use_container_width=True, hide_index=True)
        
        # Test statistique
        st.markdown("### 📊 Test Statistique Chi²")
        
        woy_neg = metriques_woy['neg']
        woy_non_neg = metriques_woy['total'] - woy_neg
        post_neg = metriques_post['neg']
        post_non_neg = metriques_post['total'] - post_neg
        
        contingency = pd.DataFrame([
            [woy_neg, woy_non_neg],
            [post_neg, post_non_neg]
        ], index=['Woyofal', 'Postpayé'], columns=['Négatif', 'Non Négatif'])
        
        chi2, p_value, dof, _ = chi2_contingency(contingency)
        
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        
        with col_stat1:
            st.metric("📊 Chi²", f"{chi2:.3f}")
        with col_stat2:
            st.metric("🎯 p-value", f"{p_value:.6f}")
        with col_stat3:
            if p_value < 0.001:
                st.error("🔴 Différence HAUTEMENT SIGNIFICATIVE")
            elif p_value < 0.05:
                st.warning("🟠 Différence SIGNIFICATIVE")
            else:
                st.success("🟢 Différence NON significative")
    
    with tab4:
        # Évolution temporelle 
        st.markdown("### 📈 Évolution du Sentiment Négatif")
        
        # Préparer les données temporelles avec la fonction sécurisée
        df_woy_temp = preparer_donnees_temporelles(df_woyofal)
        df_post_temp = preparer_donnees_temporelles(df_postpaye)
        
        if df_woy_temp is not None and df_post_temp is not None and len(df_woy_temp) > 0 and len(df_post_temp) > 0:
            # Calculer % négatif par mois
            woy_monthly = df_woy_temp.groupby('mois').apply(
                lambda x: (x['sentiment_pred'].astype(str).str.lower() == 'negative').mean() * 100
            ).reset_index(name='woyofal_neg')
            
            post_monthly = df_post_temp.groupby('mois').apply(
                lambda x: (x['sentiment_pred'].astype(str).str.lower() == 'negative').mean() * 100
            ).reset_index(name='postpaye_neg')
            
            # Fusionner
            timeline_df = pd.merge(woy_monthly, post_monthly, on='mois', how='outer').fillna(0)
            
            if len(timeline_df) > 0:
                fig_evol = go.Figure()
                fig_evol.add_trace(go.Scatter(
                    x=timeline_df['mois'],
                    y=timeline_df['woyofal_neg'],
                    name='💳 Woyofal',
                    mode='lines+markers',
                    line=dict(color='#e74c3c', width=3),
                    marker=dict(size=8)
                ))
                fig_evol.add_trace(go.Scatter(
                    x=timeline_df['mois'],
                    y=timeline_df['postpaye_neg'],
                    name='📋 Postpayé',
                    mode='lines+markers',
                    line=dict(color='#3498db', width=3),
                    marker=dict(size=8)
                ))
                
                fig_evol.update_layout(
                    height=500,
                    xaxis_title="Mois",
                    yaxis_title="% Sentiment Négatif",
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig_evol, use_container_width=True)
                
                # Statistiques temporelles
                col_temp1, col_temp2, col_temp3 = st.columns(3)
                
                with col_temp1:
                    st.metric("📅 Première date", df_woy_temp['date_publication'].min().strftime('%d/%m/%Y'))
                with col_temp2:
                    st.metric("📅 Dernière date", df_woy_temp['date_publication'].max().strftime('%d/%m/%Y'))
                with col_temp3:
                    duree = (df_woy_temp['date_publication'].max() - df_woy_temp['date_publication'].min()).days
                    st.metric("⏱️ Période", f"{duree} jours")
            else:
                st.info("Données temporelles insuffisantes pour l'évolution")
        else:
            st.info("Données temporelles non disponibles pour l'analyse d'évolution")
    
    st.markdown("---")
    
    # ================================================
    # ÉTAPE 5: RECOMMANDATIONS
    # ================================================
    
    st.markdown("## 💡 Recommandations Stratégiques")
    
    col_rec1, col_rec2 = st.columns(2)
    
    with col_rec1:
        st.markdown("### 🎯 Actions pour Woyofal")
        
        if metriques_woy['neg_pct'] > 60:
            st.error("""
            **🔴 CRITIQUE**
            
            1. Audit technique immédiat
            2. Hotline dédiée 24/7
            3. Correction des bugs de codes
            4. Communication de crise
            """)
        elif metriques_woy['neg_pct'] > 45:
            st.warning("""
            **🟠 URGENT**
            
            1. Amélioration de l'application mobile
            2. Formation des agents
            3. Campagne d'information
            4. SMS d'alerte avant épuisement
            """)
        else:
            st.success("""
            **🟢 SOUS CONTRÔLE**
            
            1. Monitoring continu
            2. Optimisations UX
            3. Enquêtes satisfaction
            """)
    
    with col_rec2:
        st.markdown("### 📋 Actions pour Postpayé")
        
        if metriques_post['neg_pct'] > 50:
            st.warning("""
            **🟠 ATTENTION**
            
            1. Simplification des factures
            2. Réduction délais de traitement
            3. Transparence tarifaire
            4. Modernisation du service client
            """)
        else:
            st.success("""
            **🟢 STABLE**
            
            1. Capitaliser sur les points forts
            2. Transfert des bonnes pratiques
            3. Harmonisation des services
            """)
    
    st.markdown("---")
    
    # ================================================
    # ÉTAPE 6: EXPORT
    # ================================================
    
    st.markdown("## 💾 Export des Données")
    
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
        if len(df_postpaye) > 0:
            csv_post = df_postpaye.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 Données Postpayé",
                data=csv_post,
                file_name=f"postpaye_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col_exp3:
        # Rapport comparatif
        rapport_data = {
            'Métrique': ['Total', 'Négatif', 'Neutre', 'Positif', '% Négatif'],
            'Woyofal': [metriques_woy['total'], metriques_woy['neg'], metriques_woy['neu'], 
                       metriques_woy['pos'], f"{metriques_woy['neg_pct']:.1f}%"],
            'Postpayé': [metriques_post['total'], metriques_post['neg'], metriques_post['neu'],
                        metriques_post['pos'], f"{metriques_post['neg_pct']:.1f}%"]
        }
        df_rapport = pd.DataFrame(rapport_data)
        csv_rapport = df_rapport.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 Rapport Comparatif",
            data=csv_rapport,
            file_name=f"rapport_woyofal_postpaye_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )