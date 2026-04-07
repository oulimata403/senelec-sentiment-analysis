"""
Page Analyse Thématique - Dashboard SENELEC
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import re
from collections import Counter

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.config import THEMES


def normalize_dates(df, date_column='date_publication'):
    """
    Normalise les dates en supprimant les fuseaux horaires
    """
    if date_column in df.columns:
        # utc=True gère les timezones mixtes (ex: Twitter + Facebook)
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce', utc=True)
        # Supprime le timezone pour éviter les conflits dans les graphiques
        df[date_column] = df[date_column].dt.tz_localize(None)
    return df


def extraire_sous_themes_woyofal(textes):
    """
    Extrait les sous-thèmes spécifiques à Woyofal à partir des textes
    """
    sous_themes = {
        'Codes non reconnus': ['code', 'code non reconnu', 'code incorrect', 'code ne marche pas', 'code invalide', 'mauvais code'],
        'Compteurs bloqués': ['compteur bloqué', 'compteur bloque', 'compteur ne marche', 'compteur HS', 'compteur défectueux', 'compteur bloquer'],
        'Difficultés de recharge': ['recharge', 'achat', 'crédit', 'crédit pas', 'recharge difficile', 'pas pu recharger', 'acheter', 'boutique'],
        'Complexité du système': ['tranche', 'comprendre', 'compliqué', 'complexe', 'pas clair', 'explication', 'comment ça marche', 'kwh'],
        'Problèmes techniques': ['bug', 'panne', 'dysfonctionnement', 'technique', 'système ne marche', 'problème technique'],
        'Service client Woyofal': ['hotline', 'appeler', 'assistance', 'service client woyofal', 'numéro vert', 'appel', 'standard'],
        'Facturation Woyofal': ['facture', 'tarif', 'prix', 'cher', 'coût', 'débit', 'consommation', 'argent'],
        'Communication': ['info', 'information', 'sensibilisation', 'campagne', 'ignorance', 'méconnaissance', 'savoir']
    }
    
    resultats = {theme: 0 for theme in sous_themes.keys()}
    
    for texte in textes:
        if pd.isna(texte) or texte == '':
            continue
        texte = str(texte).lower()
        for theme, keywords in sous_themes.items():
            for keyword in keywords:
                if keyword in texte:
                    resultats[theme] += 1
                    break
    
    return resultats


def analyser_mots_cles_woyofal(df_woyofal, text_column):
    """
    Analyse les mots-clés les plus fréquents dans les publications Woyofal
    """
    if text_column is None or df_woyofal.empty:
        return []
    
    mots_importants = []
    stopwords = ['le', 'la', 'les', 'du', 'de', 'des', 'un', 'une', 'et', 'est', 'sont', 
                 'ce', 'cette', 'ces', 'pour', 'dans', 'sur', 'par', 'avec', 'avoir', 
                 'être', 'faire', 'plus', 'très', 'tout', 'comme', 'mais', 'ou', 'où',
                 'qui', 'que', 'quoi', 'dont', 'donc', 'car', 'ni', 'ne', 'pas', 'plus',
                 'mon', 'ma', 'mes', 'ton', 'ta', 'tes', 'son', 'sa', 'ses', 'notre',
                 'votre', 'leur', 'leurs', 'au', 'aux', 'chez']
    
    for texte in df_woyofal[text_column].dropna():
        if pd.isna(texte):
            continue
        # Nettoyage basique
        mots = re.findall(r'\b[a-zA-ZÀ-ÿ]{3,}\b', str(texte).lower())
        mots = [m for m in mots if m not in stopwords and len(m) > 2]
        mots_importants.extend(mots)
    
    counter = Counter(mots_importants)
    return counter.most_common(15)


def show_thematique(df):
    """
    Affiche l'analyse thématique - Version ULTIME CORRIGÉE
    """
    
    if df is None:
        st.error("❌ Impossible de charger les données")
        return
    
    if 'theme' not in df.columns:
        st.error("❌ Données thématiques non disponibles - colonne 'theme' manquante")
        st.write("Colonnes disponibles:", list(df.columns))
        return
    
    if 'date_publication' in df.columns:
        df = normalize_dates(df.copy())
    
    # HEADER
    # st.header("🔍 Analyse Thématique Approfondie")
    st.markdown("Exploration détaillée des thèmes identifiés par Intelligence Artificielle")
    
    st.markdown("---")
    
    # ================================================
    # SECTION 1: ANALYSE GLOBALE DES THÈMES
    # ================================================
    st.markdown("## 📊 Vue d'Ensemble des Thèmes")
    
    # Distribution globale des thèmes
    theme_dist = df['theme'].value_counts().reset_index()
    theme_dist.columns = ['theme', 'count']
    theme_dist['pourcentage'] = (theme_dist['count'] / len(df) * 100).round(1)
    
    # Ajouter les labels
    theme_dist['label'] = theme_dist['theme'].apply(
        lambda x: THEMES.get(x, {}).get('label_fr', x.capitalize())
    )
    
    # Graphique de distribution
    fig_dist = px.bar(
        theme_dist,
        x='count',
        y='label',
        orientation='h',
        text='pourcentage',
        title="Distribution des Thèmes dans le Corpus",
        labels={'count': 'Nombre de publications', 'label': ''},
        color='count',
        color_continuous_scale='Viridis'
    )
    fig_dist.update_traces(texttemplate='%{text}%', textposition='outside')
    fig_dist.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
    
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Tableau récapitulatif
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(
            theme_dist[['label', 'count', 'pourcentage']].rename(
                columns={'label': 'Thème', 'count': 'Publications', 'pourcentage': '% du Corpus'}
            ).style.format({'% du Corpus': '{:.1f}%'}),
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        theme_dominant = theme_dist.iloc[0]
        st.success(f"""
        ### 🏆 Thème dominant
        **{theme_dominant['label']}**
        
        {theme_dominant['count']:,} publications
        {theme_dominant['pourcentage']}% du corpus
        """)
    
    st.markdown("---")
    
    # ================================================
    # SECTION 2: ANALYSE SPÉCIFIQUE DE WOYOFAL
    # ================================================
    
    # Filtrer les données Woyofal
    df_woyofal = df[df['theme'] == 'woyofal'].copy()
    total_woyofal = len(df_woyofal)
    
    st.markdown("## ⚡ Analyse Approfondie du Système Woyofal")
    st.markdown(f"**{total_woyofal:,} publications** analysées ({total_woyofal/len(df)*100:.1f}% du corpus)")
    
    if total_woyofal > 0:
        # Métriques de sentiment pour Woyofal
        if 'sentiment_pred' in df_woyofal.columns:
            df_woyofal['sentiment_pred'] = df_woyofal['sentiment_pred'].astype(str).str.lower()
            
            woy_neg = (df_woyofal['sentiment_pred'] == 'negative').sum()
            woy_neu = (df_woyofal['sentiment_pred'] == 'neutral').sum()
            woy_pos = (df_woyofal['sentiment_pred'] == 'positive').sum()
            
            woy_neg_pct = (woy_neg / total_woyofal * 100) if total_woyofal > 0 else 0
            woy_neu_pct = (woy_neu / total_woyofal * 100) if total_woyofal > 0 else 0
            woy_pos_pct = (woy_pos / total_woyofal * 100) if total_woyofal > 0 else 0
        else:
            woy_neg = woy_neu = woy_pos = 0
            woy_neg_pct = woy_neu_pct = woy_pos_pct = 0
        
        # KPIs Woyofal
        col_k1, col_k2, col_k3, col_k4 = st.columns(4)
        with col_k1:
            st.metric("📊 Total Publications", f"{total_woyofal:,}")
        with col_k2:
            st.metric("😡 Négatif", f"{woy_neg:,}", delta=f"{woy_neg_pct:.1f}%", delta_color="inverse")
        with col_k3:
            st.metric("😐 Neutre", f"{woy_neu:,}", delta=f"{woy_neu_pct:.1f}%")
        with col_k4:
            st.metric("😊 Positif", f"{woy_pos:,}", delta=f"{woy_pos_pct:.1f}%")
        
        st.markdown("---")
        
        # ============================================
        # SOUS-THÈMES WOYOFAL
        # ============================================
        st.markdown("### 🔍 Sous-Thèmes Identifiés dans Woyofal")
        
        # Trouver la colonne de texte
        text_column = None
        for col in ['texte_nettoye', 'texte', 'content', 'message', 'text']:
            if col in df_woyofal.columns:
                text_column = col
                break
        
        if text_column:
            # Extraire les sous-thèmes
            sous_themes = extraire_sous_themes_woyofal(df_woyofal[text_column])
            
            # Créer DataFrame pour visualisation
            df_sous_themes = pd.DataFrame([
                {'Sous-thème': theme, 'Mentions': count}
                for theme, count in sous_themes.items() if count > 0
            ]).sort_values('Mentions', ascending=False)
            
            total_mentions = df_sous_themes['Mentions'].sum()
            df_sous_themes['Pourcentage'] = (df_sous_themes['Mentions'] / total_mentions * 100).round(1) if total_mentions > 0 else 0
            
            if not df_sous_themes.empty:
                # Graphique des sous-thèmes
                fig_sous = px.bar(
                    df_sous_themes,
                    x='Mentions',
                    y='Sous-thème',
                    orientation='h',
                    text='Pourcentage',
                    title="Sous-thèmes Spécifiques à Woyofal",
                    color='Mentions',
                    color_continuous_scale='Reds'
                )
                fig_sous.update_traces(texttemplate='%{text}%', textposition='outside')
                fig_sous.update_layout(height=450)
                
                st.plotly_chart(fig_sous, use_container_width=True)
                
                # Tableau détaillé
                st.dataframe(
                    df_sous_themes,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Analyse des sous-thèmes critiques
                st.markdown("#### 📌 Analyse des Points Critiques")
                
                top_critiques = df_sous_themes.head(3)
                cols_crit = st.columns(3)
                
                for i, (_, row) in enumerate(top_critiques.iterrows()):
                    with cols_crit[i]:
                        severity = "🔴" if i == 0 else "🟠" if i == 1 else "🟡"
                        st.error(f"""
                        {severity} **{row['Sous-thème']}**
                        
                        {row['Mentions']} mentions
                        {row['Pourcentage']}% des cas
                        """)
            else:
                st.info("Aucun sous-thème spécifique identifié")
            
            # ============================================
            # MOTS-CLÉS FRÉQUENTS
            # ============================================
            st.markdown("### 🔑 Mots-Clés les Plus Fréquents")
            
            mots_cles = analyser_mots_cles_woyofal(df_woyofal, text_column)
            
            if mots_cles:
                df_mots = pd.DataFrame(mots_cles, columns=['Mot', 'Fréquence'])
                
                fig_mots = px.bar(
                    df_mots.head(10),
                    x='Fréquence',
                    y='Mot',
                    orientation='h',
                    title="Top 10 des Mots-Clés dans les Publications Woyofal",
                    color='Fréquence',
                    color_continuous_scale='Viridis'
                )
                fig_mots.update_layout(height=400)
                
                st.plotly_chart(fig_mots, use_container_width=True)
        else:
            st.warning("⚠️ Aucune colonne de texte trouvée pour l'analyse des sous-thèmes")
        
        st.markdown("---")
        
        # ============================================
        # ANALYSE TEMPORELLE WOYOFAL
        # ============================================
        if 'date_publication' in df_woyofal.columns:
            st.markdown("### 📈 Évolution Temporelle de Woyofal")
            
            # Normaliser les dates avec utc=True
            df_woyofal['date_publication'] = pd.to_datetime(
                df_woyofal['date_publication'], errors='coerce', utc=True
            ).dt.tz_localize(None)
            
            df_woyofal_dated = df_woyofal.dropna(subset=['date_publication']).copy()
            
            if len(df_woyofal_dated) > 0:
                # Agrégation par mois
                df_woyofal_dated['mois'] = df_woyofal_dated['date_publication'].dt.to_period('M').astype(str)
                
                timeline_woy = df_woyofal_dated.groupby(['mois', 'sentiment_pred']).size().reset_index(name='count')
                timeline_pivot = timeline_woy.pivot(index='mois', columns='sentiment_pred', values='count').fillna(0)
                
                # S'assurer que toutes les colonnes existent
                for sentiment in ['negative', 'neutral', 'positive']:
                    if sentiment not in timeline_pivot.columns:
                        timeline_pivot[sentiment] = 0
                
                timeline_pivot = timeline_pivot[['negative', 'neutral', 'positive']]
                
                fig_timeline_woy = go.Figure()
                
                colors = {'negative': '#e74c3c', 'neutral': '#95a5a6', 'positive': '#27ae60'}
                names = {'negative': 'Négatif', 'neutral': 'Neutre', 'positive': 'Positif'}
                
                for sentiment in ['negative', 'neutral', 'positive']:
                    fig_timeline_woy.add_trace(go.Scatter(
                        x=timeline_pivot.index,
                        y=timeline_pivot[sentiment],
                        name=names[sentiment],
                        mode='lines+markers',
                        line=dict(color=colors[sentiment], width=3),
                        marker=dict(size=8)
                    ))
                
                fig_timeline_woy.update_layout(
                    title="Évolution des Sentiments Woyofal dans le Temps",
                    xaxis_title="Mois",
                    yaxis_title="Nombre de Publications",
                    height=450,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_timeline_woy, use_container_width=True)
    
    st.markdown("---")
    
    # ================================================
    # SECTION 3: COMPARAISON AVEC AUTRES THÈMES
    # ================================================
    st.markdown("## ⚖️ Comparaison avec les Autres Thèmes")
    
    themes_list = df['theme'].unique().tolist()
    
    if len(themes_list) > 1:
        # Calculer les statistiques par thème
        theme_stats = []
        for theme in themes_list:
            df_t = df[df['theme'] == theme]
            total_t = len(df_t)
            
            if 'sentiment_pred' in df_t.columns:
                df_t['sentiment_pred'] = df_t['sentiment_pred'].astype(str).str.lower()
                neg_t = (df_t['sentiment_pred'] == 'negative').sum()
                pos_t = (df_t['sentiment_pred'] == 'positive').sum()
                neg_pct_t = (neg_t / total_t * 100) if total_t > 0 else 0
                pos_pct_t = (pos_t / total_t * 100) if total_t > 0 else 0
            else:
                neg_t = pos_t = 0
                neg_pct_t = pos_pct_t = 0
            
            label = THEMES.get(theme, {}).get('label_fr', theme.capitalize())
            
            theme_stats.append({
                'Thème': label,
                'Code': theme,
                'Total': total_t,
                '% du Corpus': (total_t / len(df) * 100) if len(df) > 0 else 0,
                'Négatif': neg_t,
                '% Négatif': neg_pct_t,
                'Positif': pos_t,
                '% Positif': pos_pct_t,
                'Criticité': round(neg_pct_t * (total_t / len(df)), 1)
            })
        
        df_stats = pd.DataFrame(theme_stats).sort_values('Total', ascending=False)
        
        # Graphique comparatif
        fig_compare = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Pourcentage de Négativité", "Volume de Publications"),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Pourcentage de négativité
        fig_compare.add_trace(
            go.Bar(
                x=df_stats['Thème'],
                y=df_stats['% Négatif'],
                name='% Négatif',
                marker_color='#e74c3c',
                text=df_stats['% Négatif'].round(1).astype(str) + '%',
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # Volume
        fig_compare.add_trace(
            go.Bar(
                x=df_stats['Thème'],
                y=df_stats['Total'],
                name='Volume',
                marker_color='#3498db',
                text=df_stats['Total'],
                textposition='outside'
            ),
            row=1, col=2
        )
        
        fig_compare.update_layout(
            title="Comparaison Thématique: Négativité vs Volume",
            height=450,
            showlegend=False,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_compare, use_container_width=True)
        
        # Tableau comparatif
        st.dataframe(
            df_stats[['Thème', 'Total', '% du Corpus', '% Négatif', '% Positif', 'Criticité']].style.format({
                'Total': '{:,.0f}',
                '% du Corpus': '{:.1f}%',
                '% Négatif': '{:.1f}%',
                '% Positif': '{:.1f}%',
                'Criticité': '{:.1f}'
            }).background_gradient(subset=['Criticité'], cmap='Reds'),
            use_container_width=True,
            hide_index=True
        )
        
        # Sauvegarder df_stats pour l'export
        stats_comparatives = df_stats.copy()
    else:
        st.info("Un seul thème disponible dans les données")
        stats_comparatives = None
    
    st.markdown("---")
    
    # ================================================
    # SECTION 4: RECOMMANDATIONS STRATÉGIQUES
    # ================================================
    st.markdown("## 💡 Recommandations Stratégiques")
    
    col_r1, col_r2 = st.columns(2)
    
    with col_r1:
        st.markdown("""
        ### 🎯 Actions Prioritaires pour Woyofal
        
        **1. Résoudre les problèmes techniques critiques**
        - ✅ Correction des bugs de reconnaissance de codes (problème #1)
        - ✅ Mise à jour des compteurs bloqués/défectueux
        - ✅ Système d'alerte automatique en cas de dysfonctionnement
        
        **2. Simplifier l'expérience utilisateur**
        - 📱 Application mobile dédiée avec interface intuitive
        - 🔔 Notifications SMS avant épuisement du crédit
        - 📞 Hotline Woyofal 24h/24 et 7j/7
        
        **3. Améliorer la communication**
        - 📢 Campagne d'information sur les tranches tarifaires
        - 📚 Tutoriels vidéo en français et wolof
        - 💬 FAQ interactive sur le site web
        """)
    
    with col_r2:
        st.markdown("""
        ### 🚀 Actions à Moyen Terme
        
        **1. Modernisation du service client**
        - ⏱️ Réduction du temps d'attente à moins de 10 minutes
        - 🎓 Formation spécifique des agents sur Woyofal
        - 📊 Suivi en temps réel des réclamations
        
        **2. Approche régionalisée**
        - 🗺️ Plans spécifiques pour Ziguinchor, Fatick, Saint-Louis
        - 🔧 Renforcement des équipes techniques locales
        - 🤝 Comités régionaux d'usagers
        
        **3. Veille citoyenne institutionnalisée**
        - 📈 Intégration du tableau de bord dans les processus décisionnels
        - 🔍 Cellule dédiée à l'analyse des données citoyennes
        - 📊 Publication trimestrielle d'un baromètre de satisfaction
        """)
    
    st.markdown("---")
    
    # ================================================
    # SECTION 5: PUBLICATIONS REPRÉSENTATIVES
    # ================================================
    with st.expander("📰 Exemples de Publications par Thème"):
        theme_example = st.selectbox(
            "Choisir un thème pour voir des exemples:",
            themes_list,
            format_func=lambda x: THEMES.get(x, {}).get('label_fr', x.capitalize()),
            key="theme_example"
        )
        
        df_example = df[df['theme'] == theme_example].copy()
        
        if not df_example.empty:
            # Trouver la colonne de texte
            text_col_example = None
            for col in ['texte_nettoye', 'texte', 'content', 'message', 'text']:
                if col in df_example.columns:
                    text_col_example = col
                    break
            
            if text_col_example:
                # Exemples négatifs
                st.markdown(f"#### 😡 Exemples Négatifs - {THEMES.get(theme_example, {}).get('label_fr', theme_example.capitalize())}")
                df_neg_ex = df_example[df_example['sentiment_pred'].astype(str).str.lower() == 'negative']
                
                if len(df_neg_ex) > 0:
                    for idx, row in df_neg_ex.head(3).iterrows():
                        texte = str(row.get(text_col_example, ''))
                        if pd.isna(texte) or texte == '' or texte == 'nan':
                            texte = "[Texte non disponible]"
                        
                        date_info = ""
                        if 'date_publication' in row and pd.notna(row['date_publication']):
                            date_info = f" *({pd.to_datetime(row['date_publication']).strftime('%d/%m/%Y')})*"
                        
                        st.markdown(f"""
                        **😡{date_info}**
                        > {texte[:200]}{'...' if len(texte) > 200 else ''}
                        """)
                        st.markdown("---")
                else:
                    st.info("Aucun exemple négatif pour ce thème")
                
                # Exemples positifs
                st.markdown(f"#### 😊 Exemples Positifs - {THEMES.get(theme_example, {}).get('label_fr', theme_example.capitalize())}")
                df_pos_ex = df_example[df_example['sentiment_pred'].astype(str).str.lower() == 'positive']
                
                if len(df_pos_ex) > 0:
                    for idx, row in df_pos_ex.head(3).iterrows():
                        texte = str(row.get(text_col_example, ''))
                        if pd.isna(texte) or texte == '' or texte == 'nan':
                            texte = "[Texte non disponible]"
                        
                        date_info = ""
                        if 'date_publication' in row and pd.notna(row['date_publication']):
                            date_info = f" *({pd.to_datetime(row['date_publication']).strftime('%d/%m/%Y')})*"
                        
                        st.markdown(f"""
                        **😊{date_info}**
                        > {texte[:200]}{'...' if len(texte) > 200 else ''}
                        """)
                        st.markdown("---")
                else:
                    st.info("Aucun exemple positif pour ce thème")
    
    st.markdown("---")
    
    # ================================================
    # SECTION 6: EXPORT DES DONNÉES
    # ================================================
    st.markdown("## 💾 Export des Données")
    
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    
    with col_exp1:
        # Export des données Woyofal
        if 'df_woyofal' in locals() and not df_woyofal.empty:
            csv_woyofal = df_woyofal.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 Données Woyofal",
                data=csv_woyofal,
                file_name=f"woyofal_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col_exp2:
        # Export des statistiques comparatives
        if 'stats_comparatives' in locals() and stats_comparatives is not None:
            csv_stats = stats_comparatives.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 Statistiques comparatives",
                data=csv_stats,
                file_name=f"senelec_stats_comparatives_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col_exp3:
        # Export des sous-thèmes Woyofal
        if 'df_sous_themes' in locals() and not df_sous_themes.empty:
            csv_sous = df_sous_themes.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 Sous-thèmes Woyofal",
                data=csv_sous,
                file_name=f"sous_themes_woyofal_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )