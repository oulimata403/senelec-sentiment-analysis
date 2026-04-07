"""
Composant KPI Cards - Dashboard SENELEC
Affichage des indicateurs clés de performance
"""

import streamlit as st
import pandas as pd


def show_kpi_cards(df):
    """
    Affiche les cartes KPI principales
    
    Args:
        df: DataFrame avec colonnes 'sentiment_pred', 'theme', 'plateforme'
    """
    
    if df is None or len(df) == 0:
        st.warning("Aucune donnée à afficher")
        return
    
    # Calculer métriques
    total = len(df)
    
    neg_count = (df['sentiment_pred'] == 'negative').sum()
    neu_count = (df['sentiment_pred'] == 'neutral').sum()
    pos_count = (df['sentiment_pred'] == 'positive').sum()
    
    neg_pct = (neg_count / total) * 100
    neu_pct = (neu_count / total) * 100
    pos_pct = (pos_count / total) * 100
    
    # Nombre de thèmes
    if 'theme' in df.columns:
        n_themes = df['theme'].nunique()
        theme_dominant = df['theme'].value_counts().index[0] if len(df) > 0 else "N/A"
    else:
        n_themes = 0
        theme_dominant = "N/A"
    
    # Nombre de plateformes
    if 'plateforme' in df.columns:
        n_plateformes = df['plateforme'].nunique()
    else:
        n_plateformes = 0
    
    # Afficher les KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="📊 Total Publications",
            value=f"{total:,}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="😡 Sentiment Négatif",
            value=f"{neg_pct:.1f}%",
            delta=f"{neg_count:,} publications",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="😐 Sentiment Neutre",
            value=f"{neu_pct:.1f}%",
            delta=f"{neu_count:,} publications",
            delta_color="off"
        )
    
    with col4:
        st.metric(
            label="😊 Sentiment Positif",
            value=f"{pos_pct:.1f}%",
            delta=f"{pos_count:,} publications",
            delta_color="normal"
        )
    
    with col5:
        st.metric(
            label="🔍 Thèmes Identifiés",
            value=n_themes,
            delta=f"Dominant: {theme_dominant[:15]}..." if len(str(theme_dominant)) > 15 else theme_dominant
        )


def show_compact_kpis(df, show_count=4):
    """
    Version compacte des KPIs (pour sidebar ou sections réduites)
    
    Args:
        df: DataFrame
        show_count: Nombre de KPIs à afficher
    """
    
    if df is None or len(df) == 0:
        return
    
    total = len(df)
    neg_pct = (df['sentiment_pred'] == 'negative').sum() / total * 100
    pos_pct = (df['sentiment_pred'] == 'positive').sum() / total * 100
    
    cols = st.columns(show_count)
    
    with cols[0]:
        st.metric("Total", f"{total:,}")
    
    if show_count > 1:
        with cols[1]:
            st.metric("Négatif", f"{neg_pct:.0f}%")
    
    if show_count > 2:
        with cols[2]:
            st.metric("Positif", f"{pos_pct:.0f}%")
    
    if show_count > 3 and 'theme' in df.columns:
        with cols[3]:
            st.metric("Thèmes", df['theme'].nunique())


def show_comparison_kpis(df1, df2, label1="Groupe 1", label2="Groupe 2"):
    """
    Affiche des KPIs comparatifs entre deux groupes
    
    Args:
        df1, df2: DataFrames à comparer
        label1, label2: Labels pour chaque groupe
    """
    
    st.markdown(f"### Comparaison : {label1} vs {label2}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"#### {label1}")
        if df1 is not None and len(df1) > 0:
            neg_pct1 = (df1['sentiment_pred'] == 'negative').sum() / len(df1) * 100
            pos_pct1 = (df1['sentiment_pred'] == 'positive').sum() / len(df1) * 100
            
            st.metric("Total", f"{len(df1):,}")
            st.metric("Négatif", f"{neg_pct1:.1f}%", delta_color="inverse")
            st.metric("Positif", f"{pos_pct1:.1f}%")
        else:
            st.info("Pas de données")
    
    with col2:
        st.markdown(f"#### {label2}")
        if df2 is not None and len(df2) > 0:
            neg_pct2 = (df2['sentiment_pred'] == 'negative').sum() / len(df2) * 100
            pos_pct2 = (df2['sentiment_pred'] == 'positive').sum() / len(df2) * 100
            
            st.metric("Total", f"{len(df2):,}")
            st.metric("Négatif", f"{neg_pct2:.1f}%", delta_color="inverse")
            st.metric("Positif", f"{pos_pct2:.1f}%")
            
            # Différence
            if df1 is not None and len(df1) > 0:
                diff_neg = neg_pct2 - neg_pct1
                if abs(diff_neg) > 5:
                    if diff_neg > 0:
                        st.error(f"⚠️ +{diff_neg:.1f} points de négatif")
                    else:
                        st.success(f"✅ {diff_neg:.1f} points de négatif")
        else:
            st.info("Pas de données")