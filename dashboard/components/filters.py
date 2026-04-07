"""
Composant Filters - Dashboard SENELEC
"""

import streamlit as st
import pandas as pd
from datetime import datetime


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


def create_date_filter(df, date_column='date_publication'):
    """
    Crée un filtre de plage de dates
    Retourne None pour signifier "pas de filtre date"
    """
    
    if df is None or date_column not in df.columns:
        return None
    
    # Normaliser les dates
    df = normalize_dates(df.copy(), date_column)
    df_dated = df.dropna(subset=[date_column])
    
    if len(df_dated) == 0:
        st.warning("⚠️ Aucune donnée temporelle disponible")
        return None
    
    date_min = df_dated[date_column].min().date()
    date_max = df_dated[date_column].max().date()
    
    # Afficher les dates disponibles
    st.sidebar.info(f"📅 Données disponibles: {date_min} → {date_max}")
    
    st.markdown("**📅 Filtre temporel**")
    
    use_date_filter = st.checkbox("Activer le filtre date", value=False, 
                                  help="Cochez pour filtrer par période")
    
    if not use_date_filter:
        st.success(f"🌍 Toutes les dates : {date_min} → {date_max}")
        return None
    
    date_range = st.date_input(
        "Sélectionnez la période",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max,
        format="DD/MM/YYYY"
    )
    
    if isinstance(date_range, tuple) and len(date_range) == 2:
        return date_range
    else:
        return (date_min, date_max)


def create_platform_filter(df, platform_column='plateforme'):
    """Filtre pour sélectionner les plateformes"""
    
    if df is None or platform_column not in df.columns:
        return "Toutes"
    
    platforms = ['Toutes'] + sorted(df[platform_column].unique().tolist())
    
    selected = st.selectbox(
        "📱 Plateforme",
        platforms
    )
    
    return selected


def create_sentiment_filter(df, sentiment_column='sentiment_pred'):
    """Filtre multi-sélection pour les sentiments"""
    
    if df is None or sentiment_column not in df.columns:
        return ['negative', 'neutral', 'positive']
    
    # S'assurer que toutes les valeurs sont des strings
    df[sentiment_column] = df[sentiment_column].astype(str)
    
    sentiment_labels = {
        'negative': '😡 Négatif',
        'neutral': '😐 Neutre',
        'positive': '😊 Positif'
    }
    
    selected_labels = st.multiselect(
        "😊 Sentiments",
        list(sentiment_labels.values()),
        default=list(sentiment_labels.values())
    )
    
    # Convertir les labels en valeurs
    reverse_map = {v: k for k, v in sentiment_labels.items()}
    selected_sentiments = [reverse_map[label] for label in selected_labels]
    
    return selected_sentiments


def create_theme_filter(df, theme_column='theme'):
    """
    Filtre pour sélectionner les thèmes
    """
    if df is None or theme_column not in df.columns:
        return []
    
    # Obtenir tous les thèmes uniques
    themes = sorted(df[theme_column].unique().tolist())
    
    # Créer des labels plus lisibles 
    theme_labels = {}
    try:
        from config.config import THEMES
        for t in themes:
            if t in THEMES:
                theme_labels[t] = THEMES[t].get('label_fr', t.capitalize())
            else:
                theme_labels[t] = t.capitalize()
    except:
        theme_labels = {t: t.capitalize() for t in themes}
    
    selected_themes = st.multiselect(
        "🔍 Thèmes",
        themes,
        default=themes,
        format_func=lambda x: theme_labels.get(x, x.capitalize()),
        help="Sélectionnez les thèmes à analyser"
    )
    
    return selected_themes


def create_region_filter(df, region_column='region'):
    """
    Filtre pour sélectionner les régions
    """
    if df is None or region_column not in df.columns:
        return "Toutes"
    
    regions = ['Toutes'] + sorted(df[region_column].dropna().unique().tolist())
    
    selected = st.selectbox(
        "🗺️ Région",
        regions
    )
    
    return selected


def apply_filters(df, platform_filter="Toutes", date_range=None, sentiment_filter=None, themes=None):
    """
    Applique les filtres sur le DataFrame
    """
    if df is None:
        return None
    
    df_filtered = df.copy()
    
    # 1. Normaliser les dates
    if 'date_publication' in df_filtered.columns:
        df_filtered = normalize_dates(df_filtered, 'date_publication')
    
    # 2. Filtre plateforme
    if platform_filter != "Toutes":
        df_filtered = df_filtered[df_filtered['plateforme'] == platform_filter.lower()]
    
    # 3. Filtre sentiments
    if sentiment_filter and len(sentiment_filter) > 0:
        df_filtered = df_filtered[df_filtered['sentiment_pred'].isin(sentiment_filter)]
    
    # 4. Filtre thèmes 
    if themes is not None and len(themes) > 0 and 'theme' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['theme'].isin(themes)]
    
    # 5. Filtre date 
    if date_range is not None and 'date_publication' in df_filtered.columns:
        date_debut = pd.Timestamp(date_range[0])
        date_fin = pd.Timestamp(date_range[1])
        
        mask_date = (
            (df_filtered['date_publication'].notna()) &
            (df_filtered['date_publication'] >= date_debut) &
            (df_filtered['date_publication'] <= date_fin)
        )
        df_filtered = df_filtered[mask_date]
    
    return df_filtered


def show_filter_summary(df_original, df_filtered):
    """Affiche un résumé des filtres appliqués"""
    if df_original is None or df_filtered is None:
        return
    
    pct_kept = (len(df_filtered) / len(df_original)) * 100 if len(df_original) > 0 else 0
    
    st.info(f"""
    📊 **Résultat :** {len(df_filtered):,} publications sur {len(df_original):,} ({pct_kept:.1f}%)
    """)