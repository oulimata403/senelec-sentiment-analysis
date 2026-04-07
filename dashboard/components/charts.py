"""
Composant Charts - Dashboard SENELEC
Création de graphiques Plotly réutilisables
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def create_sentiment_pie_chart(df):
    """
    Crée un graphique en camembert pour la distribution des sentiments
    
    Args:
        df: DataFrame avec colonne 'sentiment_pred'
    
    Returns:
        Figure Plotly
    """
    
    if df is None or 'sentiment_pred' not in df.columns:
        return go.Figure()
    
    sentiment_counts = df['sentiment_pred'].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=['Négatif', 'Neutre', 'Positif'],
        values=[
            sentiment_counts.get('negative', 0),
            sentiment_counts.get('neutral', 0),
            sentiment_counts.get('positive', 0)
        ],
        marker=dict(colors=['#e74c3c', '#95a5a6', '#27ae60']),
        hole=0.4,
        textinfo='label+percent',
        textfont_size=13,
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title="Distribution des Sentiments",
        showlegend=True,
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def create_platform_bar_chart(df):
    """
    Crée un graphique en barres pour les publications par plateforme
    
    Args:
        df: DataFrame avec colonne 'plateforme'
    
    Returns:
        Figure Plotly
    """
    
    if df is None or 'plateforme' not in df.columns:
        return go.Figure()
    
    platform_counts = df['plateforme'].value_counts()
    
    colors_map = {
        'facebook': '#3b5998',
        'twitter': '#1da1f2',
        'enquete': '#f39c12'
    }
    
    colors = [colors_map.get(pf.lower(), '#95a5a6') for pf in platform_counts.index]
    
    fig = go.Figure(data=[go.Bar(
        x=platform_counts.index,
        y=platform_counts.values,
        marker=dict(
            color=colors,
            line=dict(color='white', width=2)
        ),
        text=platform_counts.values,
        textposition='outside',
        textfont=dict(size=13, color='black'),
        hovertemplate='<b>%{x}</b><br>Publications: %{y}<extra></extra>'
    )])
    
    fig.update_layout(
        title="Publications par Plateforme",
        xaxis_title="Plateforme",
        yaxis_title="Nombre de Publications",
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    fig.update_xaxes(tickangle=0, tickfont=dict(size=12))
    fig.update_yaxes(gridcolor='rgba(0,0,0,0.1)')
    
    return fig


def create_timeline_chart(df):
    """
    Crée un graphique temporel de l'évolution des sentiments
    
    Args:
        df: DataFrame avec colonnes 'date_publication' et 'sentiment_pred'
    
    Returns:
        Figure Plotly
    """
    
    if df is None or 'date_publication' not in df.columns or 'sentiment_pred' not in df.columns:
        return go.Figure()
    
    # Préparer données
    df_temp = df.copy()
    df_temp['date_publication'] = pd.to_datetime(df_temp['date_publication'], errors='coerce')
    df_temp = df_temp.dropna(subset=['date_publication'])
    
    if len(df_temp) == 0:
        return go.Figure()
    
    # Agrégation hebdomadaire
    df_temp['semaine'] = df_temp['date_publication'].dt.to_period('W')
    
    timeline_data = df_temp.groupby(['semaine', 'sentiment_pred']).size().unstack(fill_value=0)
    timeline_data.index = timeline_data.index.to_timestamp()
    
    # Créer graphique
    fig = go.Figure()
    
    colors = {
        'negative': '#e74c3c',
        'neutral': '#95a5a6',
        'positive': '#27ae60'
    }
    
    for sentiment in ['negative', 'neutral', 'positive']:
        if sentiment in timeline_data.columns:
            fig.add_trace(go.Scatter(
                x=timeline_data.index,
                y=timeline_data[sentiment],
                name=sentiment.capitalize(),
                mode='lines+markers',
                line=dict(color=colors[sentiment], width=3),
                marker=dict(size=8),
                hovertemplate='<b>%{fullData.name}</b><br>Date: %{x|%d/%m/%Y}<br>Count: %{y}<extra></extra>'
            ))
    
    fig.update_layout(
        title="Évolution Temporelle des Sentiments",
        xaxis_title="Date",
        yaxis_title="Nombre de Publications",
        height=450,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    fig.update_xaxes(gridcolor='rgba(0,0,0,0.1)')
    fig.update_yaxes(gridcolor='rgba(0,0,0,0.1)')
    
    return fig


def create_theme_heatmap(df):
    """
    Crée une heatmap thèmes × sentiments
    
    Args:
        df: DataFrame avec colonnes 'theme' et 'sentiment_pred'
    
    Returns:
        Figure Plotly
    """
    
    if df is None or 'theme' not in df.columns or 'sentiment_pred' not in df.columns:
        return go.Figure()
    
    crosstab = pd.crosstab(df['theme'], df['sentiment_pred'])
    
    fig = go.Figure(data=go.Heatmap(
        z=crosstab.values,
        x=['Négatif', 'Neutre', 'Positif'],
        y=crosstab.index,
        colorscale='RdYlGn_r',
        text=crosstab.values,
        texttemplate='%{text}',
        textfont={"size": 11},
        colorbar=dict(title="Nombre")
    ))
    
    fig.update_layout(
        title="Heatmap Thèmes × Sentiments",
        xaxis_title="Sentiment",
        yaxis_title="Thème",
        height=500,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def create_stacked_bar_chart(df, group_by, stack_by, title="Distribution"):
    """
    Crée un graphique en barres empilées générique
    
    Args:
        df: DataFrame
        group_by: Colonne pour grouper (axe X)
        stack_by: Colonne pour empiler
        title: Titre du graphique
    
    Returns:
        Figure Plotly
    """
    
    if df is None or group_by not in df.columns or stack_by not in df.columns:
        return go.Figure()
    
    crosstab = pd.crosstab(df[group_by], df[stack_by])
    
    fig = go.Figure()
    
    colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe']
    
    for i, col in enumerate(crosstab.columns):
        fig.add_trace(go.Bar(
            name=str(col),
            x=crosstab.index,
            y=crosstab[col],
            marker_color=colors[i % len(colors)]
        ))
    
    fig.update_layout(
        title=title,
        barmode='stack',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig