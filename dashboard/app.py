"""
Dashboard Interactif SENELEC - Application Principale
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# ---------------------------------------------------
# CONFIG PAGE
# ---------------------------------------------------
st.set_page_config(
    page_title="Dashboard SENELEC - Analyse de Sentiment",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.config import EXPORTS_DIR
from utils.logger import setup_logger

logger = setup_logger("dashboard")

# ---------------------------------------------------
# CSS 
# ---------------------------------------------------
def load_custom_css():
    st.markdown("""
    <style>
    /* ====== FOND SOMBRE GLOBAL ====== */
    .stApp {
        background: linear-gradient(135deg, #000a1a 0%, #000f2a 50%, #001a3d 100%);
        color: #f1f5f9;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        padding: 20px;
    }

    /* ====== HEADER SENELEC SUR FOND SOMBRE ====== */
    .header-main {
        background: linear-gradient(135deg, #001833 0%, #0072ce 40%, #f37021 100%);
        padding: 30px 35px;
        border-radius: 25px;
        box-shadow: 0 12px 40px rgba(0,91,170,0.4), 0 0 0 1px rgba(255,255,255,0.1);
        margin-bottom: 35px;
        border: 2px solid rgba(255,255,255,0.15);
        position: relative;
        overflow: hidden;
        text-align: center;
    }

    .header-main::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 70%);
        animation: rotate 25s linear infinite;
    }

    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    .header-title-main {
        color: #ffffff;
        font-size: 3rem;
        font-weight: 900;
        margin: 0;
        text-shadow: 0 3px 8px rgba(0,0,0,0.5);
        position: relative;
        z-index: 2;
        letter-spacing: -1px;
    }

    .header-subtitle-main {
        color: rgba(255,255,255,0.95);
        font-size: 1.4rem;
        margin-top: 10px;
        font-weight: 600;
        position: relative;
        z-index: 2;
    }

    /* ====== SIDEBAR FOND SOMBRE ====== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #000a1a 0%, #0f172a 50%, #020617 100%);
        border-right: 6px solid #f37021;
        border-radius: 0 25px 25px 0;
        padding: 25px;
        box-shadow: 5px 0 30px rgba(243,112,33,0.4);
    }

    section[data-testid="stSidebar"] .stRadio > div > div {
        background: rgba(15,23,42,0.8);
        border-radius: 15px;
        padding: 15px;
        margin: 8px 0;
        border: 2px solid rgba(243,112,33,0.4);
        backdrop-filter: blur(10px);
        transition: all 0.4s ease;
    }

    section[data-testid="stSidebar"] .stRadio > div > div:hover {
        background: rgba(243,112,33,0.15);
        border-color: #f37021;
        transform: translateX(8px);
        box-shadow: 0 8px 25px rgba(243,112,33,0.3);
    }

    section[data-testid="stSidebar"] * {
        color: #f1f5f9 !important;
    }

    /* ====== STATS VERTICALES DANS SIDEBAR ====== */
    .vertical-stat-card {
        background: linear-gradient(145deg, rgba(30,41,59,0.95), rgba(15,23,42,0.98));
        border-radius: 18px;
        padding: 22px;
        margin: 12px 0;
        border: 2px solid rgba(243,112,33,0.4);
        display: flex;
        align-items: center;
        gap: 18px;
        box-shadow: 0 10px 35px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.1);
        transition: all 0.3s ease;
        backdrop-filter: blur(15px);
        position: relative;
        overflow: hidden;
    }

    .vertical-stat-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #f37021, #ff8c42, #f37021);
    }

    .vertical-stat-card:hover {
        transform: translateX(6px) scale(1.02);
        box-shadow: 0 15px 45px rgba(243,112,33,0.35);
        border-color: #f37021;
    }

    .vertical-stat-card.warning {
        border-color: rgba(245,101,101,0.6);
    }

    .vertical-stat-card.warning::before {
        background: linear-gradient(90deg, #ef4444, #f87171, #ef4444);
    }

    .vertical-stat-card.success {
        border-color: rgba(34,197,94,0.6);
    }

    .vertical-stat-card.success::before {
        background: linear-gradient(90deg, #10b981, #34d399, #10b981);
    }

    .stat-icon {
        font-size: 2.2rem;
        padding: 12px;
        background: linear-gradient(135deg, rgba(243,112,33,0.2), rgba(255,140,66,0.2));
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(243,112,33,0.3);
        flex-shrink: 0;
        filter: drop-shadow(0 2px 8px rgba(0,0,0,0.3));
    }

    .vertical-stat-card.warning .stat-icon {
        background: linear-gradient(135deg, rgba(239,68,68,0.25), rgba(248,113,113,0.25));
        box-shadow: 0 4px 15px rgba(239,68,68,0.4);
    }

    .vertical-stat-card.success .stat-icon {
        background: linear-gradient(135deg, rgba(16,185,129,0.25), rgba(52,211,153,0.25));
        box-shadow: 0 4px 15px rgba(16,185,129,0.4);
    }

    .stat-metric {
        flex: 1;
        min-width: 0;
    }

    .stat-value {
        font-size: 2.2rem;
        font-weight: 900;
        color: #f37021;
        margin-bottom: 4px;
        text-shadow: 0 2px 8px rgba(243,112,33,0.5);
        background: linear-gradient(45deg, #f37021, #ff8c42);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.1;
    }

    .vertical-stat-card.warning .stat-value {
        background: linear-gradient(45deg, #ef4444, #f87171);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .vertical-stat-card.success .stat-value {
        background: linear-gradient(45deg, #10b981, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .stat-label {
        color: #e2e8f0;
        font-weight: 700;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        opacity: 0.9;
    }

    /* ====== CARTES KPI SUR FOND SOMBRE ====== */
    .kpi-container {
        display: flex;
        gap: 25px;
        margin-bottom: 35px;
    }

    .kpi-card {
        flex: 1;
        background: linear-gradient(145deg, rgba(30,41,59,0.9) 0%, rgba(15,23,42,0.95) 100%);
        padding: 30px;
        border-radius: 25px;
        border-left: 8px solid #f37021;
        box-shadow: 0 15px 50px rgba(243,112,33,0.25), 0 5px 20px rgba(0,0,0,0.4);
        margin-bottom: 25px;
        transition: all 0.4s ease;
        border: 1px solid rgba(243,112,33,0.3);
        backdrop-filter: blur(15px);
        position: relative;
        overflow: hidden;
    }

    .kpi-card::before {
        content: '⚡';
        position: absolute;
        top: 20px;
        right: 25px;
        font-size: 2.5rem;
        opacity: 0.08;
        filter: drop-shadow(0 0 10px #f37021);
    }

    .kpi-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 25px 60px rgba(243,112,33,0.35), 0 0 30px rgba(243,112,33,0.2);
    }

    /* ====== MÉTRIQUES SUR FOND SOMBRE ====== */
    [data-testid="stMetricValue"] {
        font-size: 3.2rem;
        font-weight: 900;
        color: #f37021;
        margin-bottom: 8px;
        text-shadow: 0 2px 10px rgba(243,112,33,0.4);
        background: linear-gradient(45deg, #f37021, #ff8c42);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    [data-testid="stMetricLabel"] {
        color: #cbd5e1;
        font-weight: 700;
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ====== TITRES SUR FOND SOMBRE ====== */
    h1, h2, h3, h4 {
        color: #f8fafc;
        font-weight: 900;
        text-shadow: 0 2px 10px rgba(0,0,0,0.5);
    }

    h2 {
        border-bottom: 5px solid transparent;
        background: linear-gradient(90deg, #f37021, #ff8c42);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        padding-bottom: 15px;
        margin-bottom: 30px;
        position: relative;
        font-size: 2.2rem;
    }

    h2::after {
        content: '';
        position: absolute;
        bottom: -8px;
        left: 0;
        width: 80px;
        height: 6px;
        background: linear-gradient(90deg, #f37021, #ff8c42);
        border-radius: 3px;
        box-shadow: 0 2px 10px rgba(243,112,33,0.5);
    }

    /* ====== CARTES GÉNÉRALES SUR FOND SOMBRE ====== */
    .metric-card, .info-card {
        background: linear-gradient(145deg, rgba(30,41,59,0.95) 0%, rgba(15,23,42,0.98) 100%);
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.4), 0 0 0 1px rgba(243,112,33,0.2);
        border: 1px solid rgba(243,112,33,0.25);
        backdrop-filter: blur(20px);
        transition: all 0.4s ease;
    }

    .metric-card:hover, .info-card:hover {
        box-shadow: 0 20px 60px rgba(0,91,170,0.3), 0 0 0 1px rgba(243,112,33,0.4);
        transform: translateY(-4px);
    }

    /* ====== BOUTONS SUR FOND SOMBRE ====== */
    .stButton > button {
        background: linear-gradient(135deg, #f37021 0%, #ff8c42 50%, #f37021 100%);
        color: #ffffff;
        border-radius: 15px;
        padding: 14px 32px;
        font-weight: 800;
        font-size: 1.1rem;
        border: 2px solid rgba(243,112,33,0.5);
        transition: all 0.4s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 6px 25px rgba(243,112,33,0.4);
        backdrop-filter: blur(10px);
    }

    .stButton > button:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 35px rgba(243,112,33,0.6);
        border-color: rgba(255,255,255,0.3);
    }

    /* ====== EXPANDERS SUR FOND SOMBRE ====== */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, rgba(30,41,59,0.9), rgba(15,23,42,0.9));
        border-radius: 20px;
        border-left: 6px solid #f37021;
        font-weight: 800;
        padding: 20px 25px;
        margin-bottom: 15px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.3);
        border: 1px solid rgba(243,112,33,0.3);
        color: #f1f5f9;
    }

    /* ====== FOOTER SUR FOND SOMBRE ====== */
    .footer-main {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 50%, #020617 100%);
        padding: 30px 35px;
        border-radius: 25px;
        text-align: center;
        color: #e2e8f0;
        margin-top: 60px;
        box-shadow: 0 -12px 40px rgba(0,0,0,0.5);
        border: 2px solid rgba(243,112,33,0.3);
    }

    .footer-main h4 {
        color: #f8fafc;
        margin: 0 0 15px 0;
        font-size: 1.6rem;
        font-weight: 800;
        background: linear-gradient(45deg, #f37021, #ff8c42);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* ====== PROGRES BARS SUR FOND SOMBRE ====== */
    [data-testid="stProgress"] > div > div > div > div {
        background: linear-gradient(90deg, #f37021, #ff8c42, #f37021);
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(243,112,33,0.4);
    }

    /* ====== SELECTBOX SUR FOND SOMBRE ====== */
    .stSelectbox > div > div > div {
        background: rgba(15,23,42,0.95);
        border-radius: 15px;
        border: 2px solid rgba(243,112,33,0.4);
        color: #f1f5f9;
        backdrop-filter: blur(15px);
    }

    /* ====== ALERTES SUR FOND SOMBRE ====== */
    .stAlert {
        border-radius: 15px;
        border-left: 6px solid #f37021;
        padding: 25px;
        margin-bottom: 25px;
        background: rgba(30,41,59,0.9);
        border: 1px solid rgba(243,112,33,0.3);
        backdrop-filter: blur(15px);
    }

    /* ====== GRAPHIQUES SUR FOND SOMBRE ====== */
    .stPlotlyChart {
        border-radius: 20px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.4);
        border: 1px solid rgba(243,112,33,0.3);
        overflow: hidden;
        background: rgba(15,23,42,0.8);
        backdrop-filter: blur(20px);
    }

    /* ====== TEXTES GÉNÉRAUX ====== */
    .stText {
        color: #e2e8f0;
    }

    p, div {
        color: #cbd5e1;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------
# CHARGEMENT DONNÉES
# ---------------------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(EXPORTS_DIR / "corpus_avec_themes.csv", encoding="utf-8")
        logger.info(f"Données chargées : {len(df)} lignes")
        return df
    except Exception as e:
        st.error(f"❌ Erreur chargement données : {e}")
        return None

# ---------------------------------------------------
# HEADER 
# ---------------------------------------------------
def show_header():
    st.markdown("""
    <div class="header-main">
        <h1 class="header-title-main">⚡ SENELEC - Dashboard Analyse de Sentiment</h1>
        <p class="header-subtitle-main">
            Veille Citoyenne en Temps Réel • Intelligence Artificielle • Master 2 BI
        </p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------
# SIDEBAR STATS RAPIDES 
# ---------------------------------------------------
def show_sidebar_stats(df):
    stats_container = st.container()
    
    with stats_container:
        st.markdown("### 📈 **Statistiques Clés**")
        
        st.markdown("""
        <div class="vertical-stat-card">
            <div class="stat-icon">📊</div>
            <div class="stat-metric">
                <div class="stat-value">{:,}</div>
                <div class="stat-label">Publications Totales</div>
            </div>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
        
        neg_pct = (df['sentiment_pred'] == "negative").sum() / len(df) * 100
        st.markdown("""
        <div class="vertical-stat-card warning">
            <div class="stat-icon">📉</div>
            <div class="stat-metric">
                <div class="stat-value">{:.1f}%</div>
                <div class="stat-label">Sentiment Négatif</div>
            </div>
        </div>
        """.format(neg_pct), unsafe_allow_html=True)
        
        themes_count = df["theme"].nunique()
        st.markdown("""
        <div class="vertical-stat-card success">
            <div class="stat-icon">🎯</div>
            <div class="stat-metric">
                <div class="stat-value">{}</div>
                <div class="stat-label">Thèmes Détectés</div>
            </div>
        </div>
        """.format(themes_count), unsafe_allow_html=True)

# ---------------------------------------------------
# APP PRINCIPALE
# ---------------------------------------------------
def main():
    load_custom_css()
    show_header()

    # Chargement données global
    df = load_data()
    
    if df is None:
        st.warning("⚠️ Aucune donnée disponible. Vérifiez le fichier `corpus_avec_themes.csv`.")
        st.stop()

    # ---------------- SIDEBAR ----------------
    with st.sidebar:
        # Logo SENELEC
        st.image("dashboard/assets/logo_senelec.jpeg", width=250, caption="Société Nationale d'Electricité")
        
        st.markdown("## 🚀 Navigation Principale")

        page = st.radio(
            "",
            [
                "🏠 Vue d'ensemble",
                "🔍 Analyse Thématique", 
                "⚖️ Woyofal vs Autres",
                "💳 Woyofal vs Postpayé",
                "🗺️ Géographie"
            ],
            index=0
        )

        st.markdown("---")
        
        st.markdown("### 🎓 Mémoire Master 2 BI")
        with st.expander("📋 Détails complets", expanded=False):
            st.success("""
            **👤 Auteur :** Ouly TOURÉ  
            **🏫 Université :** UCAD  
            **📅 Année :** 2024-2025  
            **👨‍🏫 Encadrants :**  
            • Pr Aliou BOLY  
            • Pr Ndiouma BAME
            """)

        # st.markdown("### 📈 Statistiques Clés")
        show_sidebar_stats(df)

    # ---------------- CONTENU PRINCIPAL ----------------
    st.markdown("---")
    
    if page == "🏠 Vue d'ensemble":
        from dashboard.pages.overview import show_overview
        st.markdown('<h2>📊 Vue d\'Ensemble Globale</h2>', unsafe_allow_html=True)
        show_overview(df)

    elif page == "🔍 Analyse Thématique":
        from dashboard.pages.thematique import show_thematique
        st.markdown('<h2>🎯 Analyse Thématique Détaillée</h2>', unsafe_allow_html=True)
        show_thematique(df)

    elif page == "⚖️ Woyofal vs Autres":
        from dashboard.pages.comparaison import show_comparaison
        st.markdown('<h2>⚖️ Comparaison Woyofal vs Autres</h2>', unsafe_allow_html=True)
        show_comparaison(df)

    elif page == "💳 Woyofal vs Postpayé":
        from dashboard.pages.woyofal_vs_postpaye import show_woyofal_vs_postpaye
        st.markdown('<h2>💳 Woyofal vs Postpayé</h2>', unsafe_allow_html=True)
        show_woyofal_vs_postpaye(df)

    elif page == "🗺️ Géographie":
        from dashboard.pages.geographie import show_geographie
        st.markdown('<h2>🗺️ Analyse Géographique</h2>', unsafe_allow_html=True)
        show_geographie(df)

    # ---------------- FOOTER ----------------
    st.markdown("""
    <div class="footer-main">
        <h4>⚡ SENELEC - Plateforme d'Analyse de Sentiment par IA</h4>
        <p>© 2025 Ouly TOURÉ | Master 2 Business Intelligence | UCAD Dakar</p>
        <p style="font-size: 0.95rem; opacity: 0.85; margin-top: 12px;">
            Powered by Streamlit • Python • Transformers • NLP
        </p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------
if __name__ == "__main__":
    main()
