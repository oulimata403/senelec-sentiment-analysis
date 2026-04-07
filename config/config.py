"""
Configuration générale du projet Analyse SENELEC
"""
import os
import logging
from pathlib import Path

# ==========================================
# CHEMINS DE BASE
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
LOGS_DIR = BASE_DIR / "logs"

# Sous-dossiers data
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LABELED_DATA_DIR = DATA_DIR / "labeled"
EXPORTS_DIR = DATA_DIR / "exports"

# Sous-dossiers models
SENTIMENT_MODEL_DIR = MODELS_DIR / "sentiment_model"
TOPIC_MODEL_DIR = MODELS_DIR / "topic_model"

# Sous-dossiers reports
FIGURES_DIR = REPORTS_DIR / "figures"
STATISTICS_DIR = REPORTS_DIR / "statistics"

# ==========================================
# CHEMINS SPÉCIFIQUES COLLECTE FACEBOOK
# ==========================================
FACEBOOK_SESSION_PATH = BASE_DIR / "scripts" / "1_collecte" / "auth" / "facebook_session.json"
FACEBOOK_POSTS_OUTPUT = RAW_DATA_DIR / "facebook_posts_commentaires.csv"
FACEBOOK_KEYWORDS_OUTPUT = RAW_DATA_DIR / "facebook_keywords.csv"


# ==========================================
# CRÉATION DES DOSSIERS
# ==========================================
for directory in [
    DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, LABELED_DATA_DIR, EXPORTS_DIR,
    MODELS_DIR, SENTIMENT_MODEL_DIR, TOPIC_MODEL_DIR,
    REPORTS_DIR, FIGURES_DIR, STATISTICS_DIR,
    LOGS_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)

# ==========================================
# CONFIGURATION COLLECTE TWITTER
# ==========================================
TWITTER_CONFIG = {
    "max_results_per_request": 100,
    "max_tweets_per_day": 1000,
    "max_tweets_total": 5000,
    "keywords": [
        "SENELEC",
        "Woyofal",
        "coupure électricité Sénégal",
        "délestage Sénégal",
        "facture SENELEC",
        "compteur prépayé Sénégal",
        "service client SENELEC",
        "électricité Dakar",
        "#SENELEC",
        "#Woyofal",
        "#ElectriciteSenegal",
    ],
    "languages": ["fr"],
    "exclude_retweets": True,
    "min_likes": 0,
    "collect_days_back": 7,
}

# ==========================================
# CONFIGURATION NLP
# ==========================================
NLP_CONFIG = {
    "model_name": "bert-base-multilingual-cased",
    "model_alternative": "camembert-base",
    "max_length": 128,
    "padding": "max_length",
    "truncation": True,
    "batch_size": 16,
    "num_epochs": 20,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "test_size": 0.15,
    "val_size": 0.15,
    "random_state": 42,
    "min_confidence": 0.5,
}

# ==========================================
# LABELS DE SENTIMENT
# ==========================================
SENTIMENT_LABELS = {"negative": 0, "neutral": 1, "positive": 2}

SENTIMENT_LABELS_REVERSE = {0: "negative", 1: "neutral", 2: "positive"}

SENTIMENT_LABELS_FR = {
    "negative": "Négatif",
    "neutral": "Neutre",
    "positive": "Positif",
}

# ==========================================
# THÈMES ET MOTS-CLÉS
# ==========================================
THEMES = {
    "coupure": {
        "keywords": [
            "coupure",
            "délestage",
            "panne",
            "blackout",
            "courant coupé",
            "doom",
            "électricité coupée",
            "pas de courant",
        ],
        "label_fr": "Coupures d'électricité",
    },
    "woyofal": {
        "keywords": [
            "woyofal",
            "prépayé",
            "code",
            "crédit",
            "compteur prépayé",
            "recharge",
            "vente crédit",
        ],
        "label_fr": "Système Woyofal",
    },
    "facturation": {
        "keywords": [
            "facture",
            "cher",
            "coût",
            "tarif",
            "prix",
            "xaalis",
            "montant",
            "payer",
            "facturation",
        ],
        "label_fr": "Facturation et Tarifs",
    },
    "service_client": {
        "keywords": [
            "service client",
            "agence",
            "réclamation",
            "plainte",
            "appel",
            "contact",
            "réponse",
        ],
        "label_fr": "Service Client",
    },
    "qualite": {
        "keywords": [
            "qualité",
            "amélioration",
            "satisfaction",
            "service",
            "tension",
            "stabilité",
        ],
        "label_fr": "Qualité du Service",
    },
    "autre": {
        "keywords": [],
        "label_fr": "Autre",
    },
}

# ==========================================
# CONFIGURATION DASHBOARD
# ==========================================
DASHBOARD_CONFIG = {
    "title": "📊 Dashboard Perception SENELEC",
    "subtitle": "Analyse de sentiment et veille citoyenne",
    "port": 8501,
    "host": "172.20.10.5",
    "debug": True,
    "colors": {
        "positive": "#27ae60",
        "neutral": "#95a5a6",
        "negative": "#e74c3c",
        "primary": "#3498db",
        "secondary": "#f39c12",
        "background": "#ecf0f1",
    },
    "chart_height": 400,
    "chart_template": "plotly_white",
}

# ==========================================
# CONFIGURATION LOGGING
# ==========================================
LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "detailed": {
            "format": "%(asctime)s [%(levelname)s] %(name)s [%(filename)s:%(lineno)d]: %(message)s",
        },
    },
    "handlers": {
        "console": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "level": "INFO",
            "formatter": "detailed",
            "class": "logging.FileHandler",
            "filename": str(LOGS_DIR / "app.log"),
            "mode": "a",
            "encoding": "utf-8",
        },
    },
    "loggers": {
        "": {  
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": False,
        }
    },
}

# ==========================================
# CONFIGURATION ENQUÊTE
# ==========================================
ENQUETE_CONFIG = {
    "min_reponses": 500,
    "target_reponses": 800,
    "quotas": {
        "region": {
            "Dakar": 0.40,
            "Thiès": 0.15,
            "Kaolack": 0.10,
            "Saint-Louis": 0.10,
            "Autres": 0.25,
        },
        "type_compteur": {
            "Woyofal": 0.60,
            "Postpayé": 0.40,
        },
        "age": {
            "18-35": 0.50,
            "36-55": 0.35,
            "56+": 0.15,
        },
    },
}

# ==========================================
# STOPWORDS PERSONNALISÉS (WOLOF + FR)
# ==========================================
CUSTOM_STOPWORDS = {
    "fr": ["alors", "comme", "donc", "encore", "mais", "plus", "tout", "très"],
    "wolof": ["dafa", "la", "nga", "ñu", "amna", "dina"],
    "senegal": ["senegal", "dakar", "thies", "ndakaru"],
}

# ==========================================
# MÉTADONNÉES PROJET
# ==========================================
PROJECT_METADATA = {
    "name": "SENELEC Sentiment Analysis",
    "version": "1.0.0",
    "author": "Ouly TOURE",
    "email": "oulimatat17@gmail..com",
    "university": "Université Cheikh Anta Diop de Dakar(UCAD)",
    "year": "2024-2025",
    "description": "Analyse de la perception des services de la SENELEC par IA",
}

# ==========================================
# VALIDATION CONFIGURATION
# ==========================================
def validate_config() -> bool:
    required_dirs = [DATA_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR]
    for d in required_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Dossier manquant : {d}")
    print("✅ Configuration valide")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("CONFIGURATION PROJET SENELEC")
    print("=" * 60)
    print(f"📂 Dossier base : {BASE_DIR}")
    print(f"📊 Données : {DATA_DIR}")
    print(f"🤖 Modèles : {MODELS_DIR}")
    print(f"📈 Rapports : {REPORTS_DIR}")
    print(f"📝 Logs : {LOGS_DIR}")
    print("=" * 60)
    validate_config()
