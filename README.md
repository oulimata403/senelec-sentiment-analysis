# ⚡ SENELEC – Plateforme d’Analyse de Sentiment par Intelligence Artificielle  
### Mémoire de Master 2 – Business Intelligence  
**Université Cheikh Anta Diop de Dakar (UCAD)**  
**Auteur : Ouly TOURÉ**  
**Année académique : 2024 – 2025**

---

# 📌 Présentation du Projet

Ce projet constitue le travail de recherche réalisé dans le cadre du Master 2 Informatique – option Business Intelligence à l’UCAD.

Il s’agit d’une **plateforme complète d’analyse de sentiment basée sur l’Intelligence Artificielle**, permettant d’évaluer en temps réel la perception des usagers des services de la SENELEC à partir :

- des publications issues des réseaux sociaux (Facebook, Twitter/X)
- des données d’enquête terrain
- d’un modèle NLP basé sur BERT

L’objectif principal est de fournir un **outil d’aide à la décision basé sur des données massives**, combinant Business Intelligence, Data Engineering et Intelligence Artificielle.

---

# 🎯 Objectifs du Projet

- Analyser la perception citoyenne des services SENELEC  
- Identifier les thèmes dominants de mécontentement  
- Comparer le système **Woyofal (prépayé)** aux autres services  
- Mesurer les disparités régionales  
- Appliquer un modèle BERT pour la classification automatique des sentiments  
- Générer des recommandations stratégiques basées sur les données  

---

# 📊 Résultats Clés

| Indicateur | Valeur |
|------------|--------|
| Total publications analysées | 2 739 |
| Sentiment négatif global | 66.8 % |
| Sentiment positif | 25.3 % |
| Thème dominant | Système Woyofal (57.9 %) |
| Négativité Woyofal | 75.0 % |
| Nombre de régions analysées | 14 |
| Répondants enquête terrain | 490 |
| Accuracy modèle BERT | 86.15 % |

---

# 🧠 Intelligence Artificielle

## Modèle utilisé
- `bert-base-multilingual-cased`
- 7 epochs (Early Stopping)
- Framework : HuggingFace Transformers

## Performance
- Accuracy : 86.15 %  
- F1-Score : 86.52 %  
- Precision : 88.92 %  
- Recall : 86.15 %  
- Loss Test : 0.3849  

---

# 🏗️ Architecture Complète du Projet  
## senelec-sentiment-analysis/

```
senelec-sentiment-analysis/
│
├── dashboard/
│   ├── app.py
│   ├── pages/
│   │   ├── overview.py
│   │   ├── thematique.py
│   │   ├── comparaison.py
│   │   ├── woyofal_vs_postpaye.py
│   │   ├── geographie.py
│   └── components/
│       ├── filters.py
│       ├── kpi_cards.py
│       └── charts.py
│
├── models/
│   ├── sentiment_model/
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   ├── tokenizer_config.json
│   │   ├── vocab.txt
│   │   └── special_tokens_map.json
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
│
├── data/
│   ├── raw/
│   │   ├── facebook_raw.csv
│   │   ├── twitter_raw.csv
│   │   └── Enquete_SENELEC_raw.csv
│   ├── processed/
│   │   ├── corpus_nettoye.csv
│   │   ├── corpus_fusion_brut.csv
│   │   ├── corpus_facebook_nettoye.csv
│   │   └── corpus_avec_langues.csv
│   └── exports/
│       ├── corpus_avec_sentiment.csv
│       └── corpus_avec_themes.csv
│
├── utils/
│   ├── datetime_helper.py
│   ├── file_handler.py
│   ├── logger.py
│   ├── text_utils.py
│
├── logs/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_statistical_tests.ipynb
│
├── tests/
│   ├── test_text_utils.py
│   ├── test_datetime_helper.py
│   └── test_model.py
│
├── scripts/
│   ├── collecte_facebook.py
│   ├── collecte_facebook_keywords.py
│   └── collecte_twitter.py
│
├── config/
│   ├── config.py
│   └── api_keys.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

# 🧩 Fonctionnalités Principales

- Dashboard interactif (Streamlit)  
- Analyse thématique détaillée  
- Comparaison Woyofal vs Autres services  
- Comparaison Woyofal vs Postpayé  
- Analyse géographique  
- Tests statistiques (Chi², ANOVA)  
- Évolution temporelle  
- Recommandations automatiques  
- Export CSV des résultats  

---

# 🔍 Thématiques Détectées

1. Système Woyofal  
2. Coupures d’électricité  
3. Service client  
4. Facturation & Tarification  

---

# 🗺️ Analyse Géographique

- 14 régions analysées  
- Identification des zones critiques  
- ANOVA significative (p < 0.001)  
- Priorisation territoriale  

---

# 📈 Méthodologie

### 1️⃣ Collecte de données
- Scraping réseaux sociaux  
- Nettoyage et normalisation  
- Fusion avec enquête terrain  

### 2️⃣ Prétraitement NLP
- Nettoyage texte  
- Tokenisation  
- Suppression ponctuation  
- Normalisation  

### 3️⃣ Modélisation IA
- Fine-tuning BERT  
- Classification sentiment  
- Évaluation métriques  

### 4️⃣ Analyse BI
- KPI avancés  
- Indice satisfaction  
- Ratio négatif/positif  
- Criticité  
- Tests statistiques  

---

# ⚙️ Installation & Exécution

```bash
git clone https://github.com/oulimata403/analysis-senelec-sentiment.git
cd senelec-sentiment-analysis
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run dashboard/app.py
```

Puis ouvrir :

```
http://localhost:8501
```

---

# 🛠️ Technologies Utilisées

Python  
Streamlit  
Pandas  
Plotly  
NumPy  
SciPy  
HuggingFace Transformers  
PyTorch  
Scikit-learn  
NLP  
Logging structuré  

---

# 📊 Indicateurs Avancés

Indice de satisfaction  
Ratio Négatif / Positif  
Criticité globale  
Diversité thématique  
Taux d’insatisfaction régional  
Test Chi²  
Test ANOVA  

---

# 💡 Recommandations Stratégiques Générées

Audit technique du système Woyofal  
Simplification de la facturation  
Modernisation du service client  
Approche régionalisée basée sur les données  
Veille citoyenne institutionnalisée  

---

# 📚 Apports Académiques

- Intégration Business Intelligence + Intelligence Artificielle  
- Exploitation de données massives non structurées  
- Application du NLP dans un contexte africain  
- Transformation de données sociales en outils décisionnels  

---

# 🏆 Contribution

Ce projet combine :

- Business Intelligence  
- Data Science  
- Intelligence Artificielle  
- Analyse citoyenne en temps réel  

---

# 🔐 Avertissement

Les données utilisées proviennent de publications publiques et d’une enquête terrain réalisée dans un cadre académique.  
Ce projet est destiné exclusivement à des fins de recherche.

---

# 👩🏽‍💻 Auteur

**Ouly TOURÉ**  
Master 2 Informatique – Business Intelligence  
Université Cheikh Anta Diop (UCAD)  
Dakar – Sénégal  

---

# 📜 Licence

Projet académique – Tous droits réservés © 2025 Ouly TOURÉ  

---

# ⚡ Conclusion

Cette plateforme démontre la capacité de l’Intelligence Artificielle à transformer des données citoyennes en outils stratégiques d’aide à la décision, contribuant ainsi à une gouvernance plus transparente, efficace et orientée données.