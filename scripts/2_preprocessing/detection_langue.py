"""
Détection de langue (Français, Wolof, Mixte)
Gestion du multilinguisme sénégalais
"""

import sys
from pathlib import Path
import pandas as pd
import re
from typing import Tuple
from langdetect import detect, LangDetectException

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.config import PROCESSED_DATA_DIR
from utils.logger import setup_logger
from utils.file_handler import save_csv

logger = setup_logger("detection_langue")


class DetecteurLangue:
    """Détecteur de langue adapté au contexte sénégalais"""
    
    def __init__(self):
        # Mots-clés wolof fréquents
        self.mots_wolof = {
            # Pronoms et particules
            "dafa", "nga", "ñu", "la", "mu", "nu", "di", "dina",
            "amna", "am", "rek", "wax", "def", "jox", "may", "ma",
            
            # Verbes courants
            "dey", "dem", "dall", "jënd", "lekk", "togg", "liggéey",
            "gis", "xam", "wax", "def", "jox", "may", "yëg",
            
            # Expressions courantes
            "ndax", "waaw", "déédéét", "yow", "yéen", "suma", "sa",
            "li", "lou", "bu", "boo", "bi", "ba", "yi", "yu",
            
            # Électricité et SENELEC
            "doom", "xaalis", "ndakaru", "dara", "amul", "dina",
            
            # Chiffres/nombres
            "benn", "ñaar", "ñett", "ñeent", "juróom", "fukk",
        }
        
        # Patterns wolof (structures grammaticales)
        self.patterns_wolof = [
            r'\b(dafa|nga|mu|di|dina)\s+\w+',  
            r'\b(la|ma|nga)\s+(am|def|jox)',   
            r'\bxaalis\b',                      
            r'\bndakaru\b',                     
            r'\bdoom\b',                        
        ]
        
        self.stats = {
            "francais": 0,
            "wolof": 0,
            "mixte": 0,
            "autre": 0,
            "erreur": 0,
        }
    
    def compter_mots_wolof(self, texte: str) -> int:
        """Compte le nombre de mots wolof dans le texte"""
        if not texte:
            return 0
        
        mots = texte.lower().split()
        count = sum(1 for mot in mots if mot in self.mots_wolof)
        
        # Vérifier patterns
        for pattern in self.patterns_wolof:
            count += len(re.findall(pattern, texte.lower()))
        
        return count
    
    def detecter_langue_base(self, texte: str) -> str:
        """Détection de langue avec langdetect (français/anglais/autre)"""
        if not texte or len(texte) < 10:
            return "inconnu"
        
        try:
            langue = detect(texte)
            return langue
        except LangDetectException:
            return "inconnu"
    
    def detecter_langue_hybride(self, texte: str) -> Tuple[str, float]:
        """
        Détection hybride adaptée au contexte sénégalais
        
        Returns:
            (langue, confiance) où langue = 'fr', 'wo', 'mixte', 'autre'
            confiance = score entre 0 et 1
        """
        if not texte or len(texte) < 10:
            return ("inconnu", 0.0)
        
        # Compter mots wolof
        nb_mots_wolof = self.compter_mots_wolof(texte)
        nb_mots_total = len(texte.split())
        
        ratio_wolof = nb_mots_wolof / max(nb_mots_total, 1)
        
        # Détection langue de base
        langue_base = self.detecter_langue_base(texte)
        
        # Décision hybride
        if ratio_wolof > 0.3:  
            langue_finale = "wo"  
            confiance = min(ratio_wolof, 0.95)
        
        elif ratio_wolof > 0.1:  
            langue_finale = "mixte"  
            confiance = 0.7
        
        elif langue_base == "fr":
            langue_finale = "fr"  
            confiance = 0.9
        
        elif langue_base in ["en", "pt", "es"]:
            langue_finale = "autre"
            confiance = 0.6
        
        else:
            # Vérifier si wolof pur (pas détecté par langdetect)
            if nb_mots_wolof >= 3:
                langue_finale = "wo"
                confiance = 0.7
            else:
                langue_finale = "inconnu"
                confiance = 0.3
        
        return (langue_finale, confiance)
    
    def detecter_batch(self, textes: pd.Series) -> pd.DataFrame:
        """Détecte la langue pour un batch de textes"""
        resultats = []
        
        for texte in textes:
            langue, confiance = self.detecter_langue_hybride(texte)
            
            # Stats
            self.stats[langue] = self.stats.get(langue, 0) + 1
            
            resultats.append({
                "langue": langue,
                "confiance_langue": confiance,
            })
        
        return pd.DataFrame(resultats)
    
    def afficher_stats(self) -> None:
        """Affiche les statistiques de détection"""
        logger.info("\n📊 STATISTIQUES DÉTECTION DE LANGUE")
        
        total = sum(self.stats.values())
        
        for langue, count in sorted(self.stats.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total) * 100 if total > 0 else 0
            label = {
                "fr": "Français",
                "wo": "Wolof",
                "mixte": "Mixte (Fr+Wo)",
                "autre": "Autre",
                "inconnu": "Inconnu",
                "erreur": "Erreur"
            }.get(langue, langue)
            
            logger.info(f"   {label:20s} : {count:5d} ({pct:5.2f}%)")


def detecter_langues_corpus(filepath: Path) -> pd.DataFrame:
    """Détecte les langues pour tout le corpus"""
    logger.info("📥 Chargement du corpus nettoyé...")
    
    df = pd.read_csv(filepath, encoding="utf-8")
    logger.info(f"   Lignes chargées : {len(df)}")
    
    # Créer détecteur
    detecteur = DetecteurLangue()
    
    logger.info("\n🔍 DÉTECTION DES LANGUES...")
    
    # Détecter langues
    df_langues = detecteur.detecter_batch(df["texte_nettoye"])
    
    # Fusionner avec corpus
    df = pd.concat([df, df_langues], axis=1)
    
    # Afficher stats
    detecteur.afficher_stats()
    
    return df


def analyser_distribution_langues(df: pd.DataFrame) -> None:
    """Analyse la distribution des langues par plateforme"""
    logger.info("\n📈 DISTRIBUTION PAR PLATEFORME")
    
    # Crosstab
    crosstab = pd.crosstab(
        df["plateforme"], 
        df["langue"], 
        normalize="index"
    ) * 100
    
    logger.info("\n" + crosstab.to_string())
    
    # Distribution par type de contenu
    logger.info("\n📈 DISTRIBUTION PAR TYPE DE CONTENU")
    
    crosstab2 = pd.crosstab(
        df["type_contenu"], 
        df["langue"], 
        normalize="index"
    ) * 100
    
    logger.info("\n" + crosstab2.to_string())


def afficher_exemples_par_langue(df: pd.DataFrame, n: int = 3) -> None:
    """Affiche des exemples pour chaque langue détectée"""
    logger.info(f"\n📝 EXEMPLES PAR LANGUE (n={n} par langue)")
    
    for langue in ["fr", "wo", "mixte"]:
        logger.info("\n" + "=" * 60)
        logger.info(f"LANGUE : {langue.upper()}")
        logger.info("=" * 60)
        
        echantillon = df[df["langue"] == langue].head(n)
        
        for idx, row in echantillon.iterrows():
            logger.info(f"\n[Confiance: {row['confiance_langue']:.2f}]")
            logger.info(f"{row['texte_nettoye'][:150]}...")


def filtrer_par_langue(df: pd.DataFrame, langues_acceptees: list = None) -> pd.DataFrame:
    """Filtre le corpus pour ne garder que certaines langues"""
    if langues_acceptees is None:
        langues_acceptees = ["fr", "wo", "mixte"]
    
    logger.info(f"\n🔍 FILTRAGE PAR LANGUE (acceptées: {langues_acceptees})")
    
    avant = len(df)
    df_filtre = df[df["langue"].isin(langues_acceptees)].copy()
    apres = len(df_filtre)
    
    logger.info(f"   Textes conservés : {apres}/{avant} ({(apres/avant)*100:.2f}%)")
    
    return df_filtre


def sauvegarder_corpus_avec_langues(df: pd.DataFrame) -> Path:
    """Sauvegarde le corpus avec détection de langue"""
    logger.info("\n💾 SAUVEGARDE DU CORPUS AVEC LANGUES")
    
    filepath = PROCESSED_DATA_DIR / "corpus_avec_langues.csv"
    save_csv(df, filepath)
    
    logger.info(f"   Fichier : {filepath}")
    logger.info(f"   Lignes  : {len(df)}")
    
    return filepath


def main():
    """Point d'entrée principal"""
    print("=" * 70)
    print("🌍 DÉTECTION DE LANGUE - SENELEC")
    print("=" * 70)
    
    try:
        # Charger corpus nettoyé
        filepath_source = PROCESSED_DATA_DIR / "corpus_nettoye.csv"
        
        if not filepath_source.exists():
            logger.error(f"❌ Fichier source introuvable : {filepath_source}")
            logger.error("   Exécutez d'abord : nettoyage_texte.py")
            sys.exit(1)
        
        # Détecter langues
        df = detecter_langues_corpus(filepath_source)
        
        # Analyser distribution
        analyser_distribution_langues(df)
        
        # Afficher exemples
        afficher_exemples_par_langue(df, n=2)
        
        # Filtrer (garder fr, wo, mixte)
        df = filtrer_par_langue(df, langues_acceptees=["fr", "wo", "mixte"])
        
        # Sauvegarder
        filepath_final = sauvegarder_corpus_avec_langues(df)
        
        print("\n✅ DÉTECTION DE LANGUE TERMINÉE")
        print(f"📁 Fichier : {filepath_final}")
        print(f"📊 Total   : {len(df)} entrées")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la détection : {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    print("=" * 70)


if __name__ == "__main__":
    main()