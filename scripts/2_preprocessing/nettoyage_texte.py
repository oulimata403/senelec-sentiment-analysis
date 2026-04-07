"""
Nettoyage avancé des textes pour l'analyse NLP
- Normalisation unicode
- Suppression liens, hashtags, mentions
- Gestion émojis
- Normalisation espaces et ponctuation
"""

import sys
from pathlib import Path
import pandas as pd
import re
import emoji
import unicodedata
from typing import List

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.config import PROCESSED_DATA_DIR, CUSTOM_STOPWORDS
from utils.logger import setup_logger
from utils.file_handler import save_csv

logger = setup_logger("nettoyage_texte")


class NettoyeurTexte:
    """Classe pour nettoyer les textes de réseaux sociaux"""
    
    def __init__(self):
        self.stats = {
            "urls_supprimees": 0,
            "mentions_supprimees": 0,
            "hashtags_supprimes": 0,
            "emojis_convertis": 0,
        }
    
    def normaliser_unicode(self, texte: str) -> str:
        """Normalise les caractères unicode (NFD -> NFC)"""
        if not texte:
            return ""
        
        texte = unicodedata.normalize("NFC", texte)
        
        # Supprimer caractères de contrôle
        texte = "".join(ch for ch in texte if unicodedata.category(ch)[0] != "C")
        
        return texte
    
    def supprimer_urls(self, texte: str) -> str:
        """Supprime toutes les URLs"""
        # Pattern pour URLs (http, https, www)
        pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        texte_nettoye = re.sub(pattern, '', texte)
        
        # URLs avec www
        pattern_www = r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        texte_nettoye = re.sub(pattern_www, '', texte_nettoye)
        
        # Compter URLs supprimées
        urls_count = len(re.findall(pattern, texte)) + len(re.findall(pattern_www, texte))
        self.stats["urls_supprimees"] += urls_count
        
        return texte_nettoye
    
    def supprimer_mentions(self, texte: str) -> str:
        """Supprime les mentions @utilisateur"""
        mentions = re.findall(r'@\w+', texte)
        self.stats["mentions_supprimees"] += len(mentions)
        
        return re.sub(r'@\w+', '', texte)
    
    def traiter_hashtags(self, texte: str, conserver: bool = True) -> str:
        """
        Traite les hashtags
        Si conserver=True : #SENELEC -> SENELEC
        Si conserver=False : supprime complètement
        """
        hashtags = re.findall(r'#\w+', texte)
        self.stats["hashtags_supprimes"] += len(hashtags)
        
        if conserver:
            # Retirer le # mais garder le mot
            return re.sub(r'#(\w+)', r'\1', texte)
        else:
            # Supprimer complètement
            return re.sub(r'#\w+', '', texte)
    
    def convertir_emojis(self, texte: str, mode: str = "texte") -> str:
        """
        Convertit les emojis
        mode='texte'   : 😊 -> :visage_souriant:
        mode='supprimer' : supprime les emojis
        mode='garder'  : garde les emojis tels quels
        """
        if mode == "texte":
            # Convertir emoji en texte descriptif
            texte_converti = emoji.demojize(texte, language="fr")
            self.stats["emojis_convertis"] += texte.count(":") // 2
            return texte_converti
        
        elif mode == "supprimer":
            return emoji.replace_emoji(texte, replace='')
        
        else:  
            return texte
    
    def nettoyer_ponctuation(self, texte: str) -> str:
        """Normalise la ponctuation excessive"""
        # Multiples points d'exclamation/interrogation
        texte = re.sub(r'!{2,}', '!', texte)
        texte = re.sub(r'\?{2,}', '?', texte)
        texte = re.sub(r'\.{2,}', '.', texte)
        
        # Espaces avant ponctuation
        texte = re.sub(r'\s+([?.!,;:])', r'\1', texte)
        
        return texte
    
    def normaliser_espaces(self, texte: str) -> str:
        """Normalise les espaces multiples"""
        # Remplacer multiples espaces par un seul
        texte = re.sub(r'\s+', ' ', texte)
        
        # Supprimer espaces début/fin
        texte = texte.strip()
        
        return texte
    
    def supprimer_chiffres_isoles(self, texte: str) -> str:
        """Supprime les chiffres isolés (artefacts)"""
        texte = re.sub(r'\b\d+\b', '', texte)
        return texte
    
    def nettoyer_caracteres_speciaux(self, texte: str) -> str:
        """Supprime les caractères spéciaux inutiles"""
        # Garder lettres, chiffres, espaces, ponctuation de base
        texte = re.sub(r'[^\w\s?.!,;:\'-]', ' ', texte)
        return texte
    
    def nettoyer_texte_complet(self, texte: str) -> str:
        """
        Pipeline complet de nettoyage
        """
        if not texte or not isinstance(texte, str):
            return ""
        
        # 1. Normalisation unicode
        texte = self.normaliser_unicode(texte)
        
        # 2. Minuscules
        texte = texte.lower()
        
        # 3. Supprimer URLs
        texte = self.supprimer_urls(texte)
        
        # 4. Supprimer mentions
        texte = self.supprimer_mentions(texte)
        
        # 5. Traiter hashtags (on garde le mot)
        texte = self.traiter_hashtags(texte, conserver=True)
        
        # 6. Convertir emojis en texte
        texte = self.convertir_emojis(texte, mode="texte")
        
        # 7. Nettoyer ponctuation
        texte = self.nettoyer_ponctuation(texte)
        
        # 8. Supprimer chiffres isolés
        texte = self.supprimer_chiffres_isoles(texte)
        
        # 9. Caractères spéciaux
        texte = self.nettoyer_caracteres_speciaux(texte)
        
        # 10. Normaliser espaces
        texte = self.normaliser_espaces(texte)
        
        return texte
    
    def afficher_stats(self) -> None:
        """Affiche les statistiques de nettoyage"""
        logger.info("\n📊 STATISTIQUES DE NETTOYAGE")
        logger.info(f"   URLs supprimées      : {self.stats['urls_supprimees']}")
        logger.info(f"   Mentions supprimées  : {self.stats['mentions_supprimees']}")
        logger.info(f"   Hashtags traités     : {self.stats['hashtags_supprimes']}")
        logger.info(f"   Emojis convertis     : {self.stats['emojis_convertis']}")


def nettoyer_corpus(filepath: Path) -> pd.DataFrame:
    """Nettoie le corpus fusionné"""
    logger.info("📥 Chargement du corpus fusionné...")
    
    df = pd.read_csv(filepath, encoding="utf-8")
    logger.info(f"   Lignes chargées : {len(df)}")
    
    # Créer instance nettoyeur
    nettoyeur = NettoyeurTexte()
    
    logger.info("\n🧹 NETTOYAGE DES TEXTES...")
    
    # Appliquer nettoyage
    df["texte_original"] = df["texte"]  
    df["texte_nettoye"] = df["texte"].apply(nettoyeur.nettoyer_texte_complet)
    
    # Calculer longueur après nettoyage
    df["longueur_nettoyee"] = df["texte_nettoye"].str.len()
    
    # Afficher stats
    nettoyeur.afficher_stats()
    
    # Stats longueur
    logger.info("\n📏 STATISTIQUES LONGUEUR APRÈS NETTOYAGE")
    logger.info(f"   Minimum  : {df['longueur_nettoyee'].min()}")
    logger.info(f"   Maximum  : {df['longueur_nettoyee'].max()}")
    logger.info(f"   Moyenne  : {df['longueur_nettoyee'].mean():.2f}")
    logger.info(f"   Médiane  : {df['longueur_nettoyee'].median():.2f}")
    
    return df


def filtrer_textes_valides(df: pd.DataFrame, longueur_min: int = 10) -> pd.DataFrame:
    """Filtre les textes trop courts après nettoyage"""
    logger.info(f"\n🔍 FILTRAGE (longueur min : {longueur_min} caractères)")
    
    avant = len(df)
    
    # Filtrer textes vides ou trop courts
    df = df[df["longueur_nettoyee"] >= longueur_min].copy()
    
    apres = len(df)
    supprimes = avant - apres
    
    logger.info(f"   Textes supprimés : {supprimes}")
    logger.info(f"   Textes conservés : {apres} ({(apres/avant)*100:.2f}%)")
    
    return df


def afficher_exemples(df: pd.DataFrame, n: int = 5) -> None:
    """Affiche quelques exemples de nettoyage"""
    logger.info(f"\n📝 EXEMPLES DE NETTOYAGE (n={n})")
    
    echantillon = df.sample(min(n, len(df)))
    
    for idx, row in echantillon.iterrows():
        logger.info("\n" + "─" * 60)
        logger.info(f"AVANT : {row['texte_original'][:100]}...")
        logger.info(f"APRÈS : {row['texte_nettoye'][:100]}...")


def sauvegarder_corpus_nettoye(df: pd.DataFrame) -> Path:
    """Sauvegarde le corpus nettoyé"""
    logger.info("\n💾 SAUVEGARDE DU CORPUS NETTOYÉ")
    
    # Colonnes à garder
    colonnes_finales = [
        "id_unique",
        "plateforme",
        "source",
        "type_contenu",
        "texte_original",
        "texte_nettoye",
        "longueur_nettoyee",
        "date_publication",
        "date_collecte",
        "strategie_collecte",
        "mot_cle_recherche",
    ]
    
    # Garder seulement les colonnes qui existent
    colonnes_disponibles = [col for col in colonnes_finales if col in df.columns]
    df_final = df[colonnes_disponibles]
    
    filepath = PROCESSED_DATA_DIR / "corpus_nettoye.csv"
    save_csv(df_final, filepath)
    
    logger.info(f"   Fichier : {filepath}")
    logger.info(f"   Lignes  : {len(df_final)}")
    
    return filepath


def main():
    """Point d'entrée principal"""
    print("=" * 70)
    print("🧹 NETTOYAGE AVANCÉ DES TEXTES - SENELEC")
    print("=" * 70)
    
    try:
        # Charger corpus fusionné
        filepath_source = PROCESSED_DATA_DIR / "corpus_fusionne_brut.csv"
        
        if not filepath_source.exists():
            logger.error(f"❌ Fichier source introuvable : {filepath_source}")
            logger.error("   Exécutez d'abord : fusion_donnees.py")
            sys.exit(1)
        
        # Nettoyer
        df = nettoyer_corpus(filepath_source)
        
        # Filtrer textes invalides
        df = filtrer_textes_valides(df, longueur_min=10)
        
        # Afficher exemples
        afficher_exemples(df, n=3)
        
        # Sauvegarder
        filepath_nettoye = sauvegarder_corpus_nettoye(df)
        
        print("\n✅ NETTOYAGE TERMINÉ AVEC SUCCÈS")
        print(f"📁 Fichier : {filepath_nettoye}")
        print(f"📊 Total   : {len(df)} entrées valides")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du nettoyage : {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    print("=" * 70)


if __name__ == "__main__":
    main()