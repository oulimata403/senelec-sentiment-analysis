"""
Script de collecte complète de tweets SENELEC
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import tweepy

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.api_keys import TWITTER_BEARER_TOKEN, validate_credentials
from config.config import TWITTER_CONFIG, RAW_DATA_DIR
from utils.logger import setup_logger


logger = setup_logger("collecte_twitter")


class TwitterCollector:
    """Collecteur de tweets sur la SENELEC"""

    def __init__(self):
        logger.info("Initialisation du collecteur Twitter")

        # Valider credentials
        try:
            validate_credentials("twitter")
            logger.info("Credentials Twitter validés")
        except ValueError as e:
            logger.error(f"Credentials invalides : {e}")
            raise

        # Créer client
        self.client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)
        self.keywords = TWITTER_CONFIG["keywords"]
        self.max_results = TWITTER_CONFIG["max_results_per_request"]

        logger.info(f"Client Twitter créé - {len(self.keywords)} mots-clés configurés")

    def build_query(self, keywords_list=None) -> str:
        """
        Construction de la requête de recherche combinée.
        Exemple: "SENELEC OR Woyofal -is:retweet lang:fr"
        """
        if keywords_list is None:
            keywords_list = self.keywords

        query_parts = []
        for kw in keywords_list:
            if " " in kw:
                query_parts.append(f'"{kw}"')
            else:
                query_parts.append(kw)

        query = " OR ".join(query_parts)

        if TWITTER_CONFIG.get("exclude_retweets", True):
            query += " -is:retweet"

        if TWITTER_CONFIG.get("languages"):
            query += " lang:fr"

        return query

    def collect_tweets(self, query: str, max_results: int = 100) -> pd.DataFrame:
        """
        Collecte de tweets selon une requête.
        Retourne un DataFrame brut avec beaucoup de colonnes,
        qui sera ensuite réduit au moment de la sauvegarde.
        """
        tweets_data = []

        try:
            logger.info(f"Requête : {query}")
            logger.info(f"Collecte de {max_results} tweets maximum...")

            search_params = {
                "query": query,
                "max_results": min(max_results, 100),
                "tweet_fields": [
                    "created_at",
                    "author_id",
                    "lang",
                    "public_metrics",
                    "geo",
                    "context_annotations",
                ],
                "expansions": ["author_id"],
                "user_fields": ["username", "location", "verified"],
            }

            response = self.client.search_recent_tweets(**search_params)

            if not response.data:
                logger.warning("Aucun tweet trouvé")
                return pd.DataFrame()

            users = {}
            if response.includes and "users" in response.includes:
                users = {user.id: user for user in response.includes["users"]}

            for tweet in response.data:
                author = users.get(tweet.author_id)
                metrics = tweet.public_metrics or {}

                tweets_data.append(
                    {
                        "id": tweet.id,
                        "text": tweet.text,
                        "created_at": tweet.created_at,
                        "author_id": tweet.author_id,
                        "username": author.username if author else None,
                        "user_location": author.location if author else None,
                        "user_verified": author.verified if author else False,
                        "langue": tweet.lang,
                        "likes": metrics.get("like_count", 0),
                        "retweets": metrics.get("retweet_count", 0),
                        "replies": metrics.get("reply_count", 0),
                        "impressions": metrics.get("impression_count", 0),
                        "date_collecte": datetime.now(),
                    }
                )

            logger.info(f"✅ {len(tweets_data)} tweets collectés")

        except tweepy.errors.TooManyRequests as e:
            logger.error(f"Limite de requêtes atteinte : {e}")
            logger.info("Attente de 15 minutes...")
            time.sleep(15 * 60)

        except tweepy.TweepyException as e:
            logger.error(f"Erreur API Twitter : {e}")

        except Exception as e:
            logger.error(f"Erreur inattendue : {e}")
            import traceback

            logger.error(traceback.format_exc())

        return pd.DataFrame(tweets_data)

    def collect_by_keyword(self, keyword: str, max_results: int = 100) -> pd.DataFrame:
        """Collecte pour un mot-clé spécifique."""
        query = f'"{keyword}"' if " " in keyword else keyword

        if TWITTER_CONFIG.get("exclude_retweets", True):
            query += " -is:retweet"

        if TWITTER_CONFIG.get("languages"):
            query += " lang:fr"

        return self.collect_tweets(query, max_results)

    def collect_all_keywords(self, max_per_keyword: int = 100) -> pd.DataFrame:
        """
        Collecte pour tous les mots-clés configurés.
        Retourne un DataFrame fusionné de tous les tweets.
        """
        all_tweets = []

        logger.info(f"Début collecte pour {len(self.keywords)} mots-clés")

        for i, keyword in enumerate(self.keywords, 1):
            logger.info("\n" + "=" * 60)
            logger.info(f"[{i}/{len(self.keywords)}] Collecte : {keyword}")
            logger.info("=" * 60)

            df = self.collect_by_keyword(keyword, max_per_keyword)

            if not df.empty:
                all_tweets.append(df)
                logger.info(f"✅ {len(df)} tweets pour '{keyword}'")
            else:
                logger.warning(f"⚠️ Aucun tweet pour '{keyword}'")

            if i < len(self.keywords):
                logger.info("Pause de 2 secondes...")
                time.sleep(2)

        if not all_tweets:
            logger.warning("Aucune donnée collectée")
            return pd.DataFrame()

        logger.info("\nFusion et dédoublonnage...")
        df_final = pd.concat(all_tweets, ignore_index=True)

        before = len(df_final)
        df_final.drop_duplicates(subset=["id"], keep="first", inplace=True)
        after = len(df_final)

        logger.info(f"Doublons supprimés : {before - after}")
        logger.info(f"Total final : {after} tweets uniques")

        return df_final

    def save_data(self, df: pd.DataFrame, prefix: str = "twitter_keywords") -> Path | None:
        """
        Sauvegarde les données Twitter dans un format harmonisé avec Facebook keywords.

        Colonnes de sortie :
        - source        : global_posts
        - strategie     : mot_cle
        - mot_cle       : keyword
        - texte         : texte du tweet
        - date_visible  : created_at
        - date_collecte : date de collecte
        """

        if df.empty:
            logger.warning("Aucune donnée à sauvegarder")
            return None

        df_out = pd.DataFrame(
            {
                "source": "global_posts",
                "strategie": "mot_cle",
                "mot_cle": None,
                "texte": df["text"],
                "date_visible": df["created_at"],
                "date_collecte": df["date_collecte"],
            }
        )

        filename = RAW_DATA_DIR / f"{prefix}.csv"

        try:
            RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

            if filename.exists():
                df_out.to_csv(
                    filename,
                    mode="a",
                    header=False,
                    index=False,
                    encoding="utf-8",
                )
                logger.info(f"✅ Données ajoutées à : {filename}")
            else:
                df_out.to_csv(filename, index=False, encoding="utf-8")
                logger.info(f"✅ Fichier créé : {filename}")

            logger.info(f"   Taille batch courant : {len(df_out)} lignes")
            return filename

        except Exception as e:
            logger.error(f"Erreur sauvegarde : {e}")
            return None

    def display_statistics(self, df: pd.DataFrame) -> None:
        """Affiche les statistiques de collecte."""
        if df.empty:
            return

        logger.info("\n" + "=" * 60)
        logger.info("📊 STATISTIQUES DE COLLECTE")
        logger.info("=" * 60)

        logger.info("\n📈 Volume :")
        logger.info(f"   • Total tweets : {len(df)}")
        logger.info(f"   • Utilisateurs uniques : {df['author_id'].nunique()}")

        logger.info("\n📅 Période :")
        logger.info(f"   • Plus ancien : {df['created_at'].min()}")
        logger.info(f"   • Plus récent : {df['created_at'].max()}")

        logger.info("\n💬 Engagement :")
        logger.info(f"   • Total likes : {df['likes'].sum()}")
        logger.info(f"   • Total retweets : {df['retweets'].sum()}")
        logger.info(f"   • Total replies : {df['replies'].sum()}")
        logger.info(f"   • Moyenne likes/tweet : {df['likes'].mean():.2f}")

        logger.info("\n🌍 Langues :")
        for lang, count in df["langue"].value_counts().head(5).items():
            logger.info(f"   • {lang} : {count} tweets")

        logger.info("\n" + "=" * 60)


def main():
    """Point d'entrée principal"""
    print("=" * 70)
    print("🚀 COLLECTE TWEETS SENELEC")
    print("=" * 70)

    logger.info("Démarrage collecte tweets SENELEC")

    try:
        collector = TwitterCollector()

        print("\n📡 Collecte en cours...")
        df_tweets = collector.collect_all_keywords(max_per_keyword=100)

        if not df_tweets.empty:
            collector.display_statistics(df_tweets)

            print("\n💾 Sauvegarde...")
            filepath = collector.save_data(df_tweets)

            if filepath:
                print("\n✅ COLLECTE TERMINÉE")
                print(f"📁 Fichier : {filepath}")
                print(f"📊 Total : {len(df_tweets)} tweets")
                logger.info("Collecte terminée avec succès")
            else:
                print("\n❌ Erreur lors de la sauvegarde")
                logger.error("Échec sauvegarde")
        else:
            print("\n⚠️ Aucun tweet collecté")
            logger.warning("Collecte vide")

    except KeyboardInterrupt:
        print("\n\n⚠️ Collecte interrompue par l'utilisateur")
        logger.warning("Collecte interrompue")

    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        logger.error(f"Erreur fatale : {e}")
        import traceback

        logger.error(traceback.format_exc())
        sys.exit(1)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
