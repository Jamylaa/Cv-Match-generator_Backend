""" Configuration et connexion MongoDB
Utilise Beanie (ODM) pour une intégration facile avec FastAPI et Pydantic
"""
import os
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Configuration MongoDB
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "matching_intelligent")
COLLECTION_PREFIX = os.getenv("COLLECTION_PREFIX", "ai_matching")

class MongoDBConfig:
    """Configuration MongoDB"""
    
    def __init__(self):
        self.url = MONGODB_URL
        self.database_name = DATABASE_NAME
        self.collection_prefix = COLLECTION_PREFIX
        self.client: Optional[AsyncIOMotorClient] = None
        self.database = None
    
    async def connect(self):
        """Établit la connexion à MongoDB"""
        try:
            # Créer le client MongoDB
            self.client = AsyncIOMotorClient(self.url)
            self.database = self.client[self.database_name]
            
            # Tester la connexion
            await self.client.admin.command('ping')
            logger.info(f"✅ Connexion MongoDB établie: {self.database_name}")
            
            return True
        except Exception as e:
            logger.error(f"❌ Erreur de connexion MongoDB: {e}")
            return False
    
    async def disconnect(self):
        """Ferme la connexion MongoDB"""
        if self.client:
            self.client.close()
            logger.info("🔌 Connexion MongoDB fermée")
    
    def get_collection_name(self, model_name: str) -> str:
        """Génère le nom de collection avec préfixe"""
        return f"{self.collection_prefix}_{model_name.lower()}"

# Instance globale de configuration
mongodb_config = MongoDBConfig()

async def init_database():
    """Initialise la base de données et les collections"""
    try:
        # Importer les modèles après la connexion
        from .models_mongodb import Candidate, Offer, MatchResult, UserSession
        
        # Établir la connexion
        if not await mongodb_config.connect():
            raise Exception("Impossible de se connecter à MongoDB")
        
        # Initialiser Beanie avec les modèles
        await init_beanie(
            database=mongodb_config.database,
            document_models=[Candidate, Offer, MatchResult, UserSession]
        )
        
        # Créer les index pour optimiser les performances
        await create_indexes()
        
        logger.info("🚀 Base de données MongoDB initialisée avec succès")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur d'initialisation de la base de données: {e}")
        return False

async def create_indexes():
    """Crée les index MongoDB pour optimiser les performances"""
    try:
        from .models_mongodb import Candidate, Offer, MatchResult, UserSession
        
        # Index pour les candidats
        await Candidate.create_index("name")
        await Candidate.create_index("skills")
        await Candidate.create_index("created_at")
        await Candidate.create_index([("name", "text"), ("text", "text")])  # Index de recherche textuelle
        
        # Index pour les offres
        await Offer.create_index("title")
        await Offer.create_index("skills")
        await Offer.create_index("created_at")
        await Offer.create_index([("title", "text"), ("description", "text")])  # Index de recherche textuelle
        
        # Index pour les résultats de matching
        await MatchResult.create_index("candidate_id")
        await MatchResult.create_index("offer_id")
        await MatchResult.create_index("score")
        await MatchResult.create_index("created_at")
        await MatchResult.create_index([("candidate_id", 1), ("score", -1)])  # Index composé
        
        # Index pour les sessions utilisateur
        await UserSession.create_index("user_id")
        await UserSession.create_index("session_id")
        await UserSession.create_index("created_at")
        
        logger.info("📊 Index MongoDB créés avec succès")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la création des index: {e}")

async def get_database():
    """Retourne l'instance de la base de données"""
    return mongodb_config.database

async def get_collection(collection_name: str):
    """Retourne une collection spécifique"""
    return mongodb_config.database[collection_name]

# Fonctions utilitaires pour la migration
async def migrate_from_json():
    """Migre les données du fichier JSON vers MongoDB"""
    try:
        import json
        from pathlib import Path
        from .models_mongodb import Candidate, Offer
        
        # Lire les données existantes
        data_file = Path("backend/data_store.json")
        if not data_file.exists():
            logger.info("Aucun fichier de données existant à migrer")
            return True
        
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Migrer les candidats
        candidates_data = data.get("candidates", {})
        migrated_candidates = 0
        for candidate_id, candidate_data in candidates_data.items():
            try:
                candidate = Candidate(**candidate_data)
                await candidate.insert()
                migrated_candidates += 1
            except Exception as e:
                logger.warning(f"Erreur migration candidat {candidate_id}: {e}")
        
        # Migrer les offres
        offers_data = data.get("offers", {})
        migrated_offers = 0
        for offer_id, offer_data in offers_data.items():
            try:
                offer = Offer(**offer_data)
                await offer.insert()
                migrated_offers += 1
            except Exception as e:
                logger.warning(f"Erreur migration offre {offer_id}: {e}")
        
        logger.info(f"✅ Migration terminée: {migrated_candidates} candidats, {migrated_offers} offres")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la migration: {e}")
        return False

# Fonctions de santé de la base de données
async def check_database_health():
    """Vérifie la santé de la base de données"""
    try:
        if not mongodb_config.client:
            return {"status": "disconnected", "error": "Client non initialisé"}
        
        # Ping de la base de données
        await mongodb_config.client.admin.command('ping')
        
        # Vérifier les collections
        collections = await mongodb_config.database.list_collection_names()
        
        return {
            "status": "healthy",
            "database": mongodb_config.database_name,
            "collections": collections,
            "url": mongodb_config.url
        }
        
    except Exception as e:
        return {"status": "error", "error": str(e)}

# Fonctions de statistiques
async def get_database_stats():
    """Retourne les statistiques de la base de données"""
    try:
        from .models_mongodb import Candidate, Offer, MatchResult, UserSession
        
        stats = {
            "candidates_count": await Candidate.count(),
            "offers_count": await Offer.count(),
            "match_results_count": await MatchResult.count(),
            "user_sessions_count": await UserSession.count(),
            "database_size": await mongodb_config.database.command("dbStats")["dataSize"]
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des statistiques: {e}")
        return {"error": str(e)}
