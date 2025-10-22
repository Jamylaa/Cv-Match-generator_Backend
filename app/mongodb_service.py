"""
Service MongoDB pour le système de matching intelligent
Gestion des opérations CRUD et des requêtes avancées
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
from .mongodb_models import (
    Candidate, Offer, MatchResult, AISkillExtraction, 
    AnomalyDetection, SystemMetrics, AIRecommendation,
    DOCUMENT_MODELS, MatchLevel
)
import os

logger = logging.getLogger(__name__)

class MongoDBService:
    """Service principal pour les opérations MongoDB"""
    
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.database = None
        self.is_connected = False
    
    async def connect(self, mongodb_url: str = None, database_name: str = "matching_ai"):
        """Établit la connexion à MongoDB"""
        try:
            if not mongodb_url:
                mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
            
            self.client = AsyncIOMotorClient(mongodb_url)
            self.database = self.client[database_name]
            
            # Initialiser Beanie avec les modèles
            await init_beanie(
                database=self.database,
                document_models=DOCUMENT_MODELS
            )
            
            self.is_connected = True
            logger.info(f"Connected to MongoDB: {database_name}")
            
            # Créer les index si nécessaire
            await self.create_indexes()
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    async def disconnect(self):
        """Ferme la connexion à MongoDB"""
        if self.client:
            self.client.close()
            self.is_connected = False
            logger.info("Disconnected from MongoDB")
    
    async def create_indexes(self):
        """Crée les index pour optimiser les performances"""
        try:
            # Index pour les candidats
            await Candidate.create_index("name")
            await Candidate.create_index("skills")
            await Candidate.create_index("location")
            await Candidate.create_index("created_at")
            await Candidate.create_index("is_active")
            
            # Index pour les offres
            await Offer.create_index("title")
            await Offer.create_index("company")
            await Offer.create_index("skills")
            await Offer.create_index("location")
            await Offer.create_index("created_at")
            await Offer.create_index("is_active")
            await Offer.create_index("expires_at")
            
            # Index pour les résultats de matching
            await MatchResult.create_index("candidate_id")
            await MatchResult.create_index("offer_id")
            await MatchResult.create_index("score")
            await MatchResult.create_index("match_level")
            await MatchResult.create_index("created_at")
            
            # Index composés pour les requêtes complexes
            await MatchResult.create_index([("candidate_id", 1), ("score", -1)])
            await MatchResult.create_index([("offer_id", 1), ("score", -1)])
            
            logger.info("MongoDB indexes created successfully")
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
    
    # Opérations sur les candidats
    async def create_candidate(self, candidate_data: Dict[str, Any]) -> Candidate:
        """Crée un nouveau candidat"""
        candidate = Candidate(**candidate_data)
        await candidate.insert()
        logger.info(f"Created candidate: {candidate.id}")
        return candidate
    
    async def get_candidate(self, candidate_id: str) -> Optional[Candidate]:
        """Récupère un candidat par ID"""
        return await Candidate.get(candidate_id)
    
    async def get_candidates(
        self, 
        skip: int = 0, 
        limit: int = 100,
        is_active: bool = True,
        skills: List[str] = None,
        location: str = None
    ) -> List[Candidate]:
        """Récupère une liste de candidats avec filtres"""
        query = {"is_active": is_active}
        
        if skills:
            query["skills"] = {"$in": skills}
        
        if location:
            query["location"] = {"$regex": location, "$options": "i"}
        
        return await Candidate.find(query).skip(skip).limit(limit).to_list()
    
    async def update_candidate(self, candidate_id: str, update_data: Dict[str, Any]) -> Optional[Candidate]:
        """Met à jour un candidat"""
        candidate = await Candidate.get(candidate_id)
        if candidate:
            update_data["updated_at"] = datetime.utcnow()
            await candidate.update({"$set": update_data})
            logger.info(f"Updated candidate: {candidate_id}")
        return candidate
    
    async def delete_candidate(self, candidate_id: str) -> bool:
        """Supprime un candidat (soft delete)"""
        candidate = await Candidate.get(candidate_id)
        if candidate:
            await candidate.update({"$set": {"is_active": False, "updated_at": datetime.utcnow()}})
            logger.info(f"Soft deleted candidate: {candidate_id}")
            return True
        return False
    
    # Opérations sur les offres
    async def create_offer(self, offer_data: Dict[str, Any]) -> Offer:
        """Crée une nouvelle offre"""
        offer = Offer(**offer_data)
        await offer.insert()
        logger.info(f"Created offer: {offer.id}")
        return offer
    
    async def get_offer(self, offer_id: str) -> Optional[Offer]:
        """Récupère une offre par ID"""
        return await Offer.get(offer_id)
    
    async def get_offers(
        self,
        skip: int = 0,
        limit: int = 100,
        is_active: bool = True,
        skills: List[str] = None,
        location: str = None,
        company: str = None
    ) -> List[Offer]:
        """Récupère une liste d'offres avec filtres"""
        query = {"is_active": is_active}
        
        if skills:
            query["skills"] = {"$in": skills}
        
        if location:
            query["location"] = {"$regex": location, "$options": "i"}
        
        if company:
            query["company"] = {"$regex": company, "$options": "i"}
        
        return await Offer.find(query).skip(skip).limit(limit).to_list()
    
    async def update_offer(self, offer_id: str, update_data: Dict[str, Any]) -> Optional[Offer]:
        """Met à jour une offre"""
        offer = await Offer.get(offer_id)
        if offer:
            update_data["updated_at"] = datetime.utcnow()
            await offer.update({"$set": update_data})
            logger.info(f"Updated offer: {offer_id}")
        return offer
    
    async def delete_offer(self, offer_id: str) -> bool:
        """Supprime une offre (soft delete)"""
        offer = await Offer.get(offer_id)
        if offer:
            await offer.update({"$set": {"is_active": False, "updated_at": datetime.utcnow()}})
            logger.info(f"Soft deleted offer: {offer_id}")
            return True
        return False
    
    # Opérations de matching
    async def save_match_result(self, match_data: Dict[str, Any]) -> MatchResult:
        """Sauvegarde un résultat de matching"""
        match_result = MatchResult(**match_data)
        await match_result.insert()
        return match_result
    
    async def get_match_history(
        self,
        candidate_id: str = None,
        offer_id: str = None,
        limit: int = 100
    ) -> List[MatchResult]:
        """Récupère l'historique des matchings"""
        query = {}
        
        if candidate_id:
            query["candidate_id"] = candidate_id
        
        if offer_id:
            query["offer_id"] = offer_id
        
        return await MatchResult.find(query).sort("created_at", -1).limit(limit).to_list()
    
    async def get_best_matches(
        self,
        candidate_id: str,
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[MatchResult]:
        """Récupère les meilleurs matches pour un candidat"""
        return await MatchResult.find({
            "candidate_id": candidate_id,
            "score": {"$gte": min_score}
        }).sort("score", -1).limit(top_k).to_list()
    
    # Opérations IA
    async def save_skill_extraction(self, extraction_data: Dict[str, Any]) -> AISkillExtraction:
        """Sauvegarde une extraction de compétences par IA"""
        extraction = AISkillExtraction(**extraction_data)
        await extraction.insert()
        return extraction
    
    async def save_anomaly_detection(self, anomaly_data: Dict[str, Any]) -> AnomalyDetection:
        """Sauvegarde une détection d'anomalie"""
        anomaly = AnomalyDetection(**anomaly_data)
        await anomaly.insert()
        return anomaly
    
    async def save_ai_recommendation(self, recommendation_data: Dict[str, Any]) -> AIRecommendation:
        """Sauvegarde une recommandation IA"""
        recommendation = AIRecommendation(**recommendation_data)
        await recommendation.insert()
        return recommendation
    
    async def get_ai_recommendations(
        self,
        candidate_id: str,
        recommendation_type: str = None,
        is_applied: bool = None
    ) -> List[AIRecommendation]:
        """Récupère les recommandations IA pour un candidat"""
        query = {"candidate_id": candidate_id}
        
        if recommendation_type:
            query["recommendation_type"] = recommendation_type
        
        if is_applied is not None:
            query["is_applied"] = is_applied
        
        return await AIRecommendation.find(query).sort("created_at", -1).to_list()
    
    # Métriques et statistiques
    async def save_system_metric(self, metric_name: str, value: float, metadata: Dict[str, Any] = None):
        """Sauvegarde une métrique système"""
        metric = SystemMetrics(
            metric_name=metric_name,
            value=value,
            metadata=metadata or {}
        )
        await metric.insert()
    
    async def get_system_metrics(
        self,
        metric_name: str = None,
        start_date: datetime = None,
        end_date: datetime = None,
        limit: int = 1000
    ) -> List[SystemMetrics]:
        """Récupère les métriques système"""
        query = {}
        
        if metric_name:
            query["metric_name"] = metric_name
        
        if start_date or end_date:
            query["timestamp"] = {}
            if start_date:
                query["timestamp"]["$gte"] = start_date
            if end_date:
                query["timestamp"]["$lte"] = end_date
        
        return await SystemMetrics.find(query).sort("timestamp", -1).limit(limit).to_list()
    
    # Statistiques avancées
    async def get_matching_statistics(self) -> Dict[str, Any]:
        """Récupère les statistiques de matching"""
        try:
            total_candidates = await Candidate.find({"is_active": True}).count()
            total_offers = await Offer.find({"is_active": True}).count()
            total_matches = await MatchResult.find().count()
            
            # Statistiques par niveau de match
            match_levels = await MatchResult.aggregate([
                {"$group": {"_id": "$match_level", "count": {"$sum": 1}}}
            ]).to_list()
            
            # Score moyen
            avg_score = await MatchResult.aggregate([
                {"$group": {"_id": None, "avg_score": {"$avg": "$score"}}}
            ]).to_list()
            
            avg_score_value = avg_score[0]["avg_score"] if avg_score else 0
            
            return {
                "total_candidates": total_candidates,
                "total_offers": total_offers,
                "total_matches": total_matches,
                "average_score": round(avg_score_value, 3),
                "match_levels": {item["_id"]: item["count"] for item in match_levels},
                "generated_at": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting matching statistics: {e}")
            return {}
    
    async def get_skill_analytics(self) -> Dict[str, Any]:
        """Récupère les analytics sur les compétences"""
        try:
            # Compétences les plus demandées
            offer_skills = await Offer.aggregate([
                {"$unwind": "$skills"},
                {"$group": {"_id": "$skills", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 20}
            ]).to_list()
            
            # Compétences les plus courantes chez les candidats
            candidate_skills = await Candidate.aggregate([
                {"$unwind": "$skills"},
                {"$group": {"_id": "$skills", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 20}
            ]).to_list()
            
            return {
                "most_demanded_skills": offer_skills,
                "most_common_candidate_skills": candidate_skills,
                "generated_at": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting skill analytics: {e}")
            return {}
    
    # Nettoyage et maintenance
    async def cleanup_old_data(self, days: int = 30):
        """Nettoie les anciennes données"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Supprimer les anciens résultats de matching
            deleted_matches = await MatchResult.find({"created_at": {"$lt": cutoff_date}}).delete()
            
            # Supprimer les anciennes métriques
            deleted_metrics = await SystemMetrics.find({"timestamp": {"$lt": cutoff_date}}).delete()
            
            logger.info(f"Cleaned up {deleted_matches} old matches and {deleted_metrics} old metrics")
            return {"deleted_matches": deleted_matches, "deleted_metrics": deleted_metrics}
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return {}

# Instance globale du service
mongodb_service = MongoDBService()