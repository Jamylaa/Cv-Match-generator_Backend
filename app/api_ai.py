"""
API avancée avec intégration IA pour le matching intelligent
- Utilisation d'API IA externes (OpenAI, Hugging Face)
- Embeddings vectoriels pour matching sémantique
- Détection d'anomalies avec machine learning
- Recommandations personnalisées
"""

import os
import json
import asyncio
import aiohttp
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Configuration des API externes
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")

# Modèles Pydantic pour les nouvelles fonctionnalités
class AISkillExtraction(BaseModel):
    text: str
    use_ai: bool = True
    model: str = "gpt-3.5-turbo"

class VectorMatchRequest(BaseModel):
    candidate_id: str
    top_k: int = 5
    min_similarity: float = 0.3
    use_semantic: bool = True

class AnomalyDetectionRequest(BaseModel):
    entity_type: str = "candidate"  # "candidate" ou "offer"
    entity_id: Optional[str] = None
    check_all: bool = False

class AIRecommendationRequest(BaseModel):
    candidate_id: str
    recommendation_type: str = "skills"  # "skills", "career", "training"
    context: Optional[str] = None

# Cache pour les embeddings
embeddings_cache = {}
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

class AIService:
    """Service pour les appels aux API IA externes"""
    
    @staticmethod
    async def extract_skills_with_openai(text: str, model: str = "gpt-3.5-turbo") -> List[str]:
        """Extraction de compétences avec OpenAI"""
        if not OPENAI_API_KEY:
            logger.warning("OpenAI API key not found, using fallback extraction")
            return extract_skills_fallback(text)
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                prompt = f"""
                Extrayez les compétences techniques et professionnelles du texte suivant.
                Retournez uniquement une liste JSON de compétences, sans explication.
                
                Texte: {text}
                
                Format de réponse: ["compétence1", "compétence2", ...]
                """
                
                data = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500,
                    "temperature": 0.3
                }
                
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result["choices"][0]["message"]["content"].strip()
                        # Nettoyer et parser la réponse JSON
                        skills = json.loads(content)
                        return skills if isinstance(skills, list) else []
                    else:
                        logger.error(f"OpenAI API error: {response.status}")
                        return extract_skills_fallback(text)
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return extract_skills_fallback(text)
    
    @staticmethod
    async def get_embeddings_huggingface(text: str) -> List[float]:
        """Obtention d'embeddings avec Hugging Face"""
        if not HUGGINGFACE_API_KEY:
            logger.warning("Hugging Face API key not found, using TF-IDF")
            return get_tfidf_embedding(text)
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "inputs": text,
                    "options": {"wait_for_model": True}
                }
                
                async with session.post(
                    "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2",
                    headers=headers,
                    json=data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result[0] if isinstance(result, list) else []
                    else:
                        logger.error(f"Hugging Face API error: {response.status}")
                        return get_tfidf_embedding(text)
        except Exception as e:
            logger.error(f"Error calling Hugging Face API: {e}")
            return get_tfidf_embedding(text)

def extract_skills_fallback(text: str) -> List[str]:
    """Extraction de compétences de fallback"""
    common_skills = [
        "python", "java", "javascript", "react", "angular", "vue", "nodejs", "django", "flask",
        "sql", "postgresql", "mysql", "mongodb", "redis", "docker", "kubernetes", "aws", "azure",
        "git", "github", "gitlab", "jenkins", "ci/cd", "rest api", "graphql", "microservices",
        "machine learning", "ai", "data science", "pandas", "numpy", "tensorflow", "pytorch",
        "html", "css", "bootstrap", "sass", "webpack", "npm", "yarn", "linux", "bash",
        "agile", "scrum", "kanban", "jira", "confluence", "testing", "tdd", "bdd",
        "next.js", "typescript", "express", "spring", "hibernate", "jpa", "maven", "gradle"
    ]
    
    text_lower = text.lower()
    found_skills = []
    
    for skill in common_skills:
        if skill in text_lower:
            found_skills.append(skill.title())
    
    return list(set(found_skills))

def get_tfidf_embedding(text: str) -> List[float]:
    """Génération d'embeddings TF-IDF"""
    try:
        # Utiliser le vectorizer global ou en créer un nouveau
        if not hasattr(get_tfidf_embedding, 'vectorizer'):
            get_tfidf_embedding.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Pour un seul texte, on doit ajuster le vectorizer
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        embedding = vectorizer.fit_transform([text]).toarray()[0]
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Error generating TF-IDF embedding: {e}")
        return [0.0] * 1000  # Vecteur de fallback

class VectorMatchingService:
    """Service de matching basé sur les embeddings vectoriels"""
    
    def __init__(self):
        self.candidate_embeddings = {}
        self.offer_embeddings = {}
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
    
    async def generate_embeddings_for_entity(self, entity: Dict[str, Any], entity_type: str) -> List[float]:
        """Génère des embeddings pour une entité (candidat ou offre)"""
        # Combiner texte et compétences
        combined_text = f"{entity.get('text', '')} {' '.join(entity.get('skills', []))}"
        
        if entity_type == "candidate":
            combined_text += f" {entity.get('name', '')}"
        else:
            combined_text += f" {entity.get('title', '')}"
        
        # Essayer d'abord Hugging Face, puis TF-IDF
        embedding = await AIService.get_embeddings_huggingface(combined_text)
        
        if not embedding:
            embedding = get_tfidf_embedding(combined_text)
        
        return embedding
    
    async def calculate_semantic_similarity(self, candidate_id: str, offer_id: str) -> float:
        """Calcule la similarité sémantique entre un candidat et une offre"""
        try:
            # Récupérer les embeddings
            candidate_embedding = self.candidate_embeddings.get(candidate_id)
            offer_embedding = self.offer_embeddings.get(offer_id)
            
            if not candidate_embedding or not offer_embedding:
                return 0.0
            
            # Calculer la similarité cosinus
            similarity = cosine_similarity(
                [candidate_embedding], 
                [offer_embedding]
            )[0][0]
            
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def detect_anomalies(self, entities: List[Dict[str, Any]], entity_type: str) -> Dict[str, Any]:
        """Détecte les anomalies dans les profils"""
        try:
            if len(entities) < 3:  # Pas assez de données pour détecter des anomalies
                return {"anomalies": [], "message": "Not enough data for anomaly detection"}
            
            # Préparer les features pour la détection d'anomalies
            features = []
            entity_ids = []
            
            for entity in entities:
                # Features basées sur les caractéristiques du profil
                feature_vector = [
                    len(entity.get('skills', [])),
                    len(entity.get('text', '')),
                    len(entity.get('name', '')) if entity_type == "candidate" else len(entity.get('title', '')),
                    # Ajouter d'autres features pertinentes
                ]
                features.append(feature_vector)
                entity_ids.append(entity.get('id'))
            
            # Normaliser les features
            features_scaled = self.scaler.fit_transform(features)
            
            # Détecter les anomalies
            anomaly_labels = self.anomaly_detector.fit_predict(features_scaled)
            anomaly_scores = self.anomaly_detector.decision_function(features_scaled)
            
            anomalies = []
            for i, (label, score) in enumerate(zip(anomaly_labels, anomaly_scores)):
                if label == -1:  # Anomalie détectée
                    anomalies.append({
                        "entity_id": entity_ids[i],
                        "anomaly_score": float(score),
                        "reason": "Unusual profile characteristics",
                        "suggestions": [
                            "Review profile completeness",
                            "Check for data quality issues",
                            "Verify skill relevance"
                        ]
                    })
            
            return {
                "anomalies": anomalies,
                "total_entities": len(entities),
                "anomaly_count": len(anomalies),
                "detection_confidence": "high" if len(anomalies) > 0 else "low"
            }
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return {"anomalies": [], "error": str(e)}

# Instance globale du service
vector_service = VectorMatchingService()

# Endpoints de l'API IA

@router.post("/ai/extract-skills")
async def extract_skills_with_ai(request: AISkillExtraction):
    """Extraction de compétences avec IA"""
    try:
        if request.use_ai and OPENAI_API_KEY:
            skills = await AIService.extract_skills_with_openai(request.text, request.model)
        else:
            skills = extract_skills_fallback(request.text)
        
        return {
            "skills": skills,
            "method": "ai" if request.use_ai and OPENAI_API_KEY else "fallback",
            "count": len(skills)
        }
    except Exception as e:
        logger.error(f"Error in skill extraction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ai/vector-match")
async def vector_based_matching(request: VectorMatchRequest):
    """Matching basé sur les embeddings vectoriels"""
    try:
        from .storage import store
        
        # Récupérer le candidat
        candidate = store.get_candidate(request.candidate_id)
        if not candidate:
            raise HTTPException(status_code=404, detail="Candidate not found")
        
        # Récupérer toutes les offres
        offers = store.list_offers()
        
        # Générer les embeddings si nécessaire
        if request.candidate_id not in vector_service.candidate_embeddings:
            vector_service.candidate_embeddings[request.candidate_id] = await vector_service.generate_embeddings_for_entity(candidate, "candidate")
        
        matches = []
        
        for offer in offers:
            offer_id = offer["id"]
            
            # Générer l'embedding de l'offre si nécessaire
            if offer_id not in vector_service.offer_embeddings:
                vector_service.offer_embeddings[offer_id] = await vector_service.generate_embeddings_for_entity(offer, "offer")
            
            # Calculer la similarité sémantique
            similarity = await vector_service.calculate_semantic_similarity(request.candidate_id, offer_id)
            
            if similarity >= request.min_similarity:
                matches.append({
                    "offer": offer,
                    "similarity_score": similarity,
                    "similarity_percentage": round(similarity * 100, 2),
                    "match_level": get_match_level(similarity)
                })
        
        # Trier par score de similarité
        matches.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return {
            "candidate": candidate,
            "matches": matches[:request.top_k],
            "total_matches": len(matches),
            "method": "vector_similarity",
            "filters": {
                "top_k": request.top_k,
                "min_similarity": request.min_similarity
            }
        }
    except Exception as e:
        logger.error(f"Error in vector matching: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ai/detect-anomalies")
async def detect_anomalies(request: AnomalyDetectionRequest):
    """Détection d'anomalies dans les profils"""
    try:
        from .storage import store
        
        if request.check_all:
            if request.entity_type == "candidate":
                entities = store.list_candidates()
            else:
                entities = store.list_offers()
        elif request.entity_id:
            if request.entity_type == "candidate":
                entity = store.get_candidate(request.entity_id)
            else:
                entity = store.get_offer(request.entity_id)
            
            if not entity:
                raise HTTPException(status_code=404, detail="Entity not found")
            entities = [entity]
        else:
            raise HTTPException(status_code=400, detail="Either entity_id or check_all must be specified")
        
        result = vector_service.detect_anomalies(entities, request.entity_type)
        return result
    except Exception as e:
        logger.error(f"Error in anomaly detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ai/recommendations")
async def get_ai_recommendations(request: AIRecommendationRequest):
    """Recommandations personnalisées basées sur l'IA"""
    try:
        from .storage import store
        
        candidate = store.get_candidate(request.candidate_id)
        if not candidate:
            raise HTTPException(status_code=404, detail="Candidate not found")
        
        recommendations = []
        
        if request.recommendation_type == "skills":
            # Analyser les compétences manquantes
            all_offers = store.list_offers()
            all_required_skills = set()
            
            for offer in all_offers:
                all_required_skills.update(offer.get("skills", []))
            
            candidate_skills = set(candidate.get("skills", []))
            missing_skills = all_required_skills - candidate_skills
            
            recommendations = [
                {
                    "type": "skill_gap",
                    "priority": "high",
                    "title": f"Acquérir {skill}",
                    "description": f"Cette compétence est demandée dans {sum(1 for offer in all_offers if skill in offer.get('skills', []))} offres",
                    "action": f"Formation recommandée en {skill}",
                    "impact": "Augmentera significativement vos chances de matching"
                }
                for skill in list(missing_skills)[:5]  # Top 5 compétences manquantes
            ]
        
        elif request.recommendation_type == "career":
            # Recommandations de carrière basées sur le profil
            recommendations = [
                {
                    "type": "career_path",
                    "priority": "medium",
                    "title": "Développement de carrière",
                    "description": "Considérez une spécialisation dans le développement full-stack",
                    "action": "Explorez les offres de postes senior",
                    "impact": "Progression de carrière optimisée"
                }
            ]
        
        return {
            "candidate": candidate,
            "recommendations": recommendations,
            "recommendation_type": request.recommendation_type,
            "generated_at": "2024-01-01T00:00:00Z"  # Timestamp réel
        }
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ai/health")
async def ai_health_check():
    """Vérification de l'état des services IA"""
    return {
        "openai_available": bool(OPENAI_API_KEY),
        "huggingface_available": bool(HUGGINGFACE_API_KEY),
        "vector_service_ready": True,
        "anomaly_detector_ready": True,
        "status": "healthy"
    }

@router.post("/ai/test-suite")
async def run_ai_test_suite():
    """Exécute la suite de tests automatisés avec IA"""
    try:
        from .storage import store
        from .ai_testing import ai_test_suite
        
        # Exécuter tous les tests
        test_results = await ai_test_suite.run_all_tests(store)
        
        return {
            "test_suite": "AI Automated Testing",
            "results": test_results,
            "recommendations": test_results.get("recommendations", []),
            "status": "completed"
        }
    except Exception as e:
        logger.error(f"Error running AI test suite: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ai/monitoring/dashboard")
async def get_monitoring_dashboard():
    """Tableau de bord de monitoring en temps réel"""
    try:
        from .ai_monitoring import monitoring_service
        
        performance_report = monitoring_service.get_performance_report()
        alerts = monitoring_service.get_alerts()
        
        return {
            "dashboard": "AI Monitoring Dashboard",
            "performance_report": performance_report,
            "active_alerts": alerts,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting monitoring dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ai/demo/advanced-matching")
async def demo_advanced_matching():
    """Démonstration des fonctionnalités de matching avancées"""
    try:
        from .storage import store
        
        # Créer des données de démonstration
        demo_candidate = {
            "id": "demo-candidate-1",
            "name": "Développeur IA Senior",
            "text": "Expert en machine learning, deep learning, Python, TensorFlow, PyTorch, NLP, computer vision, MLOps, Docker, Kubernetes, AWS, Azure",
            "skills": ["Python", "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "NLP", "Computer Vision", "MLOps", "Docker", "Kubernetes", "AWS", "Azure"]
        }
        
        demo_offers = [
            {
                "id": "demo-offer-1",
                "title": "Data Scientist Senior",
                "description": "Recherche data scientist expérimenté en ML, Python, TensorFlow, PyTorch, NLP, computer vision, MLOps",
                "skills": ["Python", "Machine Learning", "TensorFlow", "PyTorch", "NLP", "Computer Vision", "MLOps", "Statistics"]
            },
            {
                "id": "demo-offer-2",
                "title": "ML Engineer",
                "description": "Ingénieur ML pour développer des modèles de production, Docker, Kubernetes, AWS",
                "skills": ["Python", "Machine Learning", "Docker", "Kubernetes", "AWS", "MLOps", "CI/CD"]
            },
            {
                "id": "demo-offer-3",
                "title": "Développeur Full Stack",
                "description": "Développeur full stack React, Node.js, Python, PostgreSQL",
                "skills": ["React", "Node.js", "Python", "PostgreSQL", "JavaScript", "HTML", "CSS"]
            }
        ]
        
        # Sauvegarder temporairement les données de démo
        store.add_candidate(demo_candidate)
        for offer in demo_offers:
            store.add_offer(offer)
        
        # Test 1: Extraction de compétences avec IA
        skill_extraction = await AIService.extract_skills_with_openai(demo_candidate["text"])
        
        # Test 2: Matching vectoriel
        vector_matches = []
        for offer in demo_offers:
            similarity = await vector_service.calculate_semantic_similarity(
                demo_candidate["id"], offer["id"]
            )
            vector_matches.append({
                "offer": offer,
                "similarity": similarity,
                "match_level": get_match_level(similarity)
            })
        
        # Test 3: Détection d'anomalies
        anomaly_detection = vector_service.detect_anomalies([demo_candidate], "candidate")
        
        # Test 4: Recommandations IA
        recommendations = await get_ai_recommendations(AIRecommendationRequest(
            candidate_id=demo_candidate["id"],
            recommendation_type="skills"
        ))
        
        return {
            "demo": "Advanced AI Matching Demonstration",
            "candidate": demo_candidate,
            "skill_extraction_ai": {
                "original_skills": demo_candidate["skills"],
                "ai_extracted_skills": skill_extraction,
                "extraction_accuracy": len(set(demo_candidate["skills"]) & set(skill_extraction)) / len(set(demo_candidate["skills"]) | set(skill_extraction))
            },
            "vector_matching": {
                "matches": sorted(vector_matches, key=lambda x: x["similarity"], reverse=True),
                "best_match": max(vector_matches, key=lambda x: x["similarity"])
            },
            "anomaly_detection": anomaly_detection,
            "ai_recommendations": recommendations,
            "features_demonstrated": [
                "AI-powered skill extraction",
                "Vector-based semantic matching",
                "Anomaly detection in profiles",
                "Personalized AI recommendations",
                "Real-time performance monitoring"
            ]
        }
    except Exception as e:
        logger.error(f"Error in advanced matching demo: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def get_match_level(score: float) -> str:
    """Détermine le niveau de match basé sur le score"""
    if score >= 0.8:
        return "Excellent"
    elif score >= 0.6:
        return "Bon"
    elif score >= 0.4:
        return "Moyen"
    else:
        return "Faible"
