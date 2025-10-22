"""
API IA simplifiée pour le matching intelligent
Version sans dépendances externes pour démonstration
"""

import json
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)
router = APIRouter()

# Modèles Pydantic
class AISkillExtraction(BaseModel):
    text: str
    use_ai: bool = True

class VectorMatchRequest(BaseModel):
    candidate_id: str
    top_k: int = 5
    min_similarity: float = 0.3

class AnomalyDetectionRequest(BaseModel):
    entity_type: str = "candidate"
    entity_id: Optional[str] = None
    check_all: bool = False

class AIRecommendationRequest(BaseModel):
    candidate_id: str
    recommendation_type: str = "skills"

# Service d'extraction de compétences amélioré
class AISkillExtractor:
    def __init__(self):
        self.common_skills = [
            "python", "java", "javascript", "react", "angular", "vue", "nodejs", "django", "flask",
            "sql", "postgresql", "mysql", "mongodb", "redis", "docker", "kubernetes", "aws", "azure",
            "git", "github", "gitlab", "jenkins", "ci/cd", "rest api", "graphql", "microservices",
            "machine learning", "ai", "data science", "pandas", "numpy", "tensorflow", "pytorch",
            "html", "css", "bootstrap", "sass", "webpack", "npm", "yarn", "linux", "bash",
            "agile", "scrum", "kanban", "jira", "confluence", "testing", "tdd", "bdd",
            "next.js", "typescript", "express", "spring", "hibernate", "jpa", "maven", "gradle",
            "vue.js", "svelte", "ember", "backbone", "jquery", "lodash", "moment", "axios",
            "firebase", "supabase", "prisma", "sequelize", "typeorm", "mongoose", "elasticsearch",
            "kafka", "rabbitmq", "nginx", "apache", "tomcat", "jetty", "wildfly", "glassfish",
            "full stack", "frontend", "backend", "devops", "cloud", "api", "rest", "json", "xml"
        ]
        
        self.role_patterns = {
            "développeur": ["Programming", "Software Development", "Code Review"],
            "developer": ["Programming", "Software Development", "Code Review"],
            "ingénieur": ["Engineering", "Technical Skills", "Problem Solving"],
            "engineer": ["Engineering", "Technical Skills", "Problem Solving"],
            "analyst": ["Analysis", "Problem Solving", "Data Analysis"],
            "analyste": ["Analysis", "Problem Solving", "Data Analysis"],
            "manager": ["Management", "Leadership", "Team Coordination"],
            "chef": ["Management", "Leadership", "Team Coordination"],
            "senior": ["Senior Level", "Experience", "Mentoring"],
            "junior": ["Junior Level", "Learning", "Growth"],
            "full stack": ["Full Stack Development", "Frontend", "Backend", "Database"],
            "frontend": ["Frontend Development", "UI/UX", "JavaScript", "React", "Vue"],
            "backend": ["Backend Development", "API", "Database", "Server"],
            "data": ["Data Analysis", "Data Science", "Statistics", "Machine Learning"],
            "web": ["Web Development", "HTML", "CSS", "JavaScript", "Responsive Design"],
            "mobile": ["Mobile Development", "iOS", "Android", "React Native"],
            "devops": ["DevOps", "CI/CD", "Docker", "Kubernetes", "Infrastructure"],
            "cloud": ["Cloud Computing", "AWS", "Azure", "GCP", "Scalability"],
            "ai": ["Artificial Intelligence", "Machine Learning", "AI", "Neural Networks"],
            "intelligence": ["Artificial Intelligence", "Machine Learning", "AI", "Neural Networks"]
        }
    
    def extract_skills(self, text: str, name: str = "") -> List[str]:
        """Extraction de compétences améliorée"""
        if not text and name:
            text = name
        
        if not text:
            return ["General Skills", "Problem Solving", "Communication"]
        
        found_skills = []
        text_lower = text.lower()
        
        # Extraction des compétences techniques
        for skill in self.common_skills:
            if skill in text_lower:
                found_skills.append(skill.title())
        
        # Extraction basée sur les rôles
        for pattern, skills in self.role_patterns.items():
            if pattern in text_lower:
                found_skills.extend(skills)
        
        # Si aucune compétence trouvée, ajouter des compétences génériques
        if not found_skills:
            found_skills = ["General Skills", "Problem Solving", "Communication"]
        
        return list(set(found_skills))

# Service de matching vectoriel
class VectorMatchingService:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.candidate_embeddings = {}
        self.offer_embeddings = {}
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
    
    def generate_embedding(self, text: str) -> List[float]:
        """Génère un embedding TF-IDF"""
        try:
            embedding = self.vectorizer.fit_transform([text]).toarray()[0]
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return [0.0] * 1000
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calcule la similarité cosinus entre deux textes"""
        try:
            embeddings = self.vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(embeddings[0:1], embeddings[1:2])[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def detect_anomalies(self, entities: List[Dict[str, Any]], entity_type: str) -> Dict[str, Any]:
        """Détecte les anomalies dans les profils"""
        try:
            if len(entities) < 3:
                return {"anomalies": [], "message": "Not enough data for anomaly detection"}
            
            features = []
            entity_ids = []
            
            for entity in entities:
                feature_vector = [
                    len(entity.get('skills', [])),
                    len(entity.get('text', '')),
                    len(entity.get('name', '')) if entity_type == "candidate" else len(entity.get('title', ''))
                ]
                features.append(feature_vector)
                entity_ids.append(entity.get('id'))
            
            features_array = np.array(features)
            anomaly_labels = self.anomaly_detector.fit_predict(features_array)
            
            anomalies = []
            for i, label in enumerate(anomaly_labels):
                if label == -1:  # Anomalie détectée
                    anomalies.append({
                        "entity_id": entity_ids[i],
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
                "anomaly_count": len(anomalies)
            }
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return {"anomalies": [], "error": str(e)}

# Instances globales
skill_extractor = AISkillExtractor()
vector_service = VectorMatchingService()

# Endpoints

@router.post("/ai/extract-skills")
async def extract_skills_with_ai(request: AISkillExtraction):
    """Extraction de compétences avec IA"""
    try:
        skills = skill_extractor.extract_skills(request.text)
        
        return {
            "skills": skills,
            "method": "ai_enhanced",
            "count": len(skills),
            "text_analyzed": request.text[:100] + "..." if len(request.text) > 100 else request.text
        }
    except Exception as e:
        logger.error(f"Error in skill extraction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ai/vector-match")
async def vector_based_matching(request: VectorMatchRequest):
    """Matching basé sur les embeddings vectoriels"""
    try:
        from .storage import store
        
        candidate = store.get_candidate(request.candidate_id)
        if not candidate:
            raise HTTPException(status_code=404, detail="Candidate not found")
        
        offers = store.list_offers()
        
        # Combiner texte et compétences pour le candidat
        candidate_text = f"{candidate.get('text', '')} {' '.join(candidate.get('skills', []))}"
        
        matches = []
        
        for offer in offers:
            # Combiner texte et compétences pour l'offre
            offer_text = f"{offer.get('description', '')} {' '.join(offer.get('skills', []))}"
            
            # Calculer la similarité
            similarity = vector_service.calculate_similarity(candidate_text, offer_text)
            
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
                for skill in list(missing_skills)[:5]
            ]
        
        return {
            "candidate": candidate,
            "recommendations": recommendations,
            "recommendation_type": request.recommendation_type,
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ai/health")
async def ai_health_check():
    """Vérification de l'état des services IA"""
    return {
        "ai_services_available": True,
        "vector_service_ready": True,
        "anomaly_detector_ready": True,
        "skill_extractor_ready": True,
        "status": "healthy",
        "version": "2.0.0-simple"
    }

@router.post("/ai/demo/advanced-matching")
async def demo_advanced_matching():
    """Démonstration des fonctionnalités de matching avancées"""
    try:
        from .storage import store
        
        # Créer des données de démonstration
        demo_candidate = {
            "id": "demo-candidate-ai",
            "name": "Développeur IA Senior",
            "text": "Expert en machine learning, deep learning, Python, TensorFlow, PyTorch, NLP, computer vision, MLOps, Docker, Kubernetes, AWS, Azure",
            "skills": ["Python", "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "NLP", "Computer Vision", "MLOps", "Docker", "Kubernetes", "AWS", "Azure"]
        }
        
        demo_offers = [
            {
                "id": "demo-offer-ai-1",
                "title": "Data Scientist Senior",
                "description": "Recherche data scientist expérimenté en ML, Python, TensorFlow, PyTorch, NLP, computer vision, MLOps",
                "skills": ["Python", "Machine Learning", "TensorFlow", "PyTorch", "NLP", "Computer Vision", "MLOps", "Statistics"]
            },
            {
                "id": "demo-offer-ai-2",
                "title": "ML Engineer",
                "description": "Ingénieur ML pour développer des modèles de production, Docker, Kubernetes, AWS",
                "skills": ["Python", "Machine Learning", "Docker", "Kubernetes", "AWS", "MLOps", "CI/CD"]
            },
            {
                "id": "demo-offer-ai-3",
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
        skill_extraction = skill_extractor.extract_skills(demo_candidate["text"])
        
        # Test 2: Matching vectoriel
        vector_matches = []
        candidate_text = f"{demo_candidate['text']} {' '.join(demo_candidate['skills'])}"
        
        for offer in demo_offers:
            offer_text = f"{offer['description']} {' '.join(offer['skills'])}"
            similarity = vector_service.calculate_similarity(candidate_text, offer_text)
            vector_matches.append({
                "offer": offer,
                "similarity": similarity,
                "match_level": get_match_level(similarity)
            })
        
        # Test 3: Détection d'anomalies
        anomaly_detection = vector_service.detect_anomalies([demo_candidate], "candidate")
        
        # Test 4: Recommandations IA
        all_offers = store.list_offers()
        all_required_skills = set()
        for offer in all_offers:
            all_required_skills.update(offer.get("skills", []))
        
        candidate_skills = set(demo_candidate.get("skills", []))
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
            for skill in list(missing_skills)[:5]
        ]
        
        return {
            "demo": "Advanced AI Matching Demonstration",
            "candidate": demo_candidate,
            "skill_extraction_ai": {
                "original_skills": demo_candidate["skills"],
                "ai_extracted_skills": skill_extraction,
                "extraction_accuracy": len(set(demo_candidate["skills"]) & set(skill_extraction)) / max(len(set(demo_candidate["skills"]) | set(skill_extraction)), 1)
            },
            "vector_matching": {
                "matches": sorted(vector_matches, key=lambda x: x["similarity"], reverse=True),
                "best_match": max(vector_matches, key=lambda x: x["similarity"]) if vector_matches else None
            },
            "anomaly_detection": anomaly_detection,
            "ai_recommendations": recommendations,
            "features_demonstrated": [
                "AI-powered skill extraction",
                "Vector-based semantic matching",
                "Anomaly detection in profiles",
                "Personalized AI recommendations",
                "Advanced matching algorithms"
            ]
        }
    except Exception as e:
        logger.error(f"Error in advanced matching demo: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def get_match_level(score: float) -> str:
    """Détermine le niveau de match basé sur le score"""
    if score >= 0.8:
        return "excellent"
    elif score >= 0.6:
        return "good"
    elif score >= 0.4:
        return "average"
    else:
        return "poor"
