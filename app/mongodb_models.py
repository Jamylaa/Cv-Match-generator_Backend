"""
Modèles MongoDB avec Beanie ODM pour le système de matching intelligent
"""
from beanie import Document, Indexed
from pydantic import Field, BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

class MatchLevel(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"

class EntityType(str, Enum):
    CANDIDATE = "candidate"
    OFFER = "offer"

class Candidate(Document):
    """Modèle MongoDB pour les candidats"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), alias="_id")
    name: str = Field(..., min_length=1, max_length=200)
    email: Optional[str] = Field(None, max_length=255)
    phone: Optional[str] = Field(None, max_length=20)
    text: str = Field(default="", max_length=10000)
    skills: List[str] = Field(default_factory=list)
    experience_years: Optional[int] = Field(None, ge=0, le=50)
    location: Optional[str] = Field(None, max_length=100)
    salary_expectation: Optional[float] = Field(None, ge=0)
    availability: Optional[str] = Field(None, max_length=100)
    cv_file_path: Optional[str] = Field(None, max_length=500)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True)
    
    # Métadonnées pour l'IA
    ai_confidence_score: Optional[float] = Field(None, ge=0, le=1)
    skill_extraction_method: Optional[str] = Field(None, max_length=50)
    last_ai_analysis: Optional[datetime] = None
    
    class Settings:
        name = "candidates"
        indexes = [
            "name",
            "skills",
            "location",
            "created_at",
            "is_active"
        ]

class Offer(Document):
    """Modèle MongoDB pour les offres d'emploi"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), alias="_id")
    title: str = Field(..., min_length=1, max_length=200)
    company: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=10, max_length=10000)
    skills: List[str] = Field(default_factory=list)
    required_experience: Optional[int] = Field(None, ge=0, le=50)
    location: Optional[str] = Field(None, max_length=100)
    salary_min: Optional[float] = Field(None, ge=0)
    salary_max: Optional[float] = Field(None, ge=0)
    employment_type: Optional[str] = Field(None, max_length=50)  # full-time, part-time, contract, etc.
    remote_allowed: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    is_active: bool = Field(default=True)
    
    # Métadonnées pour l'IA
    ai_confidence_score: Optional[float] = Field(None, ge=0, le=1)
    skill_extraction_method: Optional[str] = Field(None, max_length=50)
    last_ai_analysis: Optional[datetime] = None
    
    class Settings:
        name = "offers"
        indexes = [
            "title",
            "company",
            "skills",
            "location",
            "created_at",
            "is_active",
            "expires_at"
        ]

class MatchResult(Document):
    """Modèle MongoDB pour stocker les résultats de matching"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), alias="_id")
    candidate_id: str = Field(..., index=True)
    offer_id: str = Field(..., index=True)
    score: float = Field(..., ge=0, le=1)
    score_percentage: float = Field(..., ge=0, le=100)
    match_level: MatchLevel
    breakdown: Dict[str, Any] = Field(default_factory=dict)
    common_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    extra_skills: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    compatibility: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    algorithm_version: str = Field(default="2.0.0")
    
    class Settings:
        name = "match_results"
        indexes = [
            "candidate_id",
            "offer_id",
            "score",
            "match_level",
            "created_at"
        ]

class AISkillExtraction(Document):
    """Modèle MongoDB pour stocker les extractions de compétences par IA"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), alias="_id")
    entity_id: str = Field(..., index=True)
    entity_type: EntityType
    original_text: str = Field(..., max_length=10000)
    extracted_skills: List[str] = Field(default_factory=list)
    extraction_method: str = Field(..., max_length=50)  # "ai", "fallback", "manual"
    confidence_score: Optional[float] = Field(None, ge=0, le=1)
    processing_time_ms: Optional[int] = Field(None, ge=0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Settings:
        name = "ai_skill_extractions"
        indexes = [
            "entity_id",
            "entity_type",
            "extraction_method",
            "created_at"
        ]

class AnomalyDetection(Document):
    """Modèle MongoDB pour stocker les détections d'anomalies"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), alias="_id")
    entity_id: str = Field(..., index=True)
    entity_type: EntityType
    anomaly_score: float = Field(..., ge=0, le=1)
    anomaly_type: str = Field(..., max_length=100)
    description: str = Field(..., max_length=1000)
    suggestions: List[str] = Field(default_factory=list)
    is_resolved: bool = Field(default=False)
    resolved_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Settings:
        name = "anomaly_detections"
        indexes = [
            "entity_id",
            "entity_type",
            "anomaly_score",
            "is_resolved",
            "created_at"
        ]

class SystemMetrics(Document):
    """Modèle MongoDB pour stocker les métriques système"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), alias="_id")
    metric_name: str = Field(..., index=True)
    value: float = Field(...)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Settings:
        name = "system_metrics"
        indexes = [
            "metric_name",
            "timestamp"
        ]

class AIRecommendation(Document):
    """Modèle MongoDB pour stocker les recommandations IA"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), alias="_id")
    candidate_id: str = Field(..., index=True)
    recommendation_type: str = Field(..., max_length=50)
    title: str = Field(..., max_length=200)
    description: str = Field(..., max_length=1000)
    priority: str = Field(..., max_length=20)  # "high", "medium", "low"
    action: str = Field(..., max_length=500)
    impact: str = Field(..., max_length=500)
    is_applied: bool = Field(default=False)
    applied_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Settings:
        name = "ai_recommendations"
        indexes = [
            "candidate_id",
            "recommendation_type",
            "priority",
            "is_applied",
            "created_at"
        ]

# Modèles Pydantic pour les requêtes API
class CandidateCreate(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    text: Optional[str] = ""
    skills: Optional[List[str]] = []
    experience_years: Optional[int] = None
    location: Optional[str] = None
    salary_expectation: Optional[float] = None
    availability: Optional[str] = None

class CandidateUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    text: Optional[str] = None
    skills: Optional[List[str]] = None
    experience_years: Optional[int] = None
    location: Optional[str] = None
    salary_expectation: Optional[float] = None
    availability: Optional[str] = None
    is_active: Optional[bool] = None

class OfferCreate(BaseModel):
    title: str
    company: str
    description: str
    skills: Optional[List[str]] = []
    required_experience: Optional[int] = None
    location: Optional[str] = None
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    employment_type: Optional[str] = None
    remote_allowed: bool = False
    expires_at: Optional[datetime] = None

class OfferUpdate(BaseModel):
    title: Optional[str] = None
    company: Optional[str] = None
    description: Optional[str] = None
    skills: Optional[List[str]] = None
    required_experience: Optional[int] = None
    location: Optional[str] = None
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    employment_type: Optional[str] = None
    remote_allowed: Optional[bool] = None
    expires_at: Optional[datetime] = None
    is_active: Optional[bool] = None

class MatchRequest(BaseModel):
    candidate_id: str
    top_k: int = Field(default=5, ge=1, le=50)
    min_score: float = Field(default=0.0, ge=0, le=1)
    use_ai: bool = True
    algorithm_version: str = "2.0.0"

class MatchResponse(BaseModel):
    candidate: Candidate
    matches: List[Dict[str, Any]]
    total_matches: int
    algorithm_used: str
    processing_time_ms: int
    filters_applied: Dict[str, Any]

# Configuration des collections MongoDB
DOCUMENT_MODELS = [
    Candidate,
    Offer,
    MatchResult,
    AISkillExtraction,
    AnomalyDetection,
    SystemMetrics,
    AIRecommendation
]
