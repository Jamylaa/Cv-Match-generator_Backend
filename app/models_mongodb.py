"""
Modèles MongoDB avec Beanie ODM
Définit les documents MongoDB avec validation Pydantic
"""

from beanie import Document, Indexed
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

class MatchLevel(str, Enum):
    """Niveaux de matching"""
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"

class UserRole(str, Enum):
    """Rôles utilisateur"""
    ADMIN = "admin"
    RECRUITER = "recruiter"
    CANDIDATE = "candidate"
    GUEST = "guest"

class Candidate(Document):
    """Modèle candidat pour MongoDB"""
    
    # Champs de base
    name: str = Field(..., min_length=2, max_length=100)
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    
    # Profil professionnel
    title: Optional[str] = None
    experience_years: Optional[int] = Field(None, ge=0, le=50)
    current_position: Optional[str] = None
    current_company: Optional[str] = None
    
    # Contenu et compétences
    text: str = Field(default="", max_length=10000)
    skills: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    
    # Documents et fichiers
    resume_url: Optional[str] = None
    portfolio_url: Optional[str] = None
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None
    
    # Métadonnées
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True
    source: str = Field(default="manual")  # manual, upload, api, import
    
    # Préférences
    preferred_locations: List[str] = Field(default_factory=list)
    salary_expectation: Optional[int] = None
    availability: Optional[str] = None
    
    # IA et matching
    ai_extracted_skills: List[str] = Field(default_factory=list)
    skill_confidence_scores: Dict[str, float] = Field(default_factory=dict)
    last_ai_analysis: Optional[datetime] = None
    
    class Settings:
        name = "candidates"
        indexes = [
            "name",
            "skills",
            "created_at",
            [("name", "text"), ("text", "text")],
            "email",
            "is_active"
        ]

class Offer(Document):
    """Modèle offre d'emploi pour MongoDB"""
    
    # Informations de base
    title: str = Field(..., min_length=3, max_length=200)
    company: str = Field(..., min_length=2, max_length=100)
    description: str = Field(..., min_length=10, max_length=20000)
    
    # Détails de l'offre
    location: Optional[str] = None
    remote_allowed: bool = False
    contract_type: Optional[str] = None  # CDI, CDD, Freelance, Stage
    experience_required: Optional[int] = Field(None, ge=0, le=20)
    
    # Compétences et exigences
    skills: List[str] = Field(default_factory=list)
    required_skills: List[str] = Field(default_factory=list)
    nice_to_have_skills: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    
    # Rémunération et avantages
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    currency: str = Field(default="EUR")
    benefits: List[str] = Field(default_factory=list)
    
    # Informations de contact
    contact_email: Optional[EmailStr] = None
    contact_phone: Optional[str] = None
    application_url: Optional[str] = None
    
    # Métadonnées
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    is_active: bool = True
    is_featured: bool = False
    source: str = Field(default="manual")
    
    # IA et matching
    ai_extracted_skills: List[str] = Field(default_factory=list)
    skill_importance_scores: Dict[str, float] = Field(default_factory=dict)
    last_ai_analysis: Optional[datetime] = None
    
    class Settings:
        name = "offers"
        indexes = [
            "title",
            "company",
            "skills",
            "created_at",
            [("title", "text"), ("description", "text")],
            "is_active",
            "expires_at"
        ]

class MatchResult(Document):
    """Modèle résultat de matching pour MongoDB"""
    
    # Références
    candidate_id: str = Field(..., index=True)
    offer_id: str = Field(..., index=True)
    
    # Scores de matching
    overall_score: float = Field(..., ge=0.0, le=1.0)
    skill_match_score: float = Field(..., ge=0.0, le=1.0)
    text_similarity_score: float = Field(..., ge=0.0, le=1.0)
    experience_match_score: float = Field(..., ge=0.0, le=1.0)
    
    # Détails du matching
    common_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    extra_skills: List[str] = Field(default_factory=list)
    match_level: MatchLevel
    
    # Recommandations
    recommendations: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    improvements: List[str] = Field(default_factory=list)
    
    # Métadonnées
    created_at: datetime = Field(default_factory=datetime.utcnow)
    algorithm_version: str = Field(default="2.0")
    processing_time_ms: Optional[int] = None
    
    # Feedback utilisateur
    user_feedback: Optional[str] = None
    feedback_score: Optional[int] = Field(None, ge=1, le=5)
    is_favorite: bool = False
    
    class Settings:
        name = "match_results"
        indexes = [
            "candidate_id",
            "offer_id",
            "overall_score",
            "created_at",
            [("candidate_id", 1), ("overall_score", -1)],
            [("offer_id", 1), ("overall_score", -1)]
        ]

class UserSession(Document):
    """Modèle session utilisateur pour MongoDB"""
    
    # Identifiants
    user_id: Optional[str] = None
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Informations de session
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    role: UserRole = UserRole.GUEST
    
    # Activité
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    actions_count: int = 0
    searches_performed: int = 0
    matches_viewed: int = 0
    
    # Préférences
    preferred_candidate_filters: Dict[str, Any] = Field(default_factory=dict)
    preferred_offer_filters: Dict[str, Any] = Field(default_factory=dict)
    
    # Métadonnées
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    is_active: bool = True
    
    class Settings:
        name = "user_sessions"
        indexes = [
            "user_id",
            "session_id",
            "last_activity",
            "is_active"
        ]

class AnalyticsEvent(Document):
    """Modèle événement d'analytics pour MongoDB"""
    
    # Type d'événement
    event_type: str = Field(..., index=True)  # search, match, view, click, etc.
    event_category: str = Field(default="user_action")
    
    # Contexte
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    candidate_id: Optional[str] = None
    offer_id: Optional[str] = None
    
    # Données de l'événement
    event_data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Métadonnées
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    class Settings:
        name = "analytics_events"
        indexes = [
            "event_type",
            "timestamp",
            "user_id",
            "session_id"
        ]

# Modèles Pydantic pour les requêtes API
class CandidateCreate(BaseModel):
    """Modèle pour créer un candidat"""
    name: str = Field(..., min_length=2, max_length=100)
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    title: Optional[str] = None
    experience_years: Optional[int] = Field(None, ge=0, le=50)
    text: str = Field(default="", max_length=10000)
    skills: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    resume_url: Optional[str] = None
    portfolio_url: Optional[str] = None
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None
    preferred_locations: List[str] = Field(default_factory=list)
    salary_expectation: Optional[int] = None
    availability: Optional[str] = None

class CandidateUpdate(BaseModel):
    """Modèle pour mettre à jour un candidat"""
    name: Optional[str] = Field(None, min_length=2, max_length=100)
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    title: Optional[str] = None
    experience_years: Optional[int] = Field(None, ge=0, le=50)
    text: Optional[str] = Field(None, max_length=10000)
    skills: Optional[List[str]] = None
    languages: Optional[List[str]] = None
    resume_url: Optional[str] = None
    portfolio_url: Optional[str] = None
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None
    preferred_locations: Optional[List[str]] = None
    salary_expectation: Optional[int] = None
    availability: Optional[str] = None
    is_active: Optional[bool] = None

class OfferCreate(BaseModel):
    """Modèle pour créer une offre"""
    title: str = Field(..., min_length=3, max_length=200)
    company: str = Field(..., min_length=2, max_length=100)
    description: str = Field(..., min_length=10, max_length=20000)
    location: Optional[str] = None
    remote_allowed: bool = False
    contract_type: Optional[str] = None
    experience_required: Optional[int] = Field(None, ge=0, le=20)
    skills: List[str] = Field(default_factory=list)
    required_skills: List[str] = Field(default_factory=list)
    nice_to_have_skills: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    currency: str = Field(default="EUR")
    benefits: List[str] = Field(default_factory=list)
    contact_email: Optional[EmailStr] = None
    contact_phone: Optional[str] = None
    application_url: Optional[str] = None
    expires_at: Optional[datetime] = None

class OfferUpdate(BaseModel):
    """Modèle pour mettre à jour une offre"""
    title: Optional[str] = Field(None, min_length=3, max_length=200)
    company: Optional[str] = Field(None, min_length=2, max_length=100)
    description: Optional[str] = Field(None, min_length=10, max_length=20000)
    location: Optional[str] = None
    remote_allowed: Optional[bool] = None
    contract_type: Optional[str] = None
    experience_required: Optional[int] = Field(None, ge=0, le=20)
    skills: Optional[List[str]] = None
    required_skills: Optional[List[str]] = None
    nice_to_have_skills: Optional[List[str]] = None
    languages: Optional[List[str]] = None
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    currency: Optional[str] = None
    benefits: Optional[List[str]] = None
    contact_email: Optional[EmailStr] = None
    contact_phone: Optional[str] = None
    application_url: Optional[str] = None
    expires_at: Optional[datetime] = None
    is_active: Optional[bool] = None
    is_featured: Optional[bool] = None

class MatchRequest(BaseModel):
    """Modèle pour une requête de matching"""
    candidate_id: str
    top_k: int = Field(default=5, ge=1, le=50)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    filters: Optional[Dict[str, Any]] = None
    algorithm_version: Optional[str] = None

class MatchResponse(BaseModel):
    """Modèle pour la réponse de matching"""
    candidate: Candidate
    matches: List[MatchResult]
    total_matches: int
    algorithm_version: str
    processing_time_ms: int
    filters_applied: Dict[str, Any]
