from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api_simple import router as api_router
from .api_ai_simple import router as ai_router
from .api_mongodb import router as mongodb_router
from .mongodb_service import mongodb_service
import logging
import os

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI-Powered Matching Intelligent API",
    description="API avancée avec intégration IA pour le matching intelligent de candidats et offres",
    version="2.0.0"
)

# Autorise le frontend (React )
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:4200","http://192.168.56.1:3000"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclusion des routeurs
app.include_router(api_router, prefix="/api", tags=["Basic API"])
app.include_router(ai_router, prefix="/api/ai", tags=["AI Features"])
app.include_router(mongodb_router, prefix="/api/mongodb", tags=["MongoDB API"])

@app.on_event("startup")
async def startup_event():
    """Démarre les services au lancement de l'application"""
    logger.info("Starting AI-Powered Matching API with MongoDB...")
    
    # Connexion à MongoDB
    try:
        mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
        database_name = os.getenv("MONGODB_DATABASE", "matching_ai")
        
        await mongodb_service.connect(mongodb_url, database_name)
        logger.info(f"Connected to MongoDB: {database_name}")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        logger.warning("API will run without MongoDB (using in-memory storage)")
    
    logger.info("AI services initialized")

@app.on_event("shutdown")
async def shutdown_event():
    """Arrête les services à la fermeture de l'application"""
    logger.info("Shutting down AI-Powered Matching API...")
    
    # Déconnexion de MongoDB
    try:
        await mongodb_service.disconnect()
        logger.info("Disconnected from MongoDB")
    except Exception as e:
        logger.error(f"Error disconnecting from MongoDB: {e}")
    
    logger.info("AI services stopped")

@app.get("/")
def root():
    return {
        "message": "AI-Powered Matching Intelligent API running ✅",
        "version": "2.0.0",
        "features": [
            "Advanced AI matching with vector embeddings",
            "MongoDB database integration with Beanie ODM",
            "Anomaly detection and data quality monitoring",
            "Automated testing with AI validation",
            "Real-time performance monitoring",
            "Personalized recommendations",
            "External AI API integration (OpenAI, Hugging Face)",
            "Persistent data storage and analytics"
        ],
        "endpoints": {
            "basic": "/api/",
            "ai_features": "/api/ai/",
            "mongodb_api": "/api/mongodb/",
            "monitoring": "/api/ai/health",
            "documentation": "/docs"
        },
        "database": {
            "type": "MongoDB",
            "status": "connected" if mongodb_service.is_connected else "disconnected",
            "collections": [
                "candidates", "offers", "match_results", 
                "ai_skill_extractions", "anomaly_detections", 
                "system_metrics", "ai_recommendations"
            ]
        }
    }

@app.get("/health")
def health_check():
    """Vérification de l'état de santé de l'API"""
    return {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z",
        "services": {
            "api": "running",
            "ai_services": "running",
            "mongodb": "connected" if mongodb_service.is_connected else "disconnected",
            "vector_matching": "available",
            "anomaly_detection": "available"
        }
    }
