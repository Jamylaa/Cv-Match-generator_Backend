"""
Script de démarrage pour l'API avec MongoDB
"""
import asyncio
import uvicorn
import os
import sys
from pathlib import Path

# Ajouter le répertoire du projet au path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def main():
    """Fonction principale de démarrage"""
    print(" Démarrage de l'API Matching Intelligent avec MongoDB...")
    
    # Vérifier si MongoDB est disponible
    try:
        from app.mongodb_service import mongodb_service
        
        mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
        database_name = os.getenv("MONGODB_DATABASE", "matching_ai")
        
        print(f" Connexion à MongoDB: {mongodb_url}")
        print(f"  Base de données: {database_name}")
        
        await mongodb_service.connect(mongodb_url, database_name)
        print("✅ MongoDB connecté avec succès!")
        
    except Exception as e:
        print(f"⚠️  Erreur de connexion MongoDB: {e}")
        print(" L'API démarrera avec le stockage en mémoire")
    
    # Démarrer le serveur
    print(" Démarrage du serveur FastAPI...")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    asyncio.run(main())
