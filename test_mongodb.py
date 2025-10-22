"""
Script de test pour vérifier la connexion et les opérations MongoDB
"""

import asyncio
import sys
from pathlib import Path

# Ajouter le répertoire du projet au path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_mongodb_connection():
    """Test de connexion MongoDB"""
    try:
        from app.mongodb_service import mongodb_service
        from app.mongodb_models import Candidate, Offer
        
        print("🔍 Test de connexion MongoDB...")
        
        # Connexion
        await mongodb_service.connect()
        print(" Connexion MongoDB réussie!")
        
        # Test de création d'un candidat
        print(" Test de création d'un candidat...")
        candidate_data = {
            "name": "Test Candidate",
            "email": "test@example.com",
            "text": "Développeur Python expérimenté",
            "skills": ["Python", "Django", "FastAPI"],
            "location": "Paris"
        }
        
        candidate = await mongodb_service.create_candidate(candidate_data)
        print(f" Candidat créé: {candidate.id}")
        
        # Test de création d'une offre
        print(" Test de création d'une offre...")
        offer_data = {
            "title": "Développeur Python Senior",
            "company": "Tech Corp",
            "description": "Recherche développeur Python expérimenté pour rejoindre notre équipe",
            "skills": ["Python", "Django", "PostgreSQL"],
            "location": "Paris",
            "salary_min": 50000,
            "salary_max": 70000
        }
        
        offer = await mongodb_service.create_offer(offer_data)
        print(f" Offre créée: {offer.id}")
        
        # Test de récupération
        print(" Test de récupération des données...")
        candidates = await mongodb_service.get_candidates(limit=10)
        offers = await mongodb_service.get_offers(limit=10)
        
        print(f" {len(candidates)} candidats trouvés")
        print(f" {len(offers)} offres trouvées")
        
        # Test de matching
        print("🔗 Test de matching...")
        from app.api_ai_simple import vector_service
        
        candidate_text = f"{candidate.text} {' '.join(candidate.skills)}"
        offer_text = f"{offer.description} {' '.join(offer.skills)}"
        
        similarity = vector_service.calculate_similarity(candidate_text, offer_text)
        print(f" Similarité calculée: {similarity:.3f}")
        
        # Test de statistiques
        print("  Test des statistiques...")
        stats = await mongodb_service.get_matching_statistics()
        print(f" Statistiques: {stats}")
        
        print(" Tous les tests MongoDB ont réussi!")
        
    except Exception as e:
        print(f"  Erreur lors des tests MongoDB: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Déconnexion
        try:
            await mongodb_service.disconnect()
            print("  Déconnexion MongoDB")
        except:
            pass

if __name__ == "__main__":
    asyncio.run(test_mongodb_connection())
