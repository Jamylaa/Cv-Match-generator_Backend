"""
Script de démarrage simple pour le projet Backend-IA
Usage: python start_server.py
"""
import uvicorn
import sys
from pathlib import Path

# Add the project directory to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    print("Démarrage du serveur FastAPI...")
    print("URL: http://localhost:8000")
    print("Documentation: http://localhost:8000/docs")
    print("Arrêt: Ctrl+C")
    print("-" * 50)
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )