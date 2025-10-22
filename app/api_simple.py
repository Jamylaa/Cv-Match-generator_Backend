from fastapi import APIRouter, UploadFile, File, Form
from typing import List
from .models import CandidateCreate, Candidate, OfferCreate, Offer
from .storage import store
import pdfplumber
import io
import re

router = APIRouter()

# Version simplifiée sans spaCy pour compatibilité Python 3.12
def extract_skills_from_text(text: str, name: str = "") -> List[str]:
    """Extraction améliorée de compétences basée sur des mots-clés communs"""
    found_skills = []
    
    # Si pas de texte, essayer d'extraire du nom
    if not text and name:
        text = name
    
    if not text:
        return []
    
    # Liste de compétences communes
    common_skills = [
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
    
    text_lower = text.lower()
    
    # Extraction des compétences
    for skill in common_skills:
        if skill in text_lower:
            found_skills.append(skill.title())
    
    # Si aucune compétence trouvée, essayer des patterns plus génériques
    if not found_skills:
        # Patterns basés sur des mots-clés dans le nom ou texte
        generic_patterns = {
            "développeur": ["Programming", "Software Development"],
            "developer": ["Programming", "Software Development"],
            "ingénieur": ["Engineering", "Technical Skills"],
            "engineer": ["Engineering", "Technical Skills"],
            "analyst": ["Analysis", "Problem Solving"],
            "analyste": ["Analysis", "Problem Solving"],
            "manager": ["Management", "Leadership"],
            "chef": ["Management", "Leadership"],
            "senior": ["Senior Level", "Experience"],
            "junior": ["Junior Level", "Learning"],
            "full stack": ["Full Stack Development", "Frontend", "Backend"],
            "frontend": ["Frontend Development", "UI/UX", "JavaScript"],
            "backend": ["Backend Development", "API", "Database"],
            "data": ["Data Analysis", "Data Science", "Statistics"],
            "web": ["Web Development", "HTML", "CSS", "JavaScript"],
            "mobile": ["Mobile Development", "iOS", "Android"],
            "devops": ["DevOps", "CI/CD", "Docker", "Kubernetes"],
            "cloud": ["Cloud Computing", "AWS", "Azure", "GCP"],
            "ai": ["Artificial Intelligence", "Machine Learning", "AI"],
            "intelligence": ["Artificial Intelligence", "Machine Learning", "AI"]
        }
        
        for pattern, skills in generic_patterns.items():
            if pattern in text_lower:
                found_skills.extend(skills)
    
    # Si toujours aucune compétence, ajouter des compétences génériques basées sur le contexte
    if not found_skills:
        found_skills = ["General Skills", "Problem Solving", "Communication"]
    
    return list(set(found_skills))

def combined_score(candidate_text: str, candidate_skills: List[str], 
                  offer_text: str, offer_skills: List[str]) -> tuple:
    """Algorithme de matching amélioré avec scoring sophistiqué"""
    
    # Normalisation des compétences
    candidate_skills_lower = [s.lower().strip() for s in candidate_skills if s.strip()]
    offer_skills_lower = [s.lower().strip() for s in offer_skills if s.strip()]
    
    # 1. Score des compétences exactes
    common_skills = set(candidate_skills_lower).intersection(set(offer_skills_lower))
    exact_skill_score = len(common_skills) / max(len(candidate_skills_lower), len(offer_skills_lower)) if candidate_skills_lower and offer_skills_lower else 0.0
    
    # 2. Score des compétences partielles (similarité)
    partial_skill_score = 0.0
    if candidate_skills_lower and offer_skills_lower:
        partial_matches = 0
        for c_skill in candidate_skills_lower:
            for o_skill in offer_skills_lower:
                if c_skill in o_skill or o_skill in c_skill:
                    partial_matches += 0.5
        partial_skill_score = min(partial_matches / max(len(candidate_skills_lower), len(offer_skills_lower)), 1.0)
    
    # 3. Score de similarité textuelle amélioré
    text_similarity = 0.0
    if candidate_text and offer_text:
        # Mots-clés techniques importants
        tech_keywords = [
            "react", "angular", "vue", "python", "java", "javascript", "typescript",
            "nodejs", "django", "flask", "spring", "sql", "mongodb", "docker",
            "aws", "azure", "git", "agile", "scrum", "testing", "tdd", "bdd"
        ]
        
        candidate_text_lower = candidate_text.lower()
        offer_text_lower = offer_text.lower()
        
        # Score basé sur les mots-clés techniques communs
        candidate_tech = [kw for kw in tech_keywords if kw in candidate_text_lower]
        offer_tech = [kw for kw in tech_keywords if kw in offer_text_lower]
        common_tech = set(candidate_tech).intersection(set(offer_tech))
        tech_score = len(common_tech) / max(len(candidate_tech), len(offer_tech)) if candidate_tech or offer_tech else 0.0
        
        # Score basé sur les mots communs
        candidate_words = set(candidate_text_lower.split())
        offer_words = set(offer_text_lower.split())
        common_words = candidate_words.intersection(offer_words)
        word_score = len(common_words) / max(len(candidate_words), len(offer_words)) if candidate_words or offer_words else 0.0
        
        text_similarity = (tech_score * 0.7) + (word_score * 0.3)
    
    # 4. Score de couverture des compétences requises
    coverage_score = 0.0
    if offer_skills_lower:
        covered_skills = len(common_skills)
        coverage_score = covered_skills / len(offer_skills_lower)
    
    # 5. Score final pondéré
    skill_score = (exact_skill_score * 0.6) + (partial_skill_score * 0.4)
    final_score = (skill_score * 0.5) + (text_similarity * 0.3) + (coverage_score * 0.2)
    
    # 6. Bonus pour les compétences rares/importantes
    rare_skills = ["kubernetes", "microservices", "graphql", "tensorflow", "pytorch", "elasticsearch"]
    rare_bonus = 0.0
    for skill in common_skills:
        if skill in rare_skills:
            rare_bonus += 0.1
    
    final_score = min(final_score + rare_bonus, 1.0)
    
    return final_score, {
        "exact_skill_match": exact_skill_score,
        "partial_skill_match": partial_skill_score,
        "text_similarity": text_similarity,
        "coverage_score": coverage_score,
        "rare_skills_bonus": rare_bonus,
        "total_skills_candidate": len(candidate_skills_lower),
        "total_skills_offer": len(offer_skills_lower),
        "common_skills_count": len(common_skills)
    }

# helpers
def read_pdf_bytes(file_bytes: bytes):
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        pages = pdf.pages
        for p in pages:
            try:
                text += p.extract_text() or ""
            except Exception:
                pass
    return text

# Candidates
@router.post("/candidates", response_model=Candidate)
def create_candidate(payload: CandidateCreate):
    skills = payload.skills or extract_skills_from_text(payload.text or "", payload.name)
    cand = Candidate(name=payload.name, text=payload.text or "", skills=skills)
    store.add_candidate(cand.dict())
    return cand

@router.post("/candidates/upload", response_model=Candidate)
async def upload_candidate(name: str = Form(...), file: UploadFile = File(...)):
    data = await file.read()
    text = ""
    if file.filename.lower().endswith(".pdf"):
        text = read_pdf_bytes(data)
    else:
        text = data.decode("utf-8", errors="ignore")
    skills = extract_skills_from_text(text, name)
    cand = Candidate(name=name, text=text, skills=skills)
    store.add_candidate(cand.dict())
    return cand

@router.get("/candidates")
def list_candidates():
    candidates = store.list_candidates()
    return candidates

@router.get("/candidates/{cid}")
def get_candidate(cid: str):
    c = store.get_candidate(cid)
    if not c:
        return {"error": "not found"}
    return c

@router.post("/candidates/{cid}/update-skills")
def update_candidate_skills(cid: str):
    """Met à jour les compétences d'un candidat existant"""
    c = store.get_candidate(cid)
    if not c:
        return {"error": "candidate not found"}
    
    # Extraire les compétences du nom et du texte
    new_skills = extract_skills_from_text(c.get("text", ""), c.get("name", ""))
    
    # Mettre à jour le candidat
    c["skills"] = new_skills
    store.add_candidate(c)  # Sauvegarder les modifications
    
    return {"message": "Skills updated", "candidate": c, "new_skills": new_skills}

# Offers
@router.post("/offers", response_model=Offer)
def create_offer(payload: OfferCreate):
    skills = payload.skills or extract_skills_from_text(payload.description or "")
    off = Offer(title=payload.title, description=payload.description, skills=skills)
    store.add_offer(off.dict())
    return off

@router.get("/offers")
def list_offers():
    offers = store.list_offers()
    return offers

@router.get("/offers/{oid}")
def get_offer(oid: str):
    o = store.get_offer(oid)
    if not o:
        return {"error": "not found"}
    return o

# Match: match one candidate to all offers, return top-k
@router.get("/match/{candidate_id}")
def match_candidate(candidate_id: str, top_k: int = 5, min_score: float = 0.0):
    candidate = store.get_candidate(candidate_id)
    if not candidate:
        return {"error": "candidate not found"}
    
    offers = store.list_offers()
    if not offers:
        return {
            "candidate": candidate,
            "matches": [],
            "summary": {
                "total_offers": 0,
                "matches_found": 0,
                "average_score": 0.0,
                "best_score": 0.0
            }
        }
    
    results = []
    for o in offers:
        score, breakdown = combined_score(
            candidate_text=candidate.get("text",""),
            candidate_skills=candidate.get("skills", []),
            offer_text=o.get("description",""),
            offer_skills=o.get("skills", [])
        )
        
        # Calcul des compétences communes et manquantes
        candidate_skills_lower = [s.lower().strip() for s in candidate.get("skills", []) if s.strip()]
        offer_skills_lower = [s.lower().strip() for s in o.get("skills", []) if s.strip()]
        common_skills = list(set(candidate_skills_lower).intersection(set(offer_skills_lower)))
        missing_skills = list(set(offer_skills_lower) - set(candidate_skills_lower))
        extra_skills = list(set(candidate_skills_lower) - set(offer_skills_lower))
        
        # Classification du niveau de match
        match_level = "Excellent" if score >= 0.8 else "Bon" if score >= 0.6 else "Moyen" if score >= 0.4 else "Faible"
        
        # Recommandations
        recommendations = []
        if missing_skills:
            recommendations.append(f"Compétences à acquérir: {', '.join(missing_skills[:3])}")
        if score < 0.5:
            recommendations.append("Profil partiellement adapté - formation recommandée")
        if len(common_skills) > 0:
            recommendations.append(f"Points forts: {', '.join(common_skills[:3])}")
        
        results.append({
            "offer": o,
            "score": round(score, 3),
            "score_percentage": round(score * 100, 1),
            "match_level": match_level,
            "breakdown": breakdown,
            "common_skills": common_skills,
            "missing_skills": missing_skills[:5],  # Limiter à 5 compétences manquantes
            "extra_skills": extra_skills[:5],      # Limiter à 5 compétences supplémentaires
            "recommendations": recommendations,
            "compatibility": {
                "skills_coverage": round(breakdown.get("coverage_score", 0) * 100, 1),
                "text_similarity": round(breakdown.get("text_similarity", 0) * 100, 1),
                "exact_matches": breakdown.get("common_skills_count", 0)
            }
        })
    
    # Filtrage par score minimum et tri
    filtered_results = [r for r in results if r["score"] >= min_score]
    results_sorted = sorted(filtered_results, key=lambda r: r["score"], reverse=True)
    
    # Statistiques de résumé
    scores = [r["score"] for r in results_sorted]
    summary = {
        "total_offers": len(offers),
        "matches_found": len(results_sorted),
        "average_score": round(sum(scores) / len(scores), 3) if scores else 0.0,
        "best_score": round(max(scores), 3) if scores else 0.0,
        "score_distribution": {
            "excellent": len([s for s in scores if s >= 0.8]),
            "good": len([s for s in scores if 0.6 <= s < 0.8]),
            "average": len([s for s in scores if 0.4 <= s < 0.6]),
            "poor": len([s for s in scores if s < 0.4])
        }
    }
    
    return {
        "candidate": candidate,
        "matches": results_sorted[:top_k],
        "summary": summary,
        "filters_applied": {
            "top_k": top_k,
            "min_score": min_score
        }
    }

# Endpoint pour obtenir des statistiques et insights
@router.get("/stats")
def get_stats():
    candidates = store.list_candidates()
    offers = store.list_offers()
    
    # Statistiques des candidats
    all_candidate_skills = []
    for c in candidates:
        all_candidate_skills.extend([s.lower() for s in c.get("skills", [])])
    
    # Statistiques des offres
    all_offer_skills = []
    for o in offers:
        all_offer_skills.extend([s.lower() for s in o.get("skills", [])])
    
    # Compétences les plus demandées
    from collections import Counter
    offer_skill_counts = Counter(all_offer_skills)
    candidate_skill_counts = Counter(all_candidate_skills)
    
    # Compétences en pénurie (demandées mais peu disponibles)
    shortage_skills = []
    for skill, demand in offer_skill_counts.items():
        supply = candidate_skill_counts.get(skill, 0)
        if demand > supply:
            shortage_skills.append({
                "skill": skill,
                "demand": demand,
                "supply": supply,
                "shortage": demand - supply
            })
    
    shortage_skills.sort(key=lambda x: x["shortage"], reverse=True)
    
    return {
        "overview": {
            "total_candidates": len(candidates),
            "total_offers": len(offers),
            "total_skills_candidates": len(set(all_candidate_skills)),
            "total_skills_offers": len(set(all_offer_skills))
        },
        "top_demanded_skills": [
            {"skill": skill, "count": count} 
            for skill, count in offer_skill_counts.most_common(10)
        ],
        "top_available_skills": [
            {"skill": skill, "count": count} 
            for skill, count in candidate_skill_counts.most_common(10)
        ],
        "skills_shortage": shortage_skills[:10],
        "market_insights": {
            "most_common_skills": list(set(all_candidate_skills).intersection(set(all_offer_skills)))[:10],
            "unique_candidate_skills": list(set(all_candidate_skills) - set(all_offer_skills))[:10],
            "unique_offer_skills": list(set(all_offer_skills) - set(all_candidate_skills))[:10]
        }
    }

# Endpoint de test pour ajouter des candidats avec compétences
@router.post("/test/add-sample-candidate")
def add_sample_candidate():
    sample_candidate = Candidate(
        name="Développeur Full Stack",
        text="Développeur expérimenté en React, Next.js, Python, Django. Connaissance en SQL, Docker, AWS. Expérience en agile et scrum.",
        skills=["React", "Next.js", "Python", "Django", "SQL", "Docker", "AWS", "Agile", "Scrum"]
    )
    store.add_candidate(sample_candidate.dict())
    return {"message": "Candidat de test ajouté", "candidate": sample_candidate}
