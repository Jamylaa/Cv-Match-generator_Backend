from sklearn.metrics.pairwise import cosine_similarity
from .embeddings import embed_text
from .extractor import extract_skills_from_text
import numpy as np

def skill_overlap_score(skills_a, skills_b):
    set_a = set([s.lower() for s in skills_a])
    set_b = set([s.lower() for s in skills_b])
    if not set_a and not set_b:
        return 0.0
    inter = set_a.intersection(set_b)
    union = set_a.union(set_b)
    return len(inter) / max(len(union), 1)

def semantic_similarity(text_a, text_b):
    vec_a = embed_text(text_a)
    vec_b = embed_text(text_b)
    sim = cosine_similarity([vec_a], [vec_b])[0][0]
    return float(sim)

def combined_score(candidate_text, candidate_skills, offer_text, offer_skills, weights=(0.6, 0.4)):
    # skills overlap weight first, semantic similarity second
    s_overlap = skill_overlap_score(candidate_skills, offer_skills)
    s_sem = semantic_similarity(candidate_text, offer_text)
    score = weights[0] * s_overlap + weights[1] * s_sem
    # normalize to 0-100
    return round(float(score) * 100, 2), {
        "skill_overlap": round(s_overlap * 100, 2),
        "semantic_similarity": round(s_sem * 100, 2)
    }
